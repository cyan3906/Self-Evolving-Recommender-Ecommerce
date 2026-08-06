"""
LangChain Agent 初始化与调用封装。

功能：
1. 使用 LangChain create_agent 自动完成：
   模型 -> 工具调用 -> ToolMessage -> 模型总结；
2. 支持同步 invoke 和异步 ainvoke；
3. 支持 thread_id 级别的多轮对话记忆；
4. 支持静态系统提示词或动态 PromptBuilder；
5. 使用 ModelCallLimitMiddleware 限制单次请求的模型调用次数；
6. 使用 TurnFinalizer 统一处理最终文本、异常、统计信息和结束 Hook。

依赖：
    pip install -U langchain langchain-openai langgraph python-dotenv
"""

from __future__ import annotations

import logging
import sys

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRequest,
    dynamic_prompt,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver


# ---------------------------------------------------------------------------
# 项目路径与项目配置
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import Settings  # noqa: E402

try:
    from .turn_finalizer import (  # noqa: E402
        LangChainTurnResult,
        TurnFinalizer,
        TurnFinalizerConfig,
        TurnHook,
    )
except ImportError:
    from turn_finalizer import (  # type: ignore[no-redef]  # noqa: E402
        LangChainTurnResult,
        TurnFinalizer,
        TurnFinalizerConfig,
        TurnHook,
    )


settings = Settings()
logger = logging.getLogger(__name__)


# create_agent 可以直接接收：
# 1. BaseTool；
# 2. 使用 @tool 装饰的函数；
# 3. 普通 Python Callable；
# 4. LangChain 支持的工具字典。
AgentTool = BaseTool | Callable[..., Any] | dict[str, Any]

# 动态系统提示词函数接收每次 invoke 传入的 runtime_context。
SystemPromptFactory = Callable[[Mapping[str, Any]], str]
SystemPrompt = str | SystemPromptFactory


@dataclass(slots=True)
class LangChainAgentConfig:
    """LangChain Agent 配置。"""

    model_name: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = 0.0
    timeout_seconds: float = 120.0
    max_retries: int = 2

    # 单次用户请求中最多允许调用模型的次数。
    #
    # 一次正常工具流程通常至少调用两次模型：
    #   第一次：模型决定调用工具；
    #   第二次：模型根据工具结果生成最终回答。
    max_model_calls_per_run: int = 8

    # True：
    #   相同 thread_id 使用 LangGraph checkpointer 保存多轮历史。
    #
    # False：
    #   不启用 checkpointer，每次请求相互独立。
    enable_memory: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str):
            raise TypeError("model_name must be a string")

        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")

        if self.timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than 0"
            )

        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        if self.max_model_calls_per_run <= 0:
            raise ValueError(
                "max_model_calls_per_run must be greater than 0"
            )


@dataclass(slots=True)
class AgentRuntimeContext:
    """
    单次调用传给动态 PromptBuilder 的运行时上下文。

    示例：
        AgentRuntimeContext(
            values={
                "user_id": "user-001",
                "request_id": "request-001",
                "scene": "product_recommendation",
            }
        )
    """

    values: dict[str, Any] = field(default_factory=dict)


class LangChainAgent:
    """
    基于 LangChain create_agent 的 Agent 封装。

    create_agent 负责：
        1. 调用模型；
        2. 读取 AIMessage.tool_calls；
        3. 执行对应工具；
        4. 将执行结果转换为 ToolMessage；
        5. 将 ToolMessage 重新发送给模型；
        6. 循环直到模型生成最终回答。

    TurnFinalizer 负责：
        1. 提取最终 AI 文本；
        2. 统一成功和失败结构；
        3. 区分完整历史与当前 turn；
        4. 统计模型调用、工具调用和 Token；
        5. 执行持久化、日志、记忆提取等结束 Hook。
    """

    def __init__(
        self,
        config: LangChainAgentConfig,
        tools: Sequence[AgentTool] | None = None,
        system_prompt: SystemPrompt = "You are a helpful assistant.",
        *,
        turn_finalizer: TurnFinalizer | None = None,
        finalizer_hooks: Sequence[TurnHook] | None = None,
    ) -> None:
        """
        初始化 Agent。

        Args:
            config:
                模型、超时、重试、记忆等配置。

            tools:
                Agent 可以调用的工具列表。

            system_prompt:
                可以是静态字符串，也可以是：
                    Callable[[Mapping[str, Any]], str]

                动态函数会在每次模型调用前读取 runtime_context。

            turn_finalizer:
                自定义 TurnFinalizer。未传入时自动创建默认实例。

            finalizer_hooks:
                Turn 结束后执行的 Hook，例如：
                - 保存数据库；
                - 写日志；
                - 提取长期记忆；
                - 收集 A/B 测试指标。

                当已经显式传入 turn_finalizer 时，不应再传
                finalizer_hooks，Hook 应直接配置到自定义 finalizer 中。
        """
        if turn_finalizer is not None and finalizer_hooks:
            raise ValueError(
                "turn_finalizer and finalizer_hooks cannot be "
                "configured at the same time"
            )

        self.config = config
        self.tools = list(tools or [])
        self.system_prompt = system_prompt

        self.turn_finalizer = (
            turn_finalizer
            if turn_finalizer is not None
            else TurnFinalizer(
                config=TurnFinalizerConfig(
                    empty_response_text=(
                        "Agent 未生成可展示的最终回答。"
                    ),
                    failure_response_text=(
                        "Agent 执行失败，请稍后重试。"
                    ),
                    expose_error_to_user=False,
                    raise_hook_error=False,
                ),
                hooks=finalizer_hooks,
            )
        )

        self.model = self._build_model()

        self.checkpointer = (
            InMemorySaver()
            if self.config.enable_memory
            else None
        )

        middleware = self._build_middleware()

        create_agent_kwargs: dict[str, Any] = {
            "model": self.model,
            "tools": self.tools,
            "middleware": middleware,
            "checkpointer": self.checkpointer,
            "context_schema": AgentRuntimeContext,
            "name": "project_agent",
        }

        # 静态 system prompt 直接传给 create_agent。
        if isinstance(self.system_prompt, str):
            create_agent_kwargs["system_prompt"] = (
                self.system_prompt
            )

        self.agent = create_agent(**create_agent_kwargs)

    def _build_model(self) -> ChatOpenAI:
        """创建 ChatOpenAI 模型实例。"""
        model_kwargs: dict[str, Any] = {
            "model": self.config.model_name,
            "timeout": self.config.timeout_seconds,
            "max_retries": self.config.max_retries,
        }

        if self.config.api_key:
            model_kwargs["api_key"] = self.config.api_key

        if self.config.base_url:
            model_kwargs["base_url"] = self.config.base_url

        if self.config.temperature is not None:
            model_kwargs["temperature"] = (
                self.config.temperature
            )

        return ChatOpenAI(**model_kwargs)

    def _build_middleware(self) -> list[Any]:
        """
        创建 Agent middleware。

        顺序：
            1. 动态 system prompt middleware；
            2. 模型调用次数限制 middleware。
        """
        middleware: list[Any] = []

        if not isinstance(self.system_prompt, str):
            prompt_factory = self.system_prompt

            @dynamic_prompt
            def runtime_system_prompt(
                request: ModelRequest,
            ) -> str:
                runtime_context: Mapping[str, Any] = {}

                if request.runtime is not None:
                    context = request.runtime.context

                    if isinstance(
                        context,
                        AgentRuntimeContext,
                    ):
                        runtime_context = context.values

                rendered_prompt = prompt_factory(
                    runtime_context
                )

                if not isinstance(rendered_prompt, str):
                    raise TypeError(
                        "system prompt factory must return "
                        "a string"
                    )

                if not rendered_prompt.strip():
                    raise ValueError(
                        "system prompt factory returned "
                        "an empty string"
                    )

                return rendered_prompt

            middleware.append(runtime_system_prompt)

        middleware.append(
            ModelCallLimitMiddleware(
                run_limit=(
                    self.config.max_model_calls_per_run
                ),
                exit_behavior="error",
            )
        )

        return middleware

    @staticmethod
    def new_thread_id() -> str:
        """创建新的 thread_id。"""
        return str(uuid4())

    @staticmethod
    def _validate_user_input(user_input: str) -> str:
        """校验并规范化用户输入。"""
        if not isinstance(user_input, str):
            raise TypeError("user_input must be a string")

        normalized = user_input.strip()

        if not normalized:
            raise ValueError("user_input cannot be empty")

        return normalized

    @staticmethod
    def _validate_thread_id(thread_id: str) -> str:
        """校验并规范化 thread_id。"""
        if not isinstance(thread_id, str):
            raise TypeError("thread_id must be a string")

        normalized = thread_id.strip()

        if not normalized:
            raise ValueError("thread_id cannot be empty")

        return normalized

    def _build_config(
        self,
        thread_id: str,
    ) -> dict[str, Any]:
        """
        构建 LangGraph RunnableConfig。

        thread_id 是 checkpointer 区分不同会话的关键字段。
        """
        normalized_thread_id = self._validate_thread_id(
            thread_id
        )

        return {
            "configurable": {
                "thread_id": normalized_thread_id,
            }
        }

    @staticmethod
    def _build_metadata(
        *,
        user_input: str,
        runtime_context: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """
        创建 TurnFinalizer 附加元数据。

        metadata 不参与模型推理，只提供给：
        - 日志；
        - 持久化 Hook；
        - 指标收集；
        - 调试与链路追踪。
        """
        return {
            "user_input": user_input,
            "runtime_context": dict(
                runtime_context or {}
            ),
        }

    def invoke(
        self,
        user_input: str,
        *,
        thread_id: str,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> LangChainTurnResult:
        """
        同步执行一次 Agent turn。

        Agent 执行成功或失败都会返回 LangChainTurnResult。
        模型异常、工具异常、超过模型调用上限等错误不会再被
        转换为另一套返回结构，而是统一写入：

            result.completed
            result.error
            result.content
        """
        normalized_input = self._validate_user_input(
            user_input
        )
        normalized_thread_id = self._validate_thread_id(
            thread_id
        )

        runtime_values = dict(runtime_context or {})

        metadata = self._build_metadata(
            user_input=normalized_input,
            runtime_context=runtime_values,
        )

        try:
            raw_result = self.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": normalized_input,
                        }
                    ]
                },
                config=self._build_config(
                    normalized_thread_id
                ),
                context=AgentRuntimeContext(
                    values=runtime_values
                ),
            )

        except Exception as exc:
            logger.exception(
                "LangChain agent execution failed: "
                "thread_id=%s",
                normalized_thread_id,
            )

            return self.turn_finalizer.finalize(
                thread_id=normalized_thread_id,
                error=exc,
                metadata=metadata,
            )

        return self.turn_finalizer.finalize(
            thread_id=normalized_thread_id,
            raw_result=raw_result,
            metadata=metadata,
        )

    async def ainvoke(
        self,
        user_input: str,
        *,
        thread_id: str,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> LangChainTurnResult:
        """
        异步执行一次 Agent turn，适合 FastAPI。

        使用 afinalize 的原因：
        - 支持同步结束 Hook；
        - 支持异步结束 Hook；
        - 不阻断 Agent 的统一返回结构。
        """
        normalized_input = self._validate_user_input(
            user_input
        )
        normalized_thread_id = self._validate_thread_id(
            thread_id
        )

        runtime_values = dict(runtime_context or {})

        metadata = self._build_metadata(
            user_input=normalized_input,
            runtime_context=runtime_values,
        )

        try:
            raw_result = await self.agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": normalized_input,
                        }
                    ]
                },
                config=self._build_config(
                    normalized_thread_id
                ),
                context=AgentRuntimeContext(
                    values=runtime_values
                ),
            )

        except Exception as exc:
            logger.exception(
                "Async LangChain agent execution failed: "
                "thread_id=%s",
                normalized_thread_id,
            )

            return await self.turn_finalizer.afinalize(
                thread_id=normalized_thread_id,
                error=exc,
                metadata=metadata,
            )

        return await self.turn_finalizer.afinalize(
            thread_id=normalized_thread_id,
            raw_result=raw_result,
            metadata=metadata,
        )


def build_agent(
    *,
    tools: Sequence[AgentTool],
    prompt_builder: Any | None = None,
    finalizer_hooks: Sequence[TurnHook] | None = None,
) -> LangChainAgent:
    """
    构建项目 Agent。

    prompt_builder 为 None：
        使用默认静态系统提示词。

    prompt_builder 不为 None：
        每次模型调用前执行：

            prompt_builder.build(
                runtime_context=dict(runtime_context)
            )

    要求该方法最终返回字符串。
    """
    if prompt_builder is None:
        system_prompt: SystemPrompt = (
            "你是一个电商推荐系统 Agent。"
            "需要检索信息时调用工具，不要编造检索结果。"
        )
    else:

        def render_system_prompt(
            runtime_context: Mapping[str, Any],
        ) -> str:
            rendered = prompt_builder.build(
                runtime_context=dict(runtime_context)
            )

            if not isinstance(rendered, str):
                raise TypeError(
                    "prompt_builder.build() must return "
                    "a string"
                )

            return rendered

        system_prompt = render_system_prompt

    config = LangChainAgentConfig(
        model_name=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.0,
        timeout_seconds=120.0,
        max_retries=2,
        max_model_calls_per_run=8,
        enable_memory=True,
    )

    return LangChainAgent(
        config=config,
        tools=tools,
        system_prompt=system_prompt,
        finalizer_hooks=finalizer_hooks,
    )


# ---------------------------------------------------------------------------
# 使用示例
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from langchain.tools import tool

    logging.basicConfig(level=logging.INFO)

    @tool
    def get_product_price(product_id: str) -> str:
        """根据商品 ID 查询商品价格。"""
        mock_prices = {
            "1001": 199.0,
            "1002": 299.0,
        }

        price = mock_prices.get(product_id)

        if price is None:
            return f"未查询到商品 {product_id}"

        return (
            f"商品 {product_id} 的价格为 "
            f"{price:.2f} 元"
        )

    project_agent = build_agent(
        tools=[get_product_price],
    )

    conversation_id = project_agent.new_thread_id()

    first = project_agent.invoke(
        "查询商品 1001 的价格。",
        thread_id=conversation_id,
        runtime_context={
            "user_id": "user-001",
        },
    )

    print("第一轮回答：", first.content)
    print("第一轮是否完成：", first.completed)
    print("第一轮模型调用次数：", first.model_calls)
    print("第一轮工具调用次数：", first.completed_tool_calls)
    print("第一轮 Token：", first.usage.to_dict())

    # 使用相同 thread_id，LangGraph 会读取上一轮历史消息。
    second = project_agent.invoke(
        "刚才那个商品多少钱？",
        thread_id=conversation_id,
        runtime_context={
            "user_id": "user-001",
        },
    )

    print("第二轮回答：", second.content)
    print("第二轮结果：", second.to_dict())