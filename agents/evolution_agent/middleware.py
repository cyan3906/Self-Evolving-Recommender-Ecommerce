from __future__ import annotations
import logging
import re
import sys
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    dynamic_prompt,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

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



class HybridSearchSkillMiddleware(AgentMiddleware):
    """
    hybrid_search 策略参数注入中间件。

    工作流程：
        1. wrap_model_call:
           读取本次真正发送给模型的 system_message；
        2. 从 system prompt 中使用正则提取 hybrid retrieval 参数；
        3. 将解析结果保存到当前 runtime.context.values；
        4. wrap_tool_call / awrap_tool_call:
           当模型调用 hybrid_search 时，在工具真正执行之前，
           使用 Skill 参数覆盖模型生成的同名参数。

    这样可以把职责拆开：
        - LLM：负责决定是否调用 hybrid_search，以及生成 query 等业务参数；
        - Skill：负责控制 ES/Milvus 权重、召回数量等策略参数；
        - Middleware：负责把 Skill 参数确定性注入工具调用。
    """

    HYBRID_TOOL_NAME = "hybrid_search"
    CONTEXT_CONFIG_KEY = "_hybrid_search_skill_config"
    CONTEXT_PROMPT_KEY = "_hybrid_search_system_prompt"

    # system prompt 中的字段名 -> hybrid_search 工具参数名。
    #
    # 如果 hybrid_search 函数参数名称与你的 Skill Markdown 不一致，
    # 只需要修改这里右侧的工具参数名。
    TOOL_ARG_MAPPING: dict[str, str] = {
        "es_weight": "es_weight",
        "milvus_weight": "milvus_weight",
        "es_top_k": "es_top_k",
        "milvus_top_k": "milvus_top_k",
        "final_top_k": "final_top_k",
    }
    
    # 只负责从 Prompt 中确定性抽取，不让 LLM 再解析一次。
    CONFIG_PATTERNS: dict[str, tuple[str, type]] = {
        "es_weight": (
            r"(?m)^\s*es_weight\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
            float,
        ),
        "milvus_weight": (
            r"(?m)^\s*milvus_weight\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
            float,
        ),
        "es_top_k": (
            r"(?m)^\s*es_top_k\s*:\s*(\d+)\s*$",
            int,
        ),
        "milvus_top_k": (
            r"(?m)^\s*milvus_top_k\s*:\s*(\d+)\s*$",
            int,
        ),
        "final_top_k": (
            r"(?m)^\s*final_top_k\s*:\s*(\d+)\s*$",
            int,
        ),
    }

    @staticmethod
    def _system_message_to_text(system_message: Any) -> str:
        """将 ModelRequest.system_message 转成纯文本。"""
        if system_message is None:
            return ""

        content = getattr(system_message, "content", system_message)

        if isinstance(content, str):
            return content

        # 兼容部分 provider 的 block/list 形式 content。
        if isinstance(content, list):
            parts: list[str] = []

            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                    continue

                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                        continue

                parts.append(str(block))

            return "\n".join(parts)

        return str(content)

    @classmethod
    def _parse_hybrid_config(
        cls,
        system_prompt: str,
    ) -> dict[str, Any]:
        """
        从 system prompt 中抽取 Hybrid Retrieval Strategy 参数。

        例如：

            es_weight: 0.7
            milvus_weight: 0.3
            es_top_k: 50
            milvus_top_k: 50
            final_top_k: 20

        返回：

            {
                "es_weight": 0.7,
                "milvus_weight": 0.3,
                "es_top_k": 50,
                "milvus_top_k": 50,
                "final_top_k": 20,
            }
        """
        if not system_prompt:
            return {}

        parsed: dict[str, Any] = {}

        for field_name, (
            pattern,
            converter,
        ) in cls.CONFIG_PATTERNS.items():
            match = re.search(
                pattern,
                system_prompt,
                flags=re.IGNORECASE,
            )

            if match is None:
                continue

            parsed[field_name] = converter(
                match.group(1)
            )

        return parsed

    @staticmethod
    def _get_context_values(
        runtime: Any,
    ) -> dict[str, Any] | None:
        """
        获取当前 run 独立的 runtime context values。

        不把解析结果保存到 Middleware 实例属性 self.xxx，
        避免 FastAPI/并发请求之间串数据。
        """
        if runtime is None:
            return None

        context = getattr(runtime, "context", None)

        if isinstance(context, AgentRuntimeContext):
            return context.values

        return None

    @classmethod
    def _save_prompt_config(
        cls,
        request: ModelRequest,
    ) -> None:
        """读取当前 system prompt，解析后保存到当前 run context。"""
        system_prompt = cls._system_message_to_text(
            request.system_message
        )

        context_values = cls._get_context_values(
            request.runtime
        )

        if context_values is None:
            return

        # 保存原始 system prompt，便于调试/日志/后续 Skill 分析。
        context_values[
            cls.CONTEXT_PROMPT_KEY
        ] = system_prompt

        hybrid_config = cls._parse_hybrid_config(
            system_prompt
        )

        context_values[
            cls.CONTEXT_CONFIG_KEY
        ] = hybrid_config

        if hybrid_config:
            logger.debug(
                "Parsed hybrid_search config from system prompt: %s",
                hybrid_config,
            )

    @classmethod
    def _build_overridden_tool_request(
        cls,
        request: ToolCallRequest,
    ) -> ToolCallRequest:
        """
        如果当前调用的是 hybrid_search，则使用 Skill 参数覆盖 tool args。
        """
        tool_call = request.tool_call
        tool_name = tool_call.get("name")

        if tool_name != cls.HYBRID_TOOL_NAME:
            return request

        context_values = cls._get_context_values(
            request.runtime
        )

        if context_values is None:
            logger.warning(
                "hybrid_search called but runtime context is unavailable; "
                "tool args will not be overridden"
            )
            return request

        skill_config = context_values.get(
            cls.CONTEXT_CONFIG_KEY,
            {},
        )

        if not isinstance(skill_config, Mapping):
            logger.warning(
                "Invalid hybrid_search skill config type: %s",
                type(skill_config).__name__,
            )
            return request

        if not skill_config:
            logger.debug(
                "hybrid_search called but no hybrid config was parsed "
                "from the current system prompt"
            )
            return request

        original_args = dict(
            tool_call.get("args") or {}
        )

        injected_args: dict[str, Any] = {}

        for skill_key, value in skill_config.items():
            tool_arg_name = cls.TOOL_ARG_MAPPING.get(
                skill_key
            )

            if tool_arg_name is None:
                continue

            injected_args[tool_arg_name] = value

        # 顺序很重要：
        #
        # original_args 在前；
        # injected_args 在后。
        #
        # 所以 Skill 配置会强制覆盖 LLM 生成的同名参数。
        final_args = {
            **original_args,
            **injected_args,
        }

        overridden_tool_call = {
            **tool_call,
            "args": final_args,
        }

        logger.info(
            "Override hybrid_search args from skill config: "
            "original=%s, injected=%s, final=%s",
            original_args,
            injected_args,
            final_args,
        )

        return request.override(
            tool_call=overridden_tool_call
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[
            [ModelRequest],
            ModelResponse,
        ],
    ) -> ModelResponse:
        """
        同步模型调用拦截。

        此时 request.system_message 就是当前模型调用看到的系统消息。
        """
        self._save_prompt_config(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[
            [ModelRequest],
            Any,
        ],
    ) -> ModelResponse:
        """异步版本，对应 LangChainAgent.ainvoke()."""
        self._save_prompt_config(request)
        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            Any,
        ],
    ) -> Any:
        """
        同步工具执行拦截。

        hybrid_search 真正执行前，在这里覆盖参数。
        """
        request = self._build_overridden_tool_request(
            request
        )
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            Any,
        ],
    ) -> Any:
        """
        异步工具执行拦截。

        与同步 invoke 行为保持一致。
        """
        request = self._build_overridden_tool_request(
            request
        )
        return await handler(request)

