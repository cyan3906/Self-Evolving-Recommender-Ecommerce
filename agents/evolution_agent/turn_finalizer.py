"""
LangChain Turn Finalizer.

适配基于 ``langchain.agents.create_agent`` 的 Agent。

LangChain / LangGraph 已经负责：
    模型 -> tool_calls -> 工具执行 -> ToolMessage -> 再次调用模型

TurnFinalizer 只负责一次 Agent 调用结束后的统一收口：
1. 从 LangGraph 返回状态中提取最终 AI 文本；
2. 统一成功、失败返回结构；
3. 统计模型调用次数、工具调用次数和 Token 使用量；
4. 保存原始 messages 与 raw state；
5. 执行 turn 结束 Hook；
6. 同时支持同步和异步调用。

依赖：
    pip install -U langchain langchain-core langgraph
"""

from __future__ import annotations

import inspect
import logging

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    message_to_dict, # 方面处理对象 变成 字符串
)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TurnUsage:
    """一次 Agent turn 累计的 Token 使用量。"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        self.input_tokens += max(int(input_tokens), 0)
        self.output_tokens += max(int(output_tokens), 0)

        if total_tokens > 0:
            self.total_tokens += int(total_tokens)
        else:
            self.total_tokens += (
                max(int(input_tokens), 0)
                + max(int(output_tokens), 0)
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(slots=True)
class LangChainTurnResult:
    """TurnFinalizer 输出的统一返回结构。"""

    thread_id: str

    # 最终返回给业务层或前端的文本。
    content: str

    # LangChain / LangGraph 当前 thread_id 的完整消息链。
    messages: list[BaseMessage]

    # 当前用户 turn 的消息链，从最后一条 HumanMessage 开始。
    turn_messages: list[BaseMessage]

    # create_agent.invoke/ainvoke 返回的原始状态。
    raw: dict[str, Any]

    # Agent 是否正常生成了最终 AI 文本。
    completed: bool

    # 失败原因；成功时为 None。
    error: str | None = None

    # 本次状态中 AIMessage 的数量。
    model_calls: int = 0

    # AIMessage 中请求的工具调用总数。
    requested_tool_calls: int = 0

    # 已经产生 ToolMessage 的工具执行总数。
    completed_tool_calls: int = 0

    # 累计 Token 使用量。
    usage: TurnUsage = field(default_factory=TurnUsage)

    # 业务侧附加信息，例如 user_id、request_id、task_id。
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(
        self,
        *,
        serialize_messages: bool = True,
        include_raw: bool = False,
    ) -> dict[str, Any]:
        """
        转换为适合日志、FastAPI 或持久化的字典。

        ``raw`` 可能包含不可 JSON 序列化对象，因此默认不返回。
        """
        if serialize_messages:
            serialized_messages: list[Any] = [
                message_to_dict(message)
                for message in self.messages
            ]
        else:
            serialized_messages = list(self.messages)

        result: dict[str, Any] = {
            "thread_id": self.thread_id,
            "content": self.content,
            "messages": serialized_messages,
            "turn_messages": [
                message_to_dict(message)
                for message in self.turn_messages
            ] if serialize_messages else list(self.turn_messages),
            "completed": self.completed,
            "error": self.error,
            "model_calls": self.model_calls,
            "requested_tool_calls": self.requested_tool_calls,
            "completed_tool_calls": self.completed_tool_calls,
            "usage": self.usage.to_dict(),
            "metadata": dict(self.metadata),
        }

        if include_raw:
            result["raw"] = self.raw

        return result


class SyncTurnFinalizerHook(Protocol):
    """同步 Turn 结束 Hook。"""

    def on_turn_finalized(
        self,
        result: LangChainTurnResult,
    ) -> None:
        ...


class AsyncTurnFinalizerHook(Protocol):
    """异步 Turn 结束 Hook。"""

    async def aon_turn_finalized(
        self,
        result: LangChainTurnResult,
    ) -> None:
        ...


SyncHookCallable = Callable[[LangChainTurnResult], None]
AsyncHookCallable = Callable[
    [LangChainTurnResult],
    Awaitable[None],
]
TurnHook = (
    SyncTurnFinalizerHook
    | AsyncTurnFinalizerHook
    | SyncHookCallable
    | AsyncHookCallable
)


@dataclass(slots=True)
class TurnFinalizerConfig:
    """TurnFinalizer 配置。"""

    # Agent 正常结束但没有找到最终 AI 文本时的展示文案。
    empty_response_text: str = "Agent 未生成可展示的最终回答。"

    # Agent 执行异常时的默认展示文案。
    failure_response_text: str = "Agent 执行失败，请稍后重试。"

    # True 时把异常详情拼接到 content 中。
    # 生产环境建议保持 False，详细异常放在 error 字段和日志中。
    expose_error_to_user: bool = False

    # Hook 异常是否继续向上抛出。
    # 默认隔离 Hook 异常，避免日志、持久化等后处理影响主响应。
    raise_hook_error: bool = False


class TurnFinalizer:
    """
    LangChain Agent 单次调用的统一收口器。

    注意：
    TurnFinalizer 不执行模型，也不执行工具，更不控制循环。
    create_agent 已经完成循环；本类只处理最终状态。
    """

    def __init__(
        self,
        *,
        config: TurnFinalizerConfig | None = None,
        hooks: Sequence[TurnHook] | None = None,
    ) -> None:
        self.config = config or TurnFinalizerConfig()
        self.hooks = list(hooks or [])

    def finalize(
        self,
        *,
        thread_id: str,
        raw_result: Mapping[str, Any] | None = None,
        error: BaseException | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LangChainTurnResult:
        """
        同步收口。

        同步版本只执行同步 Hook。
        如果传入异步 Hook，会记录警告并跳过；异步 Agent 请使用
        ``afinalize``。
        """
        result = self._build_result(
            thread_id=thread_id,
            raw_result=raw_result,
            error=error,
            metadata=metadata,
        )

        self._run_sync_hooks(result)
        return result

    async def afinalize(
        self,
        *,
        thread_id: str,
        raw_result: Mapping[str, Any] | None = None,
        error: BaseException | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LangChainTurnResult:
        """异步收口，可同时执行同步 Hook 和异步 Hook。"""
        result = self._build_result(
            thread_id=thread_id,
            raw_result=raw_result,
            error=error,
            metadata=metadata,
        )

        await self._run_async_hooks(result)
        return result

    def _build_result(
        self,
        *,
        thread_id: str,
        raw_result: Mapping[str, Any] | None,
        error: BaseException | str | None,
        metadata: Mapping[str, Any] | None,
    ) -> LangChainTurnResult:
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError("thread_id cannot be empty")

        raw = dict(raw_result or {})
        messages = self._normalize_messages(
            raw.get("messages", [])
        )

        # checkpointer 开启时，messages 通常是整个 thread 的历史。
        # Turn 级统计只计算最后一条 HumanMessage 之后的消息。
        turn_messages = self._extract_current_turn_messages(
            messages
        )

        usage = self._collect_usage(turn_messages)
        model_calls = sum(
            isinstance(message, AIMessage)
            for message in turn_messages
        )
        requested_tool_calls = self._count_requested_tool_calls(
            turn_messages
        )
        completed_tool_calls = sum(
            isinstance(message, ToolMessage)
            for message in turn_messages
        )

        normalized_error = self._normalize_error(error)

        if normalized_error is not None:
            completed = False
            content = self._build_failure_content(
                normalized_error
            )
        else:
            content = self.extract_final_content(turn_messages)

            if content:
                completed = True
            else:
                completed = False
                normalized_error = (
                    "agent completed without a final AI text response"
                )
                content = self.config.empty_response_text

        return LangChainTurnResult(
            thread_id=thread_id,
            content=content,
            messages=messages,
            turn_messages=turn_messages,
            raw=raw,
            completed=completed,
            error=normalized_error,
            model_calls=model_calls,
            requested_tool_calls=requested_tool_calls,
            completed_tool_calls=completed_tool_calls,
            usage=usage,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def extract_final_content(
        cls,
        messages: Sequence[BaseMessage],
    ) -> str:
        """
        从后向前查找真正的最终 AI 文本。

        中间的 AIMessage 通常只包含 tool_calls，不能作为最终回答；
        因此这里会跳过仍携带 tool_calls 的 AIMessage。
        """
        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue

            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                continue

            content = cls.content_to_text(
                message.content
            ).strip()

            if content:
                return content

        return ""

    @staticmethod
    def content_to_text(content: Any) -> str:
        """兼容字符串和多内容块形式的 AIMessage.content。"""
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []

            for block in content:
                if isinstance(block, str):
                    if block.strip():
                        text_parts.append(block)
                    continue

                if not isinstance(block, Mapping):
                    rendered = str(block).strip()
                    if rendered:
                        text_parts.append(rendered)
                    continue

                # LangChain/OpenAI 常见文本块结构。
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text)
                    continue

                block_content = block.get("content")
                if (
                    isinstance(block_content, str)
                    and block_content.strip()
                ):
                    text_parts.append(block_content)

            return "\n".join(text_parts)

        return str(content)

    @staticmethod
    def _extract_current_turn_messages(
        messages: Sequence[BaseMessage],
    ) -> list[BaseMessage]:
        """
        提取当前 turn 的消息。

        启用 LangGraph checkpointer 时，返回状态会带上整个 thread 的
        历史消息。一次 turn 从最后一条 HumanMessage 开始，到当前状态
        末尾结束。
        """
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], HumanMessage):
                return list(messages[index:])

        # 理论上 create_agent 的正常调用一定包含 HumanMessage。
        # 没找到时保留全部消息，避免丢失诊断信息。
        return list(messages)

    @staticmethod
    def _normalize_messages(
        raw_messages: Any,
    ) -> list[BaseMessage]:
        if raw_messages is None:
            return []

        if not isinstance(raw_messages, Sequence):
            raise TypeError(
                "raw_result['messages'] must be a sequence"
            )

        messages: list[BaseMessage] = []

        for index, message in enumerate(raw_messages):
            if not isinstance(message, BaseMessage):
                raise TypeError(
                    "raw_result['messages'] contains a non-BaseMessage "
                    f"value at index {index}: {type(message)!r}"
                )

            messages.append(message)

        return messages

    @staticmethod
    def _count_requested_tool_calls(
        messages: Sequence[BaseMessage],
    ) -> int:
        total = 0

        for message in messages:
            if not isinstance(message, AIMessage):
                continue

            tool_calls = getattr(message, "tool_calls", None)
            if isinstance(tool_calls, Sequence):
                total += len(tool_calls)

        return total

    @staticmethod
    def _collect_usage(
        messages: Sequence[BaseMessage],
    ) -> TurnUsage:
        usage = TurnUsage()

        for message in messages:
            if not isinstance(message, AIMessage):
                continue

            usage_metadata = getattr(
                message,
                "usage_metadata",
                None,
            )

            if not isinstance(usage_metadata, Mapping):
                continue

            input_tokens = TurnFinalizer._safe_int(
                usage_metadata.get("input_tokens", 0)
            )
            output_tokens = TurnFinalizer._safe_int(
                usage_metadata.get("output_tokens", 0)
            )
            total_tokens = TurnFinalizer._safe_int(
                usage_metadata.get("total_tokens", 0)
            )

            usage.add(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

        return usage

    @staticmethod
    def _safe_int(value: Any) -> int:
        if isinstance(value, bool):
            return 0

        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _normalize_error(
        error: BaseException | str | None,
    ) -> str | None:
        if error is None:
            return None

        if isinstance(error, BaseException):
            error_text = (
                f"{error.__class__.__name__}: {error}"
            )
        else:
            error_text = str(error)

        normalized = error_text.strip()
        return normalized or "unknown agent execution error"

    def _build_failure_content(
        self,
        error: str,
    ) -> str:
        if self.config.expose_error_to_user:
            return (
                f"{self.config.failure_response_text}\n"
                f"错误信息：{error}"
            )

        return self.config.failure_response_text

    def _run_sync_hooks(
        self,
        result: LangChainTurnResult,
    ) -> None:
        for hook in self.hooks:
            try:
                callback = getattr(
                    hook,
                    "on_turn_finalized",
                    None,
                )

                if callable(callback):
                    callback(result)
                    continue

                async_callback = getattr(
                    hook,
                    "aon_turn_finalized",
                    None,
                )

                if callable(async_callback):
                    logger.warning(
                        "Skipping async TurnFinalizer hook in "
                        "synchronous finalize(): %r",
                        hook,
                    )
                    continue

                if callable(hook):
                    returned = hook(result)

                    if inspect.isawaitable(returned):
                        logger.warning(
                            "Skipping awaitable returned by hook in "
                            "synchronous finalize(): %r",
                            hook,
                        )

            except Exception:
                logger.exception(
                    "TurnFinalizer hook failed: thread_id=%s",
                    result.thread_id,
                )

                if self.config.raise_hook_error:
                    raise

    async def _run_async_hooks(
        self,
        result: LangChainTurnResult,
    ) -> None:
        for hook in self.hooks:
            try:
                async_callback = getattr(
                    hook,
                    "aon_turn_finalized",
                    None,
                )

                if callable(async_callback):
                    await async_callback(result)
                    continue

                sync_callback = getattr(
                    hook,
                    "on_turn_finalized",
                    None,
                )

                if callable(sync_callback):
                    returned = sync_callback(result)

                    if inspect.isawaitable(returned):
                        await returned
                    continue

                if callable(hook):
                    returned = hook(result)

                    if inspect.isawaitable(returned):
                        await returned

            except Exception:
                logger.exception(
                    "Async TurnFinalizer hook failed: thread_id=%s",
                    result.thread_id,
                )

                if self.config.raise_hook_error:
                    raise