"""
Hermes-style conversation loop.

Responsibilities:
1. Build the effective system prompt through PromptBuilder.
2. Maintain OpenAI-compatible conversation messages.
3. Call the LLM.
4. Execute tool calls and append tool results.
5. Repeat until the model returns a normal assistant response.
6. Enforce iteration/retry budgets and preserve a valid message history.

This module intentionally does not implement:
- persistent memory storage;
- skill creation or skill review;
- session database persistence;
- context summarization;
- business-specific tool implementations.

Those concerns should be injected through ToolExecutor and lifecycle hooks.
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

try:
    # Recommended when conversation_loop.py and prompt_builder.py are in one package.
    from .prompt_builder import PromptBuilder
except ImportError:
    # Allows direct execution during local development.
    from prompt_builder import PromptBuilder


logger = logging.getLogger(__name__)

Message = dict[str, Any]
ToolDefinition = Mapping[str, Any]


class ToolExecutor(Protocol):
    """Interface expected by ConversationLoop."""

    def get_tool_definitions(self) -> Sequence[ToolDefinition]:
        """Return OpenAI-compatible tool definitions."""

    def execute(
        self,
        *,
        name: str,
        arguments: Mapping[str, Any],
        tool_call_id: str,
    ) -> Any:
        """Execute one tool call and return a JSON-serializable result."""


class ConversationHook(Protocol):
    """Optional lifecycle hook for persistence, memory review or tracing."""

    def on_turn_complete(self, result: "ConversationResult") -> None:
        """Called after a conversation turn finishes."""


@dataclass(slots=True)
class ConversationLoopConfig:
    model: str
    temperature: float = 0.0
    max_iterations: int = 20
    max_model_retries: int = 3
    retry_base_seconds: float = 1.0
    parallel_tool_calls: bool = True
    max_parallel_tools: int = 8
    tool_result_max_chars: int = 30_000

    def __post_init__(self) -> None: # 初始化完 自动调用
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")
        if self.max_model_retries <= 0:
            raise ValueError("max_model_retries must be greater than 0")
        if self.max_parallel_tools <= 0:
            raise ValueError("max_parallel_tools must be greater than 0")
        if self.tool_result_max_chars <= 0:
            raise ValueError("tool_result_max_chars must be greater than 0")


@dataclass(slots=True)
class NormalizedToolCall:
    id: str
    name: str
    raw_arguments: str
    arguments: dict[str, Any]
    argument_error: str | None = None

    def to_message_dict(self) -> Message:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments,
            },
        }


@dataclass(slots=True)
class NormalizedModelResponse:
    content: str
    tool_calls: list[NormalizedToolCall]
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConversationResult:
    task_id: str
    final_response: str
    messages: list[Message]
    iterations: int
    completed: bool
    interrupted: bool = False
    error: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    prompt_metadata: dict[str, Any] = field(default_factory=dict)


class ConversationLoop:
    """
    OpenAI-compatible conversation state machine.

    `client` must expose:

        client.chat.completions.create(
            model=...,
            messages=...,
            tools=...,
            temperature=...,
        )

    This works with the official OpenAI client and most OpenAI-compatible
    providers.
    """

    def __init__(
        self,
        *,
        client: Any,
        prompt_builder: PromptBuilder,
        tool_executor: ToolExecutor,
        config: ConversationLoopConfig,
        hooks: Sequence[ConversationHook | Callable[[ConversationResult], None]]
        | None = None,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder
        self.tool_executor = tool_executor
        self.config = config
        self.hooks = list(hooks or [])

        self._interrupt_requested = False

    def request_interrupt(self) -> None:
        """
        Ask the current loop to stop before the next model/tool iteration.

        This does not forcibly cancel an HTTP request already in progress.
        """
        self._interrupt_requested = True

    def clear_interrupt(self) -> None:
        self._interrupt_requested = False

    def run(
        self,
        user_message: str,
        *,
        conversation_history: Sequence[Mapping[str, Any]] | None = None,
        task_id: str | None = None,
        extra_project_rules: str | None = None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> ConversationResult:
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message must be a non-empty string")

        self.clear_interrupt()
        effective_task_id = task_id or f"task_{uuid.uuid4().hex}"
        tool_definitions = list(self.tool_executor.get_tool_definitions())

        prompt_bundle = self.prompt_builder.build(
            tools=tool_definitions,
            extra_project_rules=extra_project_rules,
            runtime_context={
                "task_id": effective_task_id,
                **dict(runtime_context or {}),
            },
        )

        messages = self._build_initial_messages(
            system_message=prompt_bundle.to_system_message(),
            conversation_history=conversation_history,
            user_message=user_message,
        )

        usage_total: dict[str, int] = {}
        iterations = 0
        final_response = ""
        error: str | None = None
        completed = False
        interrupted = False

        for iteration in range(1, self.config.max_iterations + 1):
            iterations = iteration

            if self._interrupt_requested:
                interrupted = True
                error = "conversation interrupted before the next model call"
                break

            try:
                model_response = self._call_model_with_retry(
                    messages=messages,
                    tools=tool_definitions,
                )
            except Exception as exc:
                error = f"model call failed: {exc}"
                logger.exception(
                    "Conversation model call failed: task_id=%s iteration=%s",
                    effective_task_id,
                    iteration,
                )
                break

            self._merge_usage(usage_total, model_response.usage)

            assistant_message: Message = {
                "role": "assistant",
                "content": model_response.content,
            }

            if model_response.tool_calls:
                assistant_message["tool_calls"] = [
                    tool_call.to_message_dict()
                    for tool_call in model_response.tool_calls
                ]

            messages.append(assistant_message)

            # No tool call means the model has completed the turn.
            if not model_response.tool_calls:
                final_response = model_response.content.strip()
                completed = True
                break

            if self._interrupt_requested:
                interrupted = True
                error = "conversation interrupted before tool execution"
                self._append_unexecuted_tool_errors(
                    messages,
                    model_response.tool_calls,
                    error,
                )
                break

            tool_messages = self._execute_tool_batch(
                model_response.tool_calls
            )
            messages.extend(tool_messages)

        if not completed and not interrupted and error is None:
            error = (
                "maximum conversation iterations reached "
                f"({self.config.max_iterations})"
            )

        if not completed:
            final_response = self._build_failure_response(
                error=error,
                interrupted=interrupted,
            )
            # Keep role alternation valid for persistence/resume.
            if not messages or messages[-1].get("role") != "assistant":
                messages.append(
                    {
                        "role": "assistant",
                        "content": final_response,
                    }
                )

        result = ConversationResult(
            task_id=effective_task_id,
            final_response=final_response,
            messages=messages,
            iterations=iterations,
            completed=completed,
            interrupted=interrupted,
            error=error,
            usage=usage_total,
            prompt_metadata=dict(prompt_bundle.metadata),
        )
        self._notify_hooks(result)
        return result

    def chat(
        self,
        user_message: str,
        **kwargs: Any,
    ) -> str:
        """Simple entry point that returns only the final assistant text."""
        return self.run(user_message, **kwargs).final_response

    def _build_initial_messages(
        self,
        *,
        system_message: Mapping[str, Any],
        conversation_history: Sequence[Mapping[str, Any]] | None,
        user_message: str,
    ) -> list[Message]:
        messages: list[Message] = [copy.deepcopy(dict(system_message))]

        if conversation_history:
            history = [
                copy.deepcopy(dict(message))
                for message in conversation_history
                if message.get("role") != "system"
            ]
            self._validate_history(history)
            messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )
        return messages

    def _validate_history(
        self,
        history: Sequence[Mapping[str, Any]],
    ) -> None:
        valid_roles = {"user", "assistant", "tool"}

        for index, message in enumerate(history):
            role = message.get("role")
            if role not in valid_roles:
                raise ValueError(
                    f"conversation_history[{index}] has invalid role: {role!r}"
                )

            if role == "tool" and not message.get("tool_call_id"):
                raise ValueError(
                    f"conversation_history[{index}] is a tool message "
                    "without tool_call_id"
                )

        # A new user message will be appended. The supplied history should
        # therefore normally end with assistant/tool rather than user.
        if history and history[-1].get("role") == "user":
            raise ValueError(
                "conversation_history cannot end with role='user' because "
                "run() appends the current user message"
            )

    def _call_model_with_retry(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> NormalizedModelResponse:
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_model_retries + 1):
            try:
                request: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": list(messages),
                    "temperature": self.config.temperature,
                }
                if tools:
                    request["tools"] = list(tools)
                    request["tool_choice"] = "auto"

                raw_response = self.client.chat.completions.create(**request)
                return self._normalize_model_response(raw_response)

            except Exception as exc:
                last_error = exc

                if attempt >= self.config.max_model_retries:
                    break

                delay = self.config.retry_base_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Model call failed; retrying: attempt=%s/%s delay=%.2fs "
                    "error=%s",
                    attempt,
                    self.config.max_model_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)

        assert last_error is not None
        raise last_error

    def _normalize_model_response(
        self,
        raw_response: Any,
    ) -> NormalizedModelResponse:
        try:
            choice = raw_response.choices[0]
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError(
                "model response does not contain choices[0]"
            ) from exc

        message = self._get_value(choice, "message")
        if message is None:
            raise ValueError("model response choice does not contain message")

        content = self._get_value(message, "content") or ""
        if not isinstance(content, str):
            # Multimodal/provider-specific blocks are preserved as readable JSON.
            content = self._serialize(content)

        finish_reason = self._get_value(choice, "finish_reason")
        raw_tool_calls = self._get_value(message, "tool_calls") or []

        normalized_tool_calls: list[NormalizedToolCall] = []
        used_ids: set[str] = set()

        for index, raw_tool_call in enumerate(raw_tool_calls):
            function = self._get_value(raw_tool_call, "function")
            if function is None:
                raise ValueError(
                    f"tool_calls[{index}] does not contain function"
                )

            name = self._get_value(function, "name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"tool_calls[{index}] has an empty function name"
                )

            raw_arguments = self._get_value(function, "arguments")
            if raw_arguments is None or raw_arguments == "":
                raw_arguments = "{}"
            elif not isinstance(raw_arguments, str):
                raw_arguments = self._serialize(raw_arguments)

            tool_call_id = self._get_value(raw_tool_call, "id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                tool_call_id = f"call_{uuid.uuid4().hex}"

            # Providers/models occasionally duplicate IDs in one batch.
            if tool_call_id in used_ids:
                tool_call_id = f"{tool_call_id}_{index}_{uuid.uuid4().hex[:8]}"
            used_ids.add(tool_call_id)

            arguments, argument_error = self._parse_tool_arguments(
                raw_arguments
            )

            normalized_tool_calls.append(
                NormalizedToolCall(
                    id=tool_call_id,
                    name=name,
                    raw_arguments=raw_arguments,
                    arguments=arguments,
                    argument_error=argument_error,
                )
            )

        usage = self._normalize_usage(
            self._get_value(raw_response, "usage")
        )

        return NormalizedModelResponse(
            content=content,
            tool_calls=normalized_tool_calls,
            finish_reason=finish_reason,
            usage=usage,
        )

    def _execute_tool_batch(
        self,
        tool_calls: Sequence[NormalizedToolCall],
    ) -> list[Message]:
        if (
            not self.config.parallel_tool_calls
            or len(tool_calls) <= 1
        ):
            return [
                self._execute_one_tool(tool_call)
                for tool_call in tool_calls
            ]

        max_workers = min(
            self.config.max_parallel_tools,
            len(tool_calls),
        )
        indexed_results: dict[int, Message] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._execute_one_tool, tool_call): index
                for index, tool_call in enumerate(tool_calls)
            }

            for future in as_completed(futures):
                index = futures[future]
                try:
                    indexed_results[index] = future.result()
                except Exception as exc:
                    tool_call = tool_calls[index]
                    logger.exception(
                        "Unexpected tool worker failure: tool=%s call_id=%s",
                        tool_call.name,
                        tool_call.id,
                    )
                    indexed_results[index] = self._make_tool_message(
                        tool_call=tool_call,
                        payload={
                            "ok": False,
                            "error": f"tool worker failed: {exc}",
                        },
                    )

        # Keep results in the same order as the assistant tool_calls array.
        return [
            indexed_results[index]
            for index in range(len(tool_calls))
        ]

    def _execute_one_tool(
        self,
        tool_call: NormalizedToolCall,
    ) -> Message:
        if tool_call.argument_error is not None:
            return self._make_tool_message(
                tool_call=tool_call,
                payload={
                    "ok": False,
                    "error": (
                        "invalid JSON tool arguments: "
                        f"{tool_call.argument_error}"
                    ),
                    "raw_arguments": tool_call.raw_arguments,
                },
            )

        try:
            result = self.tool_executor.execute(
                name=tool_call.name,
                arguments=tool_call.arguments,
                tool_call_id=tool_call.id,
            )
            payload = {
                "ok": True,
                "result": result,
            }
        except Exception as exc:
            logger.exception(
                "Tool execution failed: tool=%s call_id=%s",
                tool_call.name,
                tool_call.id,
            )
            payload = {
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

        return self._make_tool_message(
            tool_call=tool_call,
            payload=payload,
        )

    def _make_tool_message(
        self,
        *,
        tool_call: NormalizedToolCall,
        payload: Any,
    ) -> Message:
        content = self._serialize(payload)

        if len(content) > self.config.tool_result_max_chars:
            removed_chars = len(content) - self.config.tool_result_max_chars
            content = (
                content[: self.config.tool_result_max_chars]
                + "\n"
                + self._serialize(
                    {
                        "truncated": True,
                        "removed_chars": removed_chars,
                    }
                )
            )

        return {
            "role": "tool",
            "name": tool_call.name,
            "tool_call_id": tool_call.id,
            "content": content,
        }

    def _append_unexecuted_tool_errors(
        self,
        messages: list[Message],
        tool_calls: Sequence[NormalizedToolCall],
        reason: str,
    ) -> None:
        # Every assistant tool_call must have a matching tool message.
        for tool_call in tool_calls:
            messages.append(
                self._make_tool_message(
                    tool_call=tool_call,
                    payload={
                        "ok": False,
                        "error": reason,
                    },
                )
            )

    def _parse_tool_arguments(
        self,
        raw_arguments: str,
    ) -> tuple[dict[str, Any], str | None]:
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return {}, str(exc)

        if not isinstance(parsed, dict):
            return {}, (
                "tool arguments must decode to a JSON object, "
                f"got {type(parsed).__name__}"
            )

        return parsed, None

    def _normalize_usage(self, usage: Any) -> dict[str, int]:
        if usage is None:
            return {}

        keys = (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
        )
        normalized: dict[str, int] = {}

        for key in keys:
            value = self._get_value(usage, key)
            if isinstance(value, int):
                normalized[key] = value

        return normalized

    def _merge_usage(
        self,
        total: dict[str, int],
        current: Mapping[str, Any],
    ) -> None:
        for key, value in current.items():
            if isinstance(value, int):
                total[key] = total.get(key, 0) + value

    def _notify_hooks(self, result: ConversationResult) -> None:
        for hook in self.hooks:
            try:
                callback = getattr(hook, "on_turn_complete", None)
                if callable(callback):
                    callback(result)
                elif callable(hook):
                    hook(result)
                else:
                    logger.warning(
                        "Ignoring invalid conversation hook: %r",
                        hook,
                    )
            except Exception:
                # Persistence/review failures should not discard the answer.
                logger.exception(
                    "Conversation completion hook failed: task_id=%s",
                    result.task_id,
                )

    def _build_failure_response(
        self,
        *,
        error: str | None,
        interrupted: bool,
    ) -> str:
        if interrupted:
            return "当前任务已中断。"
        if error:
            return f"任务未正常完成：{error}"
        return "任务未正常完成。"

    @staticmethod
    def _get_value(obj: Any, key: str) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(key)
        return getattr(obj, key, None)

    @staticmethod
    def _serialize(value: Any) -> str:
        if isinstance(value, str):
            return value

        try:
            return json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
        except (TypeError, ValueError):
            return str(value)


class FunctionToolExecutor:
    """
    Minimal executable ToolExecutor.

    This adapter is included so conversation_loop.py can be used immediately.
    You can later move it into tool_executor.py without changing
    ConversationLoop.

    handlers:
        {
            "search_products": search_products,
            "get_user_profile": get_user_profile,
        }

    definitions:
        OpenAI-compatible tool definition dictionaries.
    """

    def __init__(
        self,
        *,
        definitions: Sequence[ToolDefinition],
        handlers: Mapping[str, Callable[..., Any]],
    ) -> None:
        self._definitions = [dict(item) for item in definitions]
        self._handlers = dict(handlers)

    def get_tool_definitions(self) -> Sequence[ToolDefinition]:
        return copy.deepcopy(self._definitions)

    def execute(
        self,
        *,
        name: str,
        arguments: Mapping[str, Any],
        tool_call_id: str,
    ) -> Any:
        del tool_call_id

        handler = self._handlers.get(name)
        if handler is None:
            available = ", ".join(sorted(self._handlers)) or "<none>"
            raise KeyError(
                f"unknown tool {name!r}; available tools: {available}"
            )

        return handler(**dict(arguments))