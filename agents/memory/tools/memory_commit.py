from __future__ import annotations

from collections.abc import Iterable

from agents.memory.models import (
    MemoryDecision,
    MemoryDecisionType,
    MemoryOperation,
    MemoryOperationType,
)
from agents.memory.store import SQLiteMemoryStore


class CommitMemoryDecisionTool:
    """Validate and audit all persisted memory mutations through one boundary."""

    name = "commit_memory_decision"

    def __init__(self, store: SQLiteMemoryStore) -> None:
        self._store = store

    def run(
        self,
        *,
        event_id: str,
        user_id: str,
        operations: Iterable[MemoryOperation],
        warnings: Iterable[str] = (),
    ) -> MemoryDecision:
        normalized_event_id = _required_text(event_id, "event_id")
        normalized_user_id = _required_text(user_id, "user_id")

        # Validate the complete batch before the first write. The Agent or LLM never
        # receives the store and therefore cannot bypass this user boundary.
        validated = tuple(_validated_operation(item) for item in operations)
        for operation in validated:
            if operation.user_id != normalized_user_id:
                raise ValueError("memory operation cannot mutate another user")

        changed = []
        deleted_count = 0
        for operation in validated:
            if operation.operation is MemoryOperationType.UPSERT:
                changed.append(self._store.apply(operation))
                continue
            if operation.operation is MemoryOperationType.SOFT_DELETE:
                deleted_count += self._store.soft_delete(
                    operation.user_id,
                    operation.memory_type,
                    operation.scope,
                    operation.key,
                    reason=operation.reason,
                    source_event_id=operation.source_event_id or normalized_event_id,
                )
                continue
            raise ValueError(f"unsupported memory operation: {operation.operation}")

        return MemoryDecision(
            event_id=normalized_event_id,
            user_id=normalized_user_id,
            decision=(
                MemoryDecisionType.REMEMBERED
                if changed or deleted_count
                else MemoryDecisionType.IGNORED
            ),
            changed_memories=changed,
            soft_deleted_count=deleted_count,
            warnings=list(warnings),
        )


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be blank")
    return value.strip()


def _validated_operation(item: MemoryOperation) -> MemoryOperation:
    operation = MemoryOperation.model_validate(item)
    return operation.model_copy(update={"key": operation.key.casefold()})
