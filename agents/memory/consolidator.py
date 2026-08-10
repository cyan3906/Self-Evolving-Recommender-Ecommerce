from __future__ import annotations

from datetime import timedelta

from agents.memory.models import (
    MemoryOperation,
    MemoryOperationType,
    MemoryRecord,
    MemoryReflection,
    MemoryScope,
    MemoryType,
    utc_now,
)
from agents.memory.store import SQLiteMemoryStore


_CONSOLIDATABLE_TYPES = {
    MemoryType.BRAND_PREFERENCE,
    MemoryType.CATEGORY_PREFERENCE,
    MemoryType.FEATURE_PREFERENCE,
    MemoryType.PRICE_RANGE,
}


class MemoryConsolidator:
    """Promote repeated recent evidence into idempotent long-term memory."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        *,
        minimum_evidence: int = 3,
        minimum_confidence: float = 0.35,
    ) -> None:
        if minimum_evidence < 2:
            raise ValueError("minimum_evidence must be at least 2")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        self._store = store
        self._minimum_evidence = minimum_evidence
        self._minimum_confidence = minimum_confidence

    def reflect(self, user_id: str) -> MemoryReflection:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("MemoryConsolidator requires a logged-in user_id")

        now = utc_now()
        expired_count = self._store.expire(user_id, now)
        reflected_event_ids = {
            event["source_event_id"]
            for event in self._store.list_events(user_id)
            if event["source_event_id"]
        }
        promoted = []
        skipped_duplicates = 0

        for record in self._store.get_active(user_id):
            if not self._eligible(record):
                continue
            reflection_id = f"reflection:{record.id}:v{record.version}"
            if reflection_id in reflected_event_ids:
                skipped_duplicates += 1
                continue
            promoted.append(self._store.apply(MemoryOperation(
                operation=MemoryOperationType.UPSERT,
                user_id=user_id,
                memory_type=record.memory_type,
                scope=MemoryScope.LONG_TERM,
                key=record.key,
                value=record.value,
                confidence=record.confidence,
                source="memory_reflection",
                reason="repeated recent evidence consolidated into long-term memory",
                source_event_id=reflection_id,
                expires_at=now + timedelta(days=90),
            )))

        return MemoryReflection(
            user_id=user_id,
            promoted_memories=promoted,
            expired_memory_count=expired_count,
            skipped_duplicate_count=skipped_duplicates,
            reflected_at=now,
        )

    def _eligible(self, record: MemoryRecord) -> bool:
        return (
            record.scope is MemoryScope.RECENT
            and record.memory_type in _CONSOLIDATABLE_TYPES
            and record.evidence_count >= self._minimum_evidence
            and record.confidence >= self._minimum_confidence
        )
