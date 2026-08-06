from __future__ import annotations

from agents.memory.agent import V1_USER_ID, extract_deterministic_query_operations
from agents.memory.models import (
    MemoryContext,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    WeightedMemory,
    utc_now,
)
from agents.memory.store import SQLiteMemoryStore


LEVEL_WEIGHT = {
    MemoryScope.DAILY: 1.0,
    MemoryScope.RECENT: 0.65,
    MemoryScope.LONG_TERM: 0.35,
    MemoryScope.DURABLE: 0.25,
}

_LEVEL_CAP = 5


class MemoryProjector:
    """Build immutable, scene-specific memory inputs without writing persisted memory."""

    def __init__(self, store: SQLiteMemoryStore) -> None:
        self._store = store

    def for_homepage(self, user_id: str) -> MemoryContext:
        self._require_v1_user(user_id)
        records = self._active_records(user_id)
        return self._context(user_id, "homepage", records)

    def for_search(self, user_id: str, query: str) -> MemoryContext:
        self._require_v1_user(user_id)
        records = self._active_records(user_id)
        constraints = self._query_constraints(user_id, query)
        return self._context(user_id, "search", records, constraints)

    def _active_records(self, user_id: str) -> list[MemoryRecord]:
        self._store.expire(user_id, utc_now())
        return self._store.get_active(user_id)

    def _context(
        self,
        user_id: str,
        scene: str,
        records: list[MemoryRecord],
        current_constraints: list[WeightedMemory] | None = None,
    ) -> MemoryContext:
        daily_intents = self._sorted(
            record for record in records
            if record.scope is MemoryScope.DAILY
            and record.memory_type is not MemoryType.NEGATIVE_PREFERENCE
        )
        recent_preferences = self._sorted(
            record for record in records
            if record.scope is MemoryScope.RECENT
            and record.memory_type is not MemoryType.NEGATIVE_PREFERENCE
        )
        long_term_preferences = self._sorted(
            record for record in records
            if record.scope is MemoryScope.LONG_TERM
            and record.memory_type is not MemoryType.NEGATIVE_PREFERENCE
        )
        negative_preferences = self._sorted(
            record for record in records
            if record.memory_type in {
                MemoryType.NEGATIVE_PREFERENCE,
                MemoryType.PURCHASED_PRODUCT,
            }
        )
        return MemoryContext(
            user_id=user_id,
            scene=scene,
            current_constraints=current_constraints or [],
            daily_intents=daily_intents,
            recent_preferences=recent_preferences,
            long_term_preferences=long_term_preferences,
            negative_preferences=negative_preferences,
            memory_version=max((record.version for record in records), default=0),
        )

    @staticmethod
    def _sorted(records: object) -> list[WeightedMemory]:
        memories = [MemoryProjector._weighted(record) for record in records]
        memories.sort(key=lambda memory: (memory.weight, memory.confidence), reverse=True)
        return memories[:_LEVEL_CAP]

    @staticmethod
    def _weighted(record: MemoryRecord) -> WeightedMemory:
        return WeightedMemory(
            memory_type=record.memory_type,
            scope=record.scope,
            key=record.key,
            value=record.value,
            confidence=record.confidence,
            weight=LEVEL_WEIGHT[record.scope] * record.confidence,
            source_memory_id=record.id,
        )

    @staticmethod
    def _query_constraints(user_id: str, query: str) -> list[WeightedMemory]:
        operations = extract_deterministic_query_operations(
            user_id, query, "projection-query", utc_now()
        )
        constraints = [
            WeightedMemory(
                memory_type=operation.memory_type,
                scope=MemoryScope.DAILY,
                key=operation.key,
                value=operation.value,
                confidence=operation.confidence,
                weight=LEVEL_WEIGHT[MemoryScope.DAILY] * operation.confidence,
                source_memory_id=f"query:{operation.key}",
            )
            for operation in operations
        ]
        constraints.sort(key=lambda memory: (memory.weight, memory.confidence), reverse=True)
        return constraints[:_LEVEL_CAP]

    @staticmethod
    def _require_v1_user(user_id: str) -> None:
        if user_id != V1_USER_ID:
            raise ValueError(f"MemoryProjector only supports user_id {V1_USER_ID!r}")
