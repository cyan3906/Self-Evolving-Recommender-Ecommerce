from __future__ import annotations

from collections.abc import Iterable

from agents.memory.extractors import extract_deterministic_query_operations
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
_MIN_PROJECTABLE_CONFIDENCE = 0.35
_PREFERENCE_TYPES = {
    MemoryType.CATEGORY_PREFERENCE,
    MemoryType.BRAND_PREFERENCE,
    MemoryType.PRICE_RANGE,
    MemoryType.FEATURE_PREFERENCE,
}


class MemoryProjector:
    """Build immutable, scene-specific memory inputs without writing persisted memory."""

    def __init__(self, store: SQLiteMemoryStore) -> None:
        self._store = store

    def for_homepage(self, user_id: str) -> MemoryContext:
        self._require_user_id(user_id)
        records = self._active_records(user_id)
        return self._context(user_id, "homepage", records)

    def for_search(self, user_id: str, query: str) -> MemoryContext:
        self._require_user_id(user_id)
        records = self._active_records(user_id)
        constraints = self._query_constraints(user_id, query)
        return self._context(
            user_id,
            "search",
            records,
            constraints,
            self._relevance_terms(query, constraints),
        )

    def _active_records(self, user_id: str) -> list[MemoryRecord]:
        self._store.expire(user_id, utc_now())
        return self._store.get_active(user_id)

    def _context(
        self,
        user_id: str,
        scene: str,
        records: list[MemoryRecord],
        current_constraints: list[WeightedMemory] | None = None,
        relevance_terms: set[str] | None = None,
    ) -> MemoryContext:
        daily_intents = self._sorted(
            record for record in records
            if record.scope is MemoryScope.DAILY
            and record.memory_type is MemoryType.SHOPPING_INTENT
        )
        recent_preferences = self._sorted(
            (
                record for record in records
                if record.scope is MemoryScope.RECENT
                and record.memory_type in _PREFERENCE_TYPES
                and record.confidence >= _MIN_PROJECTABLE_CONFIDENCE
            ),
            relevance_terms,
        )
        recent_identities = {
            (memory.memory_type, memory.key) for memory in recent_preferences
        }
        long_term_preferences = self._sorted(
            (
                record for record in records
                if record.scope is MemoryScope.LONG_TERM
                and record.memory_type in _PREFERENCE_TYPES
                and record.confidence >= _MIN_PROJECTABLE_CONFIDENCE
                and (record.memory_type, record.key) not in recent_identities
            ),
            relevance_terms,
        )
        cadence_records = [
            record for record in records
            if record.memory_type is MemoryType.PURCHASE_CADENCE
            and record.scope is MemoryScope.RECENT
            and record.confidence >= _MIN_PROJECTABLE_CONFIDENCE
            and record.value.get("status") in {"due", "lapsed"}
        ]
        if scene == "search":
            cadence_records = [
                record for record in cadence_records
                if self._is_relevant(record, relevance_terms)
            ]
        cadence_signals = self._sorted(cadence_records, relevance_terms)
        negative_preferences = self._sorted(
            record for record in records
            if record.memory_type is MemoryType.NEGATIVE_PREFERENCE
            or (
                scene == "homepage"
                and record.memory_type is MemoryType.PURCHASED_PRODUCT
                and record.scope is MemoryScope.DURABLE
            )
        )
        return MemoryContext(
            user_id=user_id,
            scene=scene,
            current_constraints=current_constraints or [],
            daily_intents=daily_intents,
            recent_preferences=recent_preferences,
            long_term_preferences=long_term_preferences,
            cadence_signals=cadence_signals,
            negative_preferences=negative_preferences,
            memory_version=max((record.version for record in records), default=0),
        )

    @staticmethod
    def _sorted(
        records: Iterable[MemoryRecord],
        relevance_terms: set[str] | None = None,
    ) -> list[WeightedMemory]:
        weighted_records = [
            (MemoryProjector._is_relevant(record, relevance_terms), MemoryProjector._weighted(record))
            for record in records
        ]
        weighted_records.sort(
            key=lambda item: (
                not item[0],
                -item[1].weight,
                -item[1].confidence,
            )
        )
        return [memory for _, memory in weighted_records[:_LEVEL_CAP]]

    @staticmethod
    def _weighted(record: MemoryRecord) -> WeightedMemory:
        weight = LEVEL_WEIGHT[record.scope] * record.confidence
        if (
            record.memory_type is MemoryType.PURCHASE_CADENCE
            and record.value.get("status") == "lapsed"
        ):
            weight *= 0.5
        return WeightedMemory(
            memory_type=record.memory_type,
            scope=record.scope,
            key=record.key,
            value=record.value,
            confidence=record.confidence,
            weight=weight,
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
    def _relevance_terms(query: str, constraints: Iterable[WeightedMemory]) -> set[str]:
        terms = {_normalize(query)}
        for constraint in constraints:
            terms.add(_normalize(constraint.key))
            terms.update(_value_terms(constraint.value))
        return {term for term in terms if term}

    @staticmethod
    def _is_relevant(record: MemoryRecord, relevance_terms: set[str] | None) -> bool:
        if relevance_terms is None:
            return False
        record_terms = {_normalize(record.key), *_value_terms(record.value)}
        return any(
            record_term in query_term or query_term in record_term
            for record_term in record_terms if record_term
            for query_term in relevance_terms
        )

    @staticmethod
    def _require_user_id(user_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("MemoryProjector requires a logged-in user_id")


def _normalize(value: object) -> str:
    return str(value).casefold().strip()


def _value_terms(value: object) -> set[str]:
    if isinstance(value, dict):
        return {term for item in value.values() for term in _value_terms(item)}
    if isinstance(value, (list, tuple, set)):
        return {term for item in value for term in _value_terms(item)}
    return {_normalize(value)}
