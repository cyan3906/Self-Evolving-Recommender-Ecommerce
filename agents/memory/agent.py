from __future__ import annotations

from agents.memory.consolidator import MemoryConsolidator
from agents.memory.expression import ExpressionProfileLearner
from agents.memory.models import (
    BehaviorEvent,
    BehaviorEventType,
    ClarificationDecision,
    MemoryContext,
    MemoryDecision,
    MemoryDecisionType,
    MemoryOperation,
    MemoryOperationType,
    MemoryRecord,
    MemoryReflection,
    MemoryScope,
    MemoryType,
)
from agents.memory.policy import MemoryWritePolicy, item_value
from agents.memory.projector import MemoryProjector
from agents.memory.store import SQLiteMemoryStore


class MemoryAgent:
    """Own the observe, recall, reflect, and forget lifecycle of user memory."""

    def __init__(
        self,
        store: SQLiteMemoryStore,
        *,
        policy: MemoryWritePolicy | None = None,
        projector: MemoryProjector | None = None,
        consolidator: MemoryConsolidator | None = None,
        expression_learner: ExpressionProfileLearner | None = None,
    ) -> None:
        self._store = store
        self._policy = policy or MemoryWritePolicy()
        self._projector = projector or MemoryProjector(store)
        self._consolidator = consolidator or MemoryConsolidator(store)
        self._expression = expression_learner or ExpressionProfileLearner(store)
        self.last_warnings: list[str] = []

    def observe(self, event: BehaviorEvent) -> MemoryDecision:
        """Turn trusted behavior evidence into one explicit memory decision."""
        self._require_user_id(event.user_id)
        warnings: list[str] = []
        changed: list[MemoryRecord] = []

        if event.event_type is BehaviorEventType.SEARCH:
            profile_record = self._expression.observe_search(event)
            if profile_record is None:
                warnings.append("blank search query did not update expression profile")
            else:
                changed.append(profile_record)
        else:
            plan = self._policy.plan(event)
            warnings.extend(plan.warnings)
            changed.extend(self._apply(plan.operations, warnings))

        if event.event_type is BehaviorEventType.PURCHASE:
            self._close_purchase_intents(event)

        self.last_warnings = warnings
        return MemoryDecision(
            event_id=event.event_id,
            user_id=event.user_id,
            decision=(
                MemoryDecisionType.REMEMBERED if changed else MemoryDecisionType.IGNORED
            ),
            changed_memories=changed,
            warnings=warnings,
        )

    def plan_search(self, user_id: str, query: str) -> ClarificationDecision:
        """Choose a non-ranking interaction policy from query completeness and habit."""
        self._require_user_id(user_id)
        return self._expression.decide(user_id, query)

    def recall_for_search(self, user_id: str, query: str) -> MemoryContext:
        self._require_user_id(user_id)
        return self._projector.for_search(user_id, query)

    def recall_for_homepage(self, user_id: str) -> MemoryContext:
        self._require_user_id(user_id)
        return self._projector.for_homepage(user_id)

    def reflect(self, user_id: str) -> MemoryReflection:
        self._require_user_id(user_id)
        return self._consolidator.reflect(user_id)

    def forget(
        self,
        user_id: str,
        memory_type: MemoryType,
        scope: MemoryScope,
        key: str,
        event_id: str,
    ) -> bool:
        self._require_user_id(user_id)
        normalized_key = _normalize(key)
        if not normalized_key:
            raise ValueError("memory key cannot be blank")
        return self._store.soft_delete(
            user_id,
            memory_type,
            scope,
            normalized_key,
            reason="explicit user memory deletion",
            source_event_id=event_id,
        )

    # Compatibility wrappers keep callers on the audited Agent boundary.
    def record_event(self, event: BehaviorEvent) -> list[MemoryRecord]:
        return self.observe(event).changed_memories

    def record_query(self, event: BehaviorEvent) -> list[MemoryRecord]:
        if event.event_type is not BehaviorEventType.SEARCH:
            raise ValueError("record_query requires a search event")
        return self.record_event(event)

    def delete_memory(
        self,
        user_id: str,
        memory_type: MemoryType,
        scope: MemoryScope,
        key: str,
        event_id: str,
    ) -> bool:
        return self.forget(user_id, memory_type, scope, key, event_id)

    def _apply(
        self,
        candidates: tuple[MemoryOperation, ...],
        warnings: list[str],
    ) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for candidate in candidates:
            operation = MemoryOperation.model_validate(candidate)
            if operation.operation is not MemoryOperationType.UPSERT:
                warnings.append(f"unsupported memory operation ignored: {operation.operation}")
                continue
            records.append(self._store.apply(operation))
        return records

    def _close_purchase_intents(self, event: BehaviorEvent) -> None:
        product_id = item_value(event.payload)
        if product_id is None:
            return
        target_key = f"shopping_intent:{product_id}"
        for record in self._store.get_active(event.user_id):
            recorded_product = record.value.get("product_id")
            matches_value = (
                isinstance(recorded_product, str)
                and _normalize(recorded_product) == product_id
            )
            if (
                record.memory_type is MemoryType.SHOPPING_INTENT
                and record.scope is MemoryScope.DAILY
                and (record.key == target_key or matches_value)
            ):
                self._store.soft_delete(
                    event.user_id,
                    record.memory_type,
                    record.scope,
                    record.key,
                    reason="purchase completed shopping intent",
                    source_event_id=event.event_id,
                )

    @staticmethod
    def _require_user_id(user_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("MemoryAgent requires a logged-in user_id")


def _normalize(value: str) -> str:
    return value.strip().casefold()
