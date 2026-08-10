from __future__ import annotations

from datetime import datetime, timedelta

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
from agents.memory.tools import (
    CommitMemoryDecisionTool,
    GetPurchaseBehaviorSummaryTool,
)


_CADENCE_MIN_PURCHASES = 3
_CADENCE_DUE_RATIO = 0.8
_CADENCE_LAPSED_RATIO = 1.8
_CADENCE_DORMANT_RATIO = 4.0


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
        commit_tool: CommitMemoryDecisionTool | None = None,
        purchase_behavior_tool: GetPurchaseBehaviorSummaryTool | None = None,
    ) -> None:
        self._store = store
        self._policy = policy or MemoryWritePolicy()
        self._projector = projector or MemoryProjector(store)
        self._consolidator = consolidator or MemoryConsolidator(store)
        self._expression = expression_learner or ExpressionProfileLearner(store)
        self._commit = commit_tool or CommitMemoryDecisionTool(store)
        self._purchase_behavior = purchase_behavior_tool
        self.last_warnings: list[str] = []

    def observe(self, event: BehaviorEvent) -> MemoryDecision:
        """Turn trusted behavior evidence into one explicit memory decision."""
        self._require_user_id(event.user_id)
        if event.event_type is BehaviorEventType.SEARCH:
            warnings: list[str] = []
            changed: list[MemoryRecord] = []
            profile_record = self._expression.observe_search(event)
            if profile_record is None:
                warnings.append("blank search query did not update expression profile")
            else:
                changed.append(profile_record)
            self.last_warnings = warnings
            return MemoryDecision(
                event_id=event.event_id,
                user_id=event.user_id,
                decision=(
                    MemoryDecisionType.REMEMBERED
                    if changed
                    else MemoryDecisionType.IGNORED
                ),
                changed_memories=changed,
                warnings=warnings,
            )

        plan = self._policy.plan(event)
        operations = list(plan.operations)
        if event.event_type is BehaviorEventType.PURCHASE:
            operations.extend(self._purchase_intent_delete_operations(event))
        decision = self._commit.run(
            event_id=event.event_id,
            user_id=event.user_id,
            operations=operations,
            warnings=plan.warnings,
        )
        self.last_warnings = decision.warnings
        return decision

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

    def refresh_purchase_cadence(
        self,
        user_id: str,
        *,
        as_of: datetime | None = None,
    ) -> MemoryDecision:
        """Refresh repeat-purchase state without turning a user into a fixed label."""
        self._require_user_id(user_id)
        if self._purchase_behavior is None:
            raise RuntimeError("purchase behavior tool is not configured")

        summary = self._purchase_behavior.run(user_id, as_of=as_of)
        existing = {
            record.key: record
            for record in self._store.get_active(user_id)
            if record.memory_type is MemoryType.PURCHASE_CADENCE
            and record.scope is MemoryScope.RECENT
        }
        operations: list[MemoryOperation] = []
        for category in summary.categories:
            expected_days = category.median_interval_days
            if (
                category.purchase_count < _CADENCE_MIN_PURCHASES
                or expected_days is None
                or expected_days <= 0
            ):
                continue

            ratio = category.days_since_purchase / expected_days
            if ratio < _CADENCE_DUE_RATIO:
                status = "active"
            elif ratio <= _CADENCE_LAPSED_RATIO:
                status = "due"
            elif ratio <= _CADENCE_DORMANT_RATIO:
                status = "lapsed"
            else:
                status = "dormant"

            key = f"purchase_cadence:{_normalize(category.category)}"
            due_at = category.last_purchase_at + timedelta(days=expected_days)
            value = {
                "category": category.category,
                "status": status,
                "last_purchase_at": category.last_purchase_at.isoformat(),
                "expected_interval_days": round(expected_days, 4),
                "purchase_count": category.purchase_count,
                "due_at": due_at.isoformat(),
            }
            if key in existing and existing[key].value == value:
                continue
            operations.append(MemoryOperation(
                operation=MemoryOperationType.UPSERT,
                user_id=user_id,
                memory_type=MemoryType.PURCHASE_CADENCE,
                scope=MemoryScope.RECENT,
                key=key,
                value=value,
                confidence=min(0.95, 0.5 + 0.1 * (category.purchase_count - 2)),
                source="purchase_behavior_tool",
                reason=f"category purchase cadence changed to {status}",
                source_event_id=(
                    f"purchase-cadence:{user_id}:{summary.as_of.date().isoformat()}"
                ),
            ))

        decision = self._commit.run(
            event_id=f"purchase-cadence:{user_id}:{summary.as_of.date().isoformat()}",
            user_id=user_id,
            operations=operations,
        )
        self.last_warnings = decision.warnings
        return decision

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
        decision = self._commit.run(
            event_id=event_id,
            user_id=user_id,
            operations=[MemoryOperation(
                operation=MemoryOperationType.SOFT_DELETE,
                user_id=user_id,
                memory_type=memory_type,
                scope=scope,
                key=normalized_key,
                value={},
                confidence=1.0,
                source="explicit_user",
                reason="explicit user memory deletion",
                source_event_id=event_id,
            )],
        )
        return decision.soft_deleted_count == 1

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

    def _purchase_intent_delete_operations(
        self,
        event: BehaviorEvent,
    ) -> list[MemoryOperation]:
        product_id = item_value(event.payload)
        if product_id is None:
            return []
        target_key = f"shopping_intent:{product_id}"
        operations: list[MemoryOperation] = []
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
                operations.append(MemoryOperation(
                    operation=MemoryOperationType.SOFT_DELETE,
                    user_id=event.user_id,
                    memory_type=record.memory_type,
                    scope=record.scope,
                    key=record.key,
                    value=record.value,
                    confidence=1.0,
                    source="purchase_completion",
                    reason="purchase completed shopping intent",
                    source_event_id=event.event_id,
                ))
        return operations

    @staticmethod
    def _require_user_id(user_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("MemoryAgent requires a logged-in user_id")


def _normalize(value: str) -> str:
    return value.strip().casefold()
