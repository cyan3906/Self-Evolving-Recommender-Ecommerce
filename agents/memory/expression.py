from __future__ import annotations

from agents.memory.extractors import analyze_query
from agents.memory.models import (
    BehaviorEvent,
    ClarificationAction,
    ClarificationDecision,
    ExpressionDimension,
    ExpressionProfile,
    MemoryOperation,
    MemoryOperationType,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    QueryAnalysis,
)
from agents.memory.store import SQLiteMemoryStore


_PROFILE_KEY = "expression:search"
_TARGET_PRIORITY = (
    ExpressionDimension.CATEGORY,
    ExpressionDimension.USE_CASE,
    ExpressionDimension.BUDGET,
    ExpressionDimension.BRAND,
)
_DETAIL_WEIGHTS = {
    ExpressionDimension.CATEGORY: 0.4,
    ExpressionDimension.USE_CASE: 0.3,
    ExpressionDimension.BUDGET: 0.2,
    ExpressionDimension.BRAND: 0.1,
}


class ExpressionProfileLearner:
    """Learn how a user searches without turning expression style into taste."""

    def __init__(self, store: SQLiteMemoryStore) -> None:
        self._store = store

    def observe_search(self, event: BehaviorEvent) -> MemoryRecord | None:
        query = event.payload.get("query", event.payload.get("keyword", ""))
        if not isinstance(query, str) or not query.strip():
            return None

        analysis = analyze_query(query)
        previous = self.get_profile(event.user_id) or ExpressionProfile(user_id=event.user_id)
        dimension_counts = dict(previous.dimension_counts)
        for dimension in analysis.present_dimensions:
            dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1

        accepted = event.payload.get("clarification_accepted") is True
        offered = event.payload.get("clarification_offered") is True or accepted
        profile = previous.model_copy(update={
            "search_count": previous.search_count + 1,
            "dimension_counts": dimension_counts,
            "clarification_offered_count": (
                previous.clarification_offered_count + int(offered)
            ),
            "clarification_accepted_count": (
                previous.clarification_accepted_count + int(accepted)
            ),
            "reformulation_count": (
                previous.reformulation_count
                + int(event.payload.get("reformulated") is True)
            ),
            "updated_at": event.occurred_at,
        })
        return self._store.apply(MemoryOperation(
            operation=MemoryOperationType.UPSERT,
            user_id=event.user_id,
            memory_type=MemoryType.EXPRESSION_PROFILE,
            scope=MemoryScope.DURABLE,
            key=_PROFILE_KEY,
            value=profile.model_dump(mode="json"),
            confidence=max(0.05, profile.confidence),
            source="expression_learning",
            reason="search dimensions and clarification response updated expression profile",
            source_event_id=event.event_id,
        ))

    def decide(self, user_id: str, query: str) -> ClarificationDecision:
        analysis = analyze_query(query)
        profile = self.get_profile(user_id)
        target = next(
            (dimension for dimension in _TARGET_PRIORITY if dimension in analysis.missing_dimensions),
            None,
        )

        if analysis.completeness_score >= 0.7 or target is None:
            return self._decision(
                ClarificationAction.PROCEED,
                analysis,
                profile,
                None,
                "current query already contains enough decision information",
            )
        if profile is None or profile.search_count < 3:
            return self._decision(
                ClarificationAction.OFFER_FILTERS,
                analysis,
                profile,
                target,
                "cold-start profile; offer non-blocking filters instead of guessing",
            )

        detail_score = sum(
            weight * (profile.disclosure_rate(dimension) or 0.0)
            for dimension, weight in _DETAIL_WEIGHTS.items()
        )
        if detail_score >= 0.7:
            return self._decision(
                ClarificationAction.PROCEED,
                analysis,
                profile,
                None,
                "user is usually explicit; omitted dimensions are treated as unconstrained",
            )

        acceptance = profile.clarification_acceptance_rate
        if acceptance is not None and profile.clarification_offered_count >= 2 and acceptance >= 0.5:
            return self._decision(
                ClarificationAction.ASK_ONE_CONSTRAINT,
                analysis,
                profile,
                target,
                "user frequently accepts clarification; ask only the highest-value constraint",
            )
        if acceptance is not None and profile.clarification_offered_count >= 3 and acceptance <= 0.25:
            return self._decision(
                ClarificationAction.SOFT_SUPPLEMENT,
                analysis,
                profile,
                target,
                "user rarely accepts clarification; proceed and use relevant memory as a soft signal",
            )
        return self._decision(
            ClarificationAction.OFFER_FILTERS,
            analysis,
            profile,
            target,
            "query is incomplete; offer reversible filters without blocking retrieval",
        )

    def get_profile(self, user_id: str) -> ExpressionProfile | None:
        records = self._store.get_active(user_id)
        record = next(
            (
                item for item in records
                if item.memory_type is MemoryType.EXPRESSION_PROFILE
                and item.scope is MemoryScope.DURABLE
                and item.key == _PROFILE_KEY
            ),
            None,
        )
        if record is None:
            return None
        return ExpressionProfile.model_validate({**record.value, "user_id": user_id})

    @staticmethod
    def _decision(
        action: ClarificationAction,
        analysis: QueryAnalysis,
        profile: ExpressionProfile | None,
        target: ExpressionDimension | None,
        reason: str,
    ) -> ClarificationDecision:
        return ClarificationDecision(
            action=action,
            analysis=analysis,
            target_dimension=target,
            profile_confidence=profile.confidence if profile else 0.0,
            reason=reason,
        )
