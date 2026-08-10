from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agents.memory.agent import MemoryAgent
from agents.memory.models import (
    BehaviorEvent,
    BehaviorEventType,
    MemoryContext,
    MemoryDecision,
    MemoryRankingTrace,
    MemoryType,
    RerankRequest,
    SearchRecommendationRequest,
    SearchRecommendationResponse,
)
from agents.memory.reranker import MemoryReranker


CandidateRetriever = Callable[[str, int], list[dict[str, Any]]]


class SearchRecommendationAgent:
    """Coordinate product recall and ranking through the Memory Agent boundary.

    The retriever is injected deliberately: this package stays independent of
    Elasticsearch, Milvus, and their connection lifecycle.
    """

    def __init__(
        self,
        retriever: CandidateRetriever,
        memory: MemoryAgent,
        reranker: MemoryReranker | None = None,
    ) -> None:
        self._retriever = retriever
        self._memory = memory
        self._reranker = reranker or MemoryReranker()

    def observe(self, event: BehaviorEvent) -> MemoryDecision:
        """Forward behavior evidence without bypassing the Memory Agent."""
        return self._memory.observe(event)

    def search(self, request: SearchRecommendationRequest) -> SearchRecommendationResponse:
        clarification = self._memory.plan_search(request.user_id, request.query)
        candidates = self._retrieve(request.query, request.top_k)
        warnings: list[str] = []

        try:
            context = self._memory.recall_for_search(request.user_id, request.query)
        except Exception as error:
            context = MemoryContext(user_id=request.user_id, scene="search")
            warnings.append(f"memory projection unavailable; retrieval order preserved: {error}")

        ranked = self._reranker.rerank(RerankRequest(
            query=request.query,
            candidates=candidates,
            memory_context=context,
        ))

        expression_profile_updated = False
        if request.remember_query:
            try:
                decision = self._memory.observe(BehaviorEvent(
                    event_id=f"search:{request.request_id}",
                    user_id=request.user_id,
                    event_type=BehaviorEventType.SEARCH,
                    occurred_at=request.occurred_at,
                    payload={"query": request.query},
                ))
                expression_profile_updated = any(
                    record.memory_type is MemoryType.EXPRESSION_PROFILE
                    for record in decision.changed_memories
                )
                warnings.extend(decision.warnings)
            except Exception as error:
                warnings.append(f"query memory write skipped: {error}")

        trace = MemoryRankingTrace(
            memory_version=context.memory_version,
            current_constraint_count=len(context.current_constraints),
            daily_intent_count=len(context.daily_intents),
            recent_preference_count=len(context.recent_preferences),
            long_term_preference_count=len(context.long_term_preferences),
            cadence_signal_count=len(context.cadence_signals),
            negative_preference_count=len(context.negative_preferences),
            changed_position_count=sum(
                item.original_rank != item.final_rank for item in ranked
            ),
            expression_profile_updated=expression_profile_updated,
            warnings=warnings,
        )
        return SearchRecommendationResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            query=request.query,
            results=ranked,
            clarification=clarification,
            trace=trace,
        )

    def _retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        candidates = self._retriever(query, top_k)
        if not isinstance(candidates, list) or any(
            not isinstance(candidate, dict) for candidate in candidates
        ):
            raise TypeError("candidate retriever must return list[dict[str, Any]]")
        return candidates[:top_k]


# Backward-compatible alias; prefer the domain-accurate name above.
MemorySearchAgent = SearchRecommendationAgent
