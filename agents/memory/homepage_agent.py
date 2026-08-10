from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agents.memory.agent import MemoryAgent
from agents.memory.models import (
    HomepageRecommendationRequest,
    HomepageRecommendationResponse,
    MemoryContext,
    MemoryRankingTrace,
    RerankRequest,
)
from agents.memory.reranker import MemoryReranker


HomepageCandidateRetriever = Callable[[str, int], list[dict[str, Any]]]


class HomepageRecommendationAgent:
    """Apply Memory Agent projections to an injected homepage candidate source."""

    def __init__(
        self,
        retriever: HomepageCandidateRetriever,
        memory: MemoryAgent,
        reranker: MemoryReranker | None = None,
    ) -> None:
        self._retriever = retriever
        self._memory = memory
        self._reranker = reranker or MemoryReranker()

    def recommend(
        self,
        request: HomepageRecommendationRequest,
    ) -> HomepageRecommendationResponse:
        candidates = self._retrieve(request.user_id, request.top_k)
        warnings: list[str] = []
        try:
            context = self._memory.recall_for_homepage(request.user_id)
        except Exception as error:
            context = MemoryContext(user_id=request.user_id, scene="homepage")
            warnings.append(
                f"memory projection unavailable; candidate order preserved: {error}"
            )

        ranked = self._reranker.rerank(RerankRequest(
            query="",
            candidates=candidates,
            memory_context=context,
        ))
        return HomepageRecommendationResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            results=ranked,
            trace=MemoryRankingTrace(
                memory_version=context.memory_version,
                daily_intent_count=len(context.daily_intents),
                recent_preference_count=len(context.recent_preferences),
                long_term_preference_count=len(context.long_term_preferences),
                cadence_signal_count=len(context.cadence_signals),
                negative_preference_count=len(context.negative_preferences),
                changed_position_count=sum(
                    item.original_rank != item.final_rank for item in ranked
                ),
                warnings=warnings,
            ),
        )

    def _retrieve(self, user_id: str, top_k: int) -> list[dict[str, Any]]:
        candidates = self._retriever(user_id, top_k)
        if not isinstance(candidates, list) or any(
            not isinstance(candidate, dict) for candidate in candidates
        ):
            raise TypeError("homepage retriever must return list[dict[str, Any]]")
        return candidates[:top_k]
