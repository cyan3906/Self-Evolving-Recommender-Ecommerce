from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from agents.memory.models import (
    MemoryScope,
    MemoryType,
    RankedProduct,
    RerankRequest,
    RerankWeights,
    WeightedMemory,
)


class MemoryReranker:
    """Apply projected memory to retrieval results without reading or mutating memory."""

    def __init__(self, weights: RerankWeights | None = None) -> None:
        self._weights = weights or RerankWeights()

    def rerank(self, request: RerankRequest) -> list[RankedProduct]:
        try:
            return self._rerank(request)
        except Exception:
            return self._unchanged(request)

    def _rerank(self, request: RerankRequest) -> list[RankedProduct]:
        retrieval_scores = _retrieval_scores(request.candidates)
        if request.memory_context.is_empty:
            return self._unchanged(request, retrieval_scores)

        ranked = [
            self._score_candidate(product, original_rank, retrieval_scores[original_rank - 1], request)
            for original_rank, product in enumerate(request.candidates, start=1)
        ]
        ranked.sort(key=lambda item: (-item.final_score, item.original_rank))
        explained: list[RankedProduct] = []
        for rank, item in enumerate(ranked, start=1):
            reasons = item.memory_reasons
            if rank != item.original_rank and not reasons:
                reasons = [
                    f"Rank changed from {item.original_rank} to {rank} because memory adjusted other candidates"
                ]
            explained.append(item.model_copy(update={"final_rank": rank, "memory_reasons": reasons}))
        return explained

    def _score_candidate(
        self,
        product: dict[str, Any],
        original_rank: int,
        retrieval_score: float,
        request: RerankRequest,
    ) -> RankedProduct:
        base_score = self._weights.retrieval * retrieval_score
        try:
            _validate_product(product)
            memory_score, reasons = self._memory_score(product, request)
        except Exception as exc:
            memory_score = 0.0
            reasons = [f"Malformed candidate ignored for memory scoring: {exc}"]
        return RankedProduct(
            product=product,
            original_rank=original_rank,
            final_rank=original_rank,
            retrieval_score=retrieval_score,
            memory_score=memory_score,
            final_score=base_score + memory_score,
            memory_reasons=reasons,
        )

    def _memory_score(
        self,
        product: dict[str, Any],
        request: RerankRequest,
    ) -> tuple[float, list[str]]:
        context = request.memory_context
        score = 0.0
        reasons: list[str] = []

        contribution, level_reasons = self._score_memories(
            product, context.current_constraints, explicit=True
        )
        score += contribution
        reasons.extend(level_reasons)

        for memories in (
            context.daily_intents,
            context.recent_preferences,
            context.long_term_preferences,
        ):
            contribution, level_reasons = self._score_memories(product, memories)
            score += contribution
            reasons.extend(level_reasons)

        for memory in context.negative_preferences:
            if _matches(product, memory):
                penalty = self._weights.negative_penalty * memory.weight
                score -= penalty
                reasons.append(
                    f"Negative {memory.memory_type.value} matched ({_memory_label(memory)}): -{penalty:.3f}"
                )
        return score, reasons

    def _score_memories(
        self,
        product: dict[str, Any],
        memories: Iterable[WeightedMemory],
        *,
        explicit: bool = False,
    ) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []
        for memory in memories:
            level_weight = _level_weight(memory.scope, self._weights)
            if memory.memory_type is MemoryType.PRICE_RANGE:
                price_result = _price_result(product, memory)
                if price_result is None:
                    continue
                matches, detail = price_result
            else:
                matches = _matches(product, memory)
                detail = _memory_label(memory)

            if matches:
                reward = level_weight * memory.weight
                score += reward
                reasons.append(
                    f"Matched {memory.memory_type.value} ({detail}): +{reward:.3f}"
                )
            elif explicit and memory.memory_type in {
                MemoryType.BRAND_PREFERENCE,
                MemoryType.CATEGORY_PREFERENCE,
                MemoryType.PRICE_RANGE,
            }:
                penalty = self._weights.negative_penalty * memory.weight
                score -= penalty
                reasons.append(
                    f"Contradicts current {memory.memory_type.value} ({detail}): -{penalty:.3f}"
                )
        return score, reasons

    def _unchanged(
        self,
        request: RerankRequest,
        retrieval_scores: list[float] | None = None,
    ) -> list[RankedProduct]:
        scores = retrieval_scores or _rank_scores(len(request.candidates))
        return [
            RankedProduct(
                product=product,
                original_rank=rank,
                final_rank=rank,
                retrieval_score=scores[rank - 1],
                memory_score=0.0,
                final_score=self._weights.retrieval * scores[rank - 1],
                memory_reasons=[],
            )
            for rank, product in enumerate(request.candidates, start=1)
        ]


def _normalize(value: object) -> str:
    return str(value).casefold().strip()


def _value(memory: WeightedMemory, field: str) -> object | None:
    value = memory.value.get(field)
    if value is None:
        value = memory.value.get("value")
    return value


def _matches(product: dict[str, Any], memory: WeightedMemory) -> bool:
    if memory.memory_type is MemoryType.BRAND_PREFERENCE:
        return _same(product["brand"], _value(memory, "brand"))
    if memory.memory_type is MemoryType.CATEGORY_PREFERENCE:
        return _same(product["category"], _value(memory, "category"))
    if memory.memory_type is MemoryType.FEATURE_PREFERENCE:
        return _contains(product, _value(memory, "feature"))
    if memory.memory_type in {MemoryType.SHOPPING_INTENT, MemoryType.PURCHASED_PRODUCT}:
        product_id = _value(memory, "product_id")
        return _same(product["id"], product_id) if product_id is not None else _contains(
            product, _value(memory, "intent")
        )
    if memory.memory_type is MemoryType.NEGATIVE_PREFERENCE:
        return _negative_match(product, memory)
    if memory.memory_type is MemoryType.PRICE_RANGE:
        result = _price_result(product, memory)
        return bool(result and result[0])
    return False


def _negative_match(product: dict[str, Any], memory: WeightedMemory) -> bool:
    kind = _normalize(memory.value.get("kind", ""))
    target = memory.value.get("value")
    if kind == "brand":
        return _same(product["brand"], target)
    if kind == "category":
        return _same(product["category"], target)
    if kind == "product_id":
        return _same(product["id"], target)
    return _contains(product, target)


def _same(actual: object, expected: object | None) -> bool:
    return expected is not None and _normalize(actual) == _normalize(expected)


def _contains(product: dict[str, Any], expected: object | None) -> bool:
    needle = _normalize(expected) if expected is not None else ""
    if not needle:
        return False
    searchable = " ".join(_normalize(product[field]) for field in (
        "id", "brand", "category", "description"
    ))
    return needle in searchable


def _price_result(
    product: dict[str, Any], memory: WeightedMemory
) -> tuple[bool, str] | None:
    minimum = _finite_number(memory.value.get("min_price"), optional=True)
    maximum = _finite_number(memory.value.get("max_price"), optional=True)
    if minimum is None and maximum is None:
        return None
    price = _finite_number(product["price"])
    if minimum is not None and price < minimum:
        return False, f"price {price:g} below min price {minimum:g}"
    if maximum is not None and price > maximum:
        return False, f"price {price:g} exceeds max price {maximum:g}"
    bounds = []
    if minimum is not None:
        bounds.append(f"min price {minimum:g}")
    if maximum is not None:
        bounds.append(f"max price {maximum:g}")
    return True, f"price {price:g} within {' and '.join(bounds)}"


def _finite_number(value: object, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("price must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("price must be a finite number")
    return number


def _memory_label(memory: WeightedMemory) -> str:
    fields = {
        MemoryType.BRAND_PREFERENCE: "brand",
        MemoryType.CATEGORY_PREFERENCE: "category",
        MemoryType.FEATURE_PREFERENCE: "feature",
        MemoryType.SHOPPING_INTENT: "intent",
        MemoryType.PURCHASED_PRODUCT: "product_id",
    }
    field = fields.get(memory.memory_type)
    value = _value(memory, field) if field else memory.value.get("value")
    return _normalize(value) or memory.memory_type.value


def _level_weight(scope: MemoryScope, weights: RerankWeights) -> float:
    if scope is MemoryScope.DAILY:
        return weights.current_and_daily
    if scope is MemoryScope.RECENT:
        return weights.recent
    return weights.long_term


def _retrieval_scores(candidates: list[dict[str, Any]]) -> list[float]:
    raw_scores = [candidate.get("rrf_score") for candidate in candidates]
    if not raw_scores or any(not _valid_score(score) for score in raw_scores):
        return _rank_scores(len(candidates))
    numeric_scores = [float(score) for score in raw_scores]
    minimum = min(numeric_scores)
    maximum = max(numeric_scores)
    if maximum == minimum:
        return _rank_scores(len(candidates))
    return [(score - minimum) / (maximum - minimum) for score in numeric_scores]


def _valid_score(score: object) -> bool:
    return (
        not isinstance(score, bool)
        and isinstance(score, (int, float))
        and math.isfinite(float(score))
    )


def _rank_scores(candidate_count: int) -> list[float]:
    return [
        1.0 - ((rank - 1) / max(1, candidate_count - 1))
        for rank in range(1, candidate_count + 1)
    ]


def _validate_product(product: dict[str, Any]) -> None:
    for field in ("id", "brand", "category", "description"):
        value = product[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
    _finite_number(product["price"])
