import pytest

from agents.memory.models import (
    MemoryContext,
    MemoryScope,
    MemoryType,
    RerankRequest,
    WeightedMemory,
)
from agents.memory.reranker import MemoryReranker


PRODUCTS = [
    {
        "id": "huawei-phone",
        "brand": "Huawei",
        "category": "phone",
        "price": 4999,
        "description": "Photography phone with a powerful camera",
    },
    {
        "id": "apple-phone",
        "brand": "Apple",
        "category": "phone",
        "price": 7999,
        "description": "Apple smartphone",
    },
    {
        "id": "sony-headphones",
        "brand": "Sony",
        "category": "headphones",
        "price": 1999,
        "description": "Noise-cancelling wireless headphones",
    },
]


def memory(
    memory_type: MemoryType,
    scope: MemoryScope,
    value: dict,
    *,
    weight: float = 1.0,
    key: str = "deliberately:unrelated:key",
) -> WeightedMemory:
    return WeightedMemory(
        memory_type=memory_type,
        scope=scope,
        key=key,
        value=value,
        confidence=1.0,
        weight=weight,
        source_memory_id=f"test:{memory_type.value}:{scope.value}",
    )


def context(**overrides) -> MemoryContext:
    return MemoryContext(user_id="user_001", scene="search", **overrides)


def rerank(candidates, memory_context):
    return MemoryReranker().rerank(RerankRequest(
        query="phone",
        candidates=candidates,
        memory_context=memory_context,
    ))


def test_empty_context_preserves_input_order_and_original_ranks():
    ranked = rerank(PRODUCTS, context())

    assert [item.product["id"] for item in ranked] == [item["id"] for item in PRODUCTS]
    assert [item.original_rank for item in ranked] == [1, 2, 3]
    assert [item.final_rank for item in ranked] == [1, 2, 3]
    assert all(item.memory_score == 0 for item in ranked)
    assert all(item.memory_reasons == [] for item in ranked)


def test_current_apple_constraint_outweighs_contradictory_long_term_huawei_preference():
    memory_context = context(
        current_constraints=[
            memory(
                MemoryType.BRAND_PREFERENCE,
                MemoryScope.DAILY,
                {"brand": "Apple"},
                key="not-a-brand-key",
            ),
        ],
        long_term_preferences=[
            memory(
                MemoryType.BRAND_PREFERENCE,
                MemoryScope.LONG_TERM,
                {"brand": "Huawei"},
                key="brand:apple",
            ),
        ],
    )

    ranked = rerank(PRODUCTS, memory_context)

    assert ranked[0].product["id"] == "apple-phone"
    assert any("brand" in reason.casefold() for reason in ranked[0].memory_reasons)
    huawei = next(item for item in ranked if item.product["id"] == "huawei-phone")
    assert any("contradicts" in reason.casefold() for reason in huawei.memory_reasons)


def test_recent_photography_preference_raises_matching_phone_without_brand_constraint():
    candidates = [
        {**PRODUCTS[1], "rrf_score": 0.50},
        {**PRODUCTS[0], "rrf_score": 0.49},
        {**PRODUCTS[2], "rrf_score": 0.10},
    ]
    memory_context = context(recent_preferences=[
        memory(
            MemoryType.FEATURE_PREFERENCE,
            MemoryScope.RECENT,
            {"feature": "photography"},
            key="brand:apple",
        ),
    ])

    ranked = rerank(candidates, memory_context)

    assert ranked[0].product["id"] == "huawei-phone"
    assert ranked[0].original_rank == 2
    assert any("photography" in reason.casefold() for reason in ranked[0].memory_reasons)


def test_explicit_max_price_violation_adds_penalty_reason():
    memory_context = context(current_constraints=[
        memory(
            MemoryType.PRICE_RANGE,
            MemoryScope.DAILY,
            {"max_price": 5000},
            key="feature:photography",
        ),
    ])

    ranked = rerank(PRODUCTS, memory_context)
    apple = next(item for item in ranked if item.product["id"] == "apple-phone")

    assert apple.memory_score < 0
    assert any("exceeds max price" in reason.casefold() for reason in apple.memory_reasons)
    assert all(
        item.memory_reasons
        for item in ranked
        if item.final_rank != item.original_rank
    )


def test_malformed_candidate_isolated_with_base_score_and_diagnostic_reason():
    malformed = {**PRODUCTS[1], "price": {"unexpected": "mapping"}}
    memory_context = context(current_constraints=[
        memory(MemoryType.PRICE_RANGE, MemoryScope.DAILY, {"max_price": 5000}),
    ])

    ranked = rerank([PRODUCTS[0], malformed, PRODUCTS[2]], memory_context)
    apple = next(item for item in ranked if item.product["id"] == "apple-phone")

    assert apple.memory_score == 0
    assert apple.final_score == pytest.approx(0.55 * apple.retrieval_score)
    assert any("malformed candidate" in reason.casefold() for reason in apple.memory_reasons)


def test_indirect_rank_change_is_explained_when_another_candidate_is_penalized():
    memory_context = context(negative_preferences=[
        memory(
            MemoryType.NEGATIVE_PREFERENCE,
            MemoryScope.LONG_TERM,
            {"kind": "brand", "value": "Huawei"},
        ),
    ])

    ranked = rerank(PRODUCTS, memory_context)
    apple = next(item for item in ranked if item.product["id"] == "apple-phone")

    assert apple.final_rank < apple.original_rank
    assert apple.memory_reasons


def test_equal_retrieval_scores_fall_back_to_rank_normalization_and_stable_order():
    candidates = [{**product, "rrf_score": 0.25} for product in PRODUCTS]

    ranked = rerank(candidates, context())

    assert [item.retrieval_score for item in ranked] == [1.0, 0.5, 0.0]
    assert [item.original_rank for item in ranked] == [1, 2, 3]
