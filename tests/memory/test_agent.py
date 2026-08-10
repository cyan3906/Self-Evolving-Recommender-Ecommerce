from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agents.memory.agent import MemoryAgent
from agents.memory.extractors import extract_deterministic_query_operations
from agents.memory.models import (
    BehaviorEvent,
    BehaviorEventType,
    MemoryDecisionType,
    MemoryOperation,
    MemoryOperationType,
    MemoryScope,
    MemoryType,
)


def event(
    event_id: str,
    event_type: BehaviorEventType,
    payload: dict,
    *,
    user_id: str = "user_001",
) -> BehaviorEvent:
    return BehaviorEvent(
        event_id=event_id,
        user_id=user_id,
        event_type=event_type,
        occurred_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
        payload=payload,
    )


def test_view_creates_low_confidence_recent_evidence(store):
    records = MemoryAgent(store).record_event(
        event("view-1", BehaviorEventType.VIEW, {"category": "手机", "brand": "Huawei"})
    )

    assert {(record.memory_type, record.scope, record.key) for record in records} == {
        (MemoryType.CATEGORY_PREFERENCE, MemoryScope.RECENT, "category:手机"),
        (MemoryType.BRAND_PREFERENCE, MemoryScope.RECENT, "brand:huawei"),
    }
    assert all(record.confidence == 0.15 for record in records)


def test_observe_returns_an_auditable_memory_decision(store):
    decision = MemoryAgent(store).observe(
        event("view-decision", BehaviorEventType.VIEW, {"brand": "Huawei"})
    )

    assert decision.decision is MemoryDecisionType.REMEMBERED
    assert decision.event_id == "view-decision"
    assert [record.key for record in decision.changed_memories] == ["brand:huawei"]
    assert decision.warnings == []


def test_search_updates_expression_profile_without_creating_product_preference(store):
    decision = MemoryAgent(store).observe(
        event("search-1", BehaviorEventType.SEARCH, {"query": "Huawei手机5000以内"})
    )

    assert [record.memory_type for record in decision.changed_memories] == [
        MemoryType.EXPRESSION_PROFILE
    ]
    active = store.get_active("user_001")
    assert [record.memory_type for record in active] == [MemoryType.EXPRESSION_PROFILE]
    assert active[0].value["search_count"] == 1


def test_blank_search_is_ignored_without_writing_memory(store):
    decision = MemoryAgent(store).observe(
        event("search-blank", BehaviorEventType.SEARCH, {"query": " "})
    )

    assert decision.decision is MemoryDecisionType.IGNORED
    assert store.get_active("user_001") == []
    assert "blank search query" in decision.warnings[0]


def test_cart_creates_daily_shopping_intent(store):
    records = MemoryAgent(store).record_event(
        event("cart-1", BehaviorEventType.CART, {"product_id": "SKU-7"})
    )

    assert [(record.memory_type, record.scope, record.key) for record in records] == [
        (MemoryType.SHOPPING_INTENT, MemoryScope.DAILY, "shopping_intent:sku-7")
    ]


def test_purchase_creates_durable_product_and_closes_matching_intent(store):
    agent = MemoryAgent(store)
    agent.record_event(event("cart-1", BehaviorEventType.CART, {"product_id": "SKU-7"}))

    records = agent.record_event(
        event("purchase-1", BehaviorEventType.PURCHASE, {"product_id": "SKU-7"})
    )

    assert [(record.memory_type, record.scope) for record in records] == [
        (MemoryType.PURCHASED_PRODUCT, MemoryScope.DURABLE)
    ]
    assert [record.memory_type for record in store.get_active("user_001")] == [
        MemoryType.PURCHASED_PRODUCT
    ]


def test_purchase_closes_intent_matching_product_value(store):
    store.apply(MemoryOperation(
        operation=MemoryOperationType.UPSERT,
        user_id="user_001",
        memory_type=MemoryType.SHOPPING_INTENT,
        scope=MemoryScope.DAILY,
        key="shopping_intent:custom-key",
        value={"product_id": "SKU-7"},
        confidence=0.7,
        source="seed",
        reason="seed daily intent",
        source_event_id="cart-1",
    ))

    MemoryAgent(store).record_event(
        event("purchase-1", BehaviorEventType.PURCHASE, {"product_id": "SKU-7"})
    )

    assert [record.memory_type for record in store.get_active("user_001")] == [
        MemoryType.PURCHASED_PRODUCT
    ]


def test_forget_soft_deletes_requested_key(store):
    agent = MemoryAgent(store)
    agent.record_event(event("view-1", BehaviorEventType.VIEW, {"brand": "Huawei"}))

    assert agent.forget(
        "user_001",
        MemoryType.BRAND_PREFERENCE,
        MemoryScope.RECENT,
        " BRAND:HUAWEI ",
        "delete-1",
    ) is True
    assert store.get_active("user_001") == []


@pytest.mark.parametrize("user_id", ["", "   "])
def test_public_operations_require_logged_in_user(store, user_id):
    agent = MemoryAgent(store)
    invalid_event = BehaviorEvent.model_construct(
        event_id="event-1",
        user_id=user_id,
        event_type=BehaviorEventType.VIEW,
        occurred_at=datetime(2026, 8, 10, tzinfo=timezone.utc),
        payload={"brand": "Huawei"},
    )

    with pytest.raises(ValueError, match="logged-in user_id"):
        agent.observe(invalid_event)
    with pytest.raises(ValueError, match="logged-in user_id"):
        agent.recall_for_homepage(user_id)


def test_agent_isolates_multiple_logged_in_users(store):
    agent = MemoryAgent(store)
    agent.observe(event("view-1", BehaviorEventType.VIEW, {"brand": "Huawei"}))
    agent.observe(event(
        "view-2", BehaviorEventType.VIEW, {"brand": "Apple"}, user_id="user_002"
    ))

    assert [record.key for record in store.get_active("user_001")] == ["brand:huawei"]
    assert [record.key for record in store.get_active("user_002")] == ["brand:apple"]


def test_current_query_extractor_finds_brand_category_use_case_and_budget():
    operations = extract_deterministic_query_operations(
        "user_001",
        "华为拍照手机预算5000以内",
        "query-1",
        datetime(2026, 8, 10, tzinfo=timezone.utc),
    )

    assert {(operation.memory_type, operation.key) for operation in operations} == {
        (MemoryType.BRAND_PREFERENCE, "brand:huawei"),
        (MemoryType.CATEGORY_PREFERENCE, "category:手机"),
        (MemoryType.FEATURE_PREFERENCE, "feature:photography"),
        (MemoryType.PRICE_RANGE, "price:max:5000"),
    }
    assert all(operation.source == "current_query" for operation in operations)


def test_current_query_extractor_keeps_exclusion_ephemeral():
    operations = extract_deterministic_query_operations(
        "user_001",
        "不要Huawei手机",
        "query-1",
        datetime(2026, 8, 10, tzinfo=timezone.utc),
    )

    extracted = {(operation.memory_type, operation.key) for operation in operations}
    assert (MemoryType.NEGATIVE_PREFERENCE, "negative:brand:huawei") in extracted
    assert (MemoryType.CATEGORY_PREFERENCE, "category:手机") in extracted
    assert (MemoryType.NEGATIVE_PREFERENCE, "negative:category:手机") not in extracted
