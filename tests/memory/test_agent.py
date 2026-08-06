from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agents.memory.agent import (
    LLMQueryOperationExtractor,
    MemoryAgent,
    extract_deterministic_query_operations,
)
from agents.memory.models import (
    BehaviorEvent,
    BehaviorEventType,
    MemoryOperation,
    MemoryOperationType,
    MemoryScope,
    MemoryType,
)


def event(event_id: str, event_type: BehaviorEventType, payload: dict, *, user_id: str = "user_001") -> BehaviorEvent:
    return BehaviorEvent(
        event_id=event_id,
        user_id=user_id,
        event_type=event_type,
        occurred_at=datetime(2026, 8, 6, tzinfo=timezone.utc),
        payload=payload,
    )


def test_view_creates_recent_category_and_brand_evidence_only(store):
    records = MemoryAgent(store).record_event(
        event("view-1", BehaviorEventType.VIEW, {"category": "Phones", "brand": "Huawei"})
    )

    assert {(record.memory_type, record.scope, record.key) for record in records} == {
        (MemoryType.CATEGORY_PREFERENCE, MemoryScope.RECENT, "category:phones"),
        (MemoryType.BRAND_PREFERENCE, MemoryScope.RECENT, "brand:huawei"),
    }
    assert all(record.scope is not MemoryScope.LONG_TERM for record in records)


def test_cart_creates_daily_shopping_intent(store):
    records = MemoryAgent(store).record_event(
        event("cart-1", BehaviorEventType.CART, {"product_id": "SKU-7", "category": "Phones", "brand": "Huawei"})
    )

    assert len(records) == 1
    assert records[0].memory_type is MemoryType.SHOPPING_INTENT
    assert records[0].scope is MemoryScope.DAILY
    assert records[0].key == "shopping_intent:sku-7"


def test_purchase_creates_durable_product_and_closes_matching_daily_intent(store):
    agent = MemoryAgent(store)
    agent.record_event(event("cart-1", BehaviorEventType.CART, {"product_id": "SKU-7"}))

    records = agent.record_event(event("purchase-1", BehaviorEventType.PURCHASE, {"product_id": "SKU-7", "brand": "Huawei"}))

    assert [(record.memory_type, record.scope, record.key) for record in records] == [
        (MemoryType.PURCHASED_PRODUCT, MemoryScope.DURABLE, "purchased_product:sku-7")
    ]
    assert [record.memory_type for record in store.get_active("user_001")] == [MemoryType.PURCHASED_PRODUCT]


def test_purchase_closes_active_daily_intent_matching_product_value(store):
    store.apply(MemoryOperation(
        operation=MemoryOperationType.UPSERT,
        user_id="user_001",
        memory_type=MemoryType.SHOPPING_INTENT,
        scope=MemoryScope.DAILY,
        key="shopping_intent:custom-key",
        value={"product_id": "SKU-7"},
        confidence=0.55,
        source="seed",
        reason="seed daily intent",
        source_event_id="cart-1",
    ))

    MemoryAgent(store).record_event(event("purchase-1", BehaviorEventType.PURCHASE, {"product_id": "SKU-7"}))

    active = store.get_active("user_001")
    assert [(record.memory_type, record.scope) for record in active] == [
        (MemoryType.PURCHASED_PRODUCT, MemoryScope.DURABLE)
    ]


def test_delete_memory_soft_deletes_requested_key(store):
    agent = MemoryAgent(store)
    agent.record_event(event("view-1", BehaviorEventType.VIEW, {"brand": "Huawei"}))

    assert agent.delete_memory("user_001", MemoryType.BRAND_PREFERENCE, MemoryScope.RECENT, " BRAND:HUAWEI ", "delete-1") is True
    assert store.get_active("user_001") == []


@pytest.mark.parametrize("user_id", ["", "user_002"])
def test_agent_public_operations_reject_users_other_than_v1_identity(store, user_id):
    agent = MemoryAgent(store)
    invalid_event = BehaviorEvent.model_construct(
        event_id="event-1",
        user_id=user_id,
        event_type=BehaviorEventType.VIEW,
        occurred_at=datetime(2026, 8, 6, tzinfo=timezone.utc),
        payload={"brand": "Huawei"},
    )

    with pytest.raises(ValueError, match="user_001"):
        agent.record_event(invalid_event)
    with pytest.raises(ValueError, match="user_001"):
        agent.record_query(invalid_event)
    with pytest.raises(ValueError, match="user_001"):
        agent.delete_memory(user_id, MemoryType.BRAND_PREFERENCE, MemoryScope.RECENT, "brand:huawei", "delete-1")


def test_malformed_llm_json_produces_no_llm_write_and_preserves_deterministic_writes(store):
    agent = MemoryAgent(store, query_extractor=LLMQueryOperationExtractor(lambda _: "not json"))

    records = agent.record_query(event("query-1", BehaviorEventType.SEARCH, {"query": "Huawei"}))

    assert [(record.memory_type, record.key) for record in records] == [(MemoryType.BRAND_PREFERENCE, "brand:huawei")]
    assert any("LLM" in warning for warning in agent.last_warnings)


def test_schema_invalid_llm_batch_produces_no_llm_write_and_preserves_deterministic_writes(store):
    invalid_enum = '[{"operation":"upsert","memory_type":"unknown","scope":"recent","key":"bad","value":{},"confidence":0.9,"reason":"bad"}]'
    agent = MemoryAgent(store, query_extractor=LLMQueryOperationExtractor(lambda _: invalid_enum))

    records = agent.record_query(event("query-1", BehaviorEventType.SEARCH, {"query": "Huawei"}))

    assert [(record.memory_type, record.key) for record in records] == [(MemoryType.BRAND_PREFERENCE, "brand:huawei")]
    assert any("LLM" in warning for warning in agent.last_warnings)


def test_valid_llm_operation_is_applied_with_trusted_identity(store):
    generated = '[{"operation":"upsert","user_id":"attacker","memory_type":"feature_preference","scope":"recent","key":"feature:camera","value":{"feature":"camera"},"confidence":0.8,"source":"forged","reason":"query"}]'
    agent = MemoryAgent(store, query_extractor=LLMQueryOperationExtractor(lambda _: generated))

    records = agent.record_query(event("query-1", BehaviorEventType.SEARCH, {"query": "Huawei"}))

    llm_record = next(record for record in records if record.key == "feature:camera")
    assert llm_record.user_id == "user_001"
    assert llm_record.source == "llm_query"


def test_deterministic_query_parser_extracts_huawei_phone_and_budget_without_llm(store):
    operations = extract_deterministic_query_operations(
        "user_001",
        "鍗庝负鎵嬫満棰勭畻5000浠ュ唴",
        "query-1",
        datetime(2026, 8, 6, tzinfo=timezone.utc),
    )

    assert {(operation.memory_type, operation.key, operation.value.get("value")) for operation in operations} == {
        (MemoryType.BRAND_PREFERENCE, "brand:huawei", "Huawei"),
        (MemoryType.CATEGORY_PREFERENCE, "category:鎵嬫満", "鎵嬫満"),
        (MemoryType.PRICE_RANGE, "price:max:5000", 5000),
    }
    assert all(operation.operation is MemoryOperationType.UPSERT for operation in operations)


def test_deterministic_query_parser_extracts_negative_preference(store):
    agent = MemoryAgent(store)

    records = agent.record_query(event("query-1", BehaviorEventType.SEARCH, {"query": "涓嶈Huawei"}))

    assert [(record.memory_type, record.key) for record in records] == [
        (MemoryType.NEGATIVE_PREFERENCE, "negative:brand:huawei")
    ]
