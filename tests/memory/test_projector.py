from datetime import datetime, timedelta, timezone

import pytest

from agents.memory.models import (
    MemoryOperation,
    MemoryOperationType,
    MemoryScope,
    MemoryType,
)


def seed_memory(
    store,
    *,
    memory_type: MemoryType,
    scope: MemoryScope,
    key: str,
    value: dict,
    confidence: float,
    expires_at: datetime | None = None,
):
    return store.apply(MemoryOperation(
        operation=MemoryOperationType.UPSERT,
        user_id="user_001",
        memory_type=memory_type,
        scope=scope,
        key=key,
        value=value,
        confidence=confidence,
        source="seed",
        reason="projection fixture",
        source_event_id=f"seed:{key}",
        expires_at=expires_at,
    ))


def test_search_projects_current_query_above_matching_persisted_memory(store):
    from agents.memory.projector import MemoryProjector

    seed_memory(
        store, memory_type=MemoryType.BRAND_PREFERENCE, scope=MemoryScope.LONG_TERM,
        key="brand:huawei", value={"brand": "Huawei"}, confidence=0.8,
    )
    seed_memory(
        store, memory_type=MemoryType.FEATURE_PREFERENCE, scope=MemoryScope.RECENT,
        key="feature:photography", value={"feature": "photography"}, confidence=0.9,
    )
    seed_memory(
        store, memory_type=MemoryType.SHOPPING_INTENT, scope=MemoryScope.DAILY,
        key="shopping_intent:replacement", value={"intent": "replacement"}, confidence=0.7,
    )

    context = MemoryProjector(store).for_search("user_001", "给妈妈买苹果手机，预算5000以内")

    assert {(memory.memory_type, memory.key) for memory in context.current_constraints} == {
        (MemoryType.BRAND_PREFERENCE, "brand:apple"),
        (MemoryType.CATEGORY_PREFERENCE, "category:手机"),
        (MemoryType.PRICE_RANGE, "price:max:5000"),
    }
    assert [memory.key for memory in context.daily_intents] == ["shopping_intent:replacement"]
    assert [memory.key for memory in context.recent_preferences] == ["feature:photography"]
    assert [memory.key for memory in context.long_term_preferences] == ["brand:huawei"]
    assert all(memory.scope is MemoryScope.DAILY for memory in context.current_constraints)
    assert all(memory.source_memory_id.startswith("query:") for memory in context.current_constraints)
    assert all(memory.key != "brand:huawei" for memory in context.current_constraints)
    assert context.memory_version == 1


def test_homepage_excludes_l0_and_expired_unrelated_memory(store):
    from agents.memory.projector import MemoryProjector

    seed_memory(
        store, memory_type=MemoryType.SHOPPING_INTENT, scope=MemoryScope.DAILY,
        key="shopping_intent:replacement", value={"intent": "replacement"}, confidence=0.7,
    )
    expired = seed_memory(
        store, memory_type=MemoryType.BRAND_PREFERENCE, scope=MemoryScope.RECENT,
        key="brand:expired", value={"brand": "Expired"}, confidence=0.9,
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    context = MemoryProjector(store).for_homepage("user_001")

    assert context.scene == "homepage"
    assert context.current_constraints == []
    assert [memory.key for memory in context.daily_intents] == ["shopping_intent:replacement"]
    assert all(memory.source_memory_id != expired.id for memory in context.recent_preferences)
    assert store.get_active("user_001")[0].key == "shopping_intent:replacement"


def test_projection_caps_each_level_and_sorts_by_weight_then_confidence(store):
    from agents.memory.projector import MemoryProjector

    for index, confidence in enumerate((0.1, 0.3, 0.5, 0.7, 0.9, 0.8)):
        seed_memory(
            store, memory_type=MemoryType.FEATURE_PREFERENCE, scope=MemoryScope.RECENT,
            key=f"feature:camera-{index}", value={"feature": "camera"}, confidence=confidence,
        )

    context = MemoryProjector(store).for_homepage("user_001")

    assert [memory.confidence for memory in context.recent_preferences] == [0.9, 0.8, 0.7, 0.5, 0.3]
    assert all(memory.weight == pytest.approx(0.65 * memory.confidence) for memory in context.recent_preferences)


def test_search_prioritizes_query_relevant_preference_before_historical_fallback(store):
    from agents.memory.projector import MemoryProjector

    for index in range(6):
        seed_memory(
            store, memory_type=MemoryType.FEATURE_PREFERENCE, scope=MemoryScope.RECENT,
            key=f"feature:unrelated-{index}", value={"feature": f"unrelated-{index}"},
            confidence=0.99,
        )
    seed_memory(
        store, memory_type=MemoryType.BRAND_PREFERENCE, scope=MemoryScope.RECENT,
        key="brand:apple", value={"brand": "Apple"}, confidence=0.1,
    )

    context = MemoryProjector(store).for_search("user_001", "Apple")

    assert context.recent_preferences[0].key == "brand:apple"
    assert "brand:apple" in [memory.key for memory in context.recent_preferences]
    assert len(context.recent_preferences) == 5


def test_projection_uses_strict_bucket_type_and_scope_semantics(store):
    from agents.memory.projector import MemoryProjector

    seed_memory(
        store, memory_type=MemoryType.BRAND_PREFERENCE, scope=MemoryScope.DAILY,
        key="brand:daily", value={"brand": "Daily"}, confidence=0.9,
    )
    seed_memory(
        store, memory_type=MemoryType.SHOPPING_INTENT, scope=MemoryScope.DAILY,
        key="shopping_intent:daily", value={"intent": "daily"}, confidence=0.8,
    )
    seed_memory(
        store, memory_type=MemoryType.SHOPPING_INTENT, scope=MemoryScope.RECENT,
        key="shopping_intent:recent", value={"intent": "recent"}, confidence=0.9,
    )
    seed_memory(
        store, memory_type=MemoryType.FEATURE_PREFERENCE, scope=MemoryScope.RECENT,
        key="feature:camera", value={"feature": "camera"}, confidence=0.8,
    )
    seed_memory(
        store, memory_type=MemoryType.SHOPPING_INTENT, scope=MemoryScope.LONG_TERM,
        key="shopping_intent:long", value={"intent": "long"}, confidence=0.9,
    )
    seed_memory(
        store, memory_type=MemoryType.BRAND_PREFERENCE, scope=MemoryScope.LONG_TERM,
        key="brand:huawei", value={"brand": "Huawei"}, confidence=0.8,
    )
    seed_memory(
        store, memory_type=MemoryType.NEGATIVE_PREFERENCE, scope=MemoryScope.DAILY,
        key="negative:daily", value={"value": "daily"}, confidence=0.7,
    )
    purchased = seed_memory(
        store, memory_type=MemoryType.PURCHASED_PRODUCT, scope=MemoryScope.DURABLE,
        key="purchased_product:sku-7", value={"product_id": "sku-7"}, confidence=0.9,
    )

    homepage = MemoryProjector(store).for_homepage("user_001")
    search = MemoryProjector(store).for_search("user_001", "Apple")

    assert [memory.key for memory in homepage.daily_intents] == ["shopping_intent:daily"]
    assert [memory.key for memory in homepage.recent_preferences] == ["feature:camera"]
    assert [memory.key for memory in homepage.long_term_preferences] == ["brand:huawei"]
    assert {memory.key for memory in homepage.negative_preferences} == {
        "negative:daily", purchased.key,
    }
    assert [memory.key for memory in search.negative_preferences] == ["negative:daily"]


def test_soft_deleted_memory_cannot_appear_in_projection(store):
    from agents.memory.projector import MemoryProjector

    seed_memory(
        store, memory_type=MemoryType.BRAND_PREFERENCE, scope=MemoryScope.RECENT,
        key="brand:huawei", value={"brand": "Huawei"}, confidence=0.8,
    )
    assert store.soft_delete(
        "user_001", MemoryType.BRAND_PREFERENCE, MemoryScope.RECENT,
        "brand:huawei", "user removed preference",
    ) is True

    context = MemoryProjector(store).for_search("user_001", "Huawei")

    assert all(memory.key != "brand:huawei" for memory in context.recent_preferences)


@pytest.mark.parametrize("user_id", ["", "user_002"])
def test_public_projection_methods_reject_users_other_than_v1_identity(store, user_id):
    from agents.memory.projector import MemoryProjector

    projector = MemoryProjector(store)

    with pytest.raises(ValueError, match="user_001"):
        projector.for_homepage(user_id)
    with pytest.raises(ValueError, match="user_001"):
        projector.for_search(user_id, "Apple")
