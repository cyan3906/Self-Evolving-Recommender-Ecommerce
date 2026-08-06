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


@pytest.mark.parametrize("user_id", ["", "user_002"])
def test_public_projection_methods_reject_users_other_than_v1_identity(store, user_id):
    from agents.memory.projector import MemoryProjector

    projector = MemoryProjector(store)

    with pytest.raises(ValueError, match="user_001"):
        projector.for_homepage(user_id)
    with pytest.raises(ValueError, match="user_001"):
        projector.for_search(user_id, "Apple")
