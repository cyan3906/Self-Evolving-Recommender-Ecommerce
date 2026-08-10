from agents.memory.agent import MemoryAgent
from agents.memory.models import BehaviorEvent, BehaviorEventType, MemoryScope, utc_now


def test_reflection_consolidates_repeated_recent_memory_idempotently(store):
    agent = MemoryAgent(store)
    for index in range(3):
        agent.observe(BehaviorEvent(
            event_id=f"view-{index}",
            user_id="user_001",
            event_type=BehaviorEventType.VIEW,
            occurred_at=utc_now(),
            payload={"brand": "Huawei"},
        ))

    first = agent.reflect("user_001")
    second = agent.reflect("user_001")

    assert [(memory.key, memory.scope) for memory in first.promoted_memories] == [
        ("brand:huawei", MemoryScope.LONG_TERM)
    ]
    assert second.promoted_memories == []
    assert second.skipped_duplicate_count == 1


def test_recent_memory_shadows_its_reflected_long_term_copy_in_projection(store):
    agent = MemoryAgent(store)
    for index in range(3):
        agent.observe(BehaviorEvent(
            event_id=f"view-{index}",
            user_id="user_001",
            event_type=BehaviorEventType.VIEW,
            occurred_at=utc_now(),
            payload={"brand": "Huawei"},
        ))
    agent.reflect("user_001")

    context = agent.recall_for_search("user_001", "手机")

    assert [memory.key for memory in context.recent_preferences] == ["brand:huawei"]
    assert context.long_term_preferences == []
