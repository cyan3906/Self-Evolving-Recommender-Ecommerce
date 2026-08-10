import pytest

from agents.memory.models import (
    MemoryDecisionType,
    MemoryOperation,
    MemoryOperationType,
    MemoryScope,
    MemoryType,
)
from agents.memory.tools import CommitMemoryDecisionTool


def operation(user_id: str, key: str) -> MemoryOperation:
    return MemoryOperation(
        operation=MemoryOperationType.UPSERT,
        user_id=user_id,
        memory_type=MemoryType.BRAND_PREFERENCE,
        scope=MemoryScope.RECENT,
        key=key,
        value={"brand": key.split(":", 1)[-1]},
        confidence=0.7,
        source="test",
        reason="tool contract test",
        source_event_id="event-1",
    )


def test_commit_tool_applies_typed_upsert_and_soft_delete(store):
    tool = CommitMemoryDecisionTool(store)
    created = tool.run(
        event_id="event-1",
        user_id="user_001",
        operations=[operation("user_001", "brand:huawei")],
    )
    deleted = tool.run(
        event_id="event-2",
        user_id="user_001",
        operations=[operation("user_001", "brand:huawei").model_copy(update={
            "operation": MemoryOperationType.SOFT_DELETE,
            "reason": "user removed preference",
        })],
    )

    assert created.decision is MemoryDecisionType.REMEMBERED
    assert [record.key for record in created.changed_memories] == ["brand:huawei"]
    assert deleted.soft_deleted_count == 1
    assert store.get_active("user_001") == []


def test_commit_tool_validates_entire_user_boundary_before_writing(store):
    tool = CommitMemoryDecisionTool(store)

    with pytest.raises(ValueError, match="another user"):
        tool.run(
            event_id="event-1",
            user_id="user_001",
            operations=[
                operation("user_001", "brand:huawei"),
                operation("user_002", "brand:apple"),
            ],
        )

    assert store.get_active("user_001") == []
    assert store.get_active("user_002") == []
