from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agents.memory.models import (
    BehaviorEvent,
    BehaviorEventType,
    MemoryOperation,
    MemoryOperationType,
    MemoryScope,
    MemoryType,
)


def test_memory_operation_rejects_confidence_outside_unit_interval():
    with pytest.raises(ValidationError):
        MemoryOperation(
            operation=MemoryOperationType.UPSERT,
            user_id="user_001",
            memory_type=MemoryType.BRAND_PREFERENCE,
            scope=MemoryScope.RECENT,
            key="brand:huawei",
            value={"brand": "Huawei"},
            confidence=1.1,
            source="view",
            reason="repeated views",
        )


def test_behavior_event_requires_nonempty_user_id():
    with pytest.raises(ValidationError):
        BehaviorEvent(
            event_id="evt-1",
            user_id="",
            event_type=BehaviorEventType.VIEW,
            occurred_at=datetime.now(timezone.utc),
            payload={"brand": "Huawei"},
        )
