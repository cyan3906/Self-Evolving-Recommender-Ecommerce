from datetime import datetime, timezone

import pytest

from agents.memory.models import (
    MemoryOperation,
    MemoryOperationType,
    MemoryScope,
    MemoryType,
)
from agents.memory.store import SQLiteMemoryStore


@pytest.fixture
def store(tmp_path):
    memory_store = SQLiteMemoryStore(tmp_path / "memories.sqlite3")
    memory_store.initialize()
    return memory_store


def make_brand_operation(
    confidence: float,
    event_id: str,
    *,
    key: str = "brand:huawei",
    expires_at: datetime | None = None,
) -> MemoryOperation:
    return MemoryOperation(
        operation=MemoryOperationType.UPSERT,
        user_id="user_001",
        memory_type=MemoryType.BRAND_PREFERENCE,
        scope=MemoryScope.RECENT,
        key=key,
        value={"brand": "Huawei"},
        confidence=confidence,
        source="view",
        reason="repeated product views",
        source_event_id=event_id,
        expires_at=expires_at,
    )
