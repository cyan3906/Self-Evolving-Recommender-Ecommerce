import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from agents.memory.models import MemoryScope, MemoryType

from .conftest import make_brand_operation


def test_initialize_is_idempotent(tmp_path):
    from agents.memory.store import SQLiteMemoryStore

    store = SQLiteMemoryStore(tmp_path / "memories.sqlite3")

    store.initialize()
    store.initialize()

    assert store.get_active("user_001") == []


def test_repeated_evidence_updates_one_record_and_appends_two_events(store):
    first = make_brand_operation(confidence=0.4, event_id="evt-1")
    second = make_brand_operation(confidence=0.6, event_id="evt-2")

    created = store.apply(first)
    updated = store.apply(second)

    active = store.get_active("user_001")
    events = store.list_events("user_001")

    assert created.id == updated.id
    assert len(active) == 1
    assert active[0].evidence_count == 2
    assert active[0].confidence > 0.6
    assert active[0].version == 2
    assert [event["operation"] for event in events] == ["create", "update"]


def test_apply_creates_audit_event_with_before_and_after_snapshots(store):
    record = store.apply(make_brand_operation(confidence=0.4, event_id="evt-1"))

    events = store.list_events("user_001")

    assert len(events) == 1
    assert events[0]["memory_id"] == record.id
    assert events[0]["operation"] == "create"
    assert events[0]["before_json"] is None
    assert events[0]["after_json"]["id"] == record.id
    assert events[0]["source_event_id"] == "evt-1"


def test_apply_rolls_back_memory_when_audit_insert_fails(store):
    with sqlite3.connect(store._database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER abort_memory_event_insert
            BEFORE INSERT ON memory_events
            BEGIN
                SELECT RAISE(ABORT, 'forced audit failure');
            END;
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="forced audit failure"):
        store.apply(make_brand_operation(confidence=0.4, event_id="evt-1"))

    assert store.get_active("user_001") == []
    assert store.list_events("user_001") == []


def test_soft_delete_hides_record_and_appends_audit_event(store):
    record = store.apply(make_brand_operation(confidence=0.4, event_id="evt-1"))

    deleted = store.soft_delete(
        user_id="user_001",
        memory_type=MemoryType.BRAND_PREFERENCE,
        scope=MemoryScope.RECENT,
        key="brand:huawei",
        reason="user removed preference",
        source_event_id="evt-2",
    )

    events = store.list_events("user_001")
    assert deleted is True
    assert store.get_active("user_001") == []
    assert events[-1]["memory_id"] == record.id
    assert events[-1]["operation"] == "soft_delete"
    assert events[-1]["before_json"]["deleted_at"] is None
    assert events[-1]["after_json"]["deleted_at"] is not None
    assert events[-1]["after_json"]["version"] == 2
    assert events[-1]["source_event_id"] == "evt-2"


def test_expire_soft_deletes_only_due_active_records_and_audits_each(store):
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    future = datetime.now(timezone.utc) + timedelta(minutes=1)
    due = store.apply(make_brand_operation(0.4, "evt-due", key="brand:huawei", expires_at=past))
    store.apply(make_brand_operation(0.4, "evt-future", key="brand:honor", expires_at=future))

    expired = store.expire("user_001", datetime.now(timezone.utc))

    active = store.get_active("user_001")
    events = store.list_events("user_001")
    expire_events = [event for event in events if event["operation"] == "expire"]
    assert expired == 1
    assert [record.key for record in active] == ["brand:honor"]
    assert len(expire_events) == 1
    assert expire_events[0]["memory_id"] == due.id
    assert expire_events[0]["after_json"]["deleted_at"] is not None
