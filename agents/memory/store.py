from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.memory.models import (
    MemoryOperation,
    MemoryOperationType,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    utc_now,
)


class SQLiteMemoryStore:
    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)

    def initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_count INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    deleted_at TEXT,
                    version INTEGER NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS uq_active_memory
                ON user_memories(user_id, memory_type, scope, key)
                WHERE deleted_at IS NULL;

                CREATE TABLE IF NOT EXISTS memory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    before_json TEXT,
                    after_json TEXT,
                    reason TEXT NOT NULL,
                    source_event_id TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    def apply(self, operation: MemoryOperation) -> MemoryRecord:
        if operation.operation is not MemoryOperationType.UPSERT:
            raise ValueError("apply only supports upsert operations")

        timestamp = self._timestamp(utc_now())
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT * FROM user_memories
                    WHERE user_id = ? AND memory_type = ? AND scope = ? AND key = ?
                      AND deleted_at IS NULL
                    """,
                    self._identity_params(
                        operation.user_id,
                        operation.memory_type,
                        operation.scope,
                        operation.key,
                    ),
                ).fetchone()

                if row is None:
                    record_id = str(uuid.uuid4())
                    connection.execute(
                        """
                        INSERT INTO user_memories (
                            id, user_id, memory_type, scope, key, value_json,
                            confidence, evidence_count, source, created_at, updated_at,
                            expires_at, deleted_at, version
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                        """,
                        (
                            record_id,
                            operation.user_id,
                            operation.memory_type.value,
                            operation.scope.value,
                            operation.key,
                            self._json_dumps(operation.value),
                            operation.confidence,
                            1,
                            operation.source,
                            timestamp,
                            timestamp,
                            self._optional_timestamp(operation.expires_at),
                            1,
                        ),
                    )
                    after = self._fetch_record(connection, record_id)
                    self._append_event(
                        connection,
                        after,
                        operation="create",
                        before=None,
                        reason=operation.reason,
                        source_event_id=operation.source_event_id,
                        created_at=timestamp,
                    )
                    return after

                before = self._row_to_record(row)
                new_confidence = min(
                    1.0,
                    before.confidence + operation.confidence * (1.0 - before.confidence),
                )
                connection.execute(
                    """
                    UPDATE user_memories
                    SET value_json = ?, confidence = ?, evidence_count = ?, source = ?,
                        updated_at = ?, expires_at = ?, version = ?
                    WHERE id = ?
                    """,
                    (
                        self._json_dumps(operation.value),
                        new_confidence,
                        before.evidence_count + 1,
                        operation.source,
                        timestamp,
                        self._optional_timestamp(operation.expires_at),
                        before.version + 1,
                        before.id,
                    ),
                )
                after = self._fetch_record(connection, before.id)
                self._append_event(
                    connection,
                    after,
                    operation="update",
                    before=before,
                    reason=operation.reason,
                    source_event_id=operation.source_event_id,
                    created_at=timestamp,
                )
                return after

    def get_active(self, user_id: str) -> list[MemoryRecord]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM user_memories
                WHERE user_id = ? AND deleted_at IS NULL
                ORDER BY updated_at DESC, id ASC
                """,
                (user_id,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def soft_delete(
        self,
        user_id: str,
        memory_type: MemoryType,
        scope: MemoryScope,
        key: str,
        reason: str,
        source_event_id: str | None = None,
    ) -> bool:
        timestamp = self._timestamp(utc_now())
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT * FROM user_memories
                    WHERE user_id = ? AND memory_type = ? AND scope = ? AND key = ?
                      AND deleted_at IS NULL
                    """,
                    self._identity_params(user_id, memory_type, scope, key),
                ).fetchone()
                if row is None:
                    return False

                before = self._row_to_record(row)
                connection.execute(
                    """
                    UPDATE user_memories
                    SET deleted_at = ?, updated_at = ?, version = ?
                    WHERE id = ? AND deleted_at IS NULL
                    """,
                    (timestamp, timestamp, before.version + 1, before.id),
                )
                after = self._fetch_record(connection, before.id)
                self._append_event(
                    connection,
                    after,
                    operation="soft_delete",
                    before=before,
                    reason=reason,
                    source_event_id=source_event_id,
                    created_at=timestamp,
                )
                return True

    def expire(self, user_id: str, now: datetime) -> int:
        timestamp = self._timestamp(utc_now())
        cutoff = self._timestamp(now)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    """
                    SELECT * FROM user_memories
                    WHERE user_id = ? AND deleted_at IS NULL AND expires_at IS NOT NULL
                      AND expires_at <= ?
                    """,
                    (user_id, cutoff),
                ).fetchall()
                for row in rows:
                    before = self._row_to_record(row)
                    connection.execute(
                        """
                        UPDATE user_memories
                        SET deleted_at = ?, updated_at = ?, version = ?
                        WHERE id = ? AND deleted_at IS NULL
                        """,
                        (timestamp, timestamp, before.version + 1, before.id),
                    )
                    after = self._fetch_record(connection, before.id)
                    self._append_event(
                        connection,
                        after,
                        operation="expire",
                        before=before,
                        reason="memory expired",
                        source_event_id=None,
                        created_at=timestamp,
                    )
        return len(rows)

    def list_events(self, user_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, memory_id, user_id, operation, before_json, after_json,
                       reason, source_event_id, created_at
                FROM memory_events
                WHERE user_id = ?
                ORDER BY id ASC
                """,
                (user_id,),
            ).fetchall()
        return [
            {
                **dict(row),
                "before_json": self._json_loads(row["before_json"]),
                "after_json": self._json_loads(row["after_json"]),
            }
            for row in rows
        ]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _fetch_record(self, connection: sqlite3.Connection, record_id: str) -> MemoryRecord:
        row = connection.execute(
            "SELECT * FROM user_memories WHERE id = ?", (record_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"memory record {record_id} was not found")
        return self._row_to_record(row)

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            user_id=row["user_id"],
            memory_type=MemoryType(row["memory_type"]),
            scope=MemoryScope(row["scope"]),
            key=row["key"],
            value=json.loads(row["value_json"]),
            confidence=row["confidence"],
            evidence_count=row["evidence_count"],
            source=row["source"],
            created_at=self._datetime(row["created_at"]),
            updated_at=self._datetime(row["updated_at"]),
            expires_at=self._optional_datetime(row["expires_at"]),
            deleted_at=self._optional_datetime(row["deleted_at"]),
            version=row["version"],
        )

    def _append_event(
        self,
        connection: sqlite3.Connection,
        after: MemoryRecord,
        *,
        operation: str,
        before: MemoryRecord | None,
        reason: str,
        source_event_id: str | None,
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_events (
                memory_id, user_id, operation, before_json, after_json, reason,
                source_event_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                after.id,
                after.user_id,
                operation,
                self._record_json(before) if before is not None else None,
                self._record_json(after),
                reason,
                source_event_id,
                created_at,
            ),
        )

    @staticmethod
    def _identity_params(
        user_id: str,
        memory_type: MemoryType,
        scope: MemoryScope,
        key: str,
    ) -> tuple[str, str, str, str]:
        return user_id, memory_type.value, scope.value, key

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _record_json(cls, record: MemoryRecord) -> str:
        return cls._json_dumps(record.model_dump(mode="json"))

    @staticmethod
    def _json_loads(value: str | None) -> dict[str, Any] | None:
        return json.loads(value) if value is not None else None

    @staticmethod
    def _timestamp(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    @classmethod
    def _optional_timestamp(cls, value: datetime | None) -> str | None:
        return cls._timestamp(value) if value is not None else None

    @staticmethod
    def _datetime(value: str) -> datetime:
        return datetime.fromisoformat(value)

    @classmethod
    def _optional_datetime(cls, value: str | None) -> datetime | None:
        return cls._datetime(value) if value is not None else None
