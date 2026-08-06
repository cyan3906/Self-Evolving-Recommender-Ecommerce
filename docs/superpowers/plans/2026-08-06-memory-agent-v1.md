# Memory Agent V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a SQLite-backed memory agent for one simulated logged-in user and inject four-level memory projections into an explainable reranker without coupling retrieval, storage, and ranking.

**Architecture:** `agents/memory` is a small modular package. `MemoryAgent` validates and writes memory operations through `SQLiteMemoryStore`; `MemoryProjector` converts active records into an immutable scene-specific `MemoryContext`; `MemoryReranker` consumes only a `RerankRequest` and never accesses storage. Existing ES, Milvus, and RRF code remains independently usable.

**Tech Stack:** Python 3.10+, standard-library `sqlite3`, Pydantic 2, `pytest`, existing product dictionaries returned by `server.rag.hybrid_search`.

## Global Constraints

- V1 supports the single simulated logged-in user `user_001`; missing user IDs are rejected.
- Persist formal memory in `db/user_memory.db`; tests always use a temporary SQLite file.
- Keep implementation under `agents/memory/` and tests under `tests/memory/`.
- Do not import Elasticsearch, Milvus, Redis, or `server` from the memory package.
- The LLM may propose candidate operations but cannot access the store or execute mutations.
- Current explicit query constraints always outrank daily, recent, and long-term memory.
- Reranking must fall back to the original candidate order on empty context or internal scoring failure.
- Use soft deletion and append one audit event in the same transaction as every mutation.
- Keep all scoring weights in `RerankWeights`; do not scatter numeric ranking constants through conditionals.

---

## File Map

- Create `agents/memory/__init__.py`: stable public exports.
- Create `agents/memory/models.py`: enums and Pydantic request/response models.
- Create `agents/memory/store.py`: SQLite schema, transactions, CRUD, expiry, and audit reads.
- Create `agents/memory/agent.py`: behavior/query extraction, validation, and write orchestration.
- Create `agents/memory/projector.py`: homepage/search four-level projection.
- Create `agents/memory/reranker.py`: memory scoring, fallback, and explanations.
- Create `tests/memory/conftest.py`: temporary database and reusable product fixtures.
- Create `tests/memory/test_models.py`: model validation.
- Create `tests/memory/test_store.py`: persistence, transaction, delete, and expiry tests.
- Create `tests/memory/test_agent.py`: behavior rules and LLM candidate validation.
- Create `tests/memory/test_projector.py`: L0-L3 priority and scene selection.
- Create `tests/memory/test_reranker.py`: score changes, reasons, and fallback.
- Create `tests/memory/test_memory_flow.py`: complete simulated-user integration flow without external services.

---

### Task 1: Define the typed memory and rerank contracts

**Files:**
- Create: `agents/memory/models.py`
- Create: `tests/memory/test_models.py`
- Create: `tests/memory/__init__.py`

**Interfaces:**
- Produces: `MemoryType`, `MemoryScope`, `MemoryOperationType`, `BehaviorEventType`, `MemoryRecord`, `MemoryOperation`, `BehaviorEvent`, `WeightedMemory`, `MemoryContext`, `RerankRequest`, `RankedProduct`, and `RerankWeights`.
- Consumes: Pydantic 2 only.

- [ ] **Step 1: Write failing model validation tests**

```python
# tests/memory/test_models.py
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
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run: `python -m pytest tests/memory/test_models.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'agents.memory'`.

- [ ] **Step 3: Implement the models**

Implement these exact contracts in `agents/memory/models.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryType(str, Enum):
    CATEGORY_PREFERENCE = "category_preference"
    BRAND_PREFERENCE = "brand_preference"
    PRICE_RANGE = "price_range"
    FEATURE_PREFERENCE = "feature_preference"
    SHOPPING_INTENT = "shopping_intent"
    NEGATIVE_PREFERENCE = "negative_preference"
    PURCHASED_PRODUCT = "purchased_product"


class MemoryScope(str, Enum):
    DAILY = "daily"
    RECENT = "recent"
    LONG_TERM = "long_term"
    DURABLE = "durable"


class MemoryOperationType(str, Enum):
    UPSERT = "upsert"
    SOFT_DELETE = "soft_delete"


class BehaviorEventType(str, Enum):
    VIEW = "view"
    FAVORITE = "favorite"
    CART = "cart"
    PURCHASE = "purchase"
    SEARCH = "search"
    DELETE_MEMORY = "delete_memory"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MemoryOperation(StrictModel):
    operation: MemoryOperationType
    user_id: str
    memory_type: MemoryType
    scope: MemoryScope
    key: str
    value: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    reason: str
    source_event_id: str | None = None
    expires_at: datetime | None = None

    @field_validator("user_id", "key", "source", "reason")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value.strip()


class MemoryRecord(StrictModel):
    id: str
    user_id: str
    memory_type: MemoryType
    scope: MemoryScope
    key: str
    value: dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_count: int = Field(ge=1)
    source: str
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    deleted_at: datetime | None = None
    version: int = Field(ge=1)


class BehaviorEvent(StrictModel):
    event_id: str
    user_id: str
    event_type: BehaviorEventType
    occurred_at: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_id", "user_id")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value.strip()


class WeightedMemory(StrictModel):
    key: str
    value: dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    source_memory_id: str


class MemoryContext(StrictModel):
    user_id: str
    scene: str
    current_constraints: list[WeightedMemory] = Field(default_factory=list)
    daily_intents: list[WeightedMemory] = Field(default_factory=list)
    recent_preferences: list[WeightedMemory] = Field(default_factory=list)
    long_term_preferences: list[WeightedMemory] = Field(default_factory=list)
    negative_preferences: list[WeightedMemory] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)
    memory_version: int = 0

    @property
    def is_empty(self) -> bool:
        return not any((
            self.current_constraints,
            self.daily_intents,
            self.recent_preferences,
            self.long_term_preferences,
            self.negative_preferences,
        ))


class RerankRequest(StrictModel):
    query: str
    candidates: list[dict[str, Any]]
    memory_context: MemoryContext


class RankedProduct(StrictModel):
    product: dict[str, Any]
    original_rank: int
    final_rank: int
    retrieval_score: float
    memory_score: float
    final_score: float
    memory_reasons: list[str] = Field(default_factory=list)


class RerankWeights(StrictModel):
    retrieval: float = 0.55
    current_and_daily: float = 0.25
    recent: float = 0.15
    long_term: float = 0.05
    negative_penalty: float = 0.35
```

Add `tests/memory/__init__.py` as an empty package marker.

- [ ] **Step 4: Run the model tests**

Run: `python -m pytest tests/memory/test_models.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit the typed contracts**

```bash
git add agents/memory/models.py tests/memory/__init__.py tests/memory/test_models.py
git commit -m "实现记忆数据模型"
```

---

### Task 2: Implement transactional SQLite storage and audit history

**Files:**
- Create: `agents/memory/store.py`
- Create: `tests/memory/conftest.py`
- Create: `tests/memory/test_store.py`

**Interfaces:**
- Consumes: `MemoryOperation`, `MemoryRecord`, `MemoryOperationType`, and `utc_now` from Task 1.
- Produces: `SQLiteMemoryStore.initialize() -> None`, `apply(operation: MemoryOperation) -> MemoryRecord`, `get_active(user_id: str) -> list[MemoryRecord]`, `soft_delete(user_id: str, memory_type: MemoryType, scope: MemoryScope, key: str, reason: str, source_event_id: str | None = None) -> bool`, `expire(user_id: str, now: datetime) -> int`, and `list_events(user_id: str) -> list[dict[str, Any]]`.

- [ ] **Step 1: Write failing store tests**

Create a temporary-store fixture and tests covering idempotent initialization, upsert-without-duplication, atomic audit creation, soft deletion, and expiry. The central test must assert:

```python
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
```

- [ ] **Step 2: Run the store tests and confirm failure**

Run: `python -m pytest tests/memory/test_store.py -q`

Expected: collection fails because `agents.memory.store` does not exist.

- [ ] **Step 3: Implement the SQLite store**

Use `sqlite3.connect`, set `row_factory=sqlite3.Row`, enable foreign keys, and initialize these tables:

```sql
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
```

`apply()` must open one transaction, read the active uniqueness key, and either create a UUID record or update it. Use this bounded confidence rule for compatible evidence:

```python
new_confidence = min(1.0, old_confidence + incoming_confidence * (1.0 - old_confidence))
```

Serialize JSON with `ensure_ascii=False` and stable key ordering. Store UTC timestamps in ISO 8601 form. Append the before/after event before committing. Convert every returned row to `MemoryRecord` in one private `_row_to_record()` method.

For `soft_delete`, set `deleted_at`, increment `version`, and append a `soft_delete` event in the same transaction. `expire()` performs the same transition with operation `expire` for active rows whose `expires_at <= now`.

- [ ] **Step 4: Run the store tests**

Run: `python -m pytest tests/memory/test_store.py -q`

Expected: all store tests pass and the temporary database contains no duplicate active key.

- [ ] **Step 5: Commit storage**

```bash
git add agents/memory/store.py tests/memory/conftest.py tests/memory/test_store.py
git commit -m "实现SQLite记忆存储"
```

---

### Task 3: Implement MemoryAgent write orchestration

**Files:**
- Create: `agents/memory/agent.py`
- Create: `tests/memory/test_agent.py`

**Interfaces:**
- Consumes: `BehaviorEvent`, `MemoryOperation`, enums, and `SQLiteMemoryStore`.
- Produces: `MemoryAgent.record_event(event: BehaviorEvent) -> list[MemoryRecord]`, `record_query(event: BehaviorEvent) -> list[MemoryRecord]`, and `delete_memory(user_id: str, memory_type: MemoryType, scope: MemoryScope, key: str, event_id: str) -> bool`.
- Defines: `QueryOperationExtractor` protocol with `extract(user_id: str, query: str, event_id: str) -> list[MemoryOperation]`, `LLMQueryOperationExtractor`, and `extract_deterministic_query_operations(user_id: str, query: str, event_id: str, occurred_at: datetime) -> list[MemoryOperation]`.

- [ ] **Step 1: Write failing behavior and extractor tests**

Cover these exact behaviors:

- a single view produces recent category and brand evidence but not long-term memory.
- cart produces daily shopping intent.
- purchase creates durable purchase memory and soft-deletes matching daily intent.
- explicit delete soft-deletes the requested key.
- malformed LLM JSON and schema-invalid LLM operations are rejected and produce no LLM-derived write.
- deterministic query parsing still extracts `Huawei`, `手机`, and `5000` from `华为手机预算5000以内` when no LLM extractor is installed.

Use a fake text generator that returns one valid JSON operation; use invalid-JSON and invalid-enum generators and assert deterministic operations still persist.

- [ ] **Step 2: Run the agent tests and confirm failure**

Run: `python -m pytest tests/memory/test_agent.py -q`

Expected: collection fails because `agents.memory.agent` does not exist.

- [ ] **Step 3: Implement the agent**

Define centralized constants:

```python
SIGNAL_CONFIDENCE = {
    BehaviorEventType.VIEW: 0.15,
    BehaviorEventType.FAVORITE: 0.35,
    BehaviorEventType.CART: 0.55,
    BehaviorEventType.PURCHASE: 0.90,
    BehaviorEventType.SEARCH: 0.60,
}

SCOPE_TTL = {
    MemoryScope.DAILY: timedelta(days=1),
    MemoryScope.RECENT: timedelta(days=7),
    MemoryScope.LONG_TERM: timedelta(days=90),
    MemoryScope.DURABLE: None,
}
```

Normalize keys with lowercase stripped text while preserving the display value inside `value`. Behavior payload keys are `product_id`, `category`, `brand`, `features`, `price`, and `memory_key`.

Implement deterministic query patterns for:

- supported brands from the current product fixture: Apple, Huawei/华为, Xiaomi/小米, Samsung/三星, OPPO, Sony/索尼, Lenovo/联想, Dell/戴尔, Nike, Adidas.
- categories appearing in `tools/product_data.py`.
- budget forms `预算5000`, `5000以内`, `不超过5000`, and `3000到5000`.
- explicit negative forms `不要X`, `不考虑X`, and `排除X`.

The agent must collect candidate operations, validate them as `MemoryOperation`, and call `store.apply()` only after validation. Catch extractor exceptions, preserve deterministic output, and expose the error through `last_warnings: list[str]` for demo diagnostics.

Implement `LLMQueryOperationExtractor` without importing `server` or an SDK. Its constructor accepts `generate: Callable[[str], str]`. It builds a prompt containing the allowed memory types/scopes and requires a JSON array. Parse with `json.loads`, inject the trusted `user_id`, `source="llm_query"`, and `source_event_id`, then validate each item with `MemoryOperation.model_validate`. Reject the complete LLM batch if JSON is invalid or any item fails validation. This adapter lets the caller supply the existing LLM client while keeping tests and package imports offline.

For purchase, create `purchased_product` in durable scope and soft-delete matching daily `shopping_intent` records returned by `store.get_active()`.

- [ ] **Step 4: Run agent and store tests**

Run: `python -m pytest tests/memory/test_agent.py tests/memory/test_store.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit agent write path**

```bash
git add agents/memory/agent.py tests/memory/test_agent.py
git commit -m "实现记忆智能体写入策略"
```

---

### Task 4: Implement four-level scene projection

**Files:**
- Create: `agents/memory/projector.py`
- Create: `tests/memory/test_projector.py`

**Interfaces:**
- Consumes: `SQLiteMemoryStore`, `extract_deterministic_query_operations`, `MemoryRecord`, `MemoryContext`, and `WeightedMemory`.
- Produces: `MemoryProjector.for_homepage(user_id) -> MemoryContext` and `for_search(user_id, query) -> MemoryContext`.

- [ ] **Step 1: Write failing projection tests**

Test a user with:

- long-term Huawei preference.
- recent photography feature preference.
- daily replacement intent.
- current query `给妈妈买苹果手机，预算5000以内`.

Assert the search context contains Apple and budget in `current_constraints`, replacement in `daily_intents`, photography in `recent_preferences`, and Huawei in `long_term_preferences`. Assert current Apple remains L0 and Huawei remains only L3. Assert homepage has no L0 constraints and excludes unrelated expired memory.

- [ ] **Step 2: Run the projection tests and confirm failure**

Run: `python -m pytest tests/memory/test_projector.py -q`

Expected: collection fails because `agents.memory.projector` does not exist.

- [ ] **Step 3: Implement projection**

At the beginning of each projection, call `store.expire(user_id, utc_now())`, then load active rows. Map scope to level weights:

```python
LEVEL_WEIGHT = {
    MemoryScope.DAILY: 1.0,
    MemoryScope.RECENT: 0.65,
    MemoryScope.LONG_TERM: 0.35,
    MemoryScope.DURABLE: 0.25,
}
```

Set each `WeightedMemory.weight` to `LEVEL_WEIGHT[scope] * confidence`. For search, create ephemeral L0 `WeightedMemory` values from deterministic query parsing; use source IDs prefixed with `query:` so they cannot be confused with persisted rows. Select persisted memories by category/brand/feature overlap with the normalized query plus all active daily intents and negative preferences. Cap each level at five records, sorted by `(weight, confidence)` descending. Set `memory_version` to the maximum active record version, or zero for no records.

For homepage, include daily intent, recent preference, long-term preference, negative preference, and durable purchased-product suppression; leave `current_constraints` empty.

- [ ] **Step 4: Run projection tests**

Run: `python -m pytest tests/memory/test_projector.py tests/memory/test_agent.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit projection**

```bash
git add agents/memory/projector.py tests/memory/test_projector.py
git commit -m "实现分层记忆投影"
```

---

### Task 5: Implement explainable memory-aware reranking

**Files:**
- Create: `agents/memory/reranker.py`
- Create: `tests/memory/test_reranker.py`

**Interfaces:**
- Consumes: `RerankRequest`, `RankedProduct`, `RerankWeights`, `MemoryContext`, and product dictionaries containing `id`, `brand`, `category`, `price`, `description`, and optionally `rrf_score`.
- Produces: `MemoryReranker.rerank(request) -> list[RankedProduct]`.

- [ ] **Step 1: Write failing reranker tests**

Use three synthetic candidates: Huawei photography phone at 4,999; Apple phone at 7,999; Sony headphones at 1,999. Test:

- empty context preserves input order and original ranks.
- current Apple constraint outranks contradictory long-term Huawei preference.
- recent photography preference raises the Huawei photography phone when there is no explicit brand constraint.
- explicit max-price violation creates a penalty reason.
- an intentionally malformed candidate does not crash the full request and retains a base score.

- [ ] **Step 2: Run reranker tests and confirm failure**

Run: `python -m pytest tests/memory/test_reranker.py -q`

Expected: collection fails because `agents.memory.reranker` does not exist.

- [ ] **Step 3: Implement scoring and fallback**

Normalize retrieval scores by rank when `rrf_score` is missing or all scores are equal:

```python
retrieval_score = 1.0 - ((original_rank - 1) / max(1, candidate_count - 1))
```

Implement pure helpers for normalized text, value matching, price constraints, per-level score accumulation, and reasons. L0 explicit brand/category matches add under `current_and_daily`; contradictions receive a strong penalty. L1 uses the same weight bucket, L2 uses `recent`, and L3 uses `long_term`. Negative preferences subtract `negative_penalty * memory.weight`.

Compute:

```python
final_score = (
    weights.retrieval * retrieval_score
    + memory_score
)
```

Sort by `final_score` descending, then `original_rank` ascending for stability. If the context is empty, return unchanged order with zero memory score. Wrap scoring for each candidate so malformed fields produce a diagnostic reason and base retrieval score; wrap the request-level operation so an unexpected exception returns unchanged order.

- [ ] **Step 4: Run reranker tests**

Run: `python -m pytest tests/memory/test_reranker.py -q`

Expected: all tests pass and each ranking change has at least one memory reason.

- [ ] **Step 5: Commit reranking**

```bash
git add agents/memory/reranker.py tests/memory/test_reranker.py
git commit -m "实现记忆感知重排"
```

---

### Task 6: Expose the package and verify the complete simulated flow

**Files:**
- Create: `agents/memory/__init__.py`
- Create: `tests/memory/test_memory_flow.py`
- Modify: `agents/__init__.py`

**Interfaces:**
- Consumes: all public types and services from Tasks 1-5.
- Produces: stable imports from `agents.memory`; demonstrates event -> memory -> projection -> rerank for `user_001`.

- [ ] **Step 1: Write the failing integration test**

The test must:

1. create a temporary store.
2. record repeated Huawei phone views, a photography search, and a cart event.
3. build a search context for `适合拍照的手机，预算5000以内`.
4. rerank synthetic RRF candidates.
5. assert Huawei photography phone becomes first, memory reasons are non-empty, active memory is inspectable, and audit events exist.
6. create an explicit Apple query and assert L0 Apple overrides long-term Huawei preference.
7. soft-delete the Huawei brand memory and assert it disappears from the next projection.

- [ ] **Step 2: Run the integration test and confirm missing public exports**

Run: `python -m pytest tests/memory/test_memory_flow.py -q`

Expected: import fails because public exports are not defined.

- [ ] **Step 3: Add public exports**

Export only these names from `agents/memory/__init__.py`:

```python
from .agent import (
    LLMQueryOperationExtractor,
    MemoryAgent,
    QueryOperationExtractor,
    extract_deterministic_query_operations,
)
from .models import (
    BehaviorEvent,
    BehaviorEventType,
    MemoryContext,
    MemoryOperation,
    MemoryOperationType,
    MemoryRecord,
    MemoryScope,
    MemoryType,
    RankedProduct,
    RerankRequest,
    RerankWeights,
    WeightedMemory,
)
from .projector import MemoryProjector
from .reranker import MemoryReranker
from .store import SQLiteMemoryStore

__all__ = [
    "BehaviorEvent",
    "BehaviorEventType",
    "LLMQueryOperationExtractor",
    "MemoryAgent",
    "MemoryContext",
    "MemoryOperation",
    "MemoryOperationType",
    "MemoryProjector",
    "MemoryRecord",
    "MemoryReranker",
    "MemoryScope",
    "MemoryType",
    "QueryOperationExtractor",
    "RankedProduct",
    "RerankRequest",
    "RerankWeights",
    "SQLiteMemoryStore",
    "WeightedMemory",
    "extract_deterministic_query_operations",
]
```

Keep `agents/__init__.py` free of eager database initialization; it may re-export the `memory` package name only.

- [ ] **Step 4: Run all memory tests**

Run: `python -m pytest tests/memory -q`

Expected: all memory tests pass without network access or running services.

- [ ] **Step 5: Run regression checks**

Run: `python -m pytest tests/test_ab_test.py -q`

Expected: either pass, or report an existing upstream import incompatibility unrelated to `agents.memory`; record the exact outcome without modifying A/B code in this feature.

Run: `python -m compileall -q agents/memory`

Expected: exit code 0.

Run: `git diff --check`

Expected: exit code 0.

- [ ] **Step 6: Commit the complete public flow**

```bash
git add agents/__init__.py agents/memory/__init__.py tests/memory/test_memory_flow.py
git commit -m "接入初级记忆智能体流程"
```

---

## Final Verification Checklist

- [ ] `python -m pytest tests/memory -q` passes.
- [ ] `python -m compileall -q agents/memory` passes.
- [ ] `git diff --check` passes.
- [ ] The test database is temporary and `db/user_memory.db` is not committed.
- [ ] `agents.memory` imports without Elasticsearch, Milvus, Redis, or network access.
- [ ] Empty memory preserves original order.
- [ ] Explicit query intent overrides contradictory long-term memory.
- [ ] Every memory mutation is visible in audit history.
- [ ] Reranker never receives a store and never writes memory.
- [ ] Existing `server.rag.hybrid_search` remains unchanged.
