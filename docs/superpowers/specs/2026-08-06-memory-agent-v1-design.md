# Memory Agent V1 Design

## 1. Goal

Add a small, understandable memory mechanism to the existing search recommendation prototype. The first version supports one simulated logged-in user (`user_001`) and demonstrates the complete loop:

1. Receive browsing, search, favorite, cart, purchase, and explicit-delete events.
2. Convert events into validated structured memory operations.
3. Persist current memory and its change history in SQLite.
4. Project only relevant memory for homepage or search recommendation.
5. Use the projected memory as a read-only input to reranking.
6. Return explainable ranking changes without coupling the reranker to storage.

V1 is a modular monolith under `agents/memory/`. It does not implement anonymous identity, cross-device identity merging, distributed storage, or production event ingestion.

## 2. Existing System Boundary

The existing retrieval path remains unchanged:

```text
query -> Elasticsearch recall + Milvus recall -> RRF fusion
```

Memory-aware recommendation is added above that path:

```text
query
  -> hybrid_search(query)
  -> MemoryProjector builds a scene-specific MemoryContext
  -> MemoryReranker reranks the RRF candidates
  -> ranked products with memory explanations
```

`server/rag.py` continues to own recall and RRF fusion. The memory package does not write to Elasticsearch or Milvus.

## 3. Package Structure

```text
agents/
└── memory/
    ├── __init__.py
    ├── models.py
    ├── store.py
    ├── agent.py
    ├── projector.py
    └── reranker.py
```

Responsibilities:

- `models.py`: Pydantic models and enums for memory records, behavior events, candidate operations, projected contexts, rerank requests, and rerank results.
- `store.py`: SQLite schema initialization, current-memory CRUD, soft deletion, expiry, and append-only audit events.
- `agent.py`: behavior-rule extraction, optional LLM query-intent extraction, operation validation, and coordination with the store.
- `projector.py`: selects active memory, resolves priorities, and builds homepage/search read-only projections.
- `reranker.py`: scores candidates using the query, RRF result, and projected memory. It never accesses or mutates storage.

No V1 module should become a general orchestration framework. If a file approaches roughly 250 lines or gains a second unrelated responsibility, it should be split during implementation review.

## 4. Memory Model

### 4.1 Supported memory types

- `category_preference`: preferred product category.
- `brand_preference`: preferred brand.
- `price_range`: explicit or inferred budget range.
- `feature_preference`: desired product characteristics such as photography or battery life.
- `shopping_intent`: replacement, comparison, discount seeking, or purchase stage.
- `negative_preference`: explicitly rejected brand, category, or feature.
- `purchased_product`: completed purchase used to suppress immediate duplicate recommendations.

### 4.2 Memory scopes

- `daily`: strong current-shopping intent, default TTL 1 day.
- `recent`: repeated recent interest, default TTL 7 days.
- `long_term`: stable preference, default TTL 90 days.
- `durable`: explicit purchase record; no automatic expiry in V1.

### 4.3 Current-memory record

Every active or soft-deleted memory contains:

- `id`: stable UUID.
- `user_id`: V1 uses `user_001`.
- `memory_type`.
- `scope`.
- `key`: normalized identity within the type, such as `brand:huawei`.
- `value_json`: typed value serialized as JSON.
- `confidence`: value in `[0, 1]`.
- `evidence_count`.
- `source`: search, view, favorite, cart, purchase, or manual.
- `created_at`, `updated_at`, `expires_at`.
- `deleted_at`: null for active records.
- `version`: incremented on each update.

The active uniqueness key is `(user_id, memory_type, scope, key)`. Repeated evidence updates the same record instead of creating duplicates.

### 4.4 Audit event

Every mutation appends a `memory_events` row containing:

- operation: create, update, soft-delete, expire, or restore.
- memory ID and user ID.
- before and after snapshots.
- reason and source event ID.
- timestamp.

Audit events are never injected into reranking.

## 5. SQLite Design

Database location: `db/user_memory.db`.

Tables:

- `user_memories`: materialized current state, including soft-deleted records.
- `memory_events`: append-only mutation history.

The store owns transactions. Updating a memory and appending its audit event must happen in one transaction. SQLite rows use JSON text for typed values, while all external callers use Pydantic models rather than raw dictionaries.

The store exposes narrowly scoped operations:

- initialize schema.
- get active memories for a user.
- upsert a validated memory operation.
- soft-delete by memory ID or normalized key.
- expire records whose TTL has passed.
- list audit events for debugging.

## 6. Write Path and CRUD Rules

### 6.1 Behavior events

Deterministic rules handle structured events:

- view: weak evidence; repeated views can create or strengthen recent category/brand/feature preferences.
- favorite: medium evidence.
- cart: strong daily purchase intent and medium recent preference evidence.
- purchase: durable purchase memory, stronger long-term preference evidence, and closure of matching daily purchase intent.
- explicit delete: soft-delete the selected memory.

Signal weights are centralized configuration, not scattered conditionals. Initial values are test fixtures rather than claims of optimal ranking quality.

### 6.2 Search text

Query processing has two stages:

1. Deterministic parsing handles explicit brands, categories, and numeric budget bounds where possible.
2. An injected LLM extractor may produce candidate operations for semantic intents such as replacement, comparison, discount seeking, and negative preferences.

The LLM returns candidate operations only. Before persistence, the agent validates the enum values, confidence bounds, value schema, TTL, and allowed memory types. Invalid, malformed, or unavailable LLM output produces no write and does not block recommendation.

### 6.3 Update and conflict behavior

- Repeated compatible evidence increments `evidence_count`, updates `updated_at`, and raises confidence with a bounded monotonic rule.
- An explicit current-query constraint outranks inferred history.
- A new explicit daily price range replaces a conflicting inferred daily price range; the change is audited.
- Negative preferences must be explicit in V1. A skipped impression is not enough to create a negative preference.
- One isolated view cannot create a long-term preference.
- Purchase closes the matching daily shopping intent rather than deleting its audit history.

### 6.4 Delete and expiry

Deletion is soft deletion. Expired rows receive `deleted_at` and an `expire` audit event. Expiry runs on read before projection; a later background cleanup job is out of scope.

## 7. Four-Level Projection

The projector never returns raw database rows. It creates an immutable `MemoryContext` with four priority levels:

### L0: current explicit constraints

Derived from the current query: category, requested brand, budget, excluded brands/features, and other explicit requirements. L0 has highest priority and may enforce hard filtering or strong penalties.

### L1: daily shopping intent

Examples: replacement, price comparison, discount seeking, photography-phone shopping, or decision-stage intent. L1 strongly affects reranking but cannot override contradictory L0 constraints.

### L2: recent interests

Examples: repeated category views, recent favorite/cart evidence, and recent feature interest. L2 moderately affects reranking.

### L3: long-term preferences

Examples: stable brand affinity, common price range, category affinity, and price sensitivity. L3 is a weak prior and must never override L0.

Audit history, raw behavior logs, deleted memories, and unrelated memories are excluded from the projection.

### Scene-specific views

- Homepage projection uses L1, L2, and L3 because there is no current query. It may also suppress recently purchased products.
- Search projection starts from L0 and adds only query-relevant L1/L2/L3 memories. Current query semantics always dominate historical preference.

The projector caps the number of injected items per level and sorts by confidence, recency, and evidence count to keep prompts and feature vectors bounded.

## 8. Reranker Integration

The reranker accepts one request containing:

- query.
- candidate products returned by RRF.
- immutable `MemoryContext`.

It returns products with:

- original rank and RRF score.
- memory score.
- final score and final rank.
- structured `memory_reasons`.

The first rule-based formula uses centralized configurable weights:

```text
final score = normalized retrieval score
            + L0 query/constraint score
            + L1 daily-intent score
            + L2 recent-interest score
            + L3 long-term-preference score
            - negative-preference penalties
```

Initial relative influence is retrieval 55%, L0/L1 25%, L2 15%, and L3 5%. Hard exclusions and explicit budget violations are handled separately from soft preference scoring. These values are defaults for demonstration and later A/B testing.

The memory-aware rule reranker and a future LLM reranker must share the same request/response models. A/B assignment chooses the reranker implementation; it does not access memory storage.

## 9. Public V1 API

The memory package provides conceptually small entry points:

- record one structured behavior event.
- analyze and record one search query.
- delete one memory explicitly.
- get homepage memory context.
- get search memory context for a query.
- rerank candidates with a supplied context.
- inspect active memory and audit history for the demo user.

A demo flow will use `user_001`, seed a small sequence of behavior/query events, print memory before and after mutation, rerank synthetic candidates or retrieved candidates, and print ranking explanations.

## 10. Failure Handling

- SQLite failures roll back both current-state and audit writes.
- LLM timeout, invalid JSON, or schema mismatch creates no memory operation and falls back to deterministic extraction.
- Missing user ID is rejected at the package boundary.
- Unknown event and memory types are rejected before storage.
- Expired or soft-deleted records cannot appear in `MemoryContext`.
- A reranker failure returns the original RRF order rather than breaking search recommendation.
- One malformed candidate is skipped or left at its base score and reported in diagnostic metadata.

## 11. Testing

Tests use a temporary SQLite database and do not require Elasticsearch, Milvus, Redis, or a live LLM.

Required coverage:

1. schema initialization is idempotent.
2. repeated evidence updates one memory rather than duplicating it.
3. every mutation creates exactly one audit event in the same transaction.
4. conflicting explicit daily budget replaces inferred daily budget.
5. soft-deleted and expired records are excluded from projections.
6. L0 overrides contradictory long-term preference.
7. homepage and search projections select different subsets.
8. malformed LLM output creates no write and deterministic extraction still works.
9. memory-aware reranking changes an expected candidate order and returns reasons.
10. empty memory reproduces the original candidate order.
11. storage is never accessed from the reranker test path.

## 12. Non-Goals for V1

- anonymous users or identity merging.
- distributed transactions or concurrent multi-process writes beyond SQLite defaults.
- Redis caching.
- vector retrieval over memories.
- automatic deletion of source behavioral logs.
- learned ranking weights or online model training.
- production privacy/consent UI.
- allowing the LLM to execute persistence operations directly.
- modifying ES/Milvus recall based on memory.

## 13. Acceptance Criteria

V1 is complete when:

- one simulated user can create, update, soft-delete, expire, and inspect memories.
- all mutations are auditable.
- homepage and search projections follow the four-level priority model.
- an explicit query constraint demonstrably overrides contradictory long-term preference.
- memory-aware reranking can change candidate order with explainable reasons.
- empty or failed memory processing preserves the baseline RRF order.
- unit tests pass without external services.
- existing retrieval code remains independently usable.
