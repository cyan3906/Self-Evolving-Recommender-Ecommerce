# Memory and Recommendation Agent I/O

This document defines the public input and output contracts for the memory,
search recommendation, and homepage recommendation agents.

## 1. MemoryAgent

### Observe behavior

```python
MemoryAgent.observe(event: BehaviorEvent) -> MemoryDecision
```

Supported event types and payloads:

```json
{
  "event_id": "view-1001",
  "user_id": "user_001",
  "event_type": "view",
  "occurred_at": "2026-08-10T10:00:00Z",
  "payload": {
    "category": "手机",
    "brand": "Huawei"
  }
}
```

```json
{
  "event_id": "favorite-1001",
  "user_id": "user_001",
  "event_type": "favorite",
  "occurred_at": "2026-08-10T10:00:00Z",
  "payload": {
    "category": "手机",
    "brand": "Huawei",
    "features": ["photography"]
  }
}
```

```json
{
  "event_id": "cart-1001",
  "user_id": "user_001",
  "event_type": "cart",
  "occurred_at": "2026-08-10T10:00:00Z",
  "payload": {
    "product_id": "SKU-1001"
  }
}
```

```json
{
  "event_id": "purchase-1001",
  "user_id": "user_001",
  "event_type": "purchase",
  "occurred_at": "2026-08-10T10:00:00Z",
  "payload": {
    "product_id": "SKU-1001"
  }
}
```

A search event updates the expression profile. It does not persist the current
brand, category, budget, or use case as a product preference.

```json
{
  "event_id": "search-1001",
  "user_id": "user_001",
  "event_type": "search",
  "occurred_at": "2026-08-10T10:00:00Z",
  "payload": {
    "query": "华为拍照手机5000以内",
    "clarification_offered": true,
    "clarification_accepted": true,
    "reformulated": false
  }
}
```

Output:

```json
{
  "event_id": "search-1001",
  "user_id": "user_001",
  "decision": "remembered",
  "changed_memories": [
    {
      "id": "memory-uuid",
      "user_id": "user_001",
      "memory_type": "expression_profile",
      "scope": "durable",
      "key": "expression:search",
      "value": {
        "user_id": "user_001",
        "search_count": 5,
        "dimension_counts": {
          "category": 4,
          "brand": 3,
          "budget": 2,
          "use_case": 3
        },
        "clarification_offered_count": 2,
        "clarification_accepted_count": 2,
        "reformulation_count": 0,
        "updated_at": "2026-08-10T10:00:00Z"
      },
      "confidence": 0.56395,
      "evidence_count": 5,
      "source": "expression_learning",
      "created_at": "2026-08-01T10:00:00Z",
      "updated_at": "2026-08-10T10:00:00Z",
      "expires_at": null,
      "deleted_at": null,
      "version": 5
    }
  ],
  "warnings": []
}
```

`decision` is `remembered` when at least one memory changed, otherwise
`ignored`.

### Plan a search interaction

```python
MemoryAgent.plan_search(
    user_id: str,
    query: str,
) -> ClarificationDecision
```

Input:

```json
{
  "user_id": "user_001",
  "query": "手机"
}
```

Output:

```json
{
  "action": "ask_one_constraint",
  "analysis": {
    "query": "手机",
    "present_dimensions": ["category"],
    "missing_dimensions": ["brand", "budget", "use_case"],
    "completeness_score": 0.4
  },
  "target_dimension": "use_case",
  "profile_confidence": 0.25,
  "reason": "user frequently accepts clarification; ask only the highest-value constraint"
}
```

Clarification actions:

| Action | Meaning |
| --- | --- |
| `proceed` | The current query is sufficient; retrieve immediately. |
| `offer_filters` | Show reversible, non-blocking filters. |
| `ask_one_constraint` | Ask only the highest-value missing constraint. |
| `soft_supplement` | Do not interrupt; use relevant product memory as a soft signal. |

### Recall projected memory

```python
MemoryAgent.recall_for_search(user_id: str, query: str) -> MemoryContext
MemoryAgent.recall_for_homepage(user_id: str) -> MemoryContext
```

`MemoryContext` contains:

```json
{
  "user_id": "user_001",
  "scene": "search",
  "current_constraints": [],
  "daily_intents": [],
  "recent_preferences": [],
  "long_term_preferences": [],
  "negative_preferences": [],
  "generated_at": "2026-08-10T10:00:00Z",
  "memory_version": 5
}
```

### Reflect and forget

```python
MemoryAgent.reflect(user_id: str) -> MemoryReflection

MemoryAgent.forget(
    user_id: str,
    memory_type: MemoryType,
    scope: MemoryScope,
    key: str,
    event_id: str,
) -> bool
```

`reflect` consolidates qualified recent preferences into idempotent long-term
memory. `forget` performs an audited soft delete.

## 2. SearchRecommendationAgent

```python
SearchRecommendationAgent.search(
    request: SearchRecommendationRequest,
) -> SearchRecommendationResponse
```

Input:

```json
{
  "request_id": "request-1001",
  "user_id": "user_001",
  "query": "华为拍照手机5000以内",
  "top_k": 10,
  "remember_query": true,
  "occurred_at": "2026-08-10T10:00:00Z"
}
```

Output:

```json
{
  "request_id": "request-1001",
  "user_id": "user_001",
  "query": "华为拍照手机5000以内",
  "results": [
    {
      "product": {
        "id": "SKU-1001",
        "brand": "Huawei",
        "category": "手机",
        "price": 4999,
        "description": "适合摄影的华为手机"
      },
      "original_rank": 3,
      "final_rank": 1,
      "retrieval_score": 0.82,
      "memory_score": 0.21,
      "final_score": 0.661,
      "memory_reasons": [
        "matches current brand constraint",
        "matches photography preference"
      ]
    }
  ],
  "clarification": {
    "action": "proceed",
    "analysis": {
      "query": "华为拍照手机5000以内",
      "present_dimensions": ["brand", "budget", "category", "use_case"],
      "missing_dimensions": [],
      "completeness_score": 1.0
    },
    "target_dimension": null,
    "profile_confidence": 0.25,
    "reason": "current query already contains enough decision information"
  },
  "trace": {
    "memory_version": 5,
    "current_constraint_count": 4,
    "daily_intent_count": 1,
    "recent_preference_count": 2,
    "long_term_preference_count": 1,
    "negative_preference_count": 0,
    "changed_position_count": 3,
    "expression_profile_updated": true,
    "warnings": []
  }
}
```

The search retriever is injected with this contract:

```python
Callable[[query: str, top_k: int], list[dict]]
```

For the current repository it can wrap `hybrid_search(query,
final_top_k=top_k)`.

## 3. HomepageRecommendationAgent

```python
HomepageRecommendationAgent.recommend(
    request: HomepageRecommendationRequest,
) -> HomepageRecommendationResponse
```

Input:

```json
{
  "request_id": "homepage-1001",
  "user_id": "user_001",
  "top_k": 10
}
```

Output:

```json
{
  "request_id": "homepage-1001",
  "user_id": "user_001",
  "results": [
    {
      "product": {
        "id": "SKU-1001",
        "brand": "Huawei",
        "category": "手机",
        "price": 4999,
        "description": "华为手机"
      },
      "original_rank": 4,
      "final_rank": 1,
      "retrieval_score": 0.7,
      "memory_score": 0.18,
      "final_score": 0.565,
      "memory_reasons": ["matches recent brand preference"]
    }
  ],
  "trace": {
    "memory_version": 5,
    "current_constraint_count": 0,
    "daily_intent_count": 1,
    "recent_preference_count": 2,
    "long_term_preference_count": 1,
    "negative_preference_count": 1,
    "changed_position_count": 2,
    "expression_profile_updated": false,
    "warnings": []
  }
}
```

The homepage retriever is injected with this contract:

```python
Callable[[user_id: str, top_k: int], list[dict]]
```

The repository currently has no homepage recall implementation, so this
interface is the integration point for a future popularity, collaborative, or
multi-channel recall service.
