# Memory Agent for Search Recommendation

The memory package is one domain service with two consumers:

Public request and response examples are documented in [AGENT_IO.md](./AGENT_IO.md).

- `HomepageRecommendationAgent` uses shopping intent, qualified preferences,
  negative feedback, and purchase suppression.
- `SearchRecommendationAgent` keeps current query constraints at the highest
  authority, uses relevant product memory only as a soft reranking signal, and
  uses the expression profile only to choose a clarification policy.

## Write policy

| Evidence | Persistent effect | Injection rule |
| --- | --- | --- |
| One view | Low-confidence recent evidence | Not projected until confidence reaches the threshold |
| Repeated views | Strengthens the same recent assertion | Eligible after sufficient evidence |
| Favorite | Long-term product preference | Eligible immediately |
| Add to cart | Daily shopping intent | Strong homepage/search signal |
| Purchase | Durable purchase record | Closes matching shopping intent |
| Search query | Updates expression profile only | Explicit constraints remain request-local L0 |

LLM output is not a write authority. A future extractor may propose typed
evidence, but `MemoryWritePolicy` and the Pydantic contracts remain the gate.

## Expression evolution

`ExpressionProfileLearner` maintains bounded sufficient statistics instead of
storing every query summary:

- dimension disclosure counts for category, brand, budget, and use case;
- clarification offered/accepted counts;
- query reformulation count;
- sample-based profile confidence.

The resulting policy is reversible and non-blocking:

- complete query: proceed;
- cold start: offer filters;
- user usually accepts clarification: ask one high-value constraint;
- user usually rejects clarification: proceed with soft memory supplementation;
- normally explicit user omits a dimension: treat it as unconstrained.

## Runtime wiring

Infrastructure is injected so importing this package never connects to
Elasticsearch or Milvus:

```python
search_agent = SearchRecommendationAgent(
    lambda query, top_k: hybrid_search(query, final_top_k=top_k),
    memory_agent,
)
```

The homepage candidate source is intentionally another injected function because
the repository does not yet implement homepage recall.

## Evaluation

Run memory behavior as independent A/B treatments:

1. retrieval only;
2. product-memory reranking;
3. product-memory reranking plus expression-aware clarification.

Track CTR, add-to-cart rate, conversion, search reformulation, clarification
acceptance, abandonment, and the number of rankings changed by memory.
