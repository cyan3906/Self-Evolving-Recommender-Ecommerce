# Search Recommendation Memory

This context defines how observed shopping behavior becomes bounded, explainable memory for homepage and search ranking. It exists to keep evidence, durable memory, and ranking inputs distinct.

## Language

**Behavior Event**:
An observed action by a logged-in user, such as viewing, favoriting, adding to cart, purchasing, or searching. It is evidence from which memory may be derived, not a preference by itself.
_Avoid_: User preference, memory

**Memory Assertion**:
A confidence-scored statement about a user's shopping intent, preference, exclusion, or purchase, with an explicit lifetime and supporting evidence count.
_Avoid_: User profile, tag

**Memory Operation**:
A validated proposal to create, strengthen, or remove a Memory Assertion. Extractors may propose operations, but only the memory write boundary may persist them.
_Avoid_: LLM output, database command

**Memory Decision**:
The Memory Agent's auditable conclusion about whether a Behavior Event should change memory and which assertions changed.
_Avoid_: Agent response, extraction result

**Active Memory**:
A Memory Assertion that is neither soft-deleted nor expired and is eligible for scene projection.
_Avoid_: All memory, history

**Current Constraint**:
An explicit requirement in the current search, such as brand, category, exclusion, or budget. It has higher ranking authority than every historical Memory Assertion.
_Avoid_: Short-term preference, query memory

**Query Completeness**:
The coverage of decision-relevant dimensions explicitly supplied in one search, such as category, budget, use case, brand, and exclusions. It describes the current request, not the user.
_Avoid_: Query quality, user ability

**Expression Profile**:
A confidence-scored pattern describing which decision dimensions a user tends to provide or omit across searches and how they respond to optional clarification.
_Avoid_: Product preference, communication ability, user personality

**Clarification Policy**:
The selected interaction strategy for an incomplete search: proceed broadly, offer filter suggestions, cautiously supplement from memory, or request one high-value constraint.
_Avoid_: Reranking rule, user preference

**Inferred Constraint**:
A revocable, lower-authority search constraint supplied from relevant memory when the current query omits that dimension. It must remain distinguishable from a Current Constraint.
_Avoid_: Current Constraint, fact

**Memory Projection**:
A bounded, read-only view of Active Memory selected for one recommendation scene and organized by ranking authority and time horizon.
_Avoid_: Prompt dump, full user profile

**Memory Reflection**:
The consolidation of repeated, sufficiently confident short-lived Memory Assertions into idempotent long-term assertions.
_Avoid_: Summary, batch update

**Memory Agent**:
The owner of the logged-in user's memory lifecycle: it observes evidence, makes Memory Decisions, recalls projections, reflects, and forgets. It does not retrieve or rank products.
_Avoid_: Memory writer, profile service

**Candidate Product**:
A product returned by recall and fusion before memory-aware ranking is applied.
_Avoid_: Recommendation result

**Memory-Aware Reranking**:
The ordering of Candidate Products using the current Memory Projection while preserving recall as the base signal.
_Avoid_: Personalized recall, memory retrieval

**Search Recommendation Agent**:
The coordinator that retrieves Candidate Products and requests memory capabilities before reranking them. It never reads or writes the memory store directly.
_Avoid_: Memory Search Agent, Memory Agent

**Ranking Explanation**:
A structured account of the memory signals and rank delta that produced a product's final position.
_Avoid_: Chain of thought, model reasoning
