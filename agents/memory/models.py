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
    EXPRESSION_PROFILE = "expression_profile"


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


class MemoryDecisionType(str, Enum):
    REMEMBERED = "remembered"
    IGNORED = "ignored"


class ExpressionDimension(str, Enum):
    CATEGORY = "category"
    BRAND = "brand"
    BUDGET = "budget"
    USE_CASE = "use_case"
    EXCLUSION = "exclusion"


class ClarificationAction(str, Enum):
    PROCEED = "proceed"
    OFFER_FILTERS = "offer_filters"
    ASK_ONE_CONSTRAINT = "ask_one_constraint"
    SOFT_SUPPLEMENT = "soft_supplement"


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


class MemoryDecision(StrictModel):
    event_id: str
    user_id: str
    decision: MemoryDecisionType
    changed_memories: list[MemoryRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MemoryReflection(StrictModel):
    user_id: str
    promoted_memories: list[MemoryRecord] = Field(default_factory=list)
    expired_memory_count: int = Field(default=0, ge=0)
    skipped_duplicate_count: int = Field(default=0, ge=0)
    reflected_at: datetime = Field(default_factory=utc_now)


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
    memory_type: MemoryType
    scope: MemoryScope
    key: str
    value: dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    source_memory_id: str


class QueryAnalysis(StrictModel):
    query: str
    present_dimensions: list[ExpressionDimension] = Field(default_factory=list)
    missing_dimensions: list[ExpressionDimension] = Field(default_factory=list)
    completeness_score: float = Field(ge=0.0, le=1.0)


class ClarificationDecision(StrictModel):
    action: ClarificationAction
    analysis: QueryAnalysis
    target_dimension: ExpressionDimension | None = None
    profile_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str


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


class SearchRecommendationRequest(StrictModel):
    request_id: str
    user_id: str
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    remember_query: bool = True
    occurred_at: datetime = Field(default_factory=utc_now)

    @field_validator("request_id", "user_id", "query")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value.strip()


class ExpressionProfile(StrictModel):
    user_id: str
    search_count: int = Field(default=0, ge=0)
    dimension_counts: dict[ExpressionDimension, int] = Field(default_factory=dict)
    clarification_offered_count: int = Field(default=0, ge=0)
    clarification_accepted_count: int = Field(default=0, ge=0)
    reformulation_count: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=utc_now)

    @property
    def confidence(self) -> float:
        return min(1.0, self.search_count / 20.0)

    @property
    def clarification_acceptance_rate(self) -> float | None:
        if self.clarification_offered_count == 0:
            return None
        return self.clarification_accepted_count / self.clarification_offered_count

    def disclosure_rate(self, dimension: ExpressionDimension) -> float | None:
        if self.search_count == 0:
            return None
        return self.dimension_counts.get(dimension, 0) / self.search_count


class MemoryRankingTrace(StrictModel):
    memory_version: int = 0
    current_constraint_count: int = 0
    daily_intent_count: int = 0
    recent_preference_count: int = 0
    long_term_preference_count: int = 0
    negative_preference_count: int = 0
    changed_position_count: int = 0
    expression_profile_updated: bool = False
    warnings: list[str] = Field(default_factory=list)


class SearchRecommendationResponse(StrictModel):
    request_id: str
    user_id: str
    query: str
    results: list[RankedProduct]
    clarification: ClarificationDecision
    trace: MemoryRankingTrace


class HomepageRecommendationRequest(StrictModel):
    request_id: str
    user_id: str
    top_k: int = Field(default=10, ge=1, le=100)

    @field_validator("request_id", "user_id")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value.strip()


class HomepageRecommendationResponse(StrictModel):
    request_id: str
    user_id: str
    results: list[RankedProduct]
    trace: MemoryRankingTrace
