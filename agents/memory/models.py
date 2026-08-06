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
    memory_type: MemoryType
    scope: MemoryScope
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
