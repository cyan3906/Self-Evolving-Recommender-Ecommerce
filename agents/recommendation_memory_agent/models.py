from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class BehaviorEventType(str, Enum):
    """第一版只保留能直接解释商品兴趣的行为。"""

    VIEW = "view"
    CLICK = "click"
    FAVORITE = "favorite"
    CART = "cart"
    PURCHASE = "purchase"


class BehaviorEvent(BaseModel):
    """由埋点层传入的结构化行为；它是证据，不是记忆。"""

    model_config = ConfigDict(extra="forbid")

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
