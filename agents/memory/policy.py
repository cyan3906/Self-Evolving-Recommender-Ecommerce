from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from agents.memory.models import (
    BehaviorEvent,
    BehaviorEventType,
    MemoryOperation,
    MemoryOperationType,
    MemoryScope,
    MemoryType,
)


_SIGNAL_CONFIDENCE = {
    BehaviorEventType.VIEW: 0.15,
    BehaviorEventType.FAVORITE: 0.45,
    BehaviorEventType.CART: 0.70,
    BehaviorEventType.PURCHASE: 1.0,
}

_SCOPE_TTL = {
    MemoryScope.DAILY: timedelta(days=1),
    MemoryScope.RECENT: timedelta(days=7),
    MemoryScope.LONG_TERM: timedelta(days=90),
    MemoryScope.DURABLE: None,
}


@dataclass(frozen=True)
class MemoryPlan:
    operations: tuple[MemoryOperation, ...] = ()
    warnings: tuple[str, ...] = ()


class MemoryWritePolicy:
    """Pure rules that translate trusted behavior evidence into write proposals."""

    def plan(self, event: BehaviorEvent) -> MemoryPlan:
        if event.event_type is BehaviorEventType.VIEW:
            return MemoryPlan(tuple(self._preferences(event, MemoryScope.RECENT)))
        if event.event_type is BehaviorEventType.FAVORITE:
            return MemoryPlan(tuple(self._preferences(event, MemoryScope.LONG_TERM)))
        if event.event_type is BehaviorEventType.CART:
            return self._intent(event)
        if event.event_type is BehaviorEventType.PURCHASE:
            return self._purchase(event)
        if event.event_type is BehaviorEventType.SEARCH:
            return MemoryPlan()
        return MemoryPlan(warnings=("event type requires an explicit forget command",))

    def _preferences(
        self,
        event: BehaviorEvent,
        scope: MemoryScope,
    ) -> list[MemoryOperation]:
        operations: list[MemoryOperation] = []
        for field, memory_type in (
            ("category", MemoryType.CATEGORY_PREFERENCE),
            ("brand", MemoryType.BRAND_PREFERENCE),
        ):
            display = event.payload.get(field)
            if isinstance(display, str) and display.strip():
                operations.append(self._operation(
                    event,
                    memory_type,
                    scope,
                    f"{field}:{_normalize(display)}",
                    {field: display.strip()},
                    f"{event.event_type.value} product {field}",
                ))

        features = event.payload.get("features")
        if event.event_type is BehaviorEventType.FAVORITE and isinstance(features, list):
            for feature in features:
                if isinstance(feature, str) and feature.strip():
                    operations.append(self._operation(
                        event,
                        MemoryType.FEATURE_PREFERENCE,
                        scope,
                        f"feature:{_normalize(feature)}",
                        {"feature": feature.strip()},
                        "favorite product feature",
                    ))
        return operations

    def _intent(self, event: BehaviorEvent) -> MemoryPlan:
        product_id = item_value(event.payload)
        if product_id is None:
            return MemoryPlan(warnings=("cart event has no product_id or memory_key",))
        operation = self._operation(
            event,
            MemoryType.SHOPPING_INTENT,
            MemoryScope.DAILY,
            f"shopping_intent:{product_id}",
            {"product_id": event.payload.get("product_id", product_id)},
            "product added to cart",
        )
        return MemoryPlan((operation,))

    def _purchase(self, event: BehaviorEvent) -> MemoryPlan:
        product_id = item_value(event.payload)
        if product_id is None:
            return MemoryPlan(warnings=("purchase event has no product_id or memory_key",))
        operation = self._operation(
            event,
            MemoryType.PURCHASED_PRODUCT,
            MemoryScope.DURABLE,
            f"purchased_product:{product_id}",
            {"product_id": event.payload.get("product_id", product_id)},
            "product purchased",
        )
        return MemoryPlan((operation,))

    @staticmethod
    def _operation(
        event: BehaviorEvent,
        memory_type: MemoryType,
        scope: MemoryScope,
        key: str,
        value: dict[str, Any],
        reason: str,
    ) -> MemoryOperation:
        return MemoryOperation(
            operation=MemoryOperationType.UPSERT,
            user_id=event.user_id,
            memory_type=memory_type,
            scope=scope,
            key=_normalize(key),
            value=value,
            confidence=_SIGNAL_CONFIDENCE[event.event_type],
            source=event.event_type.value,
            reason=reason,
            source_event_id=event.event_id,
            expires_at=_expires_at(event.occurred_at, scope),
        )


def item_value(payload: dict[str, Any]) -> str | None:
    for field in ("product_id", "memory_key"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return _normalize(value)
    return None


def _normalize(value: str) -> str:
    return value.strip().casefold()


def _expires_at(occurred_at: datetime, scope: MemoryScope) -> datetime | None:
    ttl = _SCOPE_TTL[scope]
    if ttl is None:
        return None
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=timezone.utc)
    return occurred_at + ttl
