from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone
from statistics import median

from pydantic import Field, field_validator

from agents.memory.models import StrictModel, utc_now


class PurchaseRecord(StrictModel):
    """A normalized purchase row supplied by the commerce data adapter."""

    order_id: str
    user_id: str
    product_id: str
    category: str
    purchased_at: datetime

    @field_validator("order_id", "user_id", "product_id", "category")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value.strip()


class CategoryPurchaseCadence(StrictModel):
    category: str
    purchase_count: int = Field(ge=1)
    last_purchase_at: datetime
    days_since_purchase: float = Field(ge=0.0)
    median_interval_days: float | None = Field(default=None, ge=0.0)


class PurchaseBehaviorSummary(StrictModel):
    user_id: str
    as_of: datetime
    total_purchase_count: int = Field(ge=0)
    purchase_count_30d: int = Field(ge=0)
    purchase_count_90d: int = Field(ge=0)
    categories: list[CategoryPurchaseCadence] = Field(default_factory=list)


PurchaseHistoryReader = Callable[[str, datetime], Iterable[PurchaseRecord]]


class GetPurchaseBehaviorSummaryTool:
    """Return deterministic category-level cadence evidence from purchase history."""

    name = "get_purchase_behavior_summary"

    def __init__(self, reader: PurchaseHistoryReader) -> None:
        self._reader = reader

    def run(
        self,
        user_id: str,
        *,
        as_of: datetime | None = None,
    ) -> PurchaseBehaviorSummary:
        normalized_user_id = _required_text(user_id, "user_id")
        normalized_as_of = _aware_utc(as_of or utc_now())
        records = self._validated_records(
            normalized_user_id,
            normalized_as_of,
            self._reader(normalized_user_id, normalized_as_of),
        )

        category_records: dict[str, list[PurchaseRecord]] = defaultdict(list)
        category_labels: dict[str, str] = {}
        for record in records:
            category_key = record.category.casefold()
            category_labels.setdefault(category_key, record.category)
            category_records[category_key].append(record)

        categories: list[CategoryPurchaseCadence] = []
        for category_key, purchases in category_records.items():
            purchases.sort(key=lambda item: item.purchased_at)
            purchase_dates = [item.purchased_at for item in purchases]
            intervals = [
                (current - previous).total_seconds() / 86_400
                for previous, current in zip(purchase_dates, purchase_dates[1:])
            ]
            last_purchase_at = purchase_dates[-1]
            categories.append(CategoryPurchaseCadence(
                category=category_labels[category_key],
                purchase_count=len(purchases),
                last_purchase_at=last_purchase_at,
                days_since_purchase=(
                    normalized_as_of - last_purchase_at
                ).total_seconds() / 86_400,
                median_interval_days=median(intervals) if intervals else None,
            ))

        categories.sort(key=lambda item: (
            -item.purchase_count,
            -item.last_purchase_at.timestamp(),
            item.category.casefold(),
        ))
        return PurchaseBehaviorSummary(
            user_id=normalized_user_id,
            as_of=normalized_as_of,
            total_purchase_count=len(records),
            purchase_count_30d=_count_since(
                records, normalized_as_of - timedelta(days=30)
            ),
            purchase_count_90d=_count_since(
                records, normalized_as_of - timedelta(days=90)
            ),
            categories=categories,
        )

    @staticmethod
    def _validated_records(
        user_id: str,
        as_of: datetime,
        raw_records: Iterable[PurchaseRecord],
    ) -> list[PurchaseRecord]:
        records: list[PurchaseRecord] = []
        orders: dict[str, PurchaseRecord] = {}
        for raw_record in raw_records:
            validated = PurchaseRecord.model_validate(raw_record)
            record = validated.model_copy(update={
                "purchased_at": _aware_utc(validated.purchased_at),
            })
            if record.user_id != user_id:
                raise ValueError("purchase reader returned data for another user")
            if record.purchased_at > as_of:
                raise ValueError("purchase reader returned a future purchase")
            existing = orders.get(record.order_id)
            if existing is not None:
                if existing != record:
                    raise ValueError(f"conflicting duplicate order_id: {record.order_id}")
                continue
            orders[record.order_id] = record
            records.append(record)
        return records


def _count_since(records: Iterable[PurchaseRecord], cutoff: datetime) -> int:
    return sum(record.purchased_at >= cutoff for record in records)


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be blank")
    return value.strip()


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
