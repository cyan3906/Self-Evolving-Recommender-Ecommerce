from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Protocol

from pydantic import ValidationError

from agents.memory.models import (
    BehaviorEvent,
    BehaviorEventType,
    MemoryOperation,
    MemoryOperationType,
    MemoryRecord,
    MemoryScope,
    MemoryType,
)
from agents.memory.store import SQLiteMemoryStore


V1_USER_ID = "user_001"

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

BRAND_ALIASES = {
    "Apple": ("Apple", "鑻规灉", "苹果"),
    "Huawei": ("Huawei", "鍗庝负", "华为"),
    "Xiaomi": ("Xiaomi", "灏忕背", "小米"),
    "Samsung": ("Samsung", "涓夋槦", "三星"),
    "OPPO": ("OPPO",),
    "Sony": ("Sony", "绱㈠凹", "索尼"),
    "Lenovo": ("Lenovo", "鑱旀兂", "联想"),
    "Dell": ("Dell", "鎴村皵", "戴尔"),
    "Nike": ("Nike",),
    "Adidas": ("Adidas",),
}

# These are the categories in tools/product_data.py.  Correctly encoded aliases
# are accepted too, so a caller is not coupled to the fixture's legacy encoding.
CATEGORY_ALIASES = {
    "鎵嬫満": ("鎵嬫満", "手机"),
    "绗旇鏈數鑴?": ("绗旇鏈數鑴?", "笔记本电脑"),
    "骞虫澘鐢佃剳": ("骞虫澘鐢佃剳", "平板电脑"),
    "鏅鸿兘鎵嬭〃": ("鏅鸿兘鎵嬭〃", "智能手表"),
    "鑰虫満": ("鑰虫満", "耳机"),
    "琛ｆ湇": ("琛ｆ湇", "衣服"),
    "闉嬪瓙": ("闉嬪瓙", "鞋子"),
    "鍜栧暋鏈?": ("鍜栧暋鏈?", "咖啡机"),
    "鐢佃": ("鐢佃", "电视"),
}


class QueryOperationExtractor(Protocol):
    def extract(self, user_id: str, query: str, event_id: str) -> list[MemoryOperation]:
        """Return a complete, validated LLM-derived operation batch."""


class LLMQueryOperationExtractor:
    """Offline adapter for an application-supplied text generation function."""

    def __init__(self, generate: Callable[[str], str]) -> None:
        self._generate = generate

    def extract(self, user_id: str, query: str, event_id: str) -> list[MemoryOperation]:
        prompt = self._prompt(query)
        response = self._generate(prompt)
        try:
            payload = json.loads(response)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("LLM query extractor returned invalid JSON") from error
        if not isinstance(payload, list):
            raise ValueError("LLM query extractor must return a JSON array")

        try:
            return [
                MemoryOperation.model_validate(
                    {
                        **item,
                        "user_id": user_id,
                        "source": "llm_query",
                        "source_event_id": event_id,
                    }
                )
                for item in payload
                if isinstance(item, dict)
            ] if all(isinstance(item, dict) for item in payload) else self._invalid_item()
        except ValidationError as error:
            raise ValueError("LLM query extractor returned an invalid operation batch") from error

    @staticmethod
    def _invalid_item() -> list[MemoryOperation]:
        raise ValueError("LLM query extractor array items must be objects")

    @staticmethod
    def _prompt(query: str) -> str:
        memory_types = ", ".join(item.value for item in MemoryType)
        scopes = ", ".join(item.value for item in MemoryScope)
        return (
            "Extract memory write operations from this shopping query. Return only a JSON array. "
            f"Allowed memory_type values: {memory_types}. Allowed scope values: {scopes}. "
            "Every item must contain operation, memory_type, scope, key, value, confidence, and reason. "
            f"Query: {query}"
        )


def extract_deterministic_query_operations(
    user_id: str,
    query: str,
    event_id: str,
    occurred_at: datetime,
) -> list[MemoryOperation]:
    """Extract fixture-backed preferences without requiring an LLM or network access."""
    normalized_query = query.lower()
    negative_terms = _negative_terms(normalized_query)
    operations: list[MemoryOperation] = []

    for display, aliases in BRAND_ALIASES.items():
        if _contains_alias(normalized_query, aliases):
            if _is_negative(aliases, negative_terms):
                operations.append(_query_operation(
                    user_id, event_id, occurred_at, MemoryType.NEGATIVE_PREFERENCE,
                    MemoryScope.LONG_TERM, f"negative:brand:{_normalized_key(display)}",
                    {"kind": "brand", "value": display}, "explicitly excludes brand",
                ))
            else:
                operations.append(_query_operation(
                    user_id, event_id, occurred_at, MemoryType.BRAND_PREFERENCE,
                    MemoryScope.RECENT, f"brand:{_normalized_key(display)}",
                    {"brand": display, "value": display}, "query mentions brand",
                ))

    for display, aliases in CATEGORY_ALIASES.items():
        if _contains_alias(normalized_query, aliases):
            if _is_negative(aliases, negative_terms):
                operations.append(_query_operation(
                    user_id, event_id, occurred_at, MemoryType.NEGATIVE_PREFERENCE,
                    MemoryScope.LONG_TERM, f"negative:category:{_normalized_key(display)}",
                    {"kind": "category", "value": display}, "explicitly excludes category",
                ))
            else:
                operations.append(_query_operation(
                    user_id, event_id, occurred_at, MemoryType.CATEGORY_PREFERENCE,
                    MemoryScope.RECENT, f"category:{_normalized_key(display)}",
                    {"category": display, "value": display}, "query mentions category",
                ))

    budget = _extract_budget(normalized_query)
    if budget is not None:
        minimum, maximum = budget
        if minimum is None:
            key, value = f"price:max:{maximum}", {"value": maximum, "max_price": maximum}
        elif maximum is None:
            key, value = f"price:min:{minimum}", {"value": minimum, "min_price": minimum}
        else:
            key, value = f"price:{minimum}-{maximum}", {"min_price": minimum, "max_price": maximum}
        operations.append(_query_operation(
            user_id, event_id, occurred_at, MemoryType.PRICE_RANGE, MemoryScope.DAILY,
            key, value, "query contains budget",
        ))
    return operations


class MemoryAgent:
    def __init__(self, store: SQLiteMemoryStore, query_extractor: QueryOperationExtractor | None = None) -> None:
        self._store = store
        self._query_extractor = query_extractor
        self.last_warnings: list[str] = []

    def record_event(self, event: BehaviorEvent) -> list[MemoryRecord]:
        self._require_v1_user(event.user_id)
        self.last_warnings = []
        if event.event_type is BehaviorEventType.SEARCH:
            return self.record_query(event)

        operations = self._event_operations(event)
        records = self._apply_validated(operations)
        if event.event_type is BehaviorEventType.PURCHASE:
            self._close_purchase_intents(event)
        return records

    def record_query(self, event: BehaviorEvent) -> list[MemoryRecord]:
        self._require_v1_user(event.user_id)
        self.last_warnings = []
        query = event.payload.get("query", event.payload.get("keyword", ""))
        if not isinstance(query, str) or not query.strip():
            self.last_warnings.append("query is blank; no memory operations were extracted")
            return []

        operations = extract_deterministic_query_operations(
            event.user_id, query, event.event_id, event.occurred_at,
        )
        if self._query_extractor is not None:
            try:
                operations.extend(self._query_extractor.extract(event.user_id, query, event.event_id))
            except Exception as error:  # Adapter failures must not suppress deterministic writes.
                self.last_warnings.append(f"LLM query extraction rejected: {error}")
        return self._apply_validated(operations)

    def delete_memory(
        self,
        user_id: str,
        memory_type: MemoryType,
        scope: MemoryScope,
        key: str,
        event_id: str,
    ) -> bool:
        self._require_v1_user(user_id)
        normalized_key = _normalized_key(key)
        if not normalized_key:
            raise ValueError("memory key cannot be blank")
        return self._store.soft_delete(
            user_id, memory_type, scope, normalized_key,
            reason="explicit user memory deletion", source_event_id=event_id,
        )

    def _event_operations(self, event: BehaviorEvent) -> list[MemoryOperation]:
        payload = event.payload
        if event.event_type is BehaviorEventType.VIEW:
            return self._preference_operations(event, MemoryScope.RECENT)
        if event.event_type is BehaviorEventType.FAVORITE:
            return self._preference_operations(event, MemoryScope.LONG_TERM)
        if event.event_type is BehaviorEventType.CART:
            item = self._item_value(payload)
            if item is None:
                self.last_warnings.append("cart event has no product_id or memory_key")
                return []
            return [self._operation(
                event, MemoryType.SHOPPING_INTENT, MemoryScope.DAILY,
                f"shopping_intent:{item}", {"product_id": payload.get("product_id", item)},
                "product added to cart",
            )]
        if event.event_type is BehaviorEventType.PURCHASE:
            item = self._item_value(payload)
            if item is None:
                self.last_warnings.append("purchase event has no product_id or memory_key")
                return []
            return [self._operation(
                event, MemoryType.PURCHASED_PRODUCT, MemoryScope.DURABLE,
                f"purchased_product:{item}", {"product_id": payload.get("product_id", item)},
                "product purchased",
            )]
        return []

    def _preference_operations(self, event: BehaviorEvent, scope: MemoryScope) -> list[MemoryOperation]:
        payload = event.payload
        operations: list[MemoryOperation] = []
        for field, memory_type, prefix in (
            ("category", MemoryType.CATEGORY_PREFERENCE, "category"),
            ("brand", MemoryType.BRAND_PREFERENCE, "brand"),
        ):
            display = payload.get(field)
            if isinstance(display, str) and display.strip():
                operations.append(self._operation(
                    event, memory_type, scope, f"{prefix}:{_normalized_key(display)}",
                    {field: display.strip()}, f"{event.event_type.value} product {field}",
                ))
        features = payload.get("features")
        if event.event_type is BehaviorEventType.FAVORITE and isinstance(features, list):
            for feature in features:
                if isinstance(feature, str) and feature.strip():
                    operations.append(self._operation(
                        event, MemoryType.FEATURE_PREFERENCE, scope,
                        f"feature:{_normalized_key(feature)}", {"feature": feature.strip()},
                        "favorite product feature",
                    ))
        return operations

    def _operation(
        self,
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
            key=_normalized_key(key),
            value=value,
            confidence=SIGNAL_CONFIDENCE[event.event_type],
            source=event.event_type.value,
            reason=reason,
            source_event_id=event.event_id,
            expires_at=_expires_at(event.occurred_at, scope),
        )

    def _apply_validated(self, candidates: list[MemoryOperation]) -> list[MemoryRecord]:
        operations: list[MemoryOperation] = []
        for candidate in candidates:
            try:
                operation = MemoryOperation.model_validate(candidate)
                self._require_v1_user(operation.user_id)
                operations.append(operation)
            except (ValidationError, ValueError) as error:
                self.last_warnings.append(f"memory operation rejected: {error}")
        return [self._store.apply(operation) for operation in operations]

    def _close_purchase_intents(self, event: BehaviorEvent) -> None:
        product_id = self._item_value(event.payload)
        if product_id is None:
            return
        target_key = f"shopping_intent:{product_id}"
        for record in self._store.get_active(event.user_id):
            recorded_product = record.value.get("product_id")
            if (
                record.memory_type is MemoryType.SHOPPING_INTENT
                and record.scope is MemoryScope.DAILY
                and (
                    record.key == target_key
                    or (
                        isinstance(recorded_product, str)
                        and _normalized_key(recorded_product) == product_id
                    )
                )
            ):
                self._store.soft_delete(
                    event.user_id, record.memory_type, record.scope, record.key,
                    reason="purchase completed shopping intent", source_event_id=event.event_id,
                )

    @staticmethod
    def _item_value(payload: dict[str, Any]) -> str | None:
        for field in ("product_id", "memory_key"):
            value = payload.get(field)
            if isinstance(value, str) and value.strip():
                return _normalized_key(value)
        return None

    @staticmethod
    def _require_v1_user(user_id: str) -> None:
        if user_id != V1_USER_ID:
            raise ValueError(f"MemoryAgent only supports user_id {V1_USER_ID!r}")


def _query_operation(
    user_id: str,
    event_id: str,
    occurred_at: datetime,
    memory_type: MemoryType,
    scope: MemoryScope,
    key: str,
    value: dict[str, Any],
    reason: str,
) -> MemoryOperation:
    return MemoryOperation(
        operation=MemoryOperationType.UPSERT,
        user_id=user_id,
        memory_type=memory_type,
        scope=scope,
        key=_normalized_key(key),
        value=value,
        confidence=SIGNAL_CONFIDENCE[BehaviorEventType.SEARCH],
        source="deterministic_query",
        reason=reason,
        source_event_id=event_id,
        expires_at=_expires_at(occurred_at, scope),
    )


def _normalized_key(value: str) -> str:
    return value.strip().lower()


def _expires_at(occurred_at: datetime, scope: MemoryScope) -> datetime | None:
    ttl = SCOPE_TTL[scope]
    if ttl is None:
        return None
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=timezone.utc)
    return occurred_at + ttl


def _contains_alias(query: str, aliases: tuple[str, ...]) -> bool:
    return any(alias.lower() in query for alias in aliases)


def _negative_terms(query: str) -> list[str]:
    terms: list[str] = []
    for marker in ("涓嶈", "涓嶈€冭檻", "鎺掗櫎", "不要", "不考虑", "排除"):
        for match in re.finditer(re.escape(marker) + r"\s*([^\s,，。！？!?]+)", query):
            terms.append(match.group(1))
    return terms


def _is_negative(aliases: tuple[str, ...], negative_terms: list[str]) -> bool:
    return any(alias.lower() in term for term in negative_terms for alias in aliases)


def _extract_budget(query: str) -> tuple[int | None, int | None] | None:
    range_match = re.search(r"(\d+)\s*鍒.{0,2}?(\d+)", query)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))
    max_match = re.search(r"(?:棰勭畻\s*|涓嶈秴杩.\s*)(\d+)|(?<!\d)(\d+)\s*浠ュ唴", query)
    if max_match:
        value = next(group for group in max_match.groups() if group is not None)
        return None, int(value)
    return None
