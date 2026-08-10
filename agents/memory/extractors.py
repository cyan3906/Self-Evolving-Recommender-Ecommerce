from __future__ import annotations

import re
from datetime import datetime

from agents.memory.models import (
    ExpressionDimension,
    MemoryOperation,
    MemoryOperationType,
    MemoryScope,
    MemoryType,
    QueryAnalysis,
)


BRAND_ALIASES = {
    "Apple": ("apple", "苹果"),
    "Huawei": ("huawei", "华为"),
    "Xiaomi": ("xiaomi", "小米"),
    "Samsung": ("samsung", "三星"),
    "OPPO": ("oppo",),
    "Sony": ("sony", "索尼"),
    "Lenovo": ("lenovo", "联想"),
    "Dell": ("dell", "戴尔"),
    "Nike": ("nike",),
    "Adidas": ("adidas",),
    "DeLonghi": ("delonghi", "德龙"),
}

CATEGORY_ALIASES = {
    "手机": ("手机",),
    "笔记本电脑": ("笔记本电脑", "笔记本"),
    "平板电脑": ("平板电脑", "平板"),
    "智能手表": ("智能手表", "手表"),
    "耳机": ("耳机",),
    "衣服": ("衣服", "外套", "卫衣"),
    "鞋子": ("鞋子", "跑鞋", "篮球鞋"),
    "咖啡机": ("咖啡机",),
    "电视": ("电视",),
}

USE_CASE_ALIASES = {
    "photography": ("拍照", "摄影", "影像"),
    "gaming": ("游戏", "电竞"),
    "office": ("办公", "商务"),
    "study": ("学习", "学生"),
    "fitness": ("跑步", "健身", "运动"),
    "gift": ("送礼", "送给", "给妈妈", "给爸爸", "给朋友"),
}

_COMPLETENESS_WEIGHTS = {
    ExpressionDimension.CATEGORY: 0.4,
    ExpressionDimension.USE_CASE: 0.3,
    ExpressionDimension.BUDGET: 0.2,
    ExpressionDimension.BRAND: 0.1,
}


def analyze_query(query: str) -> QueryAnalysis:
    normalized = query.casefold().strip()
    present: set[ExpressionDimension] = set()
    if _matched_values(normalized, CATEGORY_ALIASES):
        present.add(ExpressionDimension.CATEGORY)
    if _matched_values(normalized, BRAND_ALIASES):
        present.add(ExpressionDimension.BRAND)
    if _extract_budget(normalized) is not None:
        present.add(ExpressionDimension.BUDGET)
    if _matched_values(normalized, USE_CASE_ALIASES):
        present.add(ExpressionDimension.USE_CASE)
    if _negative_terms(normalized):
        present.add(ExpressionDimension.EXCLUSION)

    scored_dimensions = set(_COMPLETENESS_WEIGHTS)
    completeness = sum(
        weight for dimension, weight in _COMPLETENESS_WEIGHTS.items()
        if dimension in present
    )
    return QueryAnalysis(
        query=query.strip(),
        present_dimensions=sorted(present, key=lambda item: item.value),
        missing_dimensions=sorted(
            scored_dimensions - present, key=lambda item: item.value
        ),
        completeness_score=round(completeness, 4),
    )


def extract_deterministic_query_operations(
    user_id: str,
    query: str,
    event_id: str,
    occurred_at: datetime,
) -> list[MemoryOperation]:
    """Extract ephemeral constraints for the current request; never persist them."""
    del occurred_at
    normalized = query.casefold()
    negative_terms = _negative_terms(normalized)
    operations: list[MemoryOperation] = []

    for display, aliases in BRAND_ALIASES.items():
        if not _contains_alias(normalized, aliases):
            continue
        if _is_negative(aliases, negative_terms):
            operations.append(_constraint(
                user_id, event_id, MemoryType.NEGATIVE_PREFERENCE,
                f"negative:brand:{_normalize(display)}",
                {"kind": "brand", "value": display},
                "current query excludes brand",
            ))
        else:
            operations.append(_constraint(
                user_id, event_id, MemoryType.BRAND_PREFERENCE,
                f"brand:{_normalize(display)}",
                {"brand": display, "value": display},
                "current query specifies brand",
            ))

    for display, aliases in CATEGORY_ALIASES.items():
        if not _contains_alias(normalized, aliases):
            continue
        if _is_negative(aliases, negative_terms):
            operations.append(_constraint(
                user_id, event_id, MemoryType.NEGATIVE_PREFERENCE,
                f"negative:category:{_normalize(display)}",
                {"kind": "category", "value": display},
                "current query excludes category",
            ))
        else:
            operations.append(_constraint(
                user_id, event_id, MemoryType.CATEGORY_PREFERENCE,
                f"category:{_normalize(display)}",
                {"category": display, "value": display},
                "current query specifies category",
            ))

    for display, aliases in USE_CASE_ALIASES.items():
        if _contains_alias(normalized, aliases):
            operations.append(_constraint(
                user_id, event_id, MemoryType.FEATURE_PREFERENCE,
                f"feature:{display}",
                {"feature": display, "value": display},
                "current query specifies use case",
            ))

    budget = _extract_budget(normalized)
    if budget is not None:
        minimum, maximum = budget
        if minimum is None:
            key, value = f"price:max:{maximum}", {"value": maximum, "max_price": maximum}
        elif maximum is None:
            key, value = f"price:min:{minimum}", {"value": minimum, "min_price": minimum}
        else:
            key = f"price:{minimum}-{maximum}"
            value = {"min_price": minimum, "max_price": maximum}
        operations.append(_constraint(
            user_id, event_id, MemoryType.PRICE_RANGE, key, value,
            "current query specifies budget",
        ))
    return operations


def _constraint(
    user_id: str,
    event_id: str,
    memory_type: MemoryType,
    key: str,
    value: dict,
    reason: str,
) -> MemoryOperation:
    return MemoryOperation(
        operation=MemoryOperationType.UPSERT,
        user_id=user_id,
        memory_type=memory_type,
        scope=MemoryScope.DAILY,
        key=_normalize(key),
        value=value,
        confidence=1.0,
        source="current_query",
        reason=reason,
        source_event_id=event_id,
    )


def _matched_values(query: str, aliases: dict[str, tuple[str, ...]]) -> list[str]:
    return [display for display, values in aliases.items() if _contains_alias(query, values)]


def _contains_alias(query: str, aliases: tuple[str, ...]) -> bool:
    return any(alias.casefold() in query for alias in aliases)


def _negative_terms(query: str) -> list[str]:
    terms: list[str] = []
    for marker in ("不要", "不考虑", "排除"):
        for match in re.finditer(re.escape(marker) + r"\s*([^\s,，。！？!?]+)", query):
            terms.append(match.group(1))
    return terms


def _is_negative(aliases: tuple[str, ...], negative_terms: list[str]) -> bool:
    return any(
        term.casefold().startswith(alias.casefold())
        for term in negative_terms
        for alias in aliases
    )


def _extract_budget(query: str) -> tuple[int | None, int | None] | None:
    range_match = re.search(r"(\d+)\s*(?:到|至)\s*(\d+)", query)
    if range_match:
        return int(range_match.group(1)), int(range_match.group(2))
    max_match = re.search(r"(?:预算|不超过)\s*(\d+)|(?<!\d)(\d+)\s*以内", query)
    if max_match:
        value = next(group for group in max_match.groups() if group is not None)
        return None, int(value)
    return None


def _normalize(value: str) -> str:
    return value.strip().casefold()
