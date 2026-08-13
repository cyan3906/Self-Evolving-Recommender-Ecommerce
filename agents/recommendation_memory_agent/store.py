from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from .models import BehaviorEvent, BehaviorEventType, utc_now


# 不同动作是不同强度的偏好证据。数值以后可以通过 A/B 实验调整。
SIGNAL_CONFIDENCE: dict[BehaviorEventType, float] = {
    BehaviorEventType.VIEW: 0.15,
    BehaviorEventType.CLICK: 0.20,
    BehaviorEventType.FAVORITE: 0.45,
    BehaviorEventType.CART: 0.70,
    BehaviorEventType.PURCHASE: 1.00,
}


class JsonMemoryStore:
    """一个 JSON 文件同时保存行为、短期记忆和长期记忆。

    这是可替换的存储适配器。未来换成 Redis/PostgreSQL 时，上层
    LangGraph 和 LangChain Tool 不需要改业务规则。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        minimum_event_nums: int = 3,
        minimum_confidence: float = 0.60,
    ) -> None:
        if minimum_event_nums < 1:
            raise ValueError("minimum_event_nums must be positive")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        self.path = Path(path)
        self.minimum_event_nums = minimum_event_nums
        self.minimum_confidence = minimum_confidence
        self._lock = threading.Lock()
        self._ensure_file()

    def record_behavior(self, raw_event: BehaviorEvent | dict[str, Any]) -> dict[str, Any]:
        """记录行为，并把其中的品牌/品类/特征合并进短期记忆。"""
        event = BehaviorEvent.model_validate(raw_event)
        with self._lock:
            data = self._load()
            user = self._user(data, event.user_id)

            # API 重试不能重复增加 event_nums。
            if any(item["event_id"] == event.event_id for item in user["behavior_events"]):
                return {
                    "user_id": event.user_id,
                    "event_id": event.event_id,
                    "duplicate": True,
                    "updated_keys": [],
                    "short_term_updates": [],
                }

            user["behavior_events"].append(event.model_dump(mode="json"))
            signal = SIGNAL_CONFIDENCE[event.event_type]
            updated: list[dict[str, Any]] = []

            for assertion in _extract_assertions(event.payload):
                key = assertion["key"]
                previous = user["short_term_memory"].get(key)
                old_confidence = float(previous["confidence"]) if previous else 0.0

                # 越接近 1，后续弱信号带来的增量越小，避免无限线性累加。
                confidence = old_confidence + signal * (1.0 - old_confidence)
                memory = {
                    **assertion,
                    "scope": "short_term",
                    "confidence": round(confidence, 6),
                    "event_nums": int(previous["event_nums"]) + 1 if previous else 1,
                    "last_event_type": event.event_type.value,
                    "last_event_at": event.occurred_at.isoformat(),
                }
                user["short_term_memory"][key] = memory
                updated.append(memory.copy())

            self._save(data)
            return {
                "user_id": event.user_id,
                "event_id": event.event_id,
                "duplicate": False,
                "updated_keys": [item["key"] for item in updated],
                "short_term_updates": updated,
            }

    def promote_eligible(self, user_id: str, keys: list[str]) -> list[dict[str, Any]]:
        """把达到双门槛的短期记忆同步为长期记忆。

        短期记忆仍保留，因为它还要描述近期兴趣；长期记忆是稳定偏好的
        独立快照，不是把短期记录从 JSON 中搬走。
        """
        with self._lock:
            data = self._load()
            user = self._user(data, user_id)
            promoted: list[dict[str, Any]] = []

            for key in keys:
                short = user["short_term_memory"].get(key)
                if not short or not self._eligible(short):
                    continue
                long_term = short.copy()
                long_term.update({
                    "scope": "long_term",
                    "promoted_at": utc_now().isoformat(),
                })
                user["long_term_memory"][key] = long_term
                promoted.append(long_term.copy())

            if promoted:
                self._save(data)
            return promoted

    def read_short_term(self, user_id: str) -> list[dict[str, Any]]:
        return self._read_memory(user_id, "short_term_memory")

    def read_long_term(self, user_id: str) -> list[dict[str, Any]]:
        return self._read_memory(user_id, "long_term_memory")

    def _read_memory(self, user_id: str, field: str) -> list[dict[str, Any]]:
        with self._lock:
            values = list(self._user(self._load(), user_id)[field].values())
        return sorted(values, key=lambda item: (-item["confidence"], item["key"]))

    def _eligible(self, memory: dict[str, Any]) -> bool:
        return (
            memory["event_nums"] >= self.minimum_event_nums
            and memory["confidence"] >= self.minimum_confidence
        )

    def _ensure_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._save({"users": {}})

    def _load(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict) or not isinstance(data.get("users"), dict):
            raise ValueError(f"invalid memory JSON: {self.path}")
        return data

    def _save(self, data: dict[str, Any]) -> None:
        # 同目录临时文件 + replace，避免进程中断留下半个 JSON。
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        os.replace(temporary, self.path)

    @staticmethod
    def _user(data: dict[str, Any], user_id: str) -> dict[str, Any]:
        return data["users"].setdefault(user_id, {
            "behavior_events": [],
            "short_term_memory": {},
            "long_term_memory": {},
        })


def _extract_assertions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """只做确定性字段转换，不让 LLM 猜测用户偏好。"""
    assertions: list[dict[str, Any]] = []
    for field, memory_type in (
        ("brand", "brand_preference"),
        ("category", "category_preference"),
    ):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            display = value.strip()
            assertions.append({
                "memory_type": memory_type,
                "key": f"{field}:{display.casefold()}",
                "value": {field: display},
            })

    features = payload.get("features", [])
    if isinstance(features, list):
        for value in features:
            if isinstance(value, str) and value.strip():
                display = value.strip()
                assertions.append({
                    "memory_type": "feature_preference",
                    "key": f"feature:{display.casefold()}",
                    "value": {"feature": display},
                })
    return assertions
