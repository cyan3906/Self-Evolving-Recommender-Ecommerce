"""最小启动入口。

运行：python -m agents.recommendation_memory_agent.run
这个演示不需要 API Key；真实推荐 Agent 使用 agent.py 的装配函数。
"""

from __future__ import annotations

import json
from pathlib import Path

from .store import JsonMemoryStore
from .tools import build_memory_tools
from .workflow import build_memory_workflow


def main() -> None:
    path = Path("db/recommendation_memory_demo.json")
    store = JsonMemoryStore(path)
    workflow = build_memory_workflow(store)

    # 三次独立行为都会写入 behavior_events，并持续增强同一短期记忆。
    events = [
        {"event_id": "demo-view", "user_id": "user_001", "event_type": "view",
         "payload": {"brand": "Huawei", "category": "手机"}},
        {"event_id": "demo-favorite", "user_id": "user_001", "event_type": "favorite",
         "payload": {"brand": "Huawei", "category": "手机"}},
        {"event_id": "demo-cart", "user_id": "user_001", "event_type": "cart",
         "payload": {"brand": "Huawei", "category": "手机"}},
    ]
    for event in events:
        workflow.invoke({"event": event})

    short_tool, long_tool = build_memory_tools(store)
    output = {
        "json_path": str(path.resolve()),
        "short_term": short_tool.invoke({"user_id": "user_001"}),
        "long_term": long_tool.invoke({"user_id": "user_001"}),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
