from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .store import JsonMemoryStore


class MemoryWorkflowState(TypedDict, total=False):
    event: dict[str, Any]
    user_id: str
    duplicate: bool
    updated_keys: list[str]
    short_term_updates: list[dict[str, Any]]
    long_term_updates: list[dict[str, Any]]


def build_memory_workflow(store: JsonMemoryStore):
    """构建最小 LangGraph：行为入短期记忆，再判断是否晋升长期记忆。"""

    def update_short_term(state: MemoryWorkflowState) -> MemoryWorkflowState:
        result = store.record_behavior(state["event"])
        return {
            "user_id": result["user_id"],
            "duplicate": result["duplicate"],
            "updated_keys": result["updated_keys"],
            "short_term_updates": result["short_term_updates"],
        }

    def update_long_term(state: MemoryWorkflowState) -> MemoryWorkflowState:
        if state.get("duplicate"):
            return {"long_term_updates": []}
        return {
            "long_term_updates": store.promote_eligible(
                state["user_id"], state.get("updated_keys", [])
            )
        }

    graph = StateGraph(MemoryWorkflowState)
    graph.add_node("update_short_term_memory", update_short_term)
    graph.add_node("promote_long_term_memory", update_long_term)
    graph.add_edge(START, "update_short_term_memory")
    graph.add_edge("update_short_term_memory", "promote_long_term_memory")
    graph.add_edge("promote_long_term_memory", END)
    return graph.compile()
