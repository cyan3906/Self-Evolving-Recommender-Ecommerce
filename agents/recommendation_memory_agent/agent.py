from __future__ import annotations

from langchain.agents import create_agent

from .store import JsonMemoryStore
from .tools import build_memory_tools


def build_recommendation_memory_agent(model, store: JsonMemoryStore):
    """把两个记忆 Tool 装进 LangChain Agent，供推荐模块直接融合。"""
    return create_agent(
        model=model,
        tools=build_memory_tools(store),
        name="recommendation_memory_agent",
        system_prompt=(
            "你是首页推荐中的记忆读取代理。"
            "需要近期兴趣时调用 read_short_term_memory；"
            "需要稳定偏好时调用 read_long_term_memory。"
            "只返回 Tool 中存在的记忆，不要编造用户偏好。"
        ),
    )
