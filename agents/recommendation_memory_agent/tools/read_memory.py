from __future__ import annotations

from langchain.tools import tool

from ..store import JsonMemoryStore


def build_memory_tools(store: JsonMemoryStore):
    """用依赖注入创建 Tool，便于替换 JSON 路径或存储实现。"""

    @tool
    def read_short_term_memory(user_id: str) -> dict:
        """读取用户近期兴趣记忆，适合首页推荐使用近期偏好时调用。"""
        return {
            "user_id": user_id,
            "scope": "short_term",
            "memories": store.read_short_term(user_id),
        }

    @tool
    def read_long_term_memory(user_id: str) -> dict:
        """读取用户稳定长期偏好，适合近期行为不足或需要长期偏好时调用。"""
        return {
            "user_id": user_id,
            "scope": "long_term",
            "memories": store.read_long_term(user_id),
        }

    return [read_short_term_memory, read_long_term_memory]
