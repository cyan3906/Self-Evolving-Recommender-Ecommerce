from agents.memory.agent import MemoryAgent
from agents.memory.models import ClarificationAction, SearchRecommendationRequest
from agents.memory.search_agent import SearchRecommendationAgent


def _product(product_id: str, brand: str, score: float) -> dict:
    return {
        "id": product_id,
        "brand": brand,
        "category": "手机",
        "description": f"{brand} camera phone",
        "price": 4999,
        "rrf_score": score,
    }


def test_search_agent_connects_recall_projection_rerank_and_query_learning(store):
    calls = []

    def retrieve(query: str, top_k: int) -> list[dict]:
        calls.append((query, top_k))
        return [_product("apple-1", "Apple", 0.02), _product("huawei-1", "Huawei", 0.01)]

    writer = MemoryAgent(store)
    search_agent = SearchRecommendationAgent(retrieve, writer)

    response = search_agent.search(SearchRecommendationRequest(
        request_id="request-1",
        user_id="user_002",
        query="Huawei 手机",
        top_k=2,
    ))

    assert calls == [("Huawei 手机", 2)]
    assert [item.product["id"] for item in response.results] == ["huawei-1", "apple-1"]
    assert response.trace.current_constraint_count == 2
    assert response.trace.changed_position_count == 2
    assert response.trace.expression_profile_updated is True
    assert response.clarification.action is ClarificationAction.OFFER_FILTERS
    assert {record.user_id for record in store.get_active("user_002")} == {"user_002"}
    assert store.get_active("user_001") == []


def test_search_agent_keeps_retrieval_available_when_memory_projection_fails(store):
    class UnavailableProjector:
        def for_search(self, user_id: str, query: str):
            raise RuntimeError("memory database unavailable")

    candidates = [_product("apple-1", "Apple", 0.02), _product("huawei-1", "Huawei", 0.01)]
    search_agent = SearchRecommendationAgent(
        lambda query, top_k: candidates,
        MemoryAgent(store, projector=UnavailableProjector()),
    )

    response = search_agent.search(SearchRecommendationRequest(
        request_id="request-2",
        user_id="user_001",
        query="手机",
        remember_query=False,
    ))

    assert [item.product["id"] for item in response.results] == ["apple-1", "huawei-1"]
    assert response.trace.memory_version == 0
    assert response.trace.expression_profile_updated is False
    assert "memory projection unavailable" in response.trace.warnings[0]
