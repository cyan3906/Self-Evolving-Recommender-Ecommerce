from agents.memory.agent import MemoryAgent
from agents.memory.homepage_agent import HomepageRecommendationAgent
from agents.memory.models import (
    BehaviorEvent,
    BehaviorEventType,
    HomepageRecommendationRequest,
    RerankWeights,
    utc_now,
)
from agents.memory.reranker import MemoryReranker


def product(product_id: str, brand: str, score: float) -> dict:
    return {
        "id": product_id,
        "brand": brand,
        "category": "手机",
        "description": f"{brand} phone",
        "price": 4999,
        "rrf_score": score,
    }


def test_homepage_agent_consumes_memory_without_owning_the_store(store):
    memory = MemoryAgent(store)
    memory.observe(BehaviorEvent(
        event_id="favorite-1",
        user_id="user_001",
        event_type=BehaviorEventType.FAVORITE,
        occurred_at=utc_now(),
        payload={"brand": "Huawei"},
    ))
    candidates = [product("apple-1", "Apple", 0.02), product("huawei-1", "Huawei", 0.01)]
    agent = HomepageRecommendationAgent(
        lambda user_id, top_k: candidates,
        memory,
        MemoryReranker(RerankWeights(retrieval=0.1, long_term=0.8)),
    )

    response = agent.recommend(HomepageRecommendationRequest(
        request_id="home-1",
        user_id="user_001",
        top_k=2,
    ))

    assert [item.product["id"] for item in response.results] == ["huawei-1", "apple-1"]
    assert response.trace.long_term_preference_count == 1
    assert response.trace.changed_position_count == 2
