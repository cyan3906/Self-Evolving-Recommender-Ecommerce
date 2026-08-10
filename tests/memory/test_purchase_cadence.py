from datetime import datetime, timezone

from agents.memory.agent import MemoryAgent
from agents.memory.models import MemoryDecisionType, MemoryType, RerankRequest
from agents.memory.reranker import MemoryReranker
from agents.memory.tools import GetPurchaseBehaviorSummaryTool, PurchaseRecord


AS_OF = datetime(2026, 8, 10, tzinfo=timezone.utc)


def test_agent_evolves_category_cadence_and_injects_only_actionable_state(store):
    purchases = [
        PurchaseRecord(
            order_id=str(index),
            user_id="user_001",
            product_id=f"coffee-{index}",
            category="coffee",
            purchased_at=purchased_at,
        )
        for index, purchased_at in enumerate((
            datetime(2026, 6, 21, tzinfo=timezone.utc),
            datetime(2026, 7, 1, tzinfo=timezone.utc),
            datetime(2026, 7, 11, tzinfo=timezone.utc),
        ), start=1)
    ]
    behavior_tool = GetPurchaseBehaviorSummaryTool(
        lambda _user_id, _as_of: list(purchases)
    )
    agent = MemoryAgent(store, purchase_behavior_tool=behavior_tool)

    first = agent.refresh_purchase_cadence("user_001", as_of=AS_OF)
    repeated = agent.refresh_purchase_cadence("user_001", as_of=AS_OF)

    assert first.decision is MemoryDecisionType.REMEMBERED
    assert first.changed_memories[0].memory_type is MemoryType.PURCHASE_CADENCE
    assert first.changed_memories[0].value["status"] == "lapsed"
    assert repeated.decision is MemoryDecisionType.IGNORED
    assert store.get_active("user_001")[0].version == 1

    search_context = agent.recall_for_search("user_001", "coffee")
    unrelated_context = agent.recall_for_search("user_001", "phone")
    homepage_context = agent.recall_for_homepage("user_001")
    assert [memory.key for memory in search_context.cadence_signals] == [
        "purchase_cadence:coffee"
    ]
    assert unrelated_context.cadence_signals == []
    assert len(homepage_context.cadence_signals) == 1

    candidates = [
        {
            "id": "phone-1",
            "brand": "Acme",
            "category": "phone",
            "price": 100,
            "description": "phone",
            "rrf_score": 0.50,
        },
        {
            "id": "coffee-1",
            "brand": "Acme",
            "category": "coffee",
            "price": 90,
            "description": "coffee beans",
            "rrf_score": 0.49,
        },
        {
            "id": "tablet-1",
            "brand": "Acme",
            "category": "tablet",
            "price": 80,
            "description": "tablet",
            "rrf_score": 0.10,
        },
    ]
    ranked = MemoryReranker().rerank(RerankRequest(
        query="coffee",
        candidates=candidates,
        memory_context=search_context,
    ))
    assert ranked[0].product["id"] == "coffee-1"
    assert any("purchase_cadence" in reason for reason in ranked[0].memory_reasons)

    purchases.append(PurchaseRecord(
        order_id="4",
        user_id="user_001",
        product_id="coffee-4",
        category="coffee",
        purchased_at=datetime(2026, 8, 5, tzinfo=timezone.utc),
    ))
    refreshed = agent.refresh_purchase_cadence("user_001", as_of=AS_OF)

    assert refreshed.changed_memories[0].value["status"] == "active"
    assert agent.recall_for_homepage("user_001").cadence_signals == []
