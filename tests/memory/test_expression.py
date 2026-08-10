from agents.memory.agent import MemoryAgent
from agents.memory.expression import ExpressionProfileLearner
from agents.memory.models import (
    BehaviorEvent,
    BehaviorEventType,
    ClarificationAction,
    ExpressionDimension,
    utc_now,
)


def search_event(
    event_id: str,
    query: str,
    *,
    offered: bool = False,
    accepted: bool = False,
) -> BehaviorEvent:
    return BehaviorEvent(
        event_id=event_id,
        user_id="user_001",
        event_type=BehaviorEventType.SEARCH,
        occurred_at=utc_now(),
        payload={
            "query": query,
            "clarification_offered": offered,
            "clarification_accepted": accepted,
        },
    )


def test_expression_profile_tracks_dimensions_not_product_preferences(store):
    agent = MemoryAgent(store)
    agent.observe(search_event("search-1", "华为拍照手机5000以内"))

    profile = ExpressionProfileLearner(store).get_profile("user_001")

    assert profile is not None
    assert profile.search_count == 1
    assert profile.dimension_counts == {
        ExpressionDimension.BRAND: 1,
        ExpressionDimension.BUDGET: 1,
        ExpressionDimension.CATEGORY: 1,
        ExpressionDimension.USE_CASE: 1,
    }


def test_complete_query_proceeds_without_clarification(store):
    decision = MemoryAgent(store).plan_search(
        "user_001", "华为拍照手机5000以内"
    )

    assert decision.action is ClarificationAction.PROCEED
    assert decision.analysis.completeness_score == 1.0
    assert decision.target_dimension is None


def test_cold_start_incomplete_query_offers_non_blocking_filters(store):
    decision = MemoryAgent(store).plan_search("user_001", "手机")

    assert decision.action is ClarificationAction.OFFER_FILTERS
    assert decision.target_dimension is ExpressionDimension.USE_CASE


def test_user_who_accepts_clarification_gets_one_high_value_question(store):
    agent = MemoryAgent(store)
    for index in range(3):
        agent.observe(search_event(
            f"search-{index}", "手机", offered=True, accepted=True
        ))

    decision = agent.plan_search("user_001", "手机")

    assert decision.action is ClarificationAction.ASK_ONE_CONSTRAINT
    assert decision.target_dimension is ExpressionDimension.USE_CASE


def test_user_who_rejects_clarification_gets_soft_supplement(store):
    agent = MemoryAgent(store)
    for index in range(4):
        agent.observe(search_event(
            f"search-{index}", "手机", offered=True, accepted=False
        ))

    decision = agent.plan_search("user_001", "手机")

    assert decision.action is ClarificationAction.SOFT_SUPPLEMENT
