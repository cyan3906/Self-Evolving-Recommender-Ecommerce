from datetime import datetime, timezone

import pytest

from agents.memory.tools import GetPurchaseBehaviorSummaryTool, PurchaseRecord


AS_OF = datetime(2026, 8, 10, tzinfo=timezone.utc)


def purchase(order_id: str, purchased_at: datetime, *, user_id: str = "user_001"):
    return PurchaseRecord(
        order_id=order_id,
        user_id=user_id,
        product_id=f"sku-{order_id}",
        category="coffee",
        purchased_at=purchased_at,
    )


def test_purchase_tool_calculates_repeat_cadence_and_time_windows():
    rows = [
        purchase("1", datetime(2026, 6, 21, tzinfo=timezone.utc)),
        purchase("2", datetime(2026, 7, 1, tzinfo=timezone.utc)),
        purchase("3", datetime(2026, 7, 11, tzinfo=timezone.utc)),
    ]
    tool = GetPurchaseBehaviorSummaryTool(lambda _user_id, _as_of: rows)

    summary = tool.run("user_001", as_of=AS_OF)

    assert summary.total_purchase_count == 3
    assert summary.purchase_count_30d == 1
    assert summary.purchase_count_90d == 3
    assert len(summary.categories) == 1
    assert summary.categories[0].median_interval_days == pytest.approx(10)
    assert summary.categories[0].days_since_purchase == pytest.approx(30)


def test_purchase_tool_rejects_cross_user_history():
    tool = GetPurchaseBehaviorSummaryTool(lambda _user_id, _as_of: [
        purchase("other", AS_OF, user_id="user_002"),
    ])

    with pytest.raises(ValueError, match="another user"):
        tool.run("user_001", as_of=AS_OF)
