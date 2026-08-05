from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np


# =========================================================
# 数据结构
# =========================================================

@dataclass
class ExperimentGroup:
    name: str
    weight: int = 50
    config: dict[str, Any] = field(default_factory=dict)

    # Thompson Sampling 状态
    successes: int = 1
    failures: int = 1


@dataclass
class Experiment:
    id: str
    name: str
    groups: list[ExperimentGroup]
    enabled: bool = True
    start_time: float = 0.0
    end_time: float = 0.0


# 推荐策略函数类型
StrategyHandler = Callable[
    [dict[str, Any], dict[str, Any]],
    dict[str, Any],
]

# 用户反馈函数类型
OutcomeHandler = Callable[
    [
        dict[str, Any],  # 原始用户数据
        dict[str, Any],  # 分桶结果
        dict[str, Any],  # 推荐结果
    ],
    dict[str, float],
]


# =========================================================
# A/B 测试引擎
# =========================================================

class ABTestEngine:
    """基于用户稳定分桶的 A/B 测试引擎。"""

    def __init__(
        self,
        bucket_count: int = 10_000,
        random_seed: int = 2026,
    ):
        if bucket_count <= 0:
            raise ValueError("bucket_count 必须大于 0")

        self.bucket_count = bucket_count

        self.experiments: dict[str, Experiment] = {}

        # 指标事件
        self._metrics: list[dict[str, Any]] = []

        # 曝光事件
        self._exposures: list[dict[str, Any]] = []

        # Thompson Sampling 随机数生成器
        self._rng = np.random.default_rng(random_seed)

        self._init_default_experiments()

    def _init_default_experiments(self) -> None:
        """初始化默认推荐策略实验。"""

        self.register_experiment(
            Experiment(
                id="rec_strategy",
                name="推荐策略实验",
                groups=[
                    ExperimentGroup(
                        name="control",
                        weight=50,
                        config={
                            "rerank": "rule_based",
                        },
                    ),
                    ExperimentGroup(
                        name="treatment_llm",
                        weight=50,
                        config={
                            "rerank": "llm",
                        },
                    ),
                ],
            )
        )

    def register_experiment(self, exp: Experiment) -> None:
        """注册实验。"""

        if not exp.id:
            raise ValueError("实验 id 不能为空")

        if not exp.groups:
            raise ValueError("实验必须至少包含一个实验组")

        if any(group.weight < 0 for group in exp.groups):
            raise ValueError("实验组权重不能小于 0")

        if sum(group.weight for group in exp.groups) <= 0:
            raise ValueError("实验组总权重必须大于 0")

        group_names = [group.name for group in exp.groups]

        if len(group_names) != len(set(group_names)):
            raise ValueError("同一个实验中的实验组名称不能重复")

        self.experiments[exp.id] = exp

    # =====================================================
    # 单用户分桶
    # =====================================================

    def assign(
        self,
        user_id: str,
        experiment_id: str = "rec_strategy",
    ) -> dict[str, Any]:
        """
        使用 user_id + experiment_id 稳定分桶。

        同一个用户在同一个实验中：
        1. bucket 不变；
        2. experiment group 不变；
        3. 推荐策略不变。
        """

        if not user_id:
            raise ValueError("user_id 不能为空")

        exp = self.experiments.get(experiment_id)

        if not exp or not self._is_active(exp):
            return {
                "experiment_id": experiment_id,
                "bucket": -1,
                "group": "control",
                "config": {},
            }

        bucket = self._hash_bucket(
            user_id=user_id,
            experiment_id=experiment_id,
        )

        group = self._bucket_to_group(
            bucket=bucket,
            groups=exp.groups,
        )

        return {
            "experiment_id": experiment_id,
            "bucket": bucket,
            "group": group.name,
            "config": dict(group.config),
        }

    # =====================================================
    # 批量分桶
    # =====================================================

    def assign_batch(
        self,
        records: Iterable[dict[str, Any]],
        experiment_id: str = "rec_strategy",
    ) -> list[dict[str, Any]]:
        """
        对一批用户数据进行分桶。

        这里只进行分桶，不执行具体推荐策略。
        """

        assigned_records: list[dict[str, Any]] = []

        for record in records:
            user_id = str(record.get("user_id", "")).strip()

            assignment = self.assign(
                user_id=user_id,
                experiment_id=experiment_id,
            )

            assigned_records.append(
                {
                    **record,
                    **assignment,
                }
            )

        return assigned_records

    # =====================================================
    # 批量执行实验
    # =====================================================

    def run_batch(
        self,
        records: Iterable[dict[str, Any]],
        strategy_handlers: dict[str, StrategyHandler],
        experiment_id: str = "rec_strategy",
        outcome_handler: OutcomeHandler | None = None,
    ) -> list[dict[str, Any]]:
        """
        批量运行实验。

        执行过程：

        用户数据
            ↓
        用户 ID 哈希分桶
            ↓
        获取实验组配置
            ↓
        根据 config 选择推荐策略
            ↓
        执行推荐策略
            ↓
        记录曝光
            ↓
        收集 CTR、CVR、GMV、停留时长等指标
        """

        results: list[dict[str, Any]] = []

        for record in records:
            user_id = str(record.get("user_id", "")).strip()

            # 1. 用户分桶
            assignment = self.assign(
                user_id=user_id,
                experiment_id=experiment_id,
            )

            # 2. 从实验配置中获取策略名称
            strategy_name = assignment["config"].get(
                "rerank",
                "rule_based",
            )

            # 3. 查找对应策略函数
            strategy_handler = strategy_handlers.get(strategy_name)

            if strategy_handler is None:
                raise KeyError(
                    f"未注册推荐策略处理器: {strategy_name}"
                )

            # 4. 执行推荐策略
            strategy_result = strategy_handler(
                record,
                assignment["config"],
            )
            
            # 5. 记录曝光事件
            self._exposures.append(
                {
                    "experiment_id": experiment_id,
                    "user_id": user_id,
                    "bucket": assignment["bucket"],
                    "group": assignment["group"],
                    "strategy": strategy_name,
                    "timestamp": time.time(),
                }
            )

            # 6. 获取用户反馈指标
            metrics: dict[str, float] = {}

            if outcome_handler is not None:
                metrics = outcome_handler(
                    record,
                    assignment,
                    strategy_result,
                )

                for metric_name, value in metrics.items():
                    self.record_metric(
                        experiment_id=experiment_id,
                        group_name=assignment["group"],
                        metric_name=metric_name,
                        value=float(value),
                        user_id=user_id,
                    )

            results.append(
                {
                    "user_id": user_id,
                    "experiment_id": experiment_id,
                    "bucket": assignment["bucket"],
                    "group": assignment["group"],
                    "config": assignment["config"],
                    "strategy": strategy_name,
                    "recommendations": strategy_result.get(
                        "recommendations",
                        [],
                    ),
                    "metrics": metrics,
                }
            )

        return results

    # =====================================================
    # Thompson Sampling
    # =====================================================

    def assign_thompson(
        self,
        user_id: str,
        experiment_id: str = "rec_strategy",
    ) -> dict[str, Any]:
        """
        使用 Thompson Sampling 动态选择实验组。

        注意：
        Thompson Sampling 默认不保证用户粘性。
        同一个用户多次访问，可能进入不同实验组。
        """

        if not user_id:
            raise ValueError("user_id 不能为空")

        exp = self.experiments.get(experiment_id)

        if not exp or not self._is_active(exp):
            return {
                "experiment_id": experiment_id,
                "group": "control",
                "config": {},
            }

        samples = []

        for group in exp.groups:
            sampled_value = self._rng.beta(
                group.successes,
                group.failures,
            )

            samples.append(
                (
                    sampled_value,
                    group,
                )
            )

        best_group = max(
            samples,
            key=lambda item: item[0],
        )[1]

        return {
            "experiment_id": experiment_id,
            "group": best_group.name,
            "config": dict(best_group.config),
        }

    def record_outcome(
        self,
        experiment_id: str,
        group_name: str,
        success: bool,
    ) -> None:
        """更新 Thompson Sampling 后验分布。"""

        exp = self.experiments.get(experiment_id)

        if not exp:
            return

        for group in exp.groups:
            if group.name != group_name:
                continue

            if success:
                group.successes += 1
            else:
                group.failures += 1

            return

    # =====================================================
    # 指标记录
    # =====================================================

    def record_metric(
        self,
        experiment_id: str,
        group_name: str,
        metric_name: str,
        value: float,
        user_id: str = "",
    ) -> None:
        """记录单个指标事件。"""

        self._metrics.append(
            {
                "experiment_id": experiment_id,
                "group": group_name,
                "metric": metric_name,
                "value": value,
                "user_id": user_id,
                "timestamp": time.time(),
            }
        )

    # =====================================================
    # 实验统计
    # =====================================================

    def get_assignment_distribution(
        self,
        results: Iterable[dict[str, Any]],
    ) -> dict[str, int]:
        """统计每个实验组分配了多少条数据。"""

        distribution: dict[str, int] = defaultdict(int)

        for result in results:
            group_name = str(result["group"])
            distribution[group_name] += 1

        return dict(distribution)

    def get_business_stats(
        self,
        experiment_id: str,
    ) -> dict[str, dict[str, float | int]]:
        """
        计算电商业务指标。

        CTR = 点击数 / 曝光数
        CVR = 成交数 / 点击数
        GMV = 成交金额总和
        单曝光 GMV = GMV / 曝光数
        平均停留时长 = 总停留时长 / 曝光数
        """

        exp = self.experiments.get(experiment_id)

        if not exp:
            return {}

        # 统计曝光
        exposure_count: dict[str, int] = defaultdict(int)

        for exposure in self._exposures:
            if exposure["experiment_id"] != experiment_id:
                continue

            exposure_count[exposure["group"]] += 1

        # 统计各指标总和
        metric_sum: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        for metric in self._metrics:
            if metric["experiment_id"] != experiment_id:
                continue

            metric_sum[metric["group"]][metric["metric"]] += metric["value"]

        result: dict[str, dict[str, float | int]] = {}

        for group in exp.groups:
            group_name = group.name

            exposures = exposure_count[group_name]
            clicks = metric_sum[group_name].get("clicked", 0.0)
            orders = metric_sum[group_name].get("converted", 0.0)
            gmv = metric_sum[group_name].get("gmv", 0.0)
            dwell_time = metric_sum[group_name].get(
                "dwell_time",
                0.0,
            )

            result[group_name] = {
                "exposures": exposures,
                "clicks": int(clicks),
                "orders": int(orders),

                "ctr": (
                    clicks / exposures
                    if exposures
                    else 0.0
                ),

                "cvr": (
                    orders / clicks
                    if clicks
                    else 0.0
                ),

                "gmv": gmv,

                "gmv_per_exposure": (
                    gmv / exposures
                    if exposures
                    else 0.0
                ),

                "avg_dwell_time": (
                    dwell_time / exposures
                    if exposures
                    else 0.0
                ),
            }

        return result

    # =====================================================
    # 内部方法
    # =====================================================

    def _is_active(self, exp: Experiment) -> bool:
        """判断实验是否处于运行状态。"""

        if not exp.enabled:
            return False

        now = time.time()

        if exp.start_time > 0 and now < exp.start_time:
            return False

        if exp.end_time > 0 and now >= exp.end_time:
            return False

        return True

    def _hash_bucket(
        self,
        user_id: str,
        experiment_id: str,
    ) -> int:
        """用户 ID 和实验 ID 共同生成稳定桶号。"""

        raw = f"{user_id}:{experiment_id}"

        digest = hashlib.md5(
            raw.encode("utf-8")
        ).hexdigest()

        return int(digest[:8], 16) % self.bucket_count

    def _bucket_to_group(
        self,
        bucket: int,
        groups: list[ExperimentGroup],
    ) -> ExperimentGroup:
        """根据桶号和权重确定实验组。"""

        total_weight = sum(
            group.weight
            for group in groups
        )

        normalized_bucket = (
            bucket * total_weight / self.bucket_count
        )

        cumulative = 0

        for group in groups:
            cumulative += group.weight

            if normalized_bucket < cumulative:
                return group

        return groups[-1]


# =========================================================
# 对照组策略：规则排序
# =========================================================

def rule_based_rerank(
    record: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    对照组：

    根据商品质量分和热度分进行排序。
    """

    candidates = record.get("candidate_items", [])

    recommendations = []

    for item in candidates:
        final_score = (
            0.70 * float(item.get("quality_score", 0.0))
            + 0.30 * float(item.get("popularity_score", 0.0))
        )

        recommendations.append(
            {
                **item,
                "final_score": final_score,
            }
        )

    recommendations.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    return {
        "recommendations": recommendations[:5],
    }


# =========================================================
# 实验组策略：LLM 模拟重排
# =========================================================

def llm_rerank(
    record: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    实验组：

    模拟 LLM 根据语义相关度和用户偏好进行重排。
    """

    candidates = record.get("candidate_items", [])

    recommendations = []

    for item in candidates:
        final_score = (
            0.45 * float(item.get("semantic_score", 0.0))
            + 0.35 * float(item.get("preference_score", 0.0))
            + 0.20 * float(item.get("quality_score", 0.0))
        )

        recommendations.append(
            {
                **item,
                "final_score": final_score,
            }
        )

    recommendations.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    return {
        "recommendations": recommendations[:5],
    }


# =========================================================
# 离线反馈模拟
# =========================================================

def deterministic_seed(*parts: str) -> int:
    """
    根据用户、实验和实验组生成稳定随机种子。

    避免同一份测试数据每次运行得到完全不同结果。
    """

    raw = ":".join(parts)

    digest = hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

    return int(digest[:16], 16)


def simulate_feedback(
    record: dict[str, Any],
    assignment: dict[str, Any],
    strategy_result: dict[str, Any],
) -> dict[str, float]:
    """
    离线演示用反馈模拟器。

    在线环境中应当替换为真实的：
    - 曝光事件
    - 点击事件
    - 下单事件
    - 支付事件
    - 页面停留事件
    """

    recommendations = strategy_result.get(
        "recommendations",
        [],
    )

    if not recommendations:
        return {
            "clicked": 0.0,
            "converted": 0.0,
            "gmv": 0.0,
            "dwell_time": 0.0,
        }

    top_item = recommendations[0]
    top_score = float(top_item.get("final_score", 0.0))

    seed = deterministic_seed(
        str(record["user_id"]),
        str(assignment["experiment_id"]),
        str(assignment["group"]),
    )

    rng = np.random.default_rng(seed)

    # 模拟点击概率
    click_probability = float(
        np.clip(
            0.05 + 0.70 * top_score,
            0.01,
            0.95,
        )
    )

    clicked = float(
        rng.random() < click_probability
    )

    # 点击后才可能成交
    conversion_probability = float(
        np.clip(
            0.03
            + 0.22 * float(
                top_item.get("preference_score", 0.0)
            )
            + 0.10 * top_score,
            0.01,
            0.60,
        )
    )

    converted = float(
        clicked
        and rng.random() < conversion_probability
    )

    price = float(top_item.get("price", 0.0))

    gmv = (
        price
        if converted
        else 0.0
    )

    # 模拟停留时长
    dwell_time = float(
        max(
            1.0,
            rng.normal(
                loc=(
                    15.0
                    + 45.0 * top_score
                    + 20.0 * clicked
                ),
                scale=8.0,
            ),
        )
    )

    return {
        "clicked": clicked,
        "converted": converted,
        "gmv": gmv,
        "dwell_time": dwell_time,
    }


# =========================================================
# 构造一批输入数据
# =========================================================

def build_demo_batch(
    user_count: int = 1_000,
    candidates_per_user: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    构造用户和候选商品数据。

    实际业务中可以替换为：
    - MySQL 查询结果
    - Elasticsearch 召回结果
    - Milvus 向量召回结果
    - RAG 混合检索结果
    """

    rng = np.random.default_rng(seed)

    batch_data: list[dict[str, Any]] = []

    for user_index in range(user_count):
        candidate_items = []

        for item_index in range(candidates_per_user):
            candidate_items.append(
                {
                    "item_id": f"product_{item_index:03d}",

                    "price": float(
                        rng.integers(100, 8_000)
                    ),

                    "quality_score": float(
                        rng.random()
                    ),

                    "popularity_score": float(
                        rng.random()
                    ),

                    "semantic_score": float(
                        rng.random()
                    ),

                    "preference_score": float(
                        rng.random()
                    ),
                }
            )

        batch_data.append(
            {
                "user_id": f"user_{user_index:06d}",
                "candidate_items": candidate_items,
            }
        )

    # print(batch_data)
    # input("点我继续")
    
    return batch_data


# =========================================================
# 运行实验
# =========================================================

def main() -> None:
    engine = ABTestEngine(
        bucket_count=10_000,
    )

    # -----------------------------------------------------
    # 1. 输入一批用户和候选商品
    # -----------------------------------------------------

    batch_data = build_demo_batch(
        user_count=1_000,
        candidates_per_user=10,
    )

    # -----------------------------------------------------
    # 2. 注册策略处理器
    # -----------------------------------------------------

    strategy_handlers = {
        "rule_based": rule_based_rerank,
        "llm": llm_rerank,
    }

    # -----------------------------------------------------
    # 3. 分桶并执行实验
    # -----------------------------------------------------

    results = engine.run_batch(
        records=batch_data,
        strategy_handlers=strategy_handlers,
        experiment_id="rec_strategy",

        # 离线演示中模拟用户反馈
        outcome_handler=simulate_feedback,
    )

    # -----------------------------------------------------
    # 4. 查看实验流量分布
    # -----------------------------------------------------

    distribution = engine.get_assignment_distribution(
        results
    )

    print("实验流量分布：")
    print(distribution)

    # -----------------------------------------------------
    # 5. 查看业务指标
    # -----------------------------------------------------

    business_stats = engine.get_business_stats(
        "rec_strategy"
    )

    print("\n实验业务指标：")

    for group_name, stats in business_stats.items():
        print(f"\n[{group_name}]")

        print(
            f"曝光数: {stats['exposures']}"
        )

        print(
            f"点击数: {stats['clicks']}"
        )

        print(
            f"订单数: {stats['orders']}"
        )

        print(
            f"CTR: {stats['ctr']:.4%}"
        )

        print(
            f"CVR: {stats['cvr']:.4%}"
        )

        print(
            f"GMV: {stats['gmv']:.2f}"
        )

        print(
            f"单曝光 GMV: "
            f"{stats['gmv_per_exposure']:.2f}"
        )

        print(
            f"平均停留时长: "
            f"{stats['avg_dwell_time']:.2f} 秒"
        )

    # -----------------------------------------------------
    # 6. 查看前 3 条执行结果
    # -----------------------------------------------------

    print("\n前 3 条实验执行结果：")

    for result in results[:3]:
        recommendations = result["recommendations"]

        top_item_id = (
            recommendations[0]["item_id"]
            if recommendations
            else None
        )

        print(
            {
                "user_id": result["user_id"],
                "bucket": result["bucket"],
                "group": result["group"],
                "strategy": result["strategy"],
                "top_item": top_item_id,
                "metrics": result["metrics"],
            }
        )


if __name__ == "__main__":
    main()
    
    
    