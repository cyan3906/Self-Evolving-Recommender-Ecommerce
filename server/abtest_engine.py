"""
A/B测试引擎
- 流量分桶：用户ID哈希取模分桶
- 实验层：Agent级别 / 模型级别 / Prompt级别实验
- MAB算法：Thompson Sampling动态分配流量
- 指标收集：CTR / CVR / GMV / 停留时长
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Experiment:
    id: str
    name: str
    groups: list[ExperimentGroup]
    enabled: bool = True
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class ExperimentGroup:
    name: str
    weight: int = 50
    config: dict[str, Any] = field(default_factory=dict)
    # Thompson Sampling state
    successes: int = 1
    failures: int = 1



class ABTestEngine:
    """基于用户稳定分桶的 A/B 测试引擎。"""

    def __init__(
        self,
        bucket_count: int = 10_000,
        random_seed: int = 2026,
        control_name: str = "control",
        treatment_name: str = "treatment_llm",
        control_experiment_config: dict[str, Any] = {},
        treatment_experiment_config: dict[str, Any] = {},
        control_weight: int = 50,
        treatment_weight: int = 50,
        experiment_id: str = "rec_strategy",
        experiment_name: str = "推荐策略实验",
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

        self.control_name = control_name
        self.treatment_name = treatment_name
        self.control_experiment_config = control_experiment_config
        self.treatment_experiment_config = treatment_experiment_config
        self.control_weight = control_weight
        self.treatment_weight = treatment_weight
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name

        self._init_default_experiments()
        
    def _init_default_experiments(self) -> None:
        """初始化默认推荐策略实验。"""

        self.register_experiment(
            Experiment(
                id=self.experiment_id,
                name=self.experiment_name,
                groups=[
                    ExperimentGroup(
                        name=self.control_name,
                        weight=self.control_weight,
                        config=self.control_experiment_config,
                    ),
                    ExperimentGroup(
                        name=self.treatment_name,
                        weight=self.treatment_weight,
                        config=self.treatment_experiment_config,
                    ),
                ],
            )
        )

    def register_experiment(self, exp: Experiment) -> None:
        """注册实验。"""

        if not exp.id:
            raise ValueError("实验 id 不能为空")

        if not exp.name:
            raise ValueError("实验 name 不能为空")

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
                "group": exp.control_name,
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
        strategy_choice: str = "search",  # 你要根据 strategy_choice 来选择策略 比如是测试 rag的 检索还是rerank
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

            # 2. 从实验配置中获取策略名称 你要根据 strategy_choice 来选择策略 比如是测试 rag的 检索还是rerank
            strategy_name = assignment["config"].get(  
                strategy_choice,
                "es",
            )

            # 3. 查找对应策略函数 也就是执行具体的策略
            strategy_handler = strategy_handlers.get(strategy_name)

            if strategy_handler is None:
                raise KeyError(
                    f"未注册推荐策略处理器: {strategy_name}"
                )

            # 4. 执行推荐策略
            strategy_result = strategy_handler(
                record, # 按照每一次来
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




# class ABTestEngine:
#     """Bucket-based A/B test engine with optional Thompson Sampling."""

#     def __init__(self, bucket_count: int = 100):
#         self.bucket_count = bucket_count
#         self.experiments: dict[str, Experiment] = {}
#         self._metrics: list[dict[str, Any]] = []
#         self._init_default_experiments()

#     def _init_default_experiments(self):
#         self.register_experiment(
#             Experiment(
#                 id="rec_strategy",
#                 name="推荐策略实验",
#                 groups=[
#                     ExperimentGroup(name="control", weight=50, config={"rerank": "rule_based"}),
#                     ExperimentGroup(name="treatment_llm", weight=50, config={"rerank": "llm"}),
#                 ],
#             )
#         )
#         self.register_experiment(
#             Experiment(
#                 id="copy_style",
#                 name="文案风格实验",
#                 groups=[
#                     ExperimentGroup(name="formal", weight=50, config={"style": "formal"}),
#                     ExperimentGroup(name="casual", weight=50, config={"style": "casual"}),
#                 ],
#             )
#         )

#     def register_experiment(self, exp: Experiment):
#         self.experiments[exp.id] = exp

#     def assign(self, user_id: str, experiment_id: str = "rec_strategy") -> dict[str, Any]:
#         """Assign user to an experiment group using consistent hashing."""
#         exp = self.experiments.get(experiment_id)
#         if not exp or not exp.enabled:
#             return {"group": "control", "config": {}}

#         bucket = self._hash_bucket(user_id, experiment_id)
#         group = self._bucket_to_group(bucket, exp.groups)
#         return {"group": group.name, "config": group.config}

#     def assign_thompson(self, user_id: str, experiment_id: str = "rec_strategy") -> dict[str, Any]:
#         """Use Thompson Sampling for dynamic traffic allocation."""
#         exp = self.experiments.get(experiment_id)
#         if not exp or not exp.enabled:
#             return {"group": "control", "config": {}}

#         samples = []
#         for g in exp.groups:
#             sample = np.random.beta(g.successes, g.failures)
#             samples.append((sample, g))

#         best = max(samples, key=lambda x: x[0])[1]
#         return {"group": best.name, "config": best.config}

#     def record_outcome(self, experiment_id: str, group_name: str, success: bool):
#         """Update Thompson Sampling posterior with observed outcome."""
#         exp = self.experiments.get(experiment_id)
#         if not exp:
#             return
#         for g in exp.groups:
#             if g.name == group_name:
#                 if success:
#                     g.successes += 1
#                 else:
#                     g.failures += 1
#                 break

#     def record_metric(
#         self,
#         experiment_id: str,
#         group_name: str,
#         metric_name: str,
#         value: float,
#         user_id: str = "",
#     ):
#         self._metrics.append({
#             "experiment_id": experiment_id,
#             "group": group_name,
#             "metric": metric_name,
#             "value": value,
#             "user_id": user_id,
#             "timestamp": time.time(),
#         })

#     def get_stats(self, experiment_id: str) -> dict[str, Any]:
#         """Aggregate metrics per group for a given experiment."""
#         exp = self.experiments.get(experiment_id)
#         if not exp:
#             return {}
#         relevant = [m for m in self._metrics if m["experiment_id"] == experiment_id]
#         stats: dict[str, dict[str, list[float]]] = {}
#         for m in relevant:
#             grp = m["group"]
#             metric = m["metric"]
#             if grp not in stats:
#                 stats[grp] = {}
#             if metric not in stats[grp]:
#                 stats[grp][metric] = []
#             stats[grp][metric].append(m["value"])

#         result: dict[str, Any] = {}
#         for grp, metrics in stats.items():
#             result[grp] = {}
#             for metric_name, values in metrics.items():
#                 arr = np.array(values)
#                 result[grp][metric_name] = {
#                     "count": len(values),
#                     "mean": float(arr.mean()),
#                     "std": float(arr.std()),
#                     "min": float(arr.min()),
#                     "max": float(arr.max()),
#                 }
#         return result

#     def _hash_bucket(self, user_id: str, experiment_id: str) -> int:
#         raw = f"{user_id}:{experiment_id}"
#         h = hashlib.md5(raw.encode()).hexdigest()
#         return int(h[:8], 16) % self.bucket_count

#     def _bucket_to_group(
#         self, bucket: int, groups: list[ExperimentGroup]
#     ) -> ExperimentGroup:
#         total_weight = sum(g.weight for g in groups)
#         cumulative = 0
#         normalized_bucket = bucket * total_weight / self.bucket_count
#         for g in groups:
#             cumulative += g.weight
#             if normalized_bucket < cumulative:
#                 return g
#         return groups[-1]
