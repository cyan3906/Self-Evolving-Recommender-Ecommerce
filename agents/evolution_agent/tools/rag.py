from __future__ import annotations

from collections import defaultdict
from typing import Any
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from server.es import search_full_text
from server.milvus import search_embedding
from typing import Any
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command

# from langchain.agents import LayerAgentState
from agent_init import LayerAgentState

# @tool(description="使用 Elasticsearch 检索商品")
def es_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """
    模拟 Elasticsearch 关键词检索。

    实际项目中，可以把这里替换成 Elasticsearch 的 search 方法。
    返回结果顺序必须按照相关性从高到低排列。
    """
    
    res = search_full_text(
        keyword = query,
        top_k = top_k
    )

    # print(res)
    # input("点我继续")
    
    return res

# @tool(description="使用 Milvus 检索商品")
def milvus_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """
    模拟 Milvus 语义向量检索。

    实际项目中，可以把这里替换成 MilvusClient.search()。
    返回结果顺序必须按照语义相关性从高到低排列。

    注意：
    Milvus 使用不同距离度量时，原始 score 的含义可能不同：
    - COSINE / IP：通常越大越相似
    - L2：通常越小越相似

    RRF 只使用排名，所以不需要统一 ES 和 Milvus 的原始分数。
    """
    
    res = search_embedding(
        query = query,
        top_k = top_k
    )

    # print(res)
    # input("点我继续")
    
    return res

def rrf_fusion(
    result_sets: dict[str, list[dict[str, Any]]],
    *,
    id_field: str = "id",
    k: int = 60,
    source_weights: dict[str, float] | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    对多个检索系统的结果进行 RRF 融合。

    参数
    ----------
    result_sets:
        不同检索系统的结果，例如：
        {
            "es": [...],
            "milvus": [...]
        }

        每个结果列表必须已经按照相关性从高到低排序。

    id_field:
        文档唯一标识字段，用于去重。

    k:
        RRF 平滑参数，一般使用 60。

    source_weights:
        各检索系统的权重，例如：
        {
            "es": 1.0,
            "milvus": 1.0
        }

    top_k:
        最终保留多少条结果。None 表示全部返回。

    返回
    ----
    按照 rrf_score 从高到低排序后的融合结果。
    """
    if k < 0:
        raise ValueError("k 必须大于或等于 0")

    source_weights = source_weights or {}

    # 保存文档基本内容
    documents: dict[str, dict[str, Any]] = {}

    # 保存每个文档的 RRF 总分
    rrf_scores: defaultdict[str, float] = defaultdict(float)

    # 保存文档在每个检索源中的详细信息
    source_details: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for source_name, results in result_sets.items():
        weight = source_weights.get(source_name, 1.0)

        if weight < 0:
            raise ValueError(f"检索源 {source_name} 的权重不能小于 0")

        # 防止同一个检索源中出现重复文档
        seen_ids: set[str] = set()

        for rank, item in enumerate(results, start=1):
            if id_field not in item:
                raise KeyError(
                    f"检索源 {source_name} 的第 {rank} 条结果缺少字段：{id_field}"
                )

            document_id = str(item[id_field])

            if document_id in seen_ids:
                continue

            seen_ids.add(document_id)

            # 同一个文档可能同时出现在 ES 和 Milvus 中。
            # 第一次出现时保存主体数据。
            if document_id not in documents:
                documents[document_id] = dict(item)
            else:
                # 补充其他检索源中存在但当前主体数据中没有的字段
                for key, value in item.items():
                    documents[document_id].setdefault(key, value)

            rrf_contribution = weight / (k + rank)
            rrf_scores[document_id] += rrf_contribution

            source_details[document_id][source_name] = {
                "rank": rank,
                "original_score": item.get("score"),
                "weight": weight,
                "rrf_contribution": rrf_contribution,
            }

    fused_results: list[dict[str, Any]] = []

    for document_id, document in documents.items():
        result = {
            **document,
            "rrf_score": rrf_scores[document_id],
            "matched_sources": list(source_details[document_id].keys()),
            "source_details": source_details[document_id],
        }
        fused_results.append(result)

    # 第一排序条件：RRF 总分越高越靠前
    # 第二排序条件：命中的检索源越多越靠前
    # 第三排序条件：id，保证相同输入下结果稳定
    fused_results.sort(
        key=lambda item: (
            -item["rrf_score"],
            -len(item["matched_sources"]),
            str(item[id_field]),
        )
    )

    if top_k is not None:
        return fused_results[:top_k]

    return fused_results

@tool(description="使用 Elasticsearch 和 Milvus 检索商品")
def hybrid_search(
    query: str,
    *,
    es_top_k: int = 10,
    milvus_top_k: int = 10,
    final_top_k: int = 10,
    rrf_k: int = 60,
    es_weight: float = 1.0,
    milvus_weight: float = 1.0,
    is_execute: bool = False,
    # 第一个 None：没有额外 Context
    # 第二个：当前 AgentState
    runtime: ToolRuntime[None, LayerAgentState],

) -> Command:
    """
    完整的混合检索入口：

    1. 获取 ES 结果；
    2. 获取 Milvus 结果；
    3. 使用 RRF 融合；
    4. 将完整检索结构写入 AgentState；
    5. 给模型返回简化后的 ToolMessage。
    """

    import time
    time.sleep(1)
    
    if not is_execute:
        return Command()

    print("11111")
    print(es_top_k)
    print(milvus_top_k)
    print(final_top_k)
    print(es_weight)
    print(milvus_weight)
    print("11111")

    # =============================
    # 1. Elasticsearch
    # =============================

    es_results = es_search(
        query=query,
        top_k=es_top_k,
    )

    # =============================
    # 2. Milvus
    # =============================

    milvus_results = milvus_search(
        query=query,
        top_k=milvus_top_k,
    )

    # =============================
    # 3. RRF
    # =============================

    final_results = rrf_fusion(
        result_sets={
            "es": es_results,
            "milvus": milvus_results,
        },
        k=rrf_k,
        source_weights={
            "es": es_weight,
            "milvus": milvus_weight,
        },
        top_k=final_top_k,
    )

    # =============================
    # 4. 构造结构化 State
    # =============================

    retrieval_state = {
        "query": query,

        "parameters": {
            "es_top_k": es_top_k,
            "milvus_top_k": milvus_top_k,
            "final_top_k": final_top_k,
            "rrf_k": rrf_k,
            "es_weight": es_weight,
            "milvus_weight": milvus_weight,
        },

        "es_results": es_results,

        "milvus_results": milvus_results,

        "final_results": final_results,
    }

    # =============================
    # 5. 更新 AgentState
    # =============================

    # return Command(
    #     update={
    #         "layer_1": [
    #             {
    #                 "tool": "hybrid_search",
    #                 "query": query,
    #                 "es_results": es_results,
    #                 "milvus_results": milvus_results,
    #                 "final_results": final_results,
    #             }
    #         ],

    #         "messages": [
    #             ToolMessage(
    #                 content="混合检索完成",
    #                 tool_call_id=runtime.tool_call_id,
    #             )
    #         ],
    #     }
    # )

    return Command(
        update={
            # 真正的数据保存在 State
            "layer_1": [retrieval_state],
            
            # ToolMessage 给模型看
            "messages": [
                ToolMessage(
                    content=(
                        f"已经调用了hybrid_search工具，下次请不要重复调用。下次请不要重复调用。下次请不要重复调用。混合检索完成，共获得 "
                        f"{len(final_results)} 条融合结果。"
                        f"完整结果已经保存到 "
                        f"AgentState.retrieval_layer。"
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
    
# @tool(description="使用 Elasticsearch 和 Milvus 检索商品")
# def hybrid_search(
#     query: str,
#     *,
#     es_top_k: int = 10,
#     milvus_top_k: int = 10,
#     final_top_k: int = 10,
#     rrf_k: int = 60,
#     es_weight: float = 1.0,
#     milvus_weight: float = 1.0,
#     # runtime: ToolRuntime[None, LayerAgentState],
# ) -> list[dict[str, Any]]:
#     """
#     完整的混合检索入口：
#     1. 获取 ES 结果；
#     2. 获取 Milvus 结果；
#     3. 使用 RRF 融合；
#     4. 返回最终排序结果。
#     """
    
#     print(11111)
#     print(es_top_k)
#     print(milvus_top_k)
#     print(final_top_k)
#     print(es_weight)
#     print(milvus_weight)
#     print(11111)
    
#     # input("点我继续")
#     es_results = es_search(query=query, top_k=es_top_k)
#     milvus_results = milvus_search(query=query, top_k=milvus_top_k)

#     return rrf_fusion(
#         result_sets={
#             "es": es_results,
#             "milvus": milvus_results,
#         },
#         k=rrf_k,
#         source_weights={
#             "es": es_weight,
#             "milvus": milvus_weight,
#         },
#         top_k=final_top_k,
#     )

def print_results(results: list[dict[str, Any]]) -> None:
    """打印融合结果。"""
    for final_rank, item in enumerate(results, start=1):
        print("=" * 80)
        print(f"最终排名：{final_rank}")
        print(f"商品 ID：{item['id']}")
        print(f"商品分类：{item['category']}")
        print(f"商品品牌：{item['brand']}")
        print(f"RRF 分数：{item['rrf_score']:.8f}")
        print(f"命中来源：{item['matched_sources']}")

        for source_name, detail in item["source_details"].items():
            print(
                f"  {source_name}: "
                f"排名={detail['rank']}, "
                f"原始分数={detail['original_score']}, "
                f"权重={detail['weight']}, "
                f"RRF贡献={detail['rrf_contribution']:.8f}"
            )


if __name__ == "__main__":
    query = "适合拍照的高端手机"

    fused_results = hybrid_search(
        query=query,
        es_top_k=10,
        milvus_top_k=10,
        final_top_k=10,
        rrf_k=60,
        es_weight=1.0,
        milvus_weight=1.0,
    )

    print_results(fused_results)