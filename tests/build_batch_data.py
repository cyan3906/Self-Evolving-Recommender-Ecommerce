import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from tools.product_data import products
from tools.user_query_data import query


def return_batch_data() -> dict:
    
    """返回批量数据。"""
    
    batch_data = []
    
    for idx,q in enumerate(query):
        batch_data.append(
            {"user_id": f"user_{idx:06d}", "query": q}
        )
    
    return batch_data


# def build_demo_batch(
#     user_count: int = 1_000,
#     candidates_per_user: int = 10,
#     seed: int = 42,
# ) -> list[dict[str, Any]]:
#     """
#     构造用户和候选商品数据。

#     实际业务中可以替换为：
#     - MySQL 查询结果
#     - Elasticsearch 召回结果
#     - Milvus 向量召回结果
#     - RAG 混合检索结果
#     """

#     rng = np.random.default_rng(seed)

#     batch_data: list[dict[str, Any]] = []

#     for user_index in range(user_count):
#         candidate_items = []

#         for item_index in range(candidates_per_user):
#             candidate_items.append(
#                 {
#                     "item_id": f"product_{item_index:03d}",

#                     "price": float(
#                         rng.integers(100, 8_000)
#                     ),

#                     "quality_score": float(
#                         rng.random()
#                     ),

#                     "popularity_score": float(
#                         rng.random()
#                     ),

#                     "semantic_score": float(
#                         rng.random()
#                     ),

#                     "preference_score": float(
#                         rng.random()
#                     ),
#                 }
#             )

#         batch_data.append(
#             {
#                 "user_id": f"user_{user_index:06d}",
#                 "candidate_items": candidate_items,
#             }
#         )

#     # print(batch_data)
#     # input("点我继续")
    
#     return batch_data


