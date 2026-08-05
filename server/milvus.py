from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import structlog
from pymilvus import MilvusClient
import sys
from config import es, Settings
from config import milvus as client


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from server.embedding_llm import generate_embedding

settings = Settings()
logger = structlog.get_logger()


def create_db() -> None:
    """
    创建 Milvus Lite 数据库所在目录。

    如果 dense_database_url 是远程 Milvus 地址，
    例如 http://localhost:19530，则不创建本地目录。
    """
    database_url = settings.dense_database_url

    if database_url.startswith(("http://", "https://")):
        return

    database_path = Path(database_url).expanduser()
    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

def delete_by_condition(filter_expr: str):
    """
    一次性条件删除，无需遍历ID
    :param filter_expr: Milvus标量过滤表达式
    """
    res = client.delete(
        collection_name=settings.milvus_collection,
        filter=filter_expr
    )
    print(f"本次删除数量：{res.get('delete_count',0)}")
    return res

def delete_all(): # 删除所有谨慎使用
    if client.has_collection(settings.milvus_collection):
        client.drop_collection(settings.milvus_collection)
    
    # print(11)
    
def create_collection(
    client: MilvusClient | None = None,
) -> bool:
    """
    创建并加载 Milvus Collection。

    返回：
        True：本次创建了 Collection
        False：Collection 已存在
    """
    if client is None:
        create_db()

    collection_name = settings.milvus_collection
    collections = client.list_collections()

    collection_created = False

    if collection_name not in collections:
        logger.info(
            "开始创建 Milvus Collection",
            collection_name=collection_name,
            dimension=settings.dimension,
            metric_type="COSINE",
        )

        client.create_collection(
            collection_name=collection_name,
            dimension=settings.dimension,
            metric_type="COSINE",
        )

        collection_created = True

        logger.info(
            "Milvus Collection 创建成功",
            collection_name=collection_name,
        )
    else:
        logger.info(
            "Milvus Collection 已存在，跳过创建",
            collection_name=collection_name,
        )

    # 搜索之前确保 Collection 已加载
    client.load_collection(
        collection_name=collection_name
    )

    logger.info(
        "Milvus Collection 加载完成",
        collection_name=collection_name,
    )

    return collection_created


def product_to_embedding_text(
    product: dict[str, Any],
) -> str:
    """
    将商品结构化字段转换成用于生成 Embedding 的文本。

    结构化字段仍然会单独存入 Milvus，
    这里只是构造向量化文本。
    """
    category = product.get("category", "")
    brand = product.get("brand", "")
    price = product.get("price", "")
    description = product.get("description", "")

    return (
        f"商品类别：{category}；"
        f"商品品牌：{brand}；"
        f"商品价格：{price}元；"
        f"商品描述：{description}"
    )


def insert_embedding(
    products: list[dict[str, Any]],
    client: MilvusClient | None = None,
) -> dict[str, Any]:
    """
    批量生成商品向量并写入 Milvus。

    使用 upsert：
        - ID 不存在：插入；
        - ID 已存在：更新；
        - 重复运行不会产生相同 ID 的重复商品。

    商品 ID 从 1 开始，与 Elasticsearch 中的商品 ID 保持一致，
    便于后续进行 RRF 融合。
    """
    if not products:
        logger.warning("商品列表为空，跳过 Milvus 写入")

        return {
            "upsert_count": 0,
        }

    embedding_texts = [
        product_to_embedding_text(product)
        for product in products
    ]

    logger.info(
        "开始生成商品 Embedding",
        product_count=len(products),
    )

    vectors = generate_embedding(
        embedding_texts
    )

    if len(vectors) != len(products):
        raise ValueError(
            "Embedding 数量与商品数量不一致："
            f"商品数量={len(products)}，"
            f"向量数量={len(vectors)}"
        )

    insert_data = []

    # 从 1 开始，保证与 ES 的 _id 保持一致
    for product_id, (product, vector, embedding_text) in enumerate(
        zip(products, vectors, embedding_texts),
        start=1,
    ):
        if len(vector) != settings.dimension:
            raise ValueError(
                f"商品 ID={product_id} 的向量维度错误："
                f"实际维度={len(vector)}，"
                f"Collection 维度={settings.dimension}"
            )

        insert_data.append(
            {
                "id": product_id,
                "vector": vector,
                "category": product.get("category"),
                "brand": product.get("brand"),
                "price": product.get("price"),
                "description": product.get("description"),

                # 保存实际用于生成向量的文本，便于排查
                "embedding_text": embedding_text,
            }
        )

    result = client.upsert(
        collection_name=settings.milvus_collection,
        data=insert_data,
    )

    logger.info(
        "Milvus 商品数据写入完成",
        product_count=len(insert_data),
        result=result,
    )

    return result


def search_embedding(
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Milvus 商品语义检索。

    返回格式统一为：
        {
            "id": "1",
            "score": 0.95,
            "category": "手机",
            "brand": "Apple",
            "price": 7999,
            "description": "..."
        }

    这样可以直接与 ES 检索结果进行 RRF 融合。
    """
    if not query or not query.strip():
        raise ValueError("检索关键词不能为空")

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    # 确保检索前 Collection 已加载
    client.load_collection(
        collection_name=settings.milvus_collection
    )

    query_vector = generate_embedding(
        [query]
    )[0]

    if len(query_vector) != settings.dimension:
        raise ValueError(
            "查询向量维度错误："
            f"实际维度={len(query_vector)}，"
            f"Collection 维度={settings.dimension}"
        )

    raw_results = client.search(
        collection_name=settings.milvus_collection,
        data=[query_vector],
        anns_field="vector",
        filter=None,
        limit=top_k,
        output_fields=[
            "category",
            "brand",
            "price",
            "description",
            "embedding_text",
        ],
        search_params={
            "metric_type": "COSINE",
        },
        consistency_level="Strong",
    )

    results: list[dict[str, Any]] = []

    if not raw_results:
        return results

    for rank, hit in enumerate(
        raw_results[0],
        start=1,
    ):
        entity = hit.get("entity", {})

        result = {
            # 转成字符串，方便和 ES 的 _id 比较
            "id": str(hit["id"]),
            "score": float(hit.get("distance", 0.0)),
            "category": entity.get("category"),
            "brand": entity.get("brand"),
            "price": entity.get("price"),
            "description": entity.get("description"),
        }

        results.append(result)

        logger.info(
            "Milvus 检索结果",
            rank=rank,
            id=result["id"],
            score=result["score"],
            category=result["category"],
            brand=result["brand"],
            price=result["price"],
        )

    return results


def milvus_run(
    products: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Milvus 初始化入口。

    执行流程：
        1. 创建本地数据库目录；
        2. 创建 MilvusClient；
        3. Collection 不存在则创建；
        4. 加载 Collection；
        5. 生成商品 Embedding；
        6. 使用 upsert 写入商品数据；
        7. 返回执行结果。
    """
    # delete_all()
    
    try:
        logger.info(
            "开始初始化 Milvus",
            database_url=settings.dense_database_url,
            collection_name=settings.milvus_collection,
        )

        # 1. 创建 Milvus Lite 数据目录
        create_db()

        # 2. 创建并加载 Collection
        collection_created = create_collection(
            client=client
        )

        # 3. 写入或更新商品数据
        upsert_result = insert_embedding(
            products=products,
            client=client,
        )

        # 4. 再次确保 Collection 为加载状态
        client.load_collection(
            collection_name=settings.milvus_collection
        )

        result = {
            "success": True,
            "database_url": settings.dense_database_url,
            "collection_name": settings.milvus_collection,
            "collection_created": collection_created,
            "product_count": len(products),
            "upsert_result": upsert_result,
        }

        logger.info(
            "Milvus 初始化完成",
            collection_name=settings.milvus_collection,
            product_count=len(products),
        )

        return result

    except Exception as exc:
        logger.exception(
            "Milvus 初始化失败",
            error=str(exc),
        )

        return {
            "success": False,
            "database_url": settings.dense_database_url,
            "collection_name": settings.milvus_collection,
            "error": str(exc),
        }


if __name__ == "__main__":
    from make_product import products

    # 自动创建 Collection 并存储商品
    run_result = milvus_run(products)

    logger.info(
        "milvus_run 执行结果",
        result=run_result,
    )

    if run_result["success"]:
        # 测试语义检索
        search_results = search_embedding(
            query="适合摄影和游戏的苹果手机",
            top_k=10,
        )

        print("\nMilvus 检索结果：")

        for item in search_results:
            print(
                f"id={item['id']} | "
                f"score={item['score']:.6f} | "
                f"category={item['category']} | "
                f"brand={item['brand']} | "
                f"price={item['price']} | "
                f"description={item['description']}"
            )