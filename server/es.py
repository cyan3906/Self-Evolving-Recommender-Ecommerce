from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from elasticsearch.exceptions import RequestError
from elasticsearch.helpers import bulk


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from config import es, Settings

from tools.product_data import products

INDEX_NAME = Settings().INDEX_NAME


def delete_index() -> bool:
    """删除商品索引。"""
    if not es.indices.exists(index=INDEX_NAME):
        print(f"索引 {INDEX_NAME} 不存在")
        return False

    response = es.indices.delete(index=INDEX_NAME)

    print(f"索引 {INDEX_NAME} 删除成功：{response}")
    return True


def create_index() -> bool:
    """
    创建商品索引。

    返回：
        True：本次创建了索引
        False：索引已经存在
    """
    if es.indices.exists(index=INDEX_NAME):
        print(f"索引 {INDEX_NAME} 已存在，跳过创建")
        return False

    mapping = {
        "properties": {
            "category": {
                "type": "keyword",
            },
            "brand": {
                "type": "keyword",
            },
            "price": {
                "type": "integer",
            },
            "description": {
                "type": "text",
                "analyzer": "standard",
            },
        }
    }

    try:
        es.indices.create(
            index=INDEX_NAME,
            mappings=mapping,
        )

        print(f"索引 {INDEX_NAME} 创建成功")
        return True

    except RequestError as exc:
        error_type = getattr(exc, "error", None)

        if error_type == "resource_already_exists_exception":
            print(f"索引 {INDEX_NAME} 已存在，跳过创建")
            return False

        raise


def insert_doc(
    doc: dict[str, Any],
    doc_id: str | int,
) -> dict[str, Any]:
    """
    插入或覆盖一条商品数据。

    固定 doc_id 可以避免重复运行时产生重复文档。
    """
    response = es.index(
        index=INDEX_NAME,
        id=str(doc_id),
        document=doc,
        refresh="wait_for",
    )

    print(
        f"商品写入完成：id={doc_id}，"
        f"result={response['result']}"
    )

    return response


def bulk_insert(
    product_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    批量写入商品数据。

    商品在列表中的序号作为 Elasticsearch 文档 ID。
    """
    if not product_list:
        return {
            "success": 0,
            "failed": 0,
            "errors": [],
        }

    actions = []

    for product_id, product in enumerate(product_list, start=1):
        action = {
            "_op_type": "index",
            "_index": INDEX_NAME,
            "_id": str(product_id),
            "_source": product,
        }

        actions.append(action)

    success_count, errors = bulk(
        client=es,
        actions=actions,
        refresh="wait_for",
        raise_on_error=False,
        raise_on_exception=False,
    )

    failed_count = len(errors)

    print(
        f"商品批量写入完成：成功 {success_count} 条，"
        f"失败 {failed_count} 条"
    )

    if errors:
        for error in errors:
            print("写入失败：", error)

    return {
        "success": success_count,
        "failed": failed_count,
        "errors": errors,
    }


def search_full_text(
    keyword: str,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    根据商品描述进行全文检索。

    category 和 brand 同时参与检索。
    """
    query = {
        "query": {
            "multi_match": {
                "query": keyword,
                "fields": [
                    "category^3",
                    "brand^2",
                    "description",
                ],
            }
        },
        "size": top_k,
        "sort": [
            {
                "_score": {
                    "order": "desc",
                }
            }
        ],
    }

    response = es.search(
        index=INDEX_NAME,
        body=query,
    )

    results = []

    print(f"\n全文检索关键词：{keyword}")

    for rank, hit in enumerate(
        response["hits"]["hits"],
        start=1,
    ):
        source = hit["_source"]

        item = {
            "id": hit["_id"],
            "score": hit["_score"],
            **source,
        }

        results.append(item)

        print(
            f"排名={rank}，"
            f"id={item['id']}，"
            f"score={item['score']:.4f}，"
            f"category={item['category']}，"
            f"brand={item['brand']}，"
            f"price={item['price']}"
        )

    return results


def search_products(
    keyword: str | None = None,
    category: str | None = None,
    brand: str | None = None,
    min_price: int | None = None,
    max_price: int | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    商品综合检索。

    支持：
        1. 描述全文检索；
        2. 商品类别过滤；
        3. 品牌过滤；
        4. 价格区间过滤。
    """
    must_conditions = []
    filter_conditions = []

    if keyword:
        must_conditions.append(
            {
                "multi_match": {
                    "query": keyword,
                    "fields": [
                        "category^3",
                        "brand^2",
                        "description",
                    ],
                }
            }
        )

    if category:
        filter_conditions.append(
            {
                "term": {
                    "category": category,
                }
            }
        )

    if brand:
        filter_conditions.append(
            {
                "term": {
                    "brand": brand,
                }
            }
        )

    if min_price is not None or max_price is not None:
        price_range = {}

        if min_price is not None:
            price_range["gte"] = min_price

        if max_price is not None:
            price_range["lte"] = max_price

        filter_conditions.append(
            {
                "range": {
                    "price": price_range,
                }
            }
        )

    if not must_conditions and not filter_conditions:
        query_body = {
            "match_all": {},
        }
    else:
        query_body = {
            "bool": {
                "must": must_conditions,
                "filter": filter_conditions,
            }
        }

    response = es.search(
        index=INDEX_NAME,
        query=query_body,
        size=top_k,
    )

    results = []

    for rank, hit in enumerate(
        response["hits"]["hits"],
        start=1,
    ):
        source = hit["_source"]

        item = {
            "id": hit["_id"],
            "score": hit["_score"],
            **source,
        }

        results.append(item)

        print(
            f"排名={rank}，"
            f"id={item['id']}，"
            f"分类={item['category']}，"
            f"品牌={item['brand']}，"
            f"价格={item['price']}，"
            f"分数={item['score']}"
        )

    return results


def es_run() -> dict[str, Any]:
    """
    Elasticsearch 商品数据初始化入口。

    流程：
        1. 检查 Elasticsearch 连接；
        2. 索引不存在则创建；
        3. 批量写入商品数据；
        4. 返回执行结果。
    """
    try:
        if not es.ping():
            raise ConnectionError(
                "无法连接 Elasticsearch，请检查连接配置"
            )

        print("Elasticsearch 连接成功")
        print(f"当前索引：{INDEX_NAME}")

        index_created = create_index()

        insert_result = bulk_insert(products)

        return {
            "success": True,
            "index_name": INDEX_NAME,
            "index_created": index_created,
            "insert_success": insert_result["success"],
            "insert_failed": insert_result["failed"],
        }

    except Exception as exc:
        print(f"Elasticsearch 初始化失败：{exc}")

        return {
            "success": False,
            "index_name": INDEX_NAME,
            "error": str(exc),
        }


if __name__ == "__main__":
    
    result = es_run()

    print("\nes_run 执行结果：")
    print(result)

    if result["success"]:
        print("\n测试全文检索：")
        search_full_text(
            keyword="适合游戏和摄影的手机",
            top_k=10,
        )

        print("\n测试条件检索：")
        search_products(
            keyword="摄影",
            category="手机",
            min_price=3000,
            max_price=8000,
            top_k=10,
        )