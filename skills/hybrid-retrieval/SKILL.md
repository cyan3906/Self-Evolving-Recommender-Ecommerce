---

name: hybrid-retrieval
description: 规定 RAG 使用 Elasticsearch 与 Milvus 混合检索，并配置两路检索的融合比例。
----------------------------------------------------------------

# Hybrid Retrieval Strategy

## 检索方式

必须使用混合检索，同时调用：

* Elasticsearch
* Milvus

## 检索权重

```yaml
elasticsearch_weight: 0.7
milvus_weight: 0.3
```

最终融合得分：

```text
final_score = es_score × 0.7 + milvus_score × 0.3
```

## 召回参数

```yaml
elasticsearch_top_k: 50
milvus_top_k: 50
final_top_k: 20
```

## Elasticsearch 配置

```yaml
retrieval_type: keyword
weight: 0.7
score_field: _score
```

Elasticsearch 主要负责：

* 商品名称匹配
* 品牌匹配
* 商品类别匹配
* 商品属性匹配
* 用户行为特征匹配
* 结构化条件过滤

## Milvus 配置

```yaml
retrieval_type: vector
weight: 0.3
metric_type: COSINE
score_field: distance
```

Milvus 主要负责：

* 查询语义匹配
* 商品描述语义匹配
* 相似商品召回
* 用户自然语言需求召回

## 融合规则

1. 两路检索结果使用统一的 `product_id` 合并。
2. Elasticsearch 和 Milvus 得分需要归一化到 `[0, 1]`。
3. Elasticsearch 得分权重为 `0.7`。
4. Milvus 得分权重为 `0.3`。
5. 某个商品未被某一路召回时，该路得分设置为 `0`。
6. 最终按照 `final_score` 降序排列。
7. 返回排序后的前 `20` 条结果。

## 降级策略

```yaml
elasticsearch_failed:
  milvus_weight: 1.0

milvus_failed:
  elasticsearch_weight: 1.0
```

正常情况下不得跳过任何一路检索。
