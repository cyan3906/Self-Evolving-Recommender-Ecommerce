

hybrid_retrieval = """
    Hybrid Retrieval Strategy

    Skill 基本信息

    skill_name: hybrid-retrieval
    skill_path: skills/hybrid-retrieval/{file_name}.md
    description: >
    商品检索场景下的混合检索策略。
    必须同时调用 Elasticsearch 与 Milvus，
    Elasticsearch 负责关键词、属性与结构化条件匹配，
    Milvus 负责语义与相似商品召回，
    最终使用加权 RRF 进行融合排序。

    检索方式

    必须使用混合检索，同时调用：

    Elasticsearch

    Milvus

    正常情况下不得跳过任何一路检索。

    检索参数

    potential_products: {potential_products}
    es_top_k: {es_top_k}
    milvus_top_k: {milvus_top_k}
    final_top_k: {final_top_k}
    rrf_k: {rrf_k}
    es_weight: {es_weight}
    milvus_weight: {milvus_weight}
    is_execute: true
"""

