
    Hybrid Retrieval Strategy

    Skill 基本信息

    skill_name: hybrid-retrieval
    skill_path: skills/hybrid-retrieval/SKILL2.md
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

    potential_products: ['商品1', '商品2', '商品3']
    es_top_k: 10
    milvus_top_k: 10
    final_top_k: 10
    rrf_k: 10
    es_weight: 0.7
    milvus_weight: 0.3
    is_execute: true
