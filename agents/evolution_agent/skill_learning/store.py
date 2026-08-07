
import os
import sys
from writer import hybrid_retrieval
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    

def save_hybird_retrieval(
    file_name: str,
    potential_products: list,
    es_top_k: int,
    milvus_top_k: int,
    final_top_k: int,
    rrf_k: int = 10,
    es_weight: float = 1.0,
    milvus_weight: float = 1.0,
) -> str:

    try:
        potential_products = [f"{product}" for product in potential_products]
        skill = hybrid_retrieval.format(
            file_name=file_name,
            potential_products=potential_products,
            es_top_k=es_top_k,
            milvus_top_k=milvus_top_k,
            final_top_k=final_top_k,
            rrf_k=rrf_k,
            es_weight=es_weight,
            milvus_weight=milvus_weight,
        )
        # 自动创建目录，避免文件夹不存在报错
        os.makedirs(PROJECT_ROOT / "skills/hybrid-retrieval", exist_ok=True)
        with open(PROJECT_ROOT / f"skills/hybrid-retrieval/{file_name}.md", "w", encoding="utf-8") as f:
            f.write(skill)
        # 成功返回
        return {"success": True, "message": "生成skill成功", "file_name": file_name}

    except Exception as e:
        # 捕获所有异常，返回失败，携带错误信息
        return {"success": False, "message": "生成skill失败", "error": str(e)}
        
