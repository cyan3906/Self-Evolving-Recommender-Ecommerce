import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from server import search_es,search_milvus,hybrid_search



def return_strategy_handlers():
    return {
        "es": search_es,
        "milvus": search_milvus,
        "hybrid": hybrid_search,
    }