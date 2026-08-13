import sys
import time
from pathlib import Path

from elasticsearch import Elasticsearch
from pymilvus import MilvusClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

from config import Settings


settings = Settings()


# ============================================================
# Elasticsearch
# ============================================================

def connect_elasticsearch(
    max_retries: int = 5,
    initial_delay: float = 2.0,
) -> Elasticsearch:
    """
    连接 Elasticsearch，失败自动重试。

    重试间隔：
    2s -> 4s -> 8s -> 16s -> 32s
    """

    delay = initial_delay
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"[Elasticsearch] 正在连接 "
                f"({attempt}/{max_retries})..."
            )

            client = Elasticsearch(
                settings.es_url,

                # 有账号密码场景：
                # basic_auth=("elastic", "你的密码"),

                # 单次请求超时
                request_timeout=10,
            )

            if client.ping():
                print("[Elasticsearch] 连接成功")
                return client

            last_error = ConnectionError(
                "Elasticsearch ping 失败"
            )

        except Exception as exc:
            last_error = exc

        if attempt < max_retries:
            print(
                f"[Elasticsearch] 连接失败: {last_error}，"
                f"{delay:.0f}s 后重试..."
            )

            time.sleep(delay)

            # 指数退避
            delay *= 2

    raise ConnectionError(
        f"Elasticsearch 连接失败，"
        f"已重试 {max_retries} 次。"
    ) from last_error


# ============================================================
# Milvus
# ============================================================

def connect_milvus(
    max_retries: int = 5,
    initial_delay: float = 2.0,
) -> MilvusClient:
    """
    连接 Milvus，失败自动重试。
    """

    delay = initial_delay
    last_error = None

    milvus_uri = PROJECT_ROOT / settings.dense_database_url

    print(f"Milvus URI: {milvus_uri}")

    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"[Milvus] 正在连接 "
                f"({attempt}/{max_retries})..."
            )

            client = MilvusClient(
                uri=str(milvus_uri)
            )

            # 主动执行一次请求，确认真正可用
            collections = client.list_collections()

            print(
                f"[Milvus] 连接成功，"
                f"collections={collections}"
            )

            return client

        except Exception as exc:
            last_error = exc

        if attempt < max_retries:
            print(
                f"[Milvus] 连接失败: {last_error}，"
                f"{delay:.0f}s 后重试..."
            )

            time.sleep(delay)

            # 指数退避
            delay *= 2

    raise ConnectionError(
        f"Milvus 连接失败，"
        f"已重试 {max_retries} 次。"
    ) from last_error


# ============================================================
# 初始化连接
# ============================================================

es_client = connect_elasticsearch(
    max_retries=5,
    initial_delay=2,
)

milvus_client = connect_milvus(
    max_retries=5,
    initial_delay=2,
)


if __name__ == "__main__":
    print(
        "Milvus collections:",
        milvus_client.list_collections(),
    )