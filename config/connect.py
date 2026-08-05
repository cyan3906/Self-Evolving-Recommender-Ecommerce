import os
import sys
from pathlib import Path
from elasticsearch import Elasticsearch
from pymilvus import MilvusClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# print(PROJECT_ROOT)
from config import Settings

es_client = Elasticsearch(
    Settings().es_url,
    # 有账号密码场景：
    # basic_auth=("elastic", "你的密码")
)

if not es_client.ping():
    raise ConnectionError("Elasticsearch 连接失败！检查地址/端口/认证")

# print(Settings().dense_database_url)


milvus_client = MilvusClient(
    uri=str(PROJECT_ROOT / Settings().dense_database_url)
)


if __name__ == "__main__":
    print(milvus_client.list_collections())

