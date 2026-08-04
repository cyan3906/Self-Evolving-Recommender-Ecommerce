
from config import Settings
from elasticsearch import Elasticsearch
from pymilvus import MilvusClient

es_client = Elasticsearch(
    Settings().es_url,
    # 有账号密码场景：
    # basic_auth=("elastic", "你的密码")
)

if not es_client.ping():
    raise ConnectionError("Elasticsearch 连接失败！检查地址/端口/认证")

milvus_client = MilvusClient(
    uri=Settings().dense_database_url
)



if __name__ == "__main__":
    print(milvus_client.list_collections())

