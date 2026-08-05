

# from server.run_llm import ChatGPT
from server import es_run,milvus_run,hybrid_search
# from config.settings import Settings

from config import milvus

from tools.product_data import products
from server.rag import print_results

if __name__ == "__main__":
    
    
    # 1.对product产品数据进行 ES 和 milvus 存储
    # es_run()
    # milvus_run(products)
    
    # print(milvus.list_collections())
    res = hybrid_search("华为")
    
    print_results(res)
    # 2. A/B test
    
    # print(111,settings.llm_api_key,settings.embedding_api_key)
    # print(ChatGPT().generate("介绍自己", "介绍你自己"))





