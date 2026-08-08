from pydantic_settings import BaseSettings
from functools import lru_cache
import logging
import structlog
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
env_path = BASE_DIR / ".env"
# print(env_path)

def setup_logging(log_level: str = "INFO"):
    log_level_name = log_level.upper()

    log_level = getattr(logging, log_level_name, logging.INFO)

    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    renderer = structlog.dev.ConsoleRenderer(colors=True)

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger() # 设置最低门槛
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


class Settings(BaseSettings):
    app_name: str = "Multi-Agent E-Commerce System"
    debug: bool = False

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.uiuihao.com/v1" 
    llm_model: str = "gpt-4o-mini" # minimax-m2
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    
    # Embedding
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.quickrouter.ai"
    embedding_model: str = "text-embedding-3-small"
    dimension: int = 1536
    
    # Redis
    redis_host: str = "127.0.0.1"
    redis_password: str = ""
    redis_port: int = 6379
    redis_db: int = 0
    feature_ttl_seconds: int = 86400

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "product_embeddings"

    # Database
    dense_database_url: str = "db/milvus_product.db"

    # ES
    es_url: str = "http://127.0.0.1:9200"
    INDEX_NAME: str = "product_demo" # "product_demo"
    
    # A/B Testing
    ab_test_enabled: bool = True
    ab_test_default_bucket_count: int = 100

    # Agent timeouts (seconds)
    agent_timeout_user_profile: float = 5.0
    agent_timeout_product_rec: float = 8.0
    agent_timeout_marketing_copy: float = 10.0
    agent_timeout_inventory: float = 5.0

    # model_config = {"env_file": ".env", "env_prefix": "ECOM_"}
    # __file__ = 当前这个 .py 文件的路径
    
    model_config = {"env_file": env_path}
    
    # Logger
    setup_logging(log_level="DEBUG")
    
@lru_cache()
def get_settings() -> Settings:
    return Settings()
