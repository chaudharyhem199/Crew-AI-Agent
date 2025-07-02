"""
Configuration settings for the vector search system.
"""

import os
from pydantic import BaseSettings
from typing import Optional


class VectorSearchConfig(BaseSettings):
    """Configuration for vector search system."""
    
    # Database settings
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "vector_search")
    postgres_user: str = os.getenv("POSTGRES_USER", "postgres")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "password")
    
    # OpenAI settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimensions: int = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    
    # Redis cache settings
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    
    # Vector search settings
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    max_results: int = int(os.getenv("MAX_RESULTS", "10"))
    similarity_metric: str = os.getenv("SIMILARITY_METRIC", "cosine")  # cosine or l2
    
    # Cache settings
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    enable_cache: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    
    @property
    def database_url(self) -> str:
        """Generate PostgreSQL connection URL."""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    class Config:
        env_file = ".env"


# Global configuration instance
config = VectorSearchConfig()