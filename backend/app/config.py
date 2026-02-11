"""
Configuration for RAG App - uses environment variables for deployment.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Keys (set via environment variables)
    VOYAGE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_URL: str = ""
    
    # Collection settings
    COLLECTION_NAME: str = "sbi_card_docs"
    
    # Chunking settings
    CHUNK_SIZE: int = 3000
    CHUNK_OVERLAP: int = 200
    MAX_CHUNKS: int = 50
    TOP_K: int = 5
    
    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
