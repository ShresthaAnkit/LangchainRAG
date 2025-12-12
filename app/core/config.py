from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from dotenv import load_dotenv
from app.schema.llm import EmbeddingProvider, LLMProvider
from app.schema.db import VectorDB

load_dotenv()


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file="app/.env",
        env_file_encoding="utf-8"
    )

    LOG_LEVEL: str = "INFO"

    VECTOR_DB: VectorDB = VectorDB.CHROMADB
    VECTORDB_PERSIST_DIRECTORY: str = "./vectorstore"
    VECTORDB_PERSIST_URL: str = "http://localhost:6333"

    EMBEDDING_PROVIDER: EmbeddingProvider = EmbeddingProvider.GOOGLE
    EMBEDDING_MODEL_NAME: str = "gemini-embedding-001"
    
    LLM_PROVIDER: LLMProvider = LLMProvider.GOOGLE
    LLM_MODEL_NAME: str = "gemini-2.5-flash"

    TAVILY_API_KEY: str = ""
    
    REDIS_URL: str = "redis://localhost:6379"

    GOOGLE_API_KEY: str = ""
    COHERE_API_KEY: str = ""

    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_BASE_URL: str = ""
    LANGFUSE_TIMEOUT: int = 5

settings = Settings()
