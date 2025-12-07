from app.core.db import get_vectorstore
from app.core.llm import get_llm
from app.core.prompt_manager import prompt_manager
from app.schema.llm import EmbeddingProvider, LLMProvider
from app.schema.db import VectorDB


def get_prompt_manager():
    return prompt_manager


def get_vectorstore_deps():
    return get_vectorstore(
        vector_db=VectorDB.CHROMADB,
        embedding_provider=EmbeddingProvider.GOOGLE,
        model_name="gemini-embedding-001",
        persist_directory="./vectorstore",
    )


def get_llm_deps():
    return get_llm(LLMProvider.GOOGLE, model_name="gemini-2.5-flash")
