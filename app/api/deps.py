from app.core.db import get_vectorstore, get_session_history
from app.core.llm import get_llm
from app.core.prompt_manager import prompt_manager
from app.core.config import settings


def get_prompt_manager_deps():
    return prompt_manager


def get_vectorstore_deps():
    return get_vectorstore(
        vector_db=settings.VECTOR_DB,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        model_name=settings.EMBEDDING_MODEL_NAME,
        persist_directory=settings.VECTORDB_PERSIST_DIRECTORY,
        persist_url=settings.VECTORDB_PERSIST_URL
    )


def get_llm_deps():
    return get_llm(settings.LLM_PROVIDER, model_name=settings.LLM_MODEL_NAME)

def get_session_history_deps(user_id: str):
    return get_session_history(user_id)