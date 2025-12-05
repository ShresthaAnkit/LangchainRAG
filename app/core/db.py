from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from app.schema.db import VectorDB
from app.schema.llm import EmbeddingProvider

_vectordb = None


def get_vectorstore(
    vector_db: VectorDB,
    embedding_provider: EmbeddingProvider,
    model_name: str,
    persist_directory: str,
):
    global _vectordb

    if embedding_provider == EmbeddingProvider.GOOGLE:
        embedding_function = GoogleGenerativeAIEmbeddings(
            model=model_name,
        )
    else:
        raise NotImplementedError()

    if vector_db == VectorDB.CHROMADB:
        _vectordb = Chroma(
            persist_directory=f"{persist_directory}-{vector_db.value}",
            embedding_function=embedding_function,
        )
    else:
        raise NotImplementedError()

    return _vectordb
