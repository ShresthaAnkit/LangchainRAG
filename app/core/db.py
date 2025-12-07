from httpx import get
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from app.schema.db import VectorDB
from app.schema.llm import EmbeddingProvider
from app.core.logging_config import get_logger

logger = get_logger(__name__)

def get_vectorstore(
    vector_db: VectorDB,
    embedding_provider: EmbeddingProvider,
    model_name: str,
    persist_directory: str,
):

    if embedding_provider == EmbeddingProvider.GOOGLE:
        embedding_function = GoogleGenerativeAIEmbeddings(
            model=model_name,
        )
    else:
        logger.error(f"Embedding provider {embedding_provider} not supported")
        raise NotImplementedError()

    vectordb_persist_directory = f"{persist_directory}-{vector_db.value}"    

    if vector_db == VectorDB.CHROMADB:

        vectordb = Chroma(
            persist_directory=vectordb_persist_directory,
            embedding_function=embedding_function,
        )
    elif vector_db == VectorDB.QDRANT:
        client = QdrantClient(path=vectordb_persist_directory)

        collection_name = f"collection-{embedding_provider.value}-{model_name}"
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        if collection_name not in collection_names:
            sample_vector = embedding_function.embed_query("test")
            vector_size = len(sample_vector)
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "size": vector_size,
                    "distance": "Cosine",
                },
            )

        vectordb = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_function,
        )
    else:
        logger.error(f"Vector DB {vector_db} not supported")
        raise NotImplementedError()

    return vectordb
