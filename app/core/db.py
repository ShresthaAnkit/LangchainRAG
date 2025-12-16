from langchain_community.chat_message_histories import RedisChatMessageHistory
from qdrant_client import QdrantClient

from app.exception import CollectionAlreadyExistsError, VectorDBError
from app.schema.llm import EmbeddingProvider
from app.schema.db import VectorDB
from app.core.config import settings
from app.core.logging_config import get_logger
from app.core.llm import get_embedding_function

logger = get_logger(__name__)


def get_session_history(session_id: str):
    """Return memory object for the user."""
    history = RedisChatMessageHistory(session_id=session_id, url=settings.REDIS_URL)

    return history


def get_vectorstore(
    vector_db: VectorDB,
    embedding_provider: EmbeddingProvider,
    collection_name: str,
    model_name: str,
    persist_directory: str = None,
    persist_url: str = None,
):
    try:
        embedding_function = get_embedding_function(
            embedding_provider=embedding_provider, model_name=model_name
        )

        # vectordb_persist_directory = f"{persist_directory}-{vector_db.value}"

        if vector_db == VectorDB.QDRANT:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient

            client = QdrantClient(url=persist_url)

            vectordb = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embedding_function,
            )
        else:
            logger.error(f"Vector DB {vector_db} not supported")
            raise VectorDBError(f"Unsupported vector database: {vector_db}")

        return vectordb
    except Exception as e:
        logger.exception("Error occurred while initializing vector store")
        raise VectorDBError(
            "An error occurred while initializing the vector store."
        ) from e


def create_collection_qdrant(collection_name: str) -> bool:
    embedding_function = get_embedding_function(
        embedding_provider=settings.EMBEDDING_PROVIDER,
        model_name=settings.EMBEDDING_MODEL_NAME,
    )

    client = QdrantClient(url=settings.VECTORDB_PERSIST_URL)

    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]
    if collection_name in collection_names:
        raise CollectionAlreadyExistsError(
            f"Collection {collection_name} already exists."
        )
    sample_vector = embedding_function.embed_query("test")
    vector_size = len(sample_vector)
    return client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "size": vector_size,
            "distance": "Cosine",
        },
    )


def list_collection_qdrant() -> list:
    client = QdrantClient(url=settings.VECTORDB_PERSIST_URL)
    collections = client.get_collections().collections
    collections_list = []
    for collection in collections:
        collections_list.append(collection.name)
    return collections_list

def delete_collection_qdrant(collection_name: str) -> bool:
    client = QdrantClient(url=settings.VECTORDB_PERSIST_URL)
    return client.delete_collection(collection_name)
