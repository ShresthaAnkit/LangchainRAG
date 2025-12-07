from app.schema.db import VectorDB
from app.schema.llm import EmbeddingProvider
from app.core.logging_config import get_logger

from app.exception import VectorDBError

logger = get_logger(__name__)


def get_vectorstore(
    vector_db: VectorDB,
    embedding_provider: EmbeddingProvider,
    model_name: str,
    persist_directory: str,
):
    try:
        if embedding_provider == EmbeddingProvider.GOOGLE:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            embedding_function = GoogleGenerativeAIEmbeddings(
                model=model_name,
            )
        elif embedding_provider == EmbeddingProvider.COHERE:
            from langchain_cohere import CohereEmbeddings

            embedding_function = CohereEmbeddings(
                model=model_name,
            )
        else:
            logger.error(f"Embedding provider {embedding_provider} not supported")
            raise VectorDBError(f"Unsupported embedding provider: {embedding_provider}")

        vectordb_persist_directory = f"{persist_directory}-{vector_db.value}"

        if vector_db == VectorDB.CHROMADB:
            from langchain_chroma import Chroma
            vectordb = Chroma(
                persist_directory=vectordb_persist_directory,
                embedding_function=embedding_function,
            )
        elif vector_db == VectorDB.QDRANT:
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient

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
            raise VectorDBError(f"Unsupported vector database: {vector_db}")

        return vectordb
    except Exception as e:
        logger.exception("Error occurred while initializing vector store")
        raise VectorDBError("An error occurred while initializing the vector store.") from e
