from app.exception.base import CustomError
from app.exception.llm import LLMProviderError
from app.exception.ingest import IngestionError
from app.exception.query import QueryError
from app.exception.vectordb import VectorDBError
from app.exception.collection import CollectionAlreadyExistsError

__all__ = (
    "CustomError",
    "IngestionError",
    "QueryError",
    "VectorDBError",
    "LLMProviderError",
    "CollectionAlreadyExistsError",
)
