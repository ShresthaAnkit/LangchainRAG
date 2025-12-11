from app.exception.base import CustomError
from app.exception.llm_provider import (
    IngestionError,
    QueryError,
    VectorDBError,
    LLMProviderError,
)
from app.exception.collection import CollectionAlreadyExistsError

__all__ = (
    "CustomError",
    "IngestionError",
    "QueryError",
    "VectorDBError",
    "LLMProviderError",
    "CollectionAlreadyExistsError",
)
