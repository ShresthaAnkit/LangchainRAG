from app.exception.base import CustomError
from app.exception.llm_provider import IngestionError

__all__ = (
    "CustomError",
    "IngestionError",
)
