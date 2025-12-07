# app/exception_handlers.py
from fastapi import Request, FastAPI, status
from fastapi.responses import JSONResponse
from app.exception import IngestionError, QueryError, VectorDBError, LLMProviderError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI):
    # Define a generic handler factory
    def create_handler(status_code):
        async def handler(request: Request, exc: Exception):
            if status_code >= 500:
                logger.exception(f"Server Error ({type(exc).__name__}): {str(exc)}")
            else:
                logger.warning(
                    f"Client Error {status_code} ({type(exc).__name__}): {str(exc)}"
                )
            return JSONResponse(status_code=status_code, content={"detail": str(exc)})

        return handler

    # Map Errors to Status Codes
    exceptions_map = {
        ValueError: status.HTTP_400_BAD_REQUEST,
        IngestionError: status.HTTP_400_BAD_REQUEST,
        QueryError: status.HTTP_400_BAD_REQUEST,
        LLMProviderError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        VectorDBError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    }

    # Register them in a loop
    for exc_class, status_code in exceptions_map.items():
        app.add_exception_handler(exc_class, create_handler(status_code))
