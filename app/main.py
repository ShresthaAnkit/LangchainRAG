from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.logging_config import get_logger
from app.api.api import api_router
from app.schema.api import ApiResponse
from app.exception_handler import register_exception_handlers

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG Bot")

    yield

    logger.info("Shutting down RAG Bot")


app = FastAPI(lifespan=lifespan)

app.include_router(api_router, prefix="/api")

register_exception_handlers(app)

@app.get("/")
def root():
    return "Welcome to RAG Bot"


@app.get("/api/health", response_model=ApiResponse)
def health_check():
    return {"success": True, "message": "Server is healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1)
