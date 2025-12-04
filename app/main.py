from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.logging_config import get_logger
from app.api.api import api_router

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG Bot")

    yield

    logger.info("Shutting down RAG Bot")

app = FastAPI(lifespan=lifespan)

app.include_router(api_router, prefix = '/api')

@app.get("/")
def root():
    return "Welcome to RAG Bot"


@app.get("/api/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, workers=1)
