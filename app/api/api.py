from fastapi import APIRouter
from app.api.endpoints import ingest, query, collection, session

api_router = APIRouter()

api_router.include_router(ingest.router, prefix="/collection", tags=["Ingest"])
api_router.include_router(query.router, prefix="/collection", tags=["Query"])
api_router.include_router(collection.router, prefix="/collection", tags=["Collections"])
api_router.include_router(session.router, prefix="/session", tags=["Session"])
