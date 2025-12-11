from fastapi import APIRouter, Depends
from langchain_core.vectorstores import VectorStore
from app.schema.query import ListCollectionResponse
from app.api.deps import get_vectorstore_deps
from app.schema.api import ApiResponse
from app.core.logging_config import get_logger
from app.core.db import create_collection_qdrant, list_collection_qdrant

logger = get_logger(__name__)

router = APIRouter()


@router.post("/{collection_name}", response_model=ApiResponse)
def create_collection_(collection_name: str):
    success = create_collection_qdrant(collection_name=collection_name)

    if success:
        return {"message": "Successfully created collection"}


@router.get("/", response_model=ListCollectionResponse)
def list_collections():
    try:
        return {"data": {"collections": list_collection_qdrant()}}
    except Exception as e:
        logger.error(
            f"List collection is not supported by the current vectorstore: {e}"
        )
        return {}


@router.delete("/{collection_name}", response_model=ApiResponse)
def delete_collection(
    collection_name: str, vectorstore: VectorStore = Depends(get_vectorstore_deps)
):
    try:
        deleted: bool = vectorstore.client.delete_collection(collection_name)
        if deleted:
            return {"messages": "Successfully Deleted Collection"}
        else:
            logger.error("Failed to delete collection.")
    except Exception as e:
        logger.error(
            f"Delete collection is not supported by the current vectorstore: {e}"
        )
    return {"success": False, "message": "Failed to Delete Collection"}
