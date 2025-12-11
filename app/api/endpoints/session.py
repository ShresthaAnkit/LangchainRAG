import uuid
from fastapi import APIRouter
from app.schema.query import SessionApiResponse

router = APIRouter()


@router.get("/", response_model=SessionApiResponse)
def get_session():
    session_id = str(uuid.uuid4())
    return {"data": {"session_id": session_id}}
