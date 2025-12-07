import uuid
from fastapi import APIRouter, Depends
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from app.api.deps import get_llm_deps, get_prompt_manager_deps, get_vectorstore_deps
from app.core.prompt_manager import PromptManager
from app.service.query_service import QueryService
from app.schema.query import QueryApiResponse, SessionApiResponse

router = APIRouter()


@router.get("/get-session", response_model=SessionApiResponse)
def get_session():
    session_id = str(uuid.uuid4())
    return {"data": {"session_id": session_id}}


@router.post("/chat", response_model=QueryApiResponse)
def chat(
    session_id: str,
    query: str,
    llm: BaseChatModel = Depends(get_llm_deps),
    vectorstore: VectorStore = Depends(get_vectorstore_deps),
    prompt_manager: PromptManager = Depends(get_prompt_manager_deps),
):
    query_service = QueryService()

    query_response = query_service.query(
        query=query,
        session_id=session_id,
        llm=llm,
        vectorstore=vectorstore,
        prompt_manager=prompt_manager,
    )

    return {"data": query_response}
