from fastapi import APIRouter, Depends
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseChatModel
from app.api.deps import get_llm_deps, get_prompt_manager_deps, get_vectorstore_deps
from app.core.prompt_manager import PromptManager
from app.service.query_service import QueryService
from app.schema.query import ChatRequest, QueryApiResponse

router = APIRouter()


@router.post("/{collection_name}/chat", response_model=QueryApiResponse)
def chat(
    request: ChatRequest,
    llm: BaseChatModel = Depends(get_llm_deps),
    vectorstore: VectorStore = Depends(get_vectorstore_deps),
    prompt_manager: PromptManager = Depends(get_prompt_manager_deps),
):
    query_service = QueryService()

    query_response = query_service.query_agentic(
        query=request.query,
        session_id=request.session_id,
        llm=llm,
        vectorstore=vectorstore,
        prompt_manager=prompt_manager,
    )

    return {"data": query_response}
