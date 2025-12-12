from pydantic import BaseModel, Field
from app.schema.api import ApiResponse


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]


class QueryApiResponse(ApiResponse[QueryResponse]):
    pass


class SessionIDResponse(BaseModel):
    session_id: str


class SessionApiResponse(ApiResponse[SessionIDResponse]):
    pass


class ChatRequest(BaseModel):
    session_id: str
    query: str


class RAGResponse(BaseModel):
    """Response validation object."""

    answer: str = Field(
        description="The answer to the user's question based on the context."
    )
    found_answer: bool = Field(
        description="True if the provided context contains the information to answer the question, False if the context is irrelevant or missing the answer."
    )
