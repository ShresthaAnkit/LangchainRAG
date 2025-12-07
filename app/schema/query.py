from pydantic import BaseModel
from app.schema.api import ApiResponse

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

class QueryApiResponse(ApiResponse[QueryResponse]):
    pass
