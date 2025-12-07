from pydantic import BaseModel
from app.schema.api import ApiResponse

class QueryResponse(BaseModel):
    answer: str

class QueryApiResponse(ApiResponse[QueryResponse]):
    pass
