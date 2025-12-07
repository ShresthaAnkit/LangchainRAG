from pydantic import BaseModel
from typing import TypeVar, Generic

T = TypeVar("T", bound = BaseModel)


class ApiError(BaseModel):
    error: str


class ApiResponse(Generic[T], BaseModel):
    success: bool = True
    message: str = "Successful"
    data: T | None = None
    errors: ApiError | None = None
