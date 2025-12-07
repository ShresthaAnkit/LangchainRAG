from pydantic import BaseModel
from typing import TypeVar, Generic

T = TypeVar("T", bound = BaseModel)


class ApiError(BaseModel):
    error: str


class ApiResponse(Generic[T], BaseModel):
    success: bool
    message: str
    data: T | None = None
    errors: ApiError | None = None
