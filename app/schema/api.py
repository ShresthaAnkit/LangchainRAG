from pydantic import BaseModel
from typing import TypeVar, Generic

T = TypeVar("T", bound=BaseModel)


class ApiResponse(Generic[T], BaseModel):
    success: bool = True
    message: str = "Successful"
    data: T | None = None
    error: str | None = None
