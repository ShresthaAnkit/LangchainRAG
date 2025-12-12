from pydantic import BaseModel
from app.schema.api import ApiResponse


class CollectionsList(BaseModel):
    collections: list


class ListCollectionResponse(ApiResponse[CollectionsList]):
    pass
