from app.exception import CustomError


class IngestionError(CustomError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class QueryError(CustomError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class VectorDBError(CustomError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

class LLMProviderError(CustomError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)