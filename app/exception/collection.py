from app.exception import CustomError


class CollectionAlreadyExistsError(CustomError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)
