from enum import Enum


class LLMProvider(str, Enum):
    GOOGLE = "google"
    COHERE = "cohere"

class EmbeddingProvider(str, Enum):
    GOOGLE = "google"
    COHERE = "cohere"