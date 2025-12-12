from enum import Enum


class LLMProvider(str, Enum):
    GOOGLE = "google"
    COHERE = "cohere"
    BEDROCK = "bedrock"

class EmbeddingProvider(str, Enum):
    GOOGLE = "google"
    COHERE = "cohere"
    BEDROCK = "bedrock"