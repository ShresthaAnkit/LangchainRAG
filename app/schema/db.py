from enum import Enum


class VectorDB(str, Enum):
    CHROMADB = "chromadb"
    QDRANT = "qdrant"
