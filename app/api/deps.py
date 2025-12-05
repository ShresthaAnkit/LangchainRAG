from app.core.db import get_chromadb

def get_vectorstore():
    return get_chromadb()