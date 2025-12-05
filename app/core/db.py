from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

_vectordb = None


def get_chromadb():
    global _vectordb
    if not isinstance(_vectordb, Chroma):
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
        )
        _vectordb = Chroma(
            persist_directory="./vectorstore", embedding_function=embedding_function
        )

    return _vectordb
