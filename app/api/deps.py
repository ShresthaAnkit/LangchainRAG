from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def get_vectorstore():
    embedding_function = GoogleGenerativeAIEmbeddings(
        model = 'gemini-embedding-001',
    )
    vectorstore = Chroma(persist_directory='./vectorstore',
                         embedding_function=embedding_function)
    
    return vectorstore