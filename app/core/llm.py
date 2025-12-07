from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from app.schema.llm import LLMProvider


def get_llm(provider: LLMProvider, model_name: str = 'gemini-2.5-flash'):
    if provider == LLMProvider.GOOGLE:
        return ChatGoogleGenerativeAI(model=model_name)
    elif provider == LLMProvider.COHERE:
        return ChatCohere(model=model_name)
    else:
        raise NotImplementedError()
