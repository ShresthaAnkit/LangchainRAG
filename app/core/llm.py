from langchain_google_genai import ChatGoogleGenerativeAI
from app.schema.llm import LLMProvider


def get_llm(provider: LLMProvider, model_name: str):
    if provider == LLMProvider.GOOGLE:
        return ChatGoogleGenerativeAI(model=model_name, max_tokens=1000)
    else:
        raise NotImplementedError()
