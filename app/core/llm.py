from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from app.schema.llm import LLMProvider
from app.exception import LLMProviderError
from app.core.logging_config import get_logger

logger = get_logger(__name__)

def get_llm(provider: LLMProvider, model_name: str = 'gemini-2.5-flash'):
    try:
        if provider == LLMProvider.GOOGLE:
            return ChatGoogleGenerativeAI(model=model_name)
        elif provider == LLMProvider.COHERE:
            return ChatCohere(model=model_name)
        else:
            raise LLMProviderError(f"LLM provider {provider} not supported")
    except Exception as e:
        logger.exception(f"Error initializing LLM provider: {provider}")
        raise LLMProviderError(f"Failed to initialize LLM provider {provider}") from e
