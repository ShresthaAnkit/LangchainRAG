import boto3
from langchain_aws import ChatBedrock
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from app.schema.llm import LLMProvider
from app.exception import LLMProviderError
from app.core.logging_config import get_logger
from app.core.config import settings

logger = get_logger(__name__)


def get_llm(provider: LLMProvider, model_name: str = "gemini-2.5-flash"):
    try:
        if provider == LLMProvider.GOOGLE:
            return ChatGoogleGenerativeAI(model=model_name)
        elif provider == LLMProvider.COHERE:
            return ChatCohere(model=model_name)
        elif provider == LLMProvider.BEDROCK:
            model_kwargs = {
                "max_tokens": 2048,
                "temperature": 0.1,
                "top_p": 0.9,
            }
            bedrock_runtime = boto3.client(
                service_name="bedrock-runtime",
                region_name=settings.REGION_NAME,
            )
            return ChatBedrock(
                client=bedrock_runtime,
                model_id=model_name,
                model_kwargs=model_kwargs,
            )
        else:
            raise LLMProviderError(f"LLM provider {provider} not supported")
    except Exception as e:
        logger.exception(f"Error initializing LLM provider: {provider}")
        raise LLMProviderError(f"Failed to initialize LLM provider {provider}") from e
