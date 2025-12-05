from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file="app/.env",
        env_file_encoding="utf-8"
    )

    LOG_LEVEL: str = "INFO"

    COHERE_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""


settings = Settings()
