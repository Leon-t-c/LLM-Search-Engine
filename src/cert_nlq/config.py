"""Settings, from the environment only. No secret is ever committed."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CERT_NLQ_", env_file=".env", extra="ignore"
    )

    registry_url: str = ""
    registry_token: str = ""
    openai_api_key: str = ""
    model: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
