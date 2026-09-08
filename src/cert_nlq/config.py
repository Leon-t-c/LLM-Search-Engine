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
    #: Cheaper model for stage-1 routing. Falls back to `model` when blank.
    router_model: str = ""

    #: Which translator backend to build: "openai" | "claude".
    provider: str = "openai"
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-5"


@lru_cache
def get_settings() -> Settings:
    return Settings()
