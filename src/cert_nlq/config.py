"""Settings, from the environment only. No secret is ever committed."""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CERT_NLQ_", env_file=".env", extra="ignore"
    )

    registry_url: str = ""
    registry_token: str = ""
    #: Inbound service token. Blank fails closed — see the API dependency.
    service_token: str = ""
    openai_api_key: str = ""
    model: str = ""
    #: Cheaper model for stage-1 routing. Falls back to `model` when blank.
    router_model: str = ""

    #: Which translator backend to build. Closed, so a near miss — a capital
    #: letter, a trailing space — fails at boot with a settings error naming
    #: the field, rather than surviving until something deeper cannot place it.
    provider: Literal["openai", "claude"] = "openai"
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-5"
    #: Let the platform answer with a substitute model when the requested one
    #: declines. Off by default: a fallback changes which model answered, so
    #: turning it on is a deliberate availability decision for one deploy, not
    #: something a scored run should inherit without saying so.
    claude_fallbacks: bool = False

    #: Where to append the per-call usage records, in addition to stderr.
    #: Blank means stderr only. Set it: terminal scrollback is not a cost
    #: record, and these numbers are the baseline a later evaluation phase
    #: compares against -- they cannot be reconstructed from the payload,
    #: only from the provider's own billing page.
    usage_log_path: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
