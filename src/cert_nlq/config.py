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

    #: The deployed `/translate` service the eval harness replays the golden
    #: set against. A separate host from `registry_url` on purpose -- the
    #: harness is an HTTP *client* of this service, never the service itself,
    #: even when both happen to run on the same box during development.
    eval_service_url: str = ""
    #: Bearer token for `eval_service_url`'s own `service_token` gate --
    #: distinct from `service_token` above (that one is *this* process's
    #: inbound gate, when this process is serving `/translate`; this one is
    #: the credential the harness presents as a caller of some other
    #: instance).
    eval_service_token: str = ""
    #: The golden set the harness replays. Relative to the working directory
    #: the CLI is run from -- `eval_data/` is gitignored except the golden
    #: file itself (see the repo's `.gitignore` comment).
    eval_golden_path: str = "eval_data/golden.jsonl"
    #: Where recorded runs and rows are stored. sqlite; `open_store` builds
    #: the URL.
    eval_db_path: str = "eval_data/evals.db"
    #: How many `/translate` calls the runner holds in flight at once. Not a
    #: knob for the provider's own rate limit alone -- it is also part of a
    #: run's identity (`store.RunRecord.concurrency`), because the report's
    #: p95 latency depends on it (spec §11: fan-out latency is not user
    #: latency).
    eval_concurrency: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
