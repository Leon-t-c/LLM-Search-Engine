"""FastAPI application.

Dependencies are passed to create_app explicitly rather than imported, so
tests construct an app that cannot reach the network.
"""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ..config import get_settings
from ..ir.schema import ScopeError
from ..registry.client import RegistryClient, RegistryUnavailable
from ..translate.provider import Provider, ProviderError
from ..translate.translator import translate

logger = logging.getLogger(__name__)


class TranslateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # "The current year as the caller sees it." Unbounded, an out-of-range
    # value (negative, or absurdly large) does not fail: it threads through
    # relative-year resolution and comes back as a resolved value in an
    # otherwise-ok payload, silently, with no error and no clarification. A
    # clock could plausibly produce a value in this window; nothing else
    # should reach here.
    now_year: int | None = Field(default=None, ge=2000, le=2100)

    @field_validator("question")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


def create_app(registry_client, provider: Provider) -> FastAPI:
    app = FastAPI(title="cert-nlq", version="0.1.0")

    @app.get("/healthz")
    def healthz():
        cached = getattr(registry_client, "cached", None)
        if cached is None:
            # `cached` is None until the first successful registry fetch.
            # An instance that has never reached the registry is not ready:
            # every call it serves is already guaranteed to fail, so the
            # status line itself must say so, not just the body — otherwise
            # a load balancer reads 200 and keeps routing live traffic here.
            return JSONResponse(
                status_code=503,
                content={"status": "degraded", "registry_version": None},
            )
        return {"status": "ok", "registry_version": cached.version}

    @app.post("/translate")
    def translate_endpoint(request: TranslateRequest) -> dict:
        try:
            registry = registry_client.fetch()
        except RegistryUnavailable as exc:
            # The exception text can carry field paths and values straight
            # out of the host's registry document (a validation error
            # stringifies with that detail). This service is built to hold
            # no schema knowledge, so that text is logged, never returned.
            logger.exception("registry unavailable")
            raise HTTPException(
                status_code=503,
                detail="The query service is temporarily unavailable.",
            ) from exc
        try:
            response = translate(
                request.question, registry, provider, now_year=request.now_year
            )
        except ProviderError as exc:
            logger.exception("translation provider failed")
            raise HTTPException(
                status_code=502,
                detail="The translation provider is temporarily unavailable.",
            ) from exc
        except ScopeError as exc:
            # A root that offers no fields to translate against is a
            # registry problem, not a bug in this service — tell the two
            # apart rather than letting this surface as an undifferentiated
            # 500. Deliberately not a blanket `except Exception`: a real bug
            # here should still 500 rather than claim to be retryable.
            logger.exception("registry-shaped scope failure")
            raise HTTPException(
                status_code=503,
                detail="The query service is temporarily unavailable.",
            ) from exc
        # `by_alias=True` is REQUIRED, not cosmetic. `Aggregate.alias` is
        # serialised as `as` (a Python keyword), and without the flag this
        # emits `{"alias": ...}` — a payload the host's compiler does not
        # understand. This has regressed silently before: every other test
        # here reads the parsed model rather than the emitted JSON, so only
        # a test against the wire format itself catches it.
        return response.model_dump(mode="json", by_alias=True)

    return app


def _build_provider(settings) -> Provider:
    """Construct the configured provider.

    Imports sit inside each branch so that an adapter that is broken, or whose
    SDK is not installed, cannot break the other one — and so neither SDK is
    imported unless it is actually in use.
    """
    if settings.provider == "claude":
        from ..translate.claude_provider import ClaudeProvider

        return ClaudeProvider(settings.anthropic_api_key, settings.claude_model)
    if settings.provider == "openai":
        from ..translate.openai_provider import OpenAIProvider

        # Every call leaves one `usage {...}` line carrying the stage, the
        # model and both token counts. That log is the running cost record —
        # the thing that is miserable to reconstruct after the fact.
        usage_log = logging.getLogger("cert_nlq.usage")
        return OpenAIProvider(
            settings.openai_api_key,
            settings.model,
            router_model=settings.router_model or None,
            on_usage=lambda record: usage_log.info("usage %s", record),
        )
    raise ProviderError(f"unknown provider {settings.provider!r}")


def make_app() -> FastAPI:
    """The production app, built from settings.

    A factory, deliberately not a module-level `app = ...`. Importing this
    module must stay free of side effects: a module-level call would
    construct a real, billable client on every import of this module,
    including every test run — every setting defaults to `""`, so it would
    not even fail outright on a missing key; it would just start spending
    money. Run it with uvicorn's factory flag:

        uvicorn cert_nlq.api.app:make_app --factory
    """
    settings = get_settings()
    return create_app(
        RegistryClient(settings.registry_url, settings.registry_token),
        _build_provider(settings),
    )
