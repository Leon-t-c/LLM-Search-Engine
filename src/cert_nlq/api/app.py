"""FastAPI application.

Dependencies are passed to create_app explicitly rather than imported, so
tests construct an app that cannot reach the network.
"""
import logging
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ..config import get_settings
from ..ir.schema import ScopeError
from ..registry.client import RegistryClient, RegistryUnavailable
from ..translate.provider import Provider, ProviderError
from ..translate.translator import translate

logger = logging.getLogger(__name__)

#: Every call leaves one `usage {...}` line carrying the stage, the model and
#: the token counts. That log is the running cost record — the thing that is
#: miserable to reconstruct after the fact. Both adapters feed it, and both
#: emit the same keys, so one line prices either of them.
usage_log = logging.getLogger("cert_nlq.usage")


class TranslateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # "The current year as the caller sees it." Bounded here because nothing
    # downstream would object: an absurd year threads through relative-year
    # resolution and comes back as a resolved value in an otherwise-ok
    # payload, with no error and no clarification to show for it. The window
    # is what a real clock could plausibly report; anything outside it is a
    # caller bug, and it should fail here where the message can say so.
    now_year: int | None = Field(default=None, ge=2000, le=2100)

    @field_validator("question")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped


def _require_token(expected: str):
    """Build the auth dependency for one configured token.

    Comparison is constant-time: a plain `==` on a secret leaks its prefix
    through timing, and this token guards an endpoint that spends money.

    A blank `expected` fails closed with a 503 rather than waving requests
    through. An unconfigured deploy should refuse to serve, not quietly serve
    everyone — that is the difference between an outage you notice and an open
    endpoint you do not.
    """

    def dependency(authorization: str = Header(default="")) -> None:
        if not expected:
            raise HTTPException(
                status_code=503, detail="The query service is not configured."
            )
        scheme, _, presented = authorization.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            presented, expected
        ):
            raise HTTPException(status_code=401, detail="Not authorized.")

    return dependency


def create_app(
    registry_client, provider: Provider, service_token: str = ""
) -> FastAPI:
    app = FastAPI(title="cert-nlq", version="0.1.0")
    guard = _require_token(service_token)

    @app.get("/healthz")
    def healthz():
        cached = getattr(registry_client, "cached", None)
        if cached is None:
            # `cached` is None until the first successful registry fetch, and
            # the only other caller of `fetch` sits behind this very gate — so
            # a probe that merely reported the cache would leave a cold process
            # unready forever: a load balancer honouring it would never send
            # the request that would warm it. Warming it here instead of in a
            # startup hook keeps a registry outage from stopping the process
            # booting at all, which is worse: a process that boots unready
            # recovers on its own.
            try:
                cached = registry_client.fetch()
            except RegistryUnavailable:
                # Not a blanket `except`: the client folds every reachability
                # and parse failure into this one type, and a real bug here
                # should still surface rather than read as a clean 503.
                logger.warning("registry still unreachable; reporting unready")
                cached = None
        if cached is None:
            # An instance that has never reached the registry is not ready:
            # every call it serves is already guaranteed to fail, so the
            # status line itself must say so, not just the body — otherwise
            # a load balancer reads 200 and keeps routing live traffic here.
            return JSONResponse(
                status_code=503,
                content={"status": "degraded", "registry_version": None},
            )
        return {"status": "ok", "registry_version": cached.version}

    @app.post("/translate", dependencies=[Depends(guard)])
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


def _log_usage(record: dict) -> None:
    usage_log.info("usage %s", record)


def _build_provider(settings) -> Provider:
    """Construct the configured provider.

    Imports sit inside each branch so that an adapter that is broken, or whose
    SDK is not installed, cannot break the other one — and so neither SDK is
    imported unless it is actually in use.
    """
    if settings.provider == "claude":
        from ..translate.claude_provider import ClaudeProvider

        return ClaudeProvider(
            settings.anthropic_api_key,
            settings.claude_model,
            use_fallbacks=settings.claude_fallbacks,
            on_usage=_log_usage,
        )
    if settings.provider == "openai":
        from ..translate.openai_provider import OpenAIProvider

        return OpenAIProvider(
            settings.openai_api_key,
            settings.model,
            router_model=settings.router_model or None,
            on_usage=_log_usage,
        )
    # Not a ProviderError: no provider failed, none was even constructed.
    # The settings type closes this branch off before it can be reached,
    # which is the point — it is here for the case that type is widened.
    raise ValueError(f"unknown provider {settings.provider!r}")


def _configure_usage_log(path: str = "") -> None:
    """Give the usage logger somewhere to write, because nothing else does.

    `usage_log` records at INFO and had no handler. Uvicorn configures only
    its own `uvicorn.*` loggers, so this one propagated to a bare root and
    Python's last-resort handler dropped everything under WARNING. Every
    usage record ever computed was handed to a logger and discarded --
    while the comment above `usage_log` claimed each call left a line, and
    that the log was the running cost record. It was inert for the whole of
    phase 1, and the first real call's token counts went with it.

    Configured here in the factory rather than at import, so that importing
    this module still has no side effects, and `create_app` -- which the
    tests drive with a fake provider -- gains no output.
    """
    if usage_log.handlers:
        return
    fmt = logging.Formatter("%(asctime)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    usage_log.addHandler(stream)
    if path:
        # Appended, never truncated: this file is an accumulating record,
        # and a restart that wiped it would take the history with it.
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        usage_log.addHandler(file_handler)
    usage_log.setLevel(logging.INFO)
    # The root logger is someone else's to configure, and a second handler
    # there would print every record twice.
    usage_log.propagate = False


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
    _configure_usage_log(settings.usage_log_path)
    return create_app(
        RegistryClient(settings.registry_url, settings.registry_token),
        _build_provider(settings),
        settings.service_token,
    )
