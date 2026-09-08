"""FastAPI application.

Dependencies are passed to create_app explicitly rather than imported, so
tests construct an app that cannot reach the network.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..config import get_settings
from ..registry.client import RegistryClient, RegistryUnavailable
from ..translate.provider import Provider, ProviderError
from ..translate.translator import translate


class TranslateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    now_year: int | None = None

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
    def healthz() -> dict:
        cached = getattr(registry_client, "cached", None)
        return {
            "status": "ok",
            "registry_version": cached.version if cached is not None else None,
        }

    @app.post("/translate")
    def translate_endpoint(request: TranslateRequest) -> dict:
        try:
            registry = registry_client.fetch()
        except RegistryUnavailable as exc:
            raise HTTPException(
                status_code=503, detail=f"Registry unavailable: {exc}"
            ) from exc
        try:
            response = translate(
                request.question, registry, provider, now_year=request.now_year
            )
        except ProviderError as exc:
            raise HTTPException(
                status_code=502, detail=f"Translation provider failed: {exc}"
            ) from exc
        # `by_alias=True` is REQUIRED, not cosmetic. `Aggregate.alias` is
        # serialised as `as` (a Python keyword), and without the flag this
        # emits `{"alias": ...}` — a payload the host's compiler does not
        # understand. Corrected 2026-09-04 after Task 4 exposed it.
        return response.model_dump(mode="json", by_alias=True)

    return app


def make_app() -> FastAPI:
    """The production app, built from settings.

    A factory, deliberately not a module-level `app = ...`. Importing this
    module must stay free of side effects: a module-level call would construct
    the real provider (and fail on a missing key) every time the test suite
    imports `create_app`. Run it with uvicorn's factory flag:

        uvicorn cert_nlq.api.app:make_app --factory

    The provider import is inside the function because
    `translate.openai_provider` does not exist until Task 13. Nothing calls
    `make_app` before then.
    """
    settings = get_settings()
    from ..translate.openai_provider import OpenAIProvider

    return create_app(
        RegistryClient(settings.registry_url, settings.registry_token),
        OpenAIProvider(settings.openai_api_key, settings.model),
    )
