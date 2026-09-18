"""Self-hosted, open-weights implementation of the Provider protocol.

Parallel to openai_provider.py and claude_provider.py — same `complete`
signature, same ProviderError contract — but the far end is a local Ollama
server rather than a vendor SDK, so this module talks raw HTTP with `httpx`
instead of a client library.

Sync, deliberately, even though the other two adapters also happen to be
sync: there is no async client here to *not* reach for. Concurrency for a
run belongs to the eval runner (`eval_concurrency`), not to this module — and
a single local GPU serialises requests anyway, so a concurrent runner gains
nothing from an async adapter and only adds a second code path to keep
correct.
"""
import json
import logging

import httpx

from .provider import ProviderError

logger = logging.getLogger(__name__)

#: The default local Ollama endpoint. Overridden by `CERT_NLQ_OLLAMA_URL`
#: for anything that is not "the server on this box" — see config.py.
DEFAULT_URL = "http://127.0.0.1:11434"

#: `/api/chat` is the structured-output-capable endpoint; `/api/generate`
#: takes a single prompt string instead of a message list and has no
#: `format` support worth relying on.
CHAT_PATH = "/api/chat"

#: Generous, not tuned: the first call on a cold server loads the model off
#: local disk into VRAM, which for a multi-gigabyte file can run well past a
#: minute before the model produces a single token. There is no SDK here and
#: therefore no retry budget to reason about alongside it (contrast
#: PROVIDER_TIMEOUT / PROVIDER_RETRIES in openai_provider.py) — this is the
#: one HTTP call the adapter makes, and it either finishes inside this window
#: or the caller should hear about it.
PROVIDER_TIMEOUT = 300.0


class OllamaProvider:
    def __init__(
        self,
        url: str,
        model: str,
        client: httpx.Client | None = None,
        on_usage=None,
    ) -> None:
        # Checked at construction, like the other two adapters: a deploy that
        # sets `CERT_NLQ_PROVIDER=ollama` but leaves the model tag blank
        # would otherwise build a provider that fails on every call instead
        # of at boot. There is no key to check alongside it — this is
        # localhost, and inventing an auth setting for a loopback service
        # would be a knob nothing reads.
        if not model:
            raise ProviderError("no model configured for the Ollama provider")
        self._url = (url or DEFAULT_URL).rstrip("/") + CHAT_PATH
        self._model = model
        self._on_usage = on_usage
        self._client = client if client is not None else httpx.Client(
            timeout=PROVIDER_TIMEOUT
        )

    def complete(
        self, system: str, question: str, schema: dict, schema_name: str
    ) -> dict:
        request = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            # The schema dict goes straight in: Ollama's structured-output
            # support takes a JSON Schema object here directly, not a
            # name/schema envelope like the OpenAI adapter builds.
            "format": schema,
            "stream": False,
            # Qwen3's -instruct variant is non-thinking already, but this is
            # sent regardless of which tag is configured: a thinking model's
            # reasoning tokens land ahead of the JSON content and fight the
            # grammar `format` imposes, so thinking is turned off at the
            # request level rather than trusted to the tag name alone.
            "think": False,
            "options": {"temperature": 0},
        }
        try:
            response = self._client.post(self._url, json=request)
        except httpx.HTTPError as exc:
            # Never the URL or the exception's own text in the message: both
            # can carry this deploy's network layout, and the message is the
            # one thing of this call that is allowed to leave the process
            # (into a caller's error response, a log a wider audience reads).
            # The detail is still worth having, so it goes to the logger
            # instead, which is this process's own.
            logger.error("Ollama request failed: %s", exc)
            raise ProviderError(
                "could not reach the local Ollama server"
            ) from exc

        if response.status_code != 200:
            logger.error(
                "Ollama returned %s: %s",
                response.status_code, response.text[:500],
            )
            raise ProviderError(
                f"Ollama returned an error status ({response.status_code})"
            )

        try:
            body = response.json()
        except ValueError as exc:
            logger.error("Ollama response was not JSON: %s", response.text[:500])
            raise ProviderError("Ollama response was not JSON") from exc

        self._report_usage(body, schema_name)

        message = body.get("message") if isinstance(body, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ProviderError("no message content in the Ollama response")
        try:
            return json.loads(content)
        except (TypeError, ValueError) as exc:
            raise ProviderError("model output was not valid JSON") from exc

    def _report_usage(self, body, schema_name: str) -> None:
        """Hand token counts to the caller. Never fail the call over this.

        The keys are the shared, non-overlapping set — see USAGE_TOKEN_KEYS
        in provider.py. This backend reports neither a cache breakdown nor a
        reasoning breakdown at all, so both stay zero rather than absent:
        every adapter always emits the full key set, and "not reported" and
        "reported as zero" are the same fact here — nothing in this stack
        distinguishes them, unlike the vendor SDKs' optional detail objects.

        `model` is read off the response body, not off the request: Ollama
        can resolve a tag to a specific quantisation, and that is the one
        that actually ran.
        """
        if self._on_usage is None:
            return
        if not isinstance(body, dict):
            return
        record = {
            "schema_name": schema_name,
            "model": body.get("model") or self._model,
            "uncached_input_tokens": _int_count(body, "prompt_eval_count"),
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "output_tokens": _int_count(body, "eval_count"),
            "reasoning_tokens": 0,
        }
        try:
            self._on_usage(record)
        except Exception:
            # The callback is a logging call in production. A misconfigured
            # handler must not turn a translation that already succeeded
            # into a failure.
            logger.exception("usage callback failed")


def _int_count(body: dict, key: str) -> int:
    """One count off the response body, or 0 when Ollama did not send it.

    Mirrors `provider.token_count`, but reads a plain JSON object rather
    than an SDK response object with attributes — `httpx.Response.json()`
    hands back a dict, so `getattr` would never find anything on it.
    """
    value = body.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
