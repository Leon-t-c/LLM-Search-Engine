"""Claude structured-output implementation of the Provider protocol.

Parallel to openai_provider.py — same `complete` signature, same ProviderError
contract, so nothing above the protocol knows which vendor is in use.
"""
import json

from .provider import ProviderError

DEFAULT_MODEL = "claude-opus-5"

#: Translation is mechanical, not a reasoning problem, so the default `high`
#: effort buys nothing. Thinking itself stays on (adaptive is the default on
#: this model): disabling it can put a tool call into visible text and leak
#: internal tags, and lowering effort cuts cost without either risk.
DEFAULT_EFFORT = "medium"

#: Server-side rescue when a safety classifier declines a request. Very
#: unlikely on this workload, but it costs one parameter. This beta gates the
#: scalar `fallbacks="default"` form specifically; the older list form is
#: gated by a different, earlier beta, and crossing the two is rejected.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

#: Payloads are small; this is headroom, not a target. The SDK refuses a
#: non-streaming request whose cap implies a run longer than its ten-minute
#: ceiling, which lands at 21_334 tokens — measured, not guessed. This sits
#: well under that, and matches the SDK's own non-streaming recommendation,
#: while leaving adaptive thinking (which spends from this same budget) room
#: to work rather than being trimmed to just clear the guard.
MAX_TOKENS = 16000


class ClaudeProvider:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        client=None,
        effort: str = DEFAULT_EFFORT,
        use_fallbacks: bool = True,
    ) -> None:
        if client is None and not api_key:
            raise ProviderError("no API key configured for the Claude provider")
        self._model = model
        self._effort = effort
        self._use_fallbacks = use_fallbacks
        if client is not None:
            self._client = client
        else:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self, system: str, question: str, schema: dict, schema_name: str
    ) -> dict:
        # `schema_name` is deliberately unused: this vendor's json_schema
        # format takes no name and no strict flag — conformance is intrinsic.
        del schema_name
        request = {
            "model": self._model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": question}],
            "output_config": {
                "effort": self._effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        }
        if self._use_fallbacks:
            endpoint = self._client.beta.messages
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        else:
            endpoint = self._client.messages

        try:
            response = endpoint.create(**request)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(str(exc)) from exc

        if response.stop_reason == "refusal":
            # `stop_details` is populated only for this stop reason, and is
            # None for every other one — so it is read only inside this branch.
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise ProviderError(f"model refused: {category or 'unspecified'}")

        # Adaptive thinking is on by default, so a thinking block may lead.
        text = next(
            (b.text for b in response.content
             if getattr(b, "type", None) == "text"),
            None,
        )
        if text is None:
            raise ProviderError("no text block in the response")
        try:
            return json.loads(text)
        except (TypeError, ValueError) as exc:
            raise ProviderError("model output was not valid JSON") from exc
