"""Claude structured-output implementation of the Provider protocol.

Parallel to openai_provider.py — same `complete` signature, same ProviderError
contract, so nothing above the protocol knows which vendor is in use.
"""
import json
import logging

from .provider import ProviderError

logger = logging.getLogger(__name__)

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

#: Caching is opt-in on this platform, where the other vendor's is automatic.
#: Without it the stable-prefix ordering costs something and buys nothing —
#: and the stage-1 system prompt is the whole registry description, which is
#: the largest avoidable cost on this path. The top-level form marks the last
#: cacheable block for us, so no per-block bookkeeping is needed here.
CACHE_CONTROL = {"type": "ephemeral"}


class ClaudeProvider:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        client=None,
        effort: str = DEFAULT_EFFORT,
        use_fallbacks: bool = True,
        on_usage=None,
    ) -> None:
        if client is None and not api_key:
            raise ProviderError("no API key configured for the Claude provider")
        self._model = model
        self._effort = effort
        self._use_fallbacks = use_fallbacks
        self._on_usage = on_usage
        if client is not None:
            self._client = client
        else:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self, system: str, question: str, schema: dict, schema_name: str
    ) -> dict:
        # `schema_name` reaches the request nowhere: this vendor's json_schema
        # format takes no name and no strict flag — conformance is intrinsic.
        # It is still carried, because the usage record is keyed by stage and
        # the name is the only stage signal the protocol offers.
        request = {
            "model": self._model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": question}],
            "output_config": {
                "effort": self._effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            "cache_control": CACHE_CONTROL,
        }
        if self._use_fallbacks:
            endpoint = self._client.beta.messages
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        else:
            endpoint = self._client.messages

        try:
            response = endpoint.create(**request)
        except Exception as exc:
            raise ProviderError(str(exc)) from exc

        self._report_usage(response, schema_name)

        if response.stop_reason == "max_tokens":
            # Checked before the text is parsed: a truncated payload is not
            # valid JSON either, and saying so sends the reader hunting for a
            # malformed model instead of a cap to raise.
            raise ProviderError(
                f"output stopped at the {MAX_TOKENS}-token cap (max_tokens); "
                "raise the cap or narrow the request"
            )
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

    def _report_usage(self, response, schema_name: str) -> None:
        """Hand token counts to the caller. Never fail the call over this.

        The keys match the other adapter's so a single log line prices either
        one. Two differences are real, not naming: `prompt_tokens` here
        excludes the cached reads rather than including them, and cache
        *creation* is billed separately and has no counterpart on the other
        side, so it gets its own key.
        """
        if self._on_usage is None:
            return
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        output_detail = getattr(usage, "output_tokens_details", None)
        record = {
            "schema_name": schema_name,
            "model": self._model,
            "prompt_tokens": getattr(usage, "input_tokens", None),
            "completion_tokens": getattr(usage, "output_tokens", None),
            "cached_tokens": getattr(usage, "cache_read_input_tokens", None),
            "cache_creation_tokens": getattr(
                usage, "cache_creation_input_tokens", None
            ),
            "reasoning_tokens": getattr(output_detail, "thinking_tokens", None),
        }
        try:
            self._on_usage(record)
        except Exception:
            # The callback is a logging call in production. A misconfigured
            # handler must not turn a translation that already succeeded
            # into a failure.
            logger.exception("usage callback failed")
