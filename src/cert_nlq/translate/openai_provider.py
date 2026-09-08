"""OpenAI structured-output implementation of the Provider protocol.

The demo provider. The Claude adapter (claude_provider.py) is the production
target, so this module is written to be cheap and replaceable rather than
optimal.

Cost measures live here: a cheaper model for stage 1, a hard output cap, a
stable message prefix so automatic prompt caching can hit, and usage
accounting on every call.
"""
import json
import logging
import re

from .provider import ProviderError, token_count
from .router import ROUTE_SCHEMA_NAME

logger = logging.getLogger(__name__)

#: Payloads are a few hundred tokens, so this is headroom rather than a
#: target — but on the reasoning-model families this task targets, the cap
#: counts reasoning tokens too, and those are not bounded by payload size. A
#: cap that a single ordinary call can hit is not a safety net, it is a coin
#: flip, so it sits far above any plausible payload. The worst case is still
#: bounded, which is all the cap was ever for; the expensive side of this
#: workload is the schema on the way in, not the tokens on the way out.
MAX_OUTPUT_TOKENS = 8000

#: The request key for the output cap. Current chat models take
#: `max_completion_tokens`; older ones take `max_tokens`. If a live call
#: returns an unknown-parameter error, switch this one constant.
OUTPUT_TOKEN_PARAM = "max_completion_tokens"

#: The vendor constrains the response-format name to this class, at most 64
#: characters. The name is derived from registry data, which nothing
#: validates on the way in, so it is sanitised rather than trusted.
_NAME_ALLOWED = re.compile(r"[^a-zA-Z0-9_-]")
_NAME_MAX = 64

#: Used when sanitising leaves nothing at all; the name is a label the vendor
#: echoes back, so any legal string will do, but it may not be empty.
_NAME_FALLBACK = "payload"


class OpenAIProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        client=None,
        router_model: str | None = None,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        on_usage=None,
    ) -> None:
        if client is None and not api_key:
            raise ProviderError("no API key configured for the OpenAI provider")
        # Checked alongside the key, and for the same reason: the setting it
        # comes from also defaults to blank, so a deploy that sets only the
        # key would otherwise build a provider that fails on every call.
        if not model:
            raise ProviderError("no model configured for the OpenAI provider")
        self._model = model
        self._router_model = router_model
        self._max_output_tokens = max_output_tokens
        self._on_usage = on_usage
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)

    @staticmethod
    def _format_name(schema_name: str) -> str:
        name = _NAME_ALLOWED.sub("", schema_name)[:_NAME_MAX]
        return name or _NAME_FALLBACK

    def _model_for(self, schema_name: str) -> str:
        """Stage 1 is a classification call and does not need the big model.

        Keyed on `schema_name` because that is the only stage signal the
        Provider protocol carries. The alternative — widening the protocol with
        a stage argument — would put a provider's cost concern into the
        interface every provider shares, so this stays local to the adapter.
        """
        if schema_name == ROUTE_SCHEMA_NAME and self._router_model:
            return self._router_model
        return self._model

    def complete(
        self, system: str, question: str, schema: dict, schema_name: str
    ) -> dict:
        model = self._model_for(schema_name)
        request = {
            "model": model,
            # System first, question last: the stable prefix is what automatic
            # prompt caching can reuse across calls.
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": self._format_name(schema_name),
                    "strict": True,
                    "schema": schema,
                },
            },
            OUTPUT_TOKEN_PARAM: self._max_output_tokens,
        }
        try:
            completion = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise ProviderError(str(exc)) from exc

        self._report_usage(completion, model, schema_name)

        # Unpacked defensively: an empty list is an IndexError and a
        # half-formed object an AttributeError, and neither is a
        # ProviderError, so either would escape this module and surface
        # above as an undifferentiated failure.
        try:
            choice = completion.choices[0]
            message = choice.message
        except (AttributeError, IndexError, TypeError) as exc:
            raise ProviderError("no choices in the response") from exc

        if getattr(message, "refusal", None):
            raise ProviderError(f"model refused: {message.refusal}")
        if getattr(choice, "finish_reason", None) == "length":
            # Checked before parsing: truncated output is not valid JSON
            # either, and reporting it that way sends the reader looking for
            # a malformed model instead of a cap to raise. Name both the
            # number and the parameter, so the message is the whole fix.
            raise ProviderError(
                f"output stopped at the {self._max_output_tokens}-token cap "
                f"({OUTPUT_TOKEN_PARAM}); raise the cap or narrow the request"
            )
        try:
            return json.loads(message.content)
        except (TypeError, ValueError) as exc:
            raise ProviderError("model output was not valid JSON") from exc

    def _report_usage(self, completion, model: str, schema_name: str) -> None:
        """Hand token counts to the caller. Never fail the call over this.

        The keys are the shared, non-overlapping set — see USAGE_TOKEN_KEYS.
        This vendor's `prompt_tokens` is a total, and *both* cache counters
        are breakdowns of it: they arrive on a detail object the SDK
        documents as the breakdown of the prompt, and the write count is
        described as a number of prompt tokens. The other vendor reports
        counters that sit beside each other instead, so both are subtracted
        back out here. Subtracting only the reads leaves the three input keys
        summing past the total, which is the overlap this key set exists to
        rule out.
        """
        if self._on_usage is None:
            return
        usage = getattr(completion, "usage", None)
        if usage is None:
            return
        # Both detail objects are optional on the response and each field
        # inside them is too, hence reading through a possibly-None object.
        prompt_detail = getattr(usage, "prompt_tokens_details", None)
        output_detail = getattr(usage, "completion_tokens_details", None)
        cached = token_count(prompt_detail, "cached_tokens")
        cache_write = token_count(prompt_detail, "cache_write_tokens")
        record = {
            "schema_name": schema_name,
            "model": model,
            # Clamped at zero: breakdowns that ever came back larger than
            # the total they belong to would otherwise make the sum negative,
            # which is worse than a slightly wrong split.
            "uncached_input_tokens": max(
                token_count(usage, "prompt_tokens") - cached - cache_write, 0
            ),
            "cached_input_tokens": cached,
            "cache_write_tokens": cache_write,
            "output_tokens": token_count(usage, "completion_tokens"),
            "reasoning_tokens": token_count(output_detail, "reasoning_tokens"),
        }
        try:
            self._on_usage(record)
        except Exception:
            # The callback is a logging call in production. A misconfigured
            # handler must not turn a translation that already succeeded
            # into a failure.
            logger.exception("usage callback failed")
