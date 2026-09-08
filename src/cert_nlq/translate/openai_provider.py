"""OpenAI structured-output implementation of the Provider protocol.

The demo provider. The Claude adapter (claude_provider.py) is the production
target, so this module is written to be cheap and replaceable rather than
optimal.

Cost measures live here: a cheaper model for stage 1, a hard output cap, a
stable message prefix so automatic prompt caching can hit, and usage
accounting on every call.
"""
import json

from .provider import ProviderError
from .router import ROUTE_SCHEMA_NAME

#: Payloads are a few hundred tokens. This is headroom, not a target.
MAX_OUTPUT_TOKENS = 1500

#: The request key for the output cap. Current chat models take
#: `max_completion_tokens`; older ones take `max_tokens`. If a live call
#: returns an unknown-parameter error, switch this one constant.
OUTPUT_TOKEN_PARAM = "max_completion_tokens"


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
        self._model = model
        self._router_model = router_model
        self._max_output_tokens = max_output_tokens
        self._on_usage = on_usage
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)

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
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
            OUTPUT_TOKEN_PARAM: self._max_output_tokens,
        }
        try:
            completion = self._client.chat.completions.create(**request)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(str(exc)) from exc

        self._report_usage(completion, model, schema_name)

        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise ProviderError(f"model refused: {message.refusal}")
        try:
            return json.loads(message.content)
        except (TypeError, ValueError) as exc:
            raise ProviderError("model output was not valid JSON") from exc

    def _report_usage(self, completion, model: str, schema_name: str) -> None:
        """Hand token counts to the caller. Never fail the call over this."""
        if self._on_usage is None:
            return
        usage = getattr(completion, "usage", None)
        if usage is None:
            return
        self._on_usage({
            "schema_name": schema_name,
            "model": model,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        })
