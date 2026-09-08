import json
from types import SimpleNamespace

import pytest

from cert_nlq.translate.claude_provider import (
    DEFAULT_MODEL,
    FALLBACK_BETA,
    ClaudeProvider,
)
from cert_nlq.translate.provider import ProviderError

SCHEMA = {"type": "object", "properties": {"root": {"type": "string"}},
          "required": ["root"], "additionalProperties": False}


def _text(payload):
    return SimpleNamespace(type="text", text=json.dumps(payload))


def _thinking():
    """Opus 5 has adaptive thinking on by default, so this block can lead."""
    return SimpleNamespace(type="thinking", thinking="considering the question")


#: Fixed counts so the usage assertions are exact.
_DEFAULT_USAGE = SimpleNamespace(input_tokens=120, output_tokens=8)


class StubMessages:
    def __init__(self, content=(), stop_reason="end_turn",
                 stop_details=None, raises=None, usage=_DEFAULT_USAGE):
        self._content = list(content)
        self._stop_reason = stop_reason
        self._stop_details = stop_details
        self._raises = raises
        self._usage = usage
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self._raises is not None:
            raise self._raises
        return SimpleNamespace(
            content=self._content,
            stop_reason=self._stop_reason,
            stop_details=self._stop_details,
            usage=self._usage,
        )


def _client(**kwargs):
    """A stub exposing both the stable and beta message endpoints."""
    stable = StubMessages(**kwargs)
    beta = StubMessages(**kwargs)
    client = SimpleNamespace(
        messages=stable,
        beta=SimpleNamespace(messages=beta),
    )
    return client, stable, beta


def test_complete_parses_the_first_text_block():
    client, _, _ = _client(content=[_text({"root": "widget"})])
    provider = ClaudeProvider("key", client=client)
    assert provider.complete("sys", "q", SCHEMA, "Route") == {"root": "widget"}


def test_thinking_blocks_are_skipped():
    """A leading thinking block must not be mistaken for the payload."""
    client, _, _ = _client(content=[_thinking(), _text({"root": "widget"})])
    provider = ClaudeProvider("key", client=client)
    assert provider.complete("sys", "q", SCHEMA, "Route") == {"root": "widget"}


def test_the_schema_is_sent_as_a_json_schema_output_format():
    client, _, beta = _client(content=[_text({"root": "widget"})])
    ClaudeProvider("key", client=client).complete("sys", "q", SCHEMA, "Route")
    fmt = beta.kwargs["output_config"]["format"]
    assert fmt == {"type": "json_schema", "schema": SCHEMA}


def test_the_system_prompt_is_top_level_and_the_question_is_the_user_turn():
    client, _, beta = _client(content=[_text({"root": "widget"})])
    ClaudeProvider("key", client=client).complete("be terse", "how many", SCHEMA, "R")
    assert beta.kwargs["system"] == "be terse"
    assert beta.kwargs["messages"] == [{"role": "user", "content": "how many"}]


def test_effort_is_set_and_defaults_below_high():
    """Translation is mechanical; the default `high` buys nothing here."""
    client, _, beta = _client(content=[_text({"root": "widget"})])
    ClaudeProvider("key", client=client).complete("sys", "q", SCHEMA, "R")
    assert beta.kwargs["output_config"]["effort"] == "medium"


def test_the_default_model_is_opus_5():
    client, _, beta = _client(content=[_text({"root": "widget"})])
    ClaudeProvider("key", client=client).complete("sys", "q", SCHEMA, "R")
    assert beta.kwargs["model"] == DEFAULT_MODEL == "claude-opus-5"


def test_fallbacks_use_the_beta_endpoint():
    client, stable, beta = _client(content=[_text({"root": "widget"})])
    ClaudeProvider("key", client=client, use_fallbacks=True).complete(
        "sys", "q", SCHEMA, "R"
    )
    assert stable.kwargs is None, "the stable endpoint must not be used"
    assert beta.kwargs["betas"] == [FALLBACK_BETA]
    assert beta.kwargs["fallbacks"] == "default"


def test_disabling_fallbacks_uses_the_stable_endpoint():
    """Escape hatch if the beta path rejects output_config."""
    client, stable, beta = _client(content=[_text({"root": "widget"})])
    ClaudeProvider("key", client=client, use_fallbacks=False).complete(
        "sys", "q", SCHEMA, "R"
    )
    assert beta.kwargs is None
    assert "betas" not in stable.kwargs
    assert "fallbacks" not in stable.kwargs


def test_a_refusal_raises_provider_error():
    client, _, _ = _client(
        content=[],
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber", explanation="no"),
    )
    with pytest.raises(ProviderError, match="refused"):
        ClaudeProvider("key", client=client).complete("sys", "q", SCHEMA, "R")


def test_a_response_with_no_text_block_raises_provider_error():
    client, _, _ = _client(content=[_thinking()])
    with pytest.raises(ProviderError, match="no text block"):
        ClaudeProvider("key", client=client).complete("sys", "q", SCHEMA, "R")


def test_unparseable_content_raises_provider_error():
    client, _, _ = _client(
        content=[SimpleNamespace(type="text", text="not json at all")]
    )
    with pytest.raises(ProviderError, match="not valid JSON"):
        ClaudeProvider("key", client=client).complete("sys", "q", SCHEMA, "R")


def test_an_upstream_exception_becomes_a_provider_error():
    client, _, _ = _client(raises=RuntimeError("connection reset"))
    with pytest.raises(ProviderError, match="connection reset"):
        ClaudeProvider("key", client=client).complete("sys", "q", SCHEMA, "R")


def test_a_missing_api_key_fails_at_construction():
    with pytest.raises(ProviderError, match="no API key"):
        ClaudeProvider("")


def test_the_output_cap_reaches_the_request():
    """A required argument: without it a real client rejects the call."""
    from cert_nlq.translate.claude_provider import MAX_TOKENS

    client, _, beta = _client(content=[_text({"root": "widget"})])
    ClaudeProvider("key", client=client).complete("sys", "q", SCHEMA, "R")
    assert beta.kwargs["max_tokens"] == MAX_TOKENS


def test_caching_is_requested_explicitly():
    """This vendor's caching is opt-in, so the stable prefix buys nothing
    unless it is asked for — and the system prompt is the bulk of the call."""
    client, _, beta = _client(content=[_text({"root": "widget"})])
    ClaudeProvider("key", client=client).complete("sys", "q", SCHEMA, "R")
    assert beta.kwargs["cache_control"] == {"type": "ephemeral"}


def test_caching_is_requested_on_the_stable_endpoint_too():
    client, stable, _ = _client(content=[_text({"root": "widget"})])
    ClaudeProvider("key", client=client, use_fallbacks=False).complete(
        "sys", "q", SCHEMA, "R"
    )
    assert stable.kwargs["cache_control"] == {"type": "ephemeral"}


def test_usage_is_reported_to_the_callback():
    """The two adapters are compared on cost, so both must account."""
    seen = []
    client, _, _ = _client(content=[_text({"root": "widget"})])
    ClaudeProvider("key", client=client, on_usage=seen.append).complete(
        "sys", "q", SCHEMA, "Route"
    )
    assert seen == [{
        "schema_name": "Route", "model": DEFAULT_MODEL,
        "prompt_tokens": 120, "completion_tokens": 8,
        "cached_tokens": None, "cache_creation_tokens": None,
        "reasoning_tokens": None,
    }]


def test_the_cache_and_reasoning_counts_reach_the_record():
    usage = SimpleNamespace(
        input_tokens=12,
        output_tokens=8,
        cache_read_input_tokens=900,
        cache_creation_input_tokens=64,
        output_tokens_details=SimpleNamespace(thinking_tokens=5),
    )
    seen = []
    client, _, _ = _client(content=[_text({"root": "widget"})], usage=usage)
    ClaudeProvider("key", client=client, on_usage=seen.append).complete(
        "sys", "q", SCHEMA, "R"
    )
    assert seen[0]["cached_tokens"] == 900
    assert seen[0]["cache_creation_tokens"] == 64
    assert seen[0]["reasoning_tokens"] == 5


def test_a_missing_usage_object_does_not_break_the_call():
    seen = []
    client, _, _ = _client(content=[_text({"root": "widget"})], usage=None)
    provider = ClaudeProvider("key", client=client, on_usage=seen.append)
    assert provider.complete("sys", "q", SCHEMA, "R") == {"root": "widget"}
    assert seen == []


def test_a_raising_usage_callback_does_not_fail_the_call():
    def explode(record):
        raise RuntimeError("the log handler is misconfigured")

    client, _, _ = _client(content=[_text({"root": "widget"})])
    provider = ClaudeProvider("key", client=client, on_usage=explode)
    assert provider.complete("sys", "q", SCHEMA, "R") == {"root": "widget"}


def test_a_truncated_response_names_the_cap_not_the_json():
    """Reporting a cut-off response as bad JSON sends the reader elsewhere."""
    from cert_nlq.translate.claude_provider import MAX_TOKENS

    client, _, _ = _client(
        content=[SimpleNamespace(type="text", text='{"root": "wid')],
        stop_reason="max_tokens",
    )
    with pytest.raises(ProviderError) as caught:
        ClaudeProvider("key", client=client).complete("sys", "q", SCHEMA, "R")
    message = str(caught.value)
    assert str(MAX_TOKENS) in message
    assert "max_tokens" in message
    assert "JSON" not in message
