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


class StubMessages:
    def __init__(self, content=(), stop_reason="end_turn",
                 stop_details=None, raises=None):
        self._content = list(content)
        self._stop_reason = stop_reason
        self._stop_details = stop_details
        self._raises = raises
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self._raises is not None:
            raise self._raises
        return SimpleNamespace(
            content=self._content,
            stop_reason=self._stop_reason,
            stop_details=self._stop_details,
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
