import json
from types import SimpleNamespace

import pytest

from cert_nlq.translate.openai_provider import OpenAIProvider
from cert_nlq.translate.provider import ProviderError

SCHEMA = {"type": "object", "properties": {"root": {"type": "string"}},
          "required": ["root"], "additionalProperties": False}


#: Fixed token counts so the usage assertions are exact.
_DEFAULT_USAGE = SimpleNamespace(prompt_tokens=120, completion_tokens=8)


class StubCompletions:
    """Records the call and returns a scripted message."""

    def __init__(self, content=None, refusal=None, raises=None,
                 usage=_DEFAULT_USAGE, finish_reason="stop", choices=None):
        self._content = content
        self._refusal = refusal
        self._raises = raises
        self._usage = usage
        self._finish_reason = finish_reason
        self._choices = choices
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self._raises is not None:
            raise self._raises
        message = SimpleNamespace(content=self._content, refusal=self._refusal)
        choices = self._choices
        if choices is None:
            choices = [SimpleNamespace(
                message=message, finish_reason=self._finish_reason
            )]
        return SimpleNamespace(choices=choices, usage=self._usage)


def _client(**kwargs):
    completions = StubCompletions(**kwargs)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_complete_parses_the_json_content():
    client, _ = _client(content=json.dumps({"root": "widget"}))
    provider = OpenAIProvider("key", "test-model", client=client)
    assert provider.complete("sys", "q", SCHEMA, "Route") == {"root": "widget"}


def test_complete_requests_a_strict_structured_output():
    client, completions = _client(content=json.dumps({"root": "widget"}))
    OpenAIProvider("key", "test-model", client=client).complete(
        "sys", "q", SCHEMA, "Route"
    )
    sent = completions.kwargs
    fmt = sent["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "Route"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"] == SCHEMA


def test_the_stable_prefix_comes_first():
    """Automatic prompt caching only helps if the prefix is stable."""
    client, completions = _client(content=json.dumps({"root": "widget"}))
    OpenAIProvider("key", "m", client=client).complete("sys", "q", SCHEMA, "R")
    assert [m["role"] for m in completions.kwargs["messages"]] == ["system", "user"]
    assert completions.kwargs["messages"][0]["content"] == "sys"


def test_output_is_capped():
    """Payloads are a few hundred tokens; output is the expensive side."""
    from cert_nlq.translate.openai_provider import (
        MAX_OUTPUT_TOKENS,
        OUTPUT_TOKEN_PARAM,
    )

    client, completions = _client(content=json.dumps({"root": "widget"}))
    OpenAIProvider("key", "m", client=client).complete("sys", "q", SCHEMA, "R")
    assert completions.kwargs[OUTPUT_TOKEN_PARAM] == MAX_OUTPUT_TOKENS == 8000


def test_stage_one_uses_the_cheaper_router_model():
    from cert_nlq.translate.router import ROUTE_SCHEMA_NAME

    client, completions = _client(content=json.dumps({"root": "widget"}))
    provider = OpenAIProvider(
        "key", "big-model", client=client, router_model="cheap-model"
    )
    provider.complete("sys", "q", SCHEMA, ROUTE_SCHEMA_NAME)
    assert completions.kwargs["model"] == "cheap-model"


def test_stage_two_uses_the_translator_model():
    client, completions = _client(content=json.dumps({"root": "widget"}))
    provider = OpenAIProvider(
        "key", "big-model", client=client, router_model="cheap-model"
    )
    provider.complete("sys", "q", SCHEMA, "widgetPayload")
    assert completions.kwargs["model"] == "big-model"


def test_without_a_router_model_both_stages_use_one_model():
    from cert_nlq.translate.router import ROUTE_SCHEMA_NAME

    client, completions = _client(content=json.dumps({"root": "widget"}))
    provider = OpenAIProvider("key", "only-model", client=client)
    provider.complete("sys", "q", SCHEMA, ROUTE_SCHEMA_NAME)
    assert completions.kwargs["model"] == "only-model"


def test_usage_is_reported_to_the_callback():
    """Token accounting is the cost control and the phase-2 baseline."""
    seen = []
    client, _ = _client(content=json.dumps({"root": "widget"}))
    provider = OpenAIProvider("key", "m", client=client, on_usage=seen.append)
    provider.complete("sys", "q", SCHEMA, "Route")
    assert seen == [{"schema_name": "Route", "model": "m",
                     "prompt_tokens": 120, "completion_tokens": 8,
                     "cached_tokens": None, "reasoning_tokens": None}]


def test_a_missing_usage_field_does_not_break_the_call():
    client, _ = _client(content=json.dumps({"root": "widget"}), usage=None)
    seen = []
    provider = OpenAIProvider("key", "m", client=client, on_usage=seen.append)
    assert provider.complete("sys", "q", SCHEMA, "R") == {"root": "widget"}
    assert seen == []


def test_a_refusal_raises_provider_error():
    client, _ = _client(content=None, refusal="I can't help with that")
    with pytest.raises(ProviderError, match="refused"):
        OpenAIProvider("key", "m", client=client).complete("sys", "q", SCHEMA, "R")


def test_unparseable_content_raises_provider_error():
    client, _ = _client(content="not json at all")
    with pytest.raises(ProviderError, match="not valid JSON"):
        OpenAIProvider("key", "m", client=client).complete("sys", "q", SCHEMA, "R")


def test_an_upstream_exception_becomes_a_provider_error():
    client, _ = _client(raises=RuntimeError("connection reset"))
    with pytest.raises(ProviderError, match="connection reset"):
        OpenAIProvider("key", "m", client=client).complete("sys", "q", SCHEMA, "R")


def test_a_missing_api_key_fails_at_construction():
    with pytest.raises(ProviderError, match="no API key"):
        OpenAIProvider("", "m")


def test_an_empty_choices_list_stays_inside_the_protocol():
    """Nothing but ProviderError may leave this module."""
    client, _ = _client(choices=[])
    with pytest.raises(ProviderError, match="no choices"):
        OpenAIProvider("key", "m", client=client).complete("sys", "q", SCHEMA, "R")


def test_a_malformed_choice_stays_inside_the_protocol():
    client, _ = _client(choices=[SimpleNamespace()])
    with pytest.raises(ProviderError, match="no choices"):
        OpenAIProvider("key", "m", client=client).complete("sys", "q", SCHEMA, "R")


def test_a_raising_usage_callback_does_not_fail_the_call():
    """The callback is a log handler in production; a broken one is not a 500."""
    def explode(record):
        raise RuntimeError("the log handler is misconfigured")

    client, _ = _client(content=json.dumps({"root": "widget"}))
    provider = OpenAIProvider("key", "m", client=client, on_usage=explode)
    assert provider.complete("sys", "q", SCHEMA, "R") == {"root": "widget"}


def test_a_truncated_response_names_the_cap_not_the_json():
    """Reporting a cut-off response as bad JSON sends the reader elsewhere."""
    from cert_nlq.translate.openai_provider import (
        MAX_OUTPUT_TOKENS,
        OUTPUT_TOKEN_PARAM,
    )

    client, _ = _client(content='{"root": "wid', finish_reason="length")
    with pytest.raises(ProviderError) as caught:
        OpenAIProvider("key", "m", client=client).complete("sys", "q", SCHEMA, "R")
    message = str(caught.value)
    assert str(MAX_OUTPUT_TOKENS) in message
    assert OUTPUT_TOKEN_PARAM in message
    assert "JSON" not in message


def test_cached_and_reasoning_counts_reach_the_record():
    """`prompt_tokens` includes cached tokens, which bill differently."""
    usage = SimpleNamespace(
        prompt_tokens=120,
        completion_tokens=8,
        prompt_tokens_details=SimpleNamespace(cached_tokens=64),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
    )
    seen = []
    client, _ = _client(content=json.dumps({"root": "widget"}), usage=usage)
    OpenAIProvider("key", "m", client=client, on_usage=seen.append).complete(
        "sys", "q", SCHEMA, "R"
    )
    assert seen[0]["cached_tokens"] == 64
    assert seen[0]["reasoning_tokens"] == 5


def test_absent_detail_blocks_do_not_break_the_call():
    """Both detail objects are optional on the response."""
    usage = SimpleNamespace(
        prompt_tokens=120, completion_tokens=8,
        prompt_tokens_details=None, completion_tokens_details=None,
    )
    seen = []
    client, _ = _client(content=json.dumps({"root": "widget"}), usage=usage)
    provider = OpenAIProvider("key", "m", client=client, on_usage=seen.append)
    assert provider.complete("sys", "q", SCHEMA, "R") == {"root": "widget"}
    assert seen[0]["cached_tokens"] is None
    assert seen[0]["reasoning_tokens"] is None
