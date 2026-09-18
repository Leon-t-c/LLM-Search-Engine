"""Ollama structured-output adapter, plus the third normaliser's fixpoint.

Two unrelated things share this file because they share an owner: the
adapter (`OllamaProvider`) and the claim that the schema it sends is one
llama.cpp's `json_schema_to_grammar` can actually consume. Everything below
`# -- schema fixpoint --` makes no network call and needs no Ollama
installation; it is a static walk over `json_schema_for`'s output, in the
spirit of `test_schema_contract.py`'s vendor-normaliser fixpoint tests, but
against a documented feature *list* rather than an importable private SDK
function -- llama.cpp's grammar converter is not a Python import this
process can reach for.

If a widget-slice schema ever grows a construct outside that list, the fix
belongs in `ir/schema.py::strictify`, which is the one place both vendor
adapters and this one already funnel through -- not in this adapter, and not
in this test.
"""
import json

import httpx
import pytest

from cert_nlq.ir.schema import json_schema_for
from cert_nlq.translate.ollama_provider import OllamaProvider
from cert_nlq.translate.provider import USAGE_TOKEN_KEYS, ProviderError

SCHEMA = {"type": "object", "properties": {"root": {"type": "string"}},
          "required": ["root"], "additionalProperties": False}

#: Distinctive enough that its accidental presence in a ProviderError
#: message is unambiguous, and nothing a real deploy would use.
URL = "http://secret-host.internal:11434"


def _ok(content=None, **extra):
    """A well-formed `/api/chat` response body."""
    body = {"model": "qwen3:4b-instruct-2507",
            "message": {"role": "assistant", "content": content},
            "done": True, "prompt_eval_count": 120, "eval_count": 8}
    body.update(extra)
    return body


def _client(handle):
    return httpx.Client(transport=httpx.MockTransport(handle))


def _handler(status=200, body=None, raises=None):
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if raises is not None:
            raise raises
        if isinstance(body, (bytes, str)):
            return httpx.Response(
                status, content=body, headers={"content-type": "text/plain"}
            )
        return httpx.Response(status, json=body)

    return handle, calls


def test_complete_parses_the_json_content():
    handle, _ = _handler(body=_ok(content=json.dumps({"root": "widget"})))
    provider = OllamaProvider(URL, "m", client=_client(handle))
    assert provider.complete("sys", "q", SCHEMA, "Route") == {"root": "widget"}


def test_the_request_shape_matches_the_documented_ollama_contract():
    handle, calls = _handler(body=_ok(content=json.dumps({"root": "widget"})))
    OllamaProvider(URL, "m", client=_client(handle)).complete(
        "sys", "q", SCHEMA, "Route"
    )
    assert len(calls) == 1
    request = calls[0]
    assert request.url.path == "/api/chat"
    sent = json.loads(request.content)
    assert sent["model"] == "m"
    assert sent["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]
    assert sent["format"] == SCHEMA
    assert sent["stream"] is False
    assert sent["think"] is False
    assert sent["options"] == {"temperature": 0}


def test_a_blank_model_fails_at_construction():
    """The setting defaults to blank, like the other two adapters' model."""
    with pytest.raises(ProviderError, match="no model"):
        OllamaProvider(URL, "")


def test_no_api_key_is_required():
    """This is localhost. Construction needs only a model tag."""
    handle, _ = _handler(body=_ok(content=json.dumps({"root": "widget"})))
    provider = OllamaProvider(URL, "m", client=_client(handle))
    assert provider.complete("sys", "q", SCHEMA, "R") == {"root": "widget"}


def test_a_connection_failure_raises_provider_error_without_the_url():
    handle, _ = _handler(raises=httpx.ConnectError("connection refused"))
    with pytest.raises(ProviderError) as caught:
        OllamaProvider(URL, "m", client=_client(handle)).complete(
            "sys", "q", SCHEMA, "R"
        )
    assert URL not in str(caught.value)
    assert "secret-host" not in str(caught.value)


def test_a_non_200_status_raises_provider_error():
    handle, _ = _handler(status=500, body={"error": "internal error, details, details"})
    with pytest.raises(ProviderError, match="500"):
        OllamaProvider(URL, "m", client=_client(handle)).complete(
            "sys", "q", SCHEMA, "R"
        )


def test_a_non_200_status_does_not_leak_the_raw_body():
    handle, _ = _handler(
        status=500, body={"error": "a very specific internal detail nobody asked for"}
    )
    with pytest.raises(ProviderError) as caught:
        OllamaProvider(URL, "m", client=_client(handle)).complete(
            "sys", "q", SCHEMA, "R"
        )
    assert "very specific internal detail" not in str(caught.value)


def test_a_non_json_response_body_raises_provider_error():
    """The HTTP call succeeded but the body itself is not JSON at all."""
    handle, _ = _handler(status=200, body="not json at all")
    with pytest.raises(ProviderError, match="not JSON"):
        OllamaProvider(URL, "m", client=_client(handle)).complete(
            "sys", "q", SCHEMA, "R"
        )


def test_unparseable_message_content_raises_provider_error():
    """The body is JSON; `message.content` -- the model's own output -- is not."""
    handle, _ = _handler(body=_ok(content="not json at all"))
    with pytest.raises(ProviderError, match="not valid JSON"):
        OllamaProvider(URL, "m", client=_client(handle)).complete(
            "sys", "q", SCHEMA, "R"
        )


def test_a_missing_message_raises_provider_error():
    handle, _ = _handler(body={"model": "m", "done": True})
    with pytest.raises(ProviderError, match="no message content"):
        OllamaProvider(URL, "m", client=_client(handle)).complete(
            "sys", "q", SCHEMA, "R"
        )


def test_usage_is_reported_to_the_callback():
    seen = []
    handle, _ = _handler(
        body=_ok(content=json.dumps({"root": "widget"}),
                  prompt_eval_count=120, eval_count=8)
    )
    provider = OllamaProvider(URL, "m", client=_client(handle), on_usage=seen.append)
    provider.complete("sys", "q", SCHEMA, "Route")
    assert seen == [{
        "schema_name": "Route", "model": "qwen3:4b-instruct-2507",
        "uncached_input_tokens": 120, "cached_input_tokens": 0,
        "cache_write_tokens": 0, "output_tokens": 8, "reasoning_tokens": 0,
    }]


def test_the_record_carries_exactly_the_shared_keys():
    seen = []
    handle, _ = _handler(body=_ok(content=json.dumps({"root": "widget"})))
    OllamaProvider(URL, "m", client=_client(handle), on_usage=seen.append).complete(
        "sys", "q", SCHEMA, "R"
    )
    assert set(seen[0]) == {"schema_name", "model", *USAGE_TOKEN_KEYS}


def test_usage_falls_back_to_the_requested_model_when_absent():
    seen = []
    body = _ok(content=json.dumps({"root": "widget"}))
    del body["model"]
    handle, _ = _handler(body=body)
    OllamaProvider(URL, "requested-tag", client=_client(handle),
                    on_usage=seen.append).complete("sys", "q", SCHEMA, "R")
    assert seen[0]["model"] == "requested-tag"


def test_missing_token_counts_come_back_as_zero_not_absent():
    seen = []
    body = {"model": "m", "message": {"content": json.dumps({"root": "widget"})}}
    handle, _ = _handler(body=body)
    OllamaProvider(URL, "m", client=_client(handle), on_usage=seen.append).complete(
        "sys", "q", SCHEMA, "R"
    )
    assert seen[0]["uncached_input_tokens"] == 0
    assert seen[0]["output_tokens"] == 0


def test_a_raising_usage_callback_does_not_fail_the_call():
    """The callback is a log handler in production; a broken one is not a 500."""
    def explode(record):
        raise RuntimeError("the log handler is misconfigured")

    handle, _ = _handler(body=_ok(content=json.dumps({"root": "widget"})))
    provider = OllamaProvider(URL, "m", client=_client(handle), on_usage=explode)
    assert provider.complete("sys", "q", SCHEMA, "R") == {"root": "widget"}


def test_build_provider_returns_ollama_provider_when_settings_say_ollama():
    from cert_nlq.api.app import _build_provider
    from cert_nlq.config import Settings

    provider = _build_provider(
        Settings(_env_file=None, provider="ollama", ollama_model="qwen3:4b-instruct-2507")
    )
    assert isinstance(provider, OllamaProvider)
    assert provider._model == "qwen3:4b-instruct-2507"
    assert provider._on_usage is not None


# -- schema fixpoint --
#
# See the module docstring. What follows checks a *documented feature list*
# for llama.cpp's `json_schema_to_grammar` (what Ollama's `format` runs
# through), not a live grammar compile -- there is no such library installed
# here, on purpose; this suite stays offline and free. The list:
#
#   permitted : object + properties + required + additionalProperties,
#               enum, anyOf, array + items, $defs + $ref, bare primitive
#               `type`
#   forbidden : oneOf, allOf, prefixItems, pattern, format, const,
#               if/then/else, numeric bounds (minimum/maximum/
#               exclusiveMinimum/exclusiveMaximum/multipleOf), string-length
#               bounds (minLength/maxLength), title, description, default
#
# `const` and the two decorative keys (`title`, `default`) are already on
# strictify's removal list for a different vendor's reasons (see
# `ir/schema.py::strictify`); their absence here is the same fact observed
# from a third angle, not a new claim.
_ALLOWED_SCHEMA_KEYS = frozenset({
    "type", "properties", "required", "additionalProperties",
    "enum", "anyOf", "items", "$defs", "$ref",
})

#: Keys whose value is a *list of sibling schemas* -- the only structural
#: positions, besides `properties` and `$defs`, that this walk descends
#: into. `oneOf` and `allOf` are named here only so a schema that grew one
#: would still be walked into (and would then fail on the key check above,
#: since neither is in `_ALLOWED_SCHEMA_KEYS`) rather than silently skipped.
_BRANCH_KEYS = ("anyOf", "oneOf", "allOf", "prefixItems")


def _walk_schema_nodes(node, visit):
    """Call `visit` on every schema-shaped position in `node`.

    Mirrors `ir/schema.py::strictify`'s own traversal: it descends into
    `$defs` values, `items`, and the branch-key lists, and into `properties`
    values -- and nowhere else, because `properties` and `$defs` are maps
    whose *keys* are field/def names, not schema keywords, and walking every
    dict value indiscriminately would inspect those names as if they were
    schema nodes.
    """
    if not isinstance(node, dict):
        return
    visit(node)
    defs = node.get("$defs")
    if isinstance(defs, dict):
        for child in defs.values():
            _walk_schema_nodes(child, visit)
    items = node.get("items")
    if isinstance(items, dict):
        _walk_schema_nodes(items, visit)
    for key in _BRANCH_KEYS:
        branch = node.get(key)
        if isinstance(branch, list):
            for child in branch:
                _walk_schema_nodes(child, visit)
    properties = node.get("properties")
    if isinstance(properties, dict):
        for child in properties.values():
            _walk_schema_nodes(child, visit)


#: A handful of distinct shapes off the one fixture root: the unfiltered
#: schema, a single narrowed group, an added join, and both at once. Not
#: exhaustive of every possible slice -- the claim being pinned is about
#: which *keywords* `build_payload_model` + `strictify` ever emit, and
#: these four already exercise every branch in `build_payload_model`
#: (conditions, the three group-depth levels, aggregate, having, sort,
#: joins) at least once.
_WIDGET_SLICES = (
    ((), ()),
    (("Lifecycle",), ()),
    ((), ("shipment",)),
    (("Commercial",), ("shipment", "vendor")),
)


@pytest.mark.parametrize("groups,joins", _WIDGET_SLICES)
def test_the_generated_schema_uses_only_ollamas_supported_constructs(
    registry, groups, joins
):
    """A static pin, not a live grammar compile -- see the module docstring.

    Failure here means `json_schema_for` started emitting a keyword outside
    the documented list above. The fix belongs in `strictify`, which both
    vendor adapters already funnel through -- not in `OllamaProvider`.
    """
    schema = json_schema_for(registry.root("widget"), groups, joins)
    offending = []

    def visit(node):
        extra = set(node) - _ALLOWED_SCHEMA_KEYS
        if extra:
            offending.append((extra, node))

    _walk_schema_nodes(schema, visit)
    assert offending == []


def test_the_fixpoint_walk_would_notice_an_unsupported_keyword():
    """Proves the walk above can fail, rather than passing on everything."""
    offending = []

    def visit(node):
        extra = set(node) - _ALLOWED_SCHEMA_KEYS
        if extra:
            offending.append(extra)

    _walk_schema_nodes({"type": "string", "pattern": "^[A-Z]+$"}, visit)
    assert offending == [{"pattern"}]


def test_the_fixpoint_walk_does_not_mistake_property_names_for_keywords():
    """`properties`/`$defs` are maps of *names* -- walking their keys as
    schema keywords would flag a field literally called `pattern`."""
    offending = []

    def visit(node):
        extra = set(node) - _ALLOWED_SCHEMA_KEYS
        if extra:
            offending.append(extra)

    _walk_schema_nodes(
        {"type": "object", "properties": {"pattern": {"type": "string"}},
         "required": ["pattern"], "additionalProperties": False},
        visit,
    )
    assert offending == []
