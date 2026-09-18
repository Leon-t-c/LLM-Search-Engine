import pytest
from fastapi.testclient import TestClient

from cert_nlq.api.app import create_app
from cert_nlq.registry.client import RegistryUnavailable
from cert_nlq.translate.provider import FakeProvider, ProviderError

ROUTE = {"root": "widget", "joins": [], "groups": ["Lifecycle"]}
GOOD = {"root": "widget", "where": {"combinator": "AND", "children": [
        {"field": "widget.status", "op": "=", "value": "Active"}]}}


class StubRegistry:
    """Stands in for RegistryClient. Same two members the app touches."""

    def __init__(self, registry, error=None, loaded=True):
        self._registry = registry
        self.error = error
        self.fetches = 0
        #: `loaded=False` represents a RegistryClient that has never
        #: completed a successful fetch, where `cached` stays `None`.
        self.cached = registry if loaded else None

    def fetch(self, force: bool = False):
        self.fetches += 1
        if self.error is not None:
            raise self.error
        # Mirrors RegistryClient: a successful fetch populates the cache.
        self.cached = self._registry
        return self._registry


#: The token these tests configure. Not a secret: it never leaves this file.
TOKEN = "s3cret"


def _client(registry, responses, error=None):
    """An app with the guard configured, and a client that satisfies it.

    The guard fails closed, so there is no such thing as an app without a
    token: every test that exercises the endpoint behind it has to present
    one. Sending the header from the client keeps that out of each test body.
    """
    return TestClient(
        create_app(
            StubRegistry(registry, error), FakeProvider(responses),
            service_token=TOKEN,
        ),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )


def test_translate_returns_ok(registry):
    response = _client(registry, [ROUTE, GOOD]).post(
        "/translate", json={"question": "active widgets", "now_year": 2026}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["payload"]["where"]["children"][0]["value"] == "A"


def test_the_aggregate_alias_goes_out_on_the_wire_as_as(registry):
    """The wire format must use `as`, not the Python attribute name.

    `Aggregate.alias` is serialised as `as` because `as` is a keyword. A
    `model_dump` without `by_alias=True` emits `{"alias": ...}`, which the
    host's compiler does not understand — and no other test would notice,
    because every other assertion reads the parsed model rather than the JSON.
    """
    agg = {"root": "widget",
           "where": {"combinator": "AND", "children": [
               {"field": "widget.status", "op": "=", "value": "Active"}]},
           "aggregate": [{"fn": "count", "field": "*", "as": "n"}]}
    response = _client(registry, [ROUTE, agg]).post(
        "/translate", json={"question": "how many active widgets", "now_year": 2026}
    )
    assert response.status_code == 200
    emitted = response.json()["payload"]["aggregate"][0]
    assert "as" in emitted, f"wire format regression: got {sorted(emitted)}"
    assert "alias" not in emitted
    assert emitted["as"] == "n"


def test_translate_returns_a_refusal_with_status_200(registry):
    """A refusal is a successful translation of an unanswerable question.

    The status code describes the HTTP call, not the answer: the service was
    asked a question and gave its considered response. A 4xx here would tell
    the caller to retry something that will never succeed.
    """
    response = _client(registry, [{"root": "nope", "joins": [], "groups": []}]).post(
        "/translate", json={"question": "what is the weather"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "refused"
    assert body["reason"] == "not_a_query"


def test_a_blank_question_is_a_422(registry):
    response = _client(registry, []).post("/translate", json={"question": "   "})
    assert response.status_code == 422


def test_a_missing_question_is_a_422(registry):
    response = _client(registry, []).post("/translate", json={})
    assert response.status_code == 422


def test_an_out_of_range_now_year_is_a_422(registry):
    response = _client(registry, []).post(
        "/translate", json={"question": "active widgets", "now_year": 99999}
    )
    assert response.status_code == 422


def test_registry_unavailable_is_a_503(registry):
    response = _client(
        registry, [], error=RegistryUnavailable("registry returned 503")
    ).post("/translate", json={"question": "active widgets"})
    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"].lower()


def test_a_registry_shaped_scope_failure_is_a_503_not_a_500(registry):
    """A root offering no fields to translate against is a registry problem.

    It must be told apart from a genuine bug in this service, which stays a
    500.
    """
    roots = tuple(
        r.model_copy(update={"fields": ()}) if r.root == "widget" else r
        for r in registry.roots
    )
    scopeless = registry.model_copy(update={"roots": roots})
    response = _client(scopeless, [ROUTE]).post(
        "/translate", json={"question": "active widgets"}
    )
    assert response.status_code == 503


def test_provider_failure_is_a_502(registry):
    response = _client(registry, [ProviderError("rate limited")]).post(
        "/translate", json={"question": "active widgets"}
    )
    assert response.status_code == 502
    assert "provider" in response.json()["detail"].lower()


def test_a_registry_failure_does_not_echo_its_internals(registry):
    """The body must not carry whatever the registry error happened to say.

    A malformed registry produces a validation error naming fields and
    values from the host's own document. That is exactly the knowledge this
    service is built not to hold, let alone hand out.
    """
    marker = "zzz_marker_field_path_should_not_leak"
    response = _client(
        registry, [], error=RegistryUnavailable(f"registry document is malformed: {marker}")
    ).post("/translate", json={"question": "active widgets"})
    assert response.status_code == 503
    assert marker not in response.text


def test_healthz_reports_the_cached_registry_version(registry):
    response = _client(registry, []).get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "registry_version": "sha256:fixture-v1"}


def test_healthz_is_degraded_while_the_registry_is_out_of_reach(registry):
    """An instance that has never loaded the registry is not ready.

    Reporting ok here sends live traffic to a process where every call is
    already known to fail.
    """
    stub = StubRegistry(registry, error=RegistryUnavailable("down"), loaded=False)
    client = TestClient(create_app(stub, FakeProvider([])))
    response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "registry_version": None}


def test_healthz_becomes_ready_on_its_own_once_the_registry_answers(registry):
    """A cold process must be able to reach ready without serving a call.

    Readiness gates traffic, and the only other caller of `fetch` sits behind
    that gate — so a probe that merely reports the cache would keep a healthy
    process out of rotation forever.
    """
    stub = StubRegistry(registry, error=RegistryUnavailable("down"), loaded=False)
    client = TestClient(create_app(stub, FakeProvider([])))
    assert client.get("/healthz").status_code == 503

    stub.error = None
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "registry_version": "sha256:fixture-v1"}


def test_healthz_does_not_refetch_once_the_registry_is_cached(registry):
    """The probe is called on a schedule; a warm process must stay free."""
    stub = StubRegistry(registry)
    client = TestClient(create_app(stub, FakeProvider([])))
    assert client.get("/healthz").status_code == 200
    assert client.get("/healthz").status_code == 200
    assert stub.fetches == 0


def test_no_database_driver_is_importable():
    """Credential isolation is structural: this service cannot reach a DB.

    The guarantee is that no dependency drags a driver in. Every name here is
    a real import name, so each assertion can actually fail — `psycopg2-binary`
    is a distribution, not a module, and would pass vacuously forever.

    SQLAlchemy is deliberately absent: the evaluation store uses it, and it
    grants no database access on its own.
    """
    for module in (
        "psycopg", "psycopg2", "asyncpg", "pyodbc", "pymssql", "MySQLdb",
        "pymysql", "mysql.connector", "pg8000",
    ):
        with pytest.raises(ImportError):
            __import__(module)


def test_importing_the_app_module_builds_no_provider():
    """The production app must be a factory, not a module-level instance.

    A module-level `app = make_app()` would construct a real, billable API
    client on every import of this module, including every test run. Every
    setting defaults to `""`, so it would not even fail outright on a missing
    key — it would just start spending money.
    """
    import cert_nlq.api.app as module

    assert not hasattr(module, "app"), "make_app must not be called at import"
    assert callable(module.make_app)


def test_make_app_builds_without_calling_the_api(monkeypatch):
    """Constructing the client issues no request, so this is free."""
    from cert_nlq.api.app import make_app
    from cert_nlq.config import get_settings

    # Pinned rather than left to the default: the factory now selects a
    # provider from this setting, so an inherited value would decide which
    # adapter this test builds.
    monkeypatch.setenv("CERT_NLQ_PROVIDER", "openai")
    monkeypatch.setenv("CERT_NLQ_REGISTRY_URL", "https://cert.example")
    monkeypatch.setenv("CERT_NLQ_REGISTRY_TOKEN", "t")
    monkeypatch.setenv("CERT_NLQ_OPENAI_API_KEY", "sk-not-a-real-key")
    monkeypatch.setenv("CERT_NLQ_MODEL", "test-model")
    get_settings.cache_clear()
    try:
        assert make_app().title == "cert-nlq"
    finally:
        get_settings.cache_clear()


def test_provider_selection_honours_the_setting():
    from cert_nlq.api.app import _build_provider
    from cert_nlq.config import Settings
    from cert_nlq.translate.claude_provider import ClaudeProvider

    provider = _build_provider(
        Settings(provider="claude", anthropic_api_key="sk-not-real")
    )
    assert isinstance(provider, ClaudeProvider)


def test_an_unknown_provider_is_a_misconfiguration_not_a_provider_failure():
    """ProviderError means the provider failed; nothing was even built here.

    The settings type makes this branch unreachable in practice, which is the
    point of both halves.
    """
    from types import SimpleNamespace

    from cert_nlq.api.app import _build_provider

    with pytest.raises(ValueError, match="unknown provider") as caught:
        _build_provider(SimpleNamespace(provider="gemini"))
    assert not isinstance(caught.value, ProviderError)


def test_both_providers_are_built_with_usage_accounting():
    """The cost comparison is the point; neither side may account silently."""
    from cert_nlq.api.app import _build_provider
    from cert_nlq.config import Settings

    claude = _build_provider(
        Settings(provider="claude", anthropic_api_key="sk-not-real")
    )
    openai = _build_provider(
        Settings(provider="openai", openai_api_key="sk-not-real", model="m")
    )
    assert claude._on_usage is not None
    assert openai._on_usage is not None


GUARDED = {"question": "active widgets", "now_year": 2026}


def _guarded_client(registry, responses, token=TOKEN):
    return TestClient(
        create_app(StubRegistry(registry), FakeProvider(responses),
                   service_token=token)
    )


def test_a_request_without_a_token_is_rejected(registry):
    response = _guarded_client(registry, []).post("/translate", json=GUARDED)
    assert response.status_code == 401


def test_a_request_with_the_wrong_token_is_rejected(registry):
    response = _guarded_client(registry, []).post(
        "/translate", json=GUARDED, headers={"Authorization": "Bearer wrong"}
    )
    assert response.status_code == 401


def test_a_rejected_request_never_reaches_the_provider(registry):
    """Auth must run before anything billable.

    A 401 that still spent a model call would defeat the point of having one.
    """
    provider = FakeProvider([])
    client = TestClient(
        create_app(StubRegistry(registry), provider, service_token=TOKEN)
    )
    client.post("/translate", json=GUARDED)
    assert provider.calls == []


def test_a_request_with_the_right_token_is_translated(registry):
    response = _guarded_client(registry, [ROUTE, GOOD]).post(
        "/translate", json=GUARDED, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_healthz_needs_no_token(registry):
    """Readiness probes should not carry a credential."""
    response = _guarded_client(registry, []).get("/healthz")
    assert response.status_code in (200, 503)


def test_an_unconfigured_token_refuses_every_request(registry):
    """Fail closed. A blank token must not mean 'authentication off'.

    A misconfigured deploy that silently accepted everything is the exact
    failure this endpoint cannot afford, and it is the one a default-empty
    setting invites.
    """
    response = _guarded_client(registry, [], token="").post(
        "/translate", json=GUARDED, headers={"Authorization": "Bearer anything"}
    )
    assert response.status_code == 503


def test_the_claude_fallback_setting_reaches_the_adapter():
    """Wired end to end, both ways: a field nothing reads is not a knob."""
    from cert_nlq.api.app import _build_provider
    from cert_nlq.config import Settings

    off = _build_provider(
        Settings(_env_file=None, provider="claude", anthropic_api_key="sk-not-real")
    )
    on = _build_provider(
        Settings(
            _env_file=None,
            provider="claude",
            anthropic_api_key="sk-not-real",
            claude_fallbacks=True,
        )
    )
    assert off._use_fallbacks is False
    assert on._use_fallbacks is True


def test_the_usage_log_actually_emits(tmp_path):
    """It did not, for the whole of phase 1.

    `usage_log` records at INFO and had no handler; uvicorn configures only
    its own loggers, so this one propagated to a bare root and Python's
    last-resort handler dropped everything below WARNING. Every usage
    record was computed, handed to a logger, and discarded -- while the
    comment above `usage_log` described it as the running cost record.
    Nothing failed, which is why it survived: a log that says nothing looks
    exactly like a system that has not been called.
    """
    from cert_nlq.api.app import _configure_usage_log, _log_usage, usage_log

    path = tmp_path / "usage.log"
    saved = list(usage_log.handlers)
    usage_log.handlers.clear()
    try:
        _configure_usage_log(str(path))
        _log_usage({"schema_name": "route", "model": "m", "output_tokens": 7})
        for handler in usage_log.handlers:
            handler.flush()
        written = path.read_text(encoding="utf-8")
    finally:
        for handler in usage_log.handlers:
            handler.close()
        usage_log.handlers[:] = saved
    assert "schema_name" in written
    assert "output_tokens" in written


class UsageReportingProvider:
    """A `FakeProvider` that also reports usage, the way a real adapter does.

    `FakeProvider` never calls `on_usage`, so it cannot exercise the capture
    path on its own. This stands in for a real provider by calling
    `_log_usage` -- the same function both adapters' `on_usage` is wired to
    -- from inside `complete`, mid-request, exactly where a real usage
    record would land.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, system, question, schema, schema_name):
        from cert_nlq.api.app import _log_usage

        self.calls.append({"schema_name": schema_name})
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        _log_usage({
            "schema_name": schema_name,
            "model": "test-model",
            "uncached_input_tokens": len(self.calls),
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "output_tokens": 1,
            "reasoning_tokens": 0,
        })
        return nxt


def test_every_translate_response_carries_a_usage_list(registry):
    """`usage` is present on every response shape, not added to one and
    forgotten on the others -- all three return from the same line."""
    ok = _client(registry, [ROUTE, GOOD]).post(
        "/translate", json={"question": "active widgets", "now_year": 2026}
    ).json()
    assert ok["status"] == "ok"
    assert ok["usage"] == []

    refused = _client(
        registry, [{"root": "nope", "joins": [], "groups": []}]
    ).post("/translate", json={"question": "what is the weather"}).json()
    assert refused["status"] == "refused"
    assert refused["usage"] == []


def test_a_usage_record_captured_during_a_request_lands_in_its_own_body(registry):
    """The provider's usage callback fires mid-request; the record it hands
    `_log_usage` must appear in the body of the request that triggered it."""
    provider = UsageReportingProvider([ROUTE, GOOD])
    client = TestClient(
        create_app(StubRegistry(registry), provider, service_token=TOKEN),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    response = client.post(
        "/translate", json={"question": "active widgets", "now_year": 2026}
    )
    assert response.status_code == 200
    usage = response.json()["usage"]
    # One record per model call this request made: routing, then translation.
    assert len(usage) == 2
    assert all(record["model"] == "test-model" for record in usage)


def test_log_usage_with_no_request_in_flight_only_logs(caplog):
    """Outside a request the ContextVar is unset (`None`), and a call must
    still just log -- never raise for lack of a bucket to append to."""
    from cert_nlq.api.app import _log_usage

    with caplog.at_level("INFO", logger="cert_nlq.usage"):
        _log_usage({"schema_name": "route", "model": "m", "output_tokens": 3})
    assert any("output_tokens" in message for message in caplog.messages)


def test_two_sequential_requests_do_not_leak_usage_into_each_other(registry):
    """Uvicorn's threadpool interleaves requests; a record from one request
    reaching another's body would silently mis-price both."""
    provider = UsageReportingProvider([ROUTE, GOOD, ROUTE, GOOD])
    client = TestClient(
        create_app(StubRegistry(registry), provider, service_token=TOKEN),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    first = client.post(
        "/translate", json={"question": "active widgets", "now_year": 2026}
    ).json()
    second = client.post(
        "/translate", json={"question": "active widgets", "now_year": 2026}
    ).json()

    assert len(first["usage"]) == 2
    assert len(second["usage"]) == 2
    # Each provider call counts total calls made so far, so the two
    # requests' records are distinguishable -- and must not overlap.
    first_counts = {record["uncached_input_tokens"] for record in first["usage"]}
    second_counts = {record["uncached_input_tokens"] for record in second["usage"]}
    assert first_counts == {1, 2}
    assert second_counts == {3, 4}
