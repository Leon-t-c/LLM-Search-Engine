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
        self._error = error
        #: `loaded=False` represents a RegistryClient that has never
        #: completed a successful fetch, where `cached` stays `None`.
        self.cached = registry if loaded else None

    def fetch(self, force: bool = False):
        if self._error is not None:
            raise self._error
        return self._registry


def _client(registry, responses, error=None):
    return TestClient(
        create_app(StubRegistry(registry, error), FakeProvider(responses))
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


def test_healthz_is_degraded_before_the_registry_has_ever_loaded(registry):
    """An instance that has never loaded the registry is not ready.

    Reporting ok here sends live traffic to a process where every call is
    already known to fail.
    """
    client = TestClient(
        create_app(StubRegistry(registry, loaded=False), FakeProvider([]))
    )
    response = client.get("/healthz")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "registry_version": None}


def test_no_database_driver_is_importable():
    """Credential isolation is structural: this service cannot reach a DB.

    The guarantee is that no dependency drags a driver in. Every name here is
    a real import name, so each assertion can actually fail — `psycopg2-binary`
    is a distribution, not a module, and would pass vacuously forever.

    SQLAlchemy is deliberately absent: the phase-2 eval store uses it, and it
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

    A module-level `app = make_app()` would construct a billable API client on
    every import of this module — including every test run — and would fail
    outright when no key is configured.
    """
    import cert_nlq.api.app as module

    assert not hasattr(module, "app"), "make_app must not be called at import"
    assert callable(module.make_app)
