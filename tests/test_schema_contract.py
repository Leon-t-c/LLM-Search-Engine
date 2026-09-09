"""Check the claims about the generated schema that nothing was checking.

Three of them, each previously believed rather than tested.

**The fixpoint claim.** `strictify` narrows the schema into the subset both
vendors accept. The existing coverage asserts that subset by hand-restating
what we believe the vendors demand — which tests our belief, not their rule.
The real check is that each vendor's own normaliser has nothing left to do:
run it over the finished schema and get the same schema back.

That couples this suite to two private SDK functions, which is normally a
reason not to write a test. Here it is the entire point. These normalisers
are the rules, and if an SDK bump changes one, this is where we want to find
out — in a run that costs nothing, rather than from a 400 on the first paid
call of the day. Both imports are guarded, so a moved private path skips with
a message naming what moved instead of failing as though the schema broke.

**Byte-determinism.** The schema is the cached prefix of every prompt, and a
cache hit needs the bytes to match exactly. Delete one `sorted()` in the
generation path and the ordering follows a set's iteration order, which
varies with the interpreter's hash seed — so every cache hit across processes
disappears and the bill roughly doubles, with nothing failing. Comparing two
calls inside one process proves nothing about that: the seed is fixed for the
life of an interpreter. It has to be separate processes, with different seeds.

**The registry client through the app.** Every other test here substitutes a
stub whose constructor sets `cached`, which is precisely the fact that hid a
defect where a cold process could never become ready. Nothing drove the real
client through the app at all. This does, over a stubbed transport, so it
stays offline and free.
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from cert_nlq.api.app import create_app
from cert_nlq.ir.schema import json_schema_for
from cert_nlq.registry.client import RegistryClient
from cert_nlq.translate.provider import FakeProvider

ROOT_NAMES = ("widget", "vendor")


@pytest.mark.parametrize("name", ROOT_NAMES)
def test_the_openai_normaliser_has_nothing_left_to_do(registry, name):
    """A fixpoint under the vendor's own pass, not under our idea of it."""
    module = pytest.importorskip(
        "openai.lib._pydantic",
        reason="the SDK's strict-schema pass has moved; check where to",
    )
    ensure = getattr(module, "_ensure_strict_json_schema", None)
    if ensure is None:
        pytest.skip("the SDK's strict-schema pass has been renamed; check to what")

    schema = json_schema_for(registry.root(name))
    # It mutates what it is given, so it gets a copy and the original stays
    # available to compare against.
    normalised = ensure(copy.deepcopy(schema), path=(), root=copy.deepcopy(schema))
    assert normalised == schema


@pytest.mark.parametrize("name", ROOT_NAMES)
def test_the_anthropic_normaliser_has_nothing_left_to_do(registry, name):
    """The same claim, for the other vendor. Their rules are not identical."""
    module = pytest.importorskip(
        "anthropic.lib._parse._transform",
        reason="the SDK's schema transform has moved; check where to",
    )
    transform = getattr(module, "transform_schema", None)
    if transform is None:
        pytest.skip("the SDK's schema transform has been renamed; check to what")

    schema = json_schema_for(registry.root(name))
    assert transform(copy.deepcopy(schema)) == schema


def test_the_anthropic_normaliser_would_notice_a_keyword_it_rejects():
    """Proves the check above can fail, rather than passing on everything.

    That pass folds a keyword it does not recognise into the description
    instead of refusing it, which leaves a slot looking constrained and being
    unconstrained — a silent failure, and the reason `strictify` removes those
    keywords rather than trusting them.
    """
    module = pytest.importorskip("anthropic.lib._parse._transform")
    unsupported = {"type": "string", "minLength": 3}
    assert module.transform_schema(dict(unsupported)) != unsupported


#: Run in a child interpreter: emit the schema exactly as it would be sent.
#: Kept as a string rather than a file so the thing being measured and the
#: thing asserting on it stay in one place.
_EMIT = """
import json, sys
from pathlib import Path
sys.path.insert(0, {src!r})
from cert_nlq.ir.schema import json_schema_for
from cert_nlq.registry.models import Registry

registry = Registry.model_validate(
    json.loads(Path({fixture!r}).read_text(encoding="utf-8"))
)
sys.stdout.write(json.dumps(json_schema_for(registry.root({name!r}))))
"""

#: Deliberately different, and never the default. A run under one seed says
#: nothing about a run under another, which is the whole point.
_SEEDS = ("0", "1", "7", "12345")


@pytest.mark.parametrize("name", ROOT_NAMES)
def test_the_schema_is_byte_identical_across_processes(name):
    """Same bytes under every hash seed, or the prompt cache never hits."""
    here = Path(__file__).resolve().parent
    source = _EMIT.format(
        src=str(here.parent / "src"),
        fixture=str(here / "fixtures" / "registry.json"),
        name=name,
    )

    emitted = []
    for seed in _SEEDS:
        environment = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            env=environment,
            check=True,
        )
        emitted.append(result.stdout)

    assert len(set(emitted)) == 1, (
        "the schema differs between interpreters with different hash seeds, "
        "so it can never be a cached prompt prefix across processes"
    )
    # Cheap guard against the whole thing passing on four empty strings.
    assert emitted[0].startswith(b"{")


def test_the_child_process_actually_ran_under_the_seed_it_was_given():
    """Otherwise the test above is four runs under one seed, proving nothing."""
    seen = set()
    for seed in _SEEDS:
        result = subprocess.run(
            [sys.executable, "-c", "import os; print(os.environ['PYTHONHASHSEED'])"],
            capture_output=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
        )
        seen.add(result.stdout.strip())
    assert len(seen) == len(_SEEDS)


TOKEN = "s3cret"
ROUTE = {"root": "widget", "joins": [], "groups": ["Lifecycle"]}
GOOD = {"root": "widget", "where": {"combinator": "AND", "children": [
        {"field": "widget.status", "op": "=", "value": "Active"}]}}


def _app_on_a_real_client(registry_json, responses, status=200):
    """The app, built around the real registry client over a fake transport.

    Nothing is stubbed above the socket: the client fetches, parses, caches
    and reports `cached` on its own behaviour rather than because a test
    constructor set it.
    """
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, json=registry_json)

    transport = httpx.Client(transport=httpx.MockTransport(handle))
    client = RegistryClient("https://host.example", token="t", client=transport)
    app = create_app(client, FakeProvider(responses), service_token=TOKEN)
    return TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}), client, calls


def test_a_question_goes_all_the_way_through_the_real_registry_client(
    registry_json,
):
    http, client, calls = _app_on_a_real_client(registry_json, [ROUTE, GOOD])

    assert client.cached is None, "a fresh client has fetched nothing yet"
    response = http.post(
        "/translate", json={"question": "active widgets", "now_year": 2026}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["registry_version"] == "sha256:fixture-v1"
    assert len(calls) == 1, "the registry is fetched once, then cached"
    assert client.cached is not None


def test_a_second_question_reuses_the_fetched_registry(registry_json):
    http, _, calls = _app_on_a_real_client(
        registry_json, [ROUTE, GOOD, ROUTE, GOOD]
    )
    for _ in range(2):
        http.post("/translate", json={"question": "active widgets", "now_year": 2026})
    assert len(calls) == 1


def test_a_cold_real_client_reports_unready_then_warms_itself(registry_json):
    """The defect the stub hid: `cached` is set by a fetch, never by a probe.

    A probe that only reported the cache would leave a cold process unready
    for ever, because the only other caller of `fetch` sits behind the token
    gate a load balancer will not pass.
    """
    http, client, calls = _app_on_a_real_client(registry_json, [])

    assert client.cached is None
    probe = http.get("/healthz")

    assert probe.status_code == 200
    assert probe.json() == {"status": "ok", "registry_version": "sha256:fixture-v1"}
    assert len(calls) == 1
    assert client.cached is not None


def test_a_cold_real_client_that_cannot_reach_the_registry_stays_unready(
    registry_json,
):
    """And says so in the status line, not only in the body."""
    http, client, calls = _app_on_a_real_client(registry_json, [], status=503)

    probe = http.get("/healthz")

    assert probe.status_code == 503
    assert probe.json()["status"] == "degraded"
    assert client.cached is None
    assert calls, "it must have tried"


def test_the_real_client_turns_an_unreachable_registry_into_a_503(registry_json):
    """The unhappy path through the real parse-and-fetch code, not a stub."""
    http, _, _ = _app_on_a_real_client(registry_json, [], status=500)
    response = http.post(
        "/translate", json={"question": "active widgets", "now_year": 2026}
    )
    assert response.status_code == 503
    # The detail must stay generic: the client's own message can carry
    # field paths out of the host's registry document.
    assert response.json()["detail"] == "The query service is temporarily unavailable."


def test_the_response_the_real_path_emits_uses_the_wire_names(registry_json):
    """`as`, not `alias` — the alias survives the whole real stack."""
    aggregating = {"root": "widget",
                   "aggregate": [{"fn": "count", "field": "*", "as": "total"}]}
    http, _, _ = _app_on_a_real_client(registry_json, [ROUTE, aggregating])
    body = http.post(
        "/translate", json={"question": "how many widgets", "now_year": 2026}
    ).json()
    assert body["payload"]["aggregate"][0] == {
        "fn": "count", "field": "*", "as": "total"
    }
    assert json.dumps(body).count('"alias"') == 0
