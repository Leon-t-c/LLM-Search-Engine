"""The async runner, exercised against `httpx.MockTransport` -- no network,
no `pytest-asyncio`/`anyio` plugin (neither is a project dependency): every
async test wraps its assertions in a local `async def` and drives it with a
bare `asyncio.run`, the same stdlib-only approach the rest of this eval
layer takes.
"""
import ast
import asyncio
import json
from pathlib import Path

import httpx

from cert_nlq.evals.golden import GoldenCase
from cert_nlq.evals.runner import canonical_only, run_benchmark, stratified_sample

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _ok_case(case_id, payload, *, question=None, source="hand-written", paraphrase_of=None):
    return GoldenCase.model_validate({
        "id": case_id,
        "question": question or f"question for {case_id}",
        "source": "paraphrase" if paraphrase_of else source,
        "paraphrase_of": paraphrase_of,
        "expect": {"kind": "ok", "payload": payload},
    })


def _simple_filter_payload():
    """Exactly one condition, no join/aggregate -- `derive_difficulty`'s
    `easy` bucket."""
    return {
        "root": "widget",
        "where": {"field": "widget.serial", "op": ">", "value": 5},
    }


def _multi_condition_payload():
    """Two conditions, flat -- `medium`."""
    return {
        "root": "widget",
        "where": {
            "combinator": "AND",
            "children": [
                {"field": "widget.serial", "op": ">", "value": 5},
                {"field": "widget.price", "op": ">", "value": 100},
            ],
        },
    }


def _nested_logic_payload():
    """Depth-2 `where` -- `hard`."""
    return {
        "root": "widget",
        "where": {
            "combinator": "AND",
            "children": [
                {"field": "widget.serial", "op": ">", "value": 5},
                {
                    "combinator": "OR",
                    "children": [
                        {"field": "widget.status", "op": "=", "value": "A"},
                        {"field": "widget.status", "op": "=", "value": "X"},
                    ],
                },
            ],
        },
    }


def _run(coro):
    return asyncio.run(coro)


def _make_client(handler) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(base_url="http://eval.test", transport=transport)


# --------------------------------------------------------------------------
# run_benchmark: ordering, concurrency, latency
# --------------------------------------------------------------------------


def test_results_keep_case_order_regardless_of_completion_order():
    # q0 is the slowest call, q2 the fastest -- completion order is the
    # reverse of submission order, but the returned list must not be.
    delays = {"q0": 0.03, "q1": 0.015, "q2": 0.0}
    cases = [
        _ok_case("c0", _simple_filter_payload(), question="q0"),
        _ok_case("c1", _simple_filter_payload(), question="q1"),
        _ok_case("c2", _simple_filter_payload(), question="q2"),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        question = body["question"]
        await asyncio.sleep(delays[question])
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "payload": {"root": "widget"},
                "route": {"root": question, "joins": [], "groups": []},
                "usage": [],
            },
        )

    async def go():
        async with _make_client(handler) as client:
            return await run_benchmark(cases, client=client, concurrency=3)

    outcomes = _run(go())
    assert [o.route["root"] for o in outcomes] == ["q0", "q1", "q2"]


def test_semaphore_bounds_in_flight_calls_to_concurrency():
    in_flight = 0
    peak = 0
    cases = [_ok_case(f"c{i}", _simple_filter_payload()) for i in range(6)]

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(
            200, json={"status": "ok", "payload": {"root": "widget"}, "usage": []}
        )

    async def go():
        async with _make_client(handler) as client:
            return await run_benchmark(cases, client=client, concurrency=2)

    outcomes = _run(go())
    assert len(outcomes) == 6
    assert peak <= 2
    # The fixture actually exercised concurrency -- not a degenerate serial
    # run that happens to satisfy "<= 2" trivially at peak 1.
    assert peak == 2


def test_latency_is_measured_per_call():
    cases = [_ok_case("c0", _simple_filter_payload())]

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(
            200, json={"status": "ok", "payload": {"root": "widget"}, "usage": []}
        )

    async def go():
        async with _make_client(handler) as client:
            return await run_benchmark(cases, client=client, concurrency=1)

    (outcome,) = _run(go())
    assert outcome.latency_ms >= 40  # 50ms nominal, generous floor for CI jitter


# --------------------------------------------------------------------------
# wire -> scoring status translation, usage/route capture, candidates
# --------------------------------------------------------------------------


def test_ok_response_translates_and_captures_usage_and_route():
    cases = [_ok_case("c0", _simple_filter_payload())]
    usage = [{"schema_name": "translate", "model": "gpt-5", "uncached_input_tokens": 100,
              "cached_input_tokens": 0, "cache_write_tokens": 0, "output_tokens": 20,
              "reasoning_tokens": 0}]
    route = {"root": "widget", "joins": [], "groups": ["Lifecycle"]}

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "payload": {"root": "widget"},
                "usage": usage,
                "route": route,
            },
        )

    async def go():
        async with _make_client(handler) as client:
            return await run_benchmark(cases, client=client, concurrency=1)

    (outcome,) = _run(go())
    assert outcome.status == "ok"
    assert outcome.payload == {"root": "widget"}
    assert outcome.usage == tuple(usage)
    assert outcome.route == route
    assert outcome.error is None


def test_needs_clarification_body_parses_into_candidates():
    cases = [_ok_case("c0", _simple_filter_payload())]
    candidates = [
        {"field": "widget.status", "op": "=", "value": "A", "label": "Active"},
        {"field": "widget.status", "op": "=", "value": "X", "label": "Retired"},
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "needs_clarification",
                "payload": {"root": "widget"},
                "unresolved": {
                    "phrase": "operational",
                    "question": "which status?",
                    "candidates": candidates,
                },
                "usage": [],
            },
        )

    async def go():
        async with _make_client(handler) as client:
            return await run_benchmark(cases, client=client, concurrency=1)

    (outcome,) = _run(go())
    assert outcome.status == "clarify"
    assert outcome.candidates == tuple(candidates)
    assert outcome.payload == {"root": "widget"}


def test_refused_response_translates_status_and_reason():
    cases = [_ok_case("c0", _simple_filter_payload())]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "refused", "reason": "not_a_query", "detail": "hi", "usage": []},
        )

    async def go():
        async with _make_client(handler) as client:
            return await run_benchmark(cases, client=client, concurrency=1)

    (outcome,) = _run(go())
    assert outcome.status == "refusal"
    assert outcome.reason == "not_a_query"


def test_wire_status_the_wire_never_leaks_its_own_spelling():
    """`Outcome.status` is a `Literal["ok", "clarify", "refusal"]` --
    constructing one with the wire's own strings (`"needs_clarification"`,
    `"refused"`) would raise. Confirms the runner never does that; the
    translated value alone reaches `Outcome`.
    """
    cases = [_ok_case("c0", _simple_filter_payload())]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"status": "needs_clarification",
                       "payload": {"root": "widget"},
                       "unresolved": {"phrase": "x", "question": "y",
                                      "candidates": [{"field": "widget.status", "op": "=",
                                                       "value": "A", "label": "Active"}]},
                       "usage": []},
        )

    async def go():
        async with _make_client(handler) as client:
            return await run_benchmark(cases, client=client, concurrency=1)

    (outcome,) = _run(go())
    assert outcome.status == "clarify"
    assert outcome.status != "needs_clarification"


# --------------------------------------------------------------------------
# error paths: never an exception
# --------------------------------------------------------------------------


def test_a_502_becomes_an_error_outcome_not_a_crash():
    cases = [
        _ok_case("c0", _simple_filter_payload()),
        _ok_case("c1", _simple_filter_payload()),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["question"] == cases[1].question:
            return httpx.Response(502, text="bad gateway")
        return httpx.Response(
            200, json={"status": "ok", "payload": {"root": "widget"}, "usage": []}
        )

    async def go():
        async with _make_client(handler) as client:
            return await run_benchmark(cases, client=client, concurrency=2)

    outcomes = _run(go())
    assert len(outcomes) == 2
    assert outcomes[0].error is None
    assert outcomes[1].error is not None
    assert "502" in outcomes[1].error
    assert outcomes[1].status is None


def test_a_timeout_becomes_an_error_outcome():
    cases = [_ok_case("c0", _simple_filter_payload())]

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated timeout", request=request)

    async def go():
        async with _make_client(handler) as client:
            return await run_benchmark(cases, client=client, concurrency=1)

    (outcome,) = _run(go())
    assert outcome.status is None
    assert outcome.error is not None


def test_malformed_json_body_becomes_an_error_outcome():
    cases = [_ok_case("c0", _simple_filter_payload())]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json at all")

    async def go():
        async with _make_client(handler) as client:
            return await run_benchmark(cases, client=client, concurrency=1)

    (outcome,) = _run(go())
    assert outcome.error is not None
    assert outcome.status is None


def test_unrecognised_wire_status_becomes_an_error_outcome():
    cases = [_ok_case("c0", _simple_filter_payload())]

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "something_new"})

    async def go():
        async with _make_client(handler) as client:
            return await run_benchmark(cases, client=client, concurrency=1)

    (outcome,) = _run(go())
    assert outcome.error is not None
    assert outcome.status is None


# --------------------------------------------------------------------------
# canonical_only()
# --------------------------------------------------------------------------


def test_canonical_only_filters_out_paraphrases():
    canonical = _ok_case("c0", _simple_filter_payload())
    paraphrase = _ok_case("c0-p1", _simple_filter_payload(), paraphrase_of="c0")
    result = canonical_only([canonical, paraphrase])
    assert [c.id for c in result] == ["c0"]


# --------------------------------------------------------------------------
# stratified_sample(): families whole, deterministic, proportional
# --------------------------------------------------------------------------


def _families(registry):
    """Three families across the three difficulty buckets, each with one
    paraphrase sibling -- six cases, three families."""
    easy = _ok_case("easy-1", _simple_filter_payload())
    easy_p = _ok_case("easy-1-p1", _simple_filter_payload(), paraphrase_of="easy-1")
    medium = _ok_case("medium-1", _multi_condition_payload())
    medium_p = _ok_case("medium-1-p1", _multi_condition_payload(), paraphrase_of="medium-1")
    hard = _ok_case("hard-1", _nested_logic_payload())
    hard_p = _ok_case("hard-1-p1", _nested_logic_payload(), paraphrase_of="hard-1")
    return [easy, easy_p, medium, medium_p, hard, hard_p]


def test_stratified_sample_keeps_every_paraphrase_family_whole(registry):
    cases = _families(registry)
    sample = stratified_sample(cases, 4, seed=1, registry=registry)
    sampled_ids = {c.id for c in sample}
    # For every family present at all, every member is present.
    families = {
        "easy-1": {"easy-1", "easy-1-p1"},
        "medium-1": {"medium-1", "medium-1-p1"},
        "hard-1": {"hard-1", "hard-1-p1"},
    }
    for members in families.values():
        overlap = sampled_ids & members
        assert overlap in (set(), members), f"family split: {overlap} of {members}"


def test_stratified_sample_is_deterministic_under_the_same_seed(registry):
    cases = _families(registry)
    first = [c.id for c in stratified_sample(cases, 4, seed=7, registry=registry)]
    second = [c.id for c in stratified_sample(cases, 4, seed=7, registry=registry)]
    assert first == second


def test_stratified_sample_of_zero_returns_nothing(registry):
    cases = _families(registry)
    assert stratified_sample(cases, 0, seed=1, registry=registry) == []


def test_stratified_sample_never_exceeds_the_available_cases(registry):
    cases = _families(registry)
    sample = stratified_sample(cases, 100, seed=1, registry=registry)
    assert {c.id for c in sample} <= {c.id for c in cases}


# --------------------------------------------------------------------------
# the async-containment tripwire (spec §11 decision)
# --------------------------------------------------------------------------


def test_only_evals_defines_async_functions():
    """Every `async def` in `src/cert_nlq` lives under `evals/`. Walked with
    `ast`, not imported: an import-based check would only see functions
    reachable from whatever happens to get imported, and this must hold for
    the whole tree.
    """
    src_root = Path(__file__).resolve().parents[1] / "src" / "cert_nlq"
    offenders = []
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                rel = path.relative_to(src_root)
                if rel.parts[0] != "evals":
                    offenders.append(f"{rel}:{node.lineno} ({node.name})")
    assert offenders == [], f"async def outside evals/: {offenders}"
