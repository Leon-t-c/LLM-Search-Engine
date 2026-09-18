"""The async benchmark runner.

**This module is the project's only async Python** -- spec §11/§15's own
decision (phase 2's replay is concurrent, `asyncio` + `httpx.AsyncClient`,
"the project's first and only async Python"). Nothing else in `src/cert_nlq`
may define an `async def`; `tests/test_evals_runner.py` walks the whole
source tree with `ast` and fails the suite if anything does. The reason to
keep it contained rather than let async creep into the translator or the API
layer: the service itself is synchronous FastAPI (uvicorn's own threadpool
already gives it concurrency), and a second concurrency model living beside
the first would be two things to reason about for no measured benefit -- the
only place concurrency actually pays for itself is replaying ~180 golden
questions against a deployed service, which is what this module alone does.

`run_benchmark` never raises out of a per-case call: an HTTP error, a
timeout, or a response this module cannot parse all become an `Outcome` with
`error` set (`scoring.Outcome`'s own contract), never an exception -- the
alternative is a benchmark run that stops at the first flaky call instead of
recording it as one. **Zero retries**, deliberately: a retry is a second
paid call and a changed latency measurement, and this module's whole job is
producing one honest number for each, not the best of several.
"""
import asyncio
import random
import time
from collections import defaultdict
from collections.abc import Sequence

import httpx
from pydantic import ValidationError

from ..registry.models import Registry
from .golden import GoldenCase, derive_difficulty, family
from .scoring import Outcome

#: Per-call timeout. Long enough that a slow provider call is measured, not
#: mistaken for an infra failure; the number itself is a judgement call
#: (task-6-report.md) with no spec-mandated value.
CALL_TIMEOUT_SECONDS = 180.0

#: Wire status (`ir.response`'s own `Literal`) -> `Outcome.status`'s
#: vocabulary. See `Outcome`'s own docstring for why this translation must
#: happen before construction, never after.
_STATUS_MAP = {
    "ok": "ok",
    "needs_clarification": "clarify",
    "refused": "refusal",
}


def _elapsed_ms(started: float) -> float:
    return (time.monotonic() - started) * 1000


def _to_outcome(response: httpx.Response, latency_ms: float) -> Outcome:
    """One `/translate` response body -> one `Outcome`.

    Raises `ValueError`/`TypeError`/`KeyError`/`pydantic.ValidationError` on
    anything this cannot make sense of -- `response.json()`'s own decode
    error included -- and the caller (`_call_one`) turns every one of those
    into an error `Outcome` rather than letting it propagate. That keeps
    this function itself simple (raise on the first thing that looks wrong)
    while still meeting `run_benchmark`'s never-raises contract at the one
    place that matters.
    """
    body = response.json()
    if not isinstance(body, dict):
        raise ValueError(f"expected a JSON object, got {type(body).__name__}")
    status = _STATUS_MAP.get(body.get("status"))
    if status is None:
        raise ValueError(f"unrecognised wire status {body.get('status')!r}")
    usage = tuple(u for u in (body.get("usage") or ()) if isinstance(u, dict))
    route = body.get("route") if isinstance(body.get("route"), dict) else None
    payload = body.get("payload") if isinstance(body.get("payload"), dict) else None

    if status == "ok":
        return Outcome(
            status="ok", payload=payload, latency_ms=latency_ms, usage=usage, route=route
        )
    if status == "clarify":
        unresolved = body.get("unresolved")
        raw_candidates = unresolved.get("candidates") if isinstance(unresolved, dict) else ()
        candidates = tuple(c for c in (raw_candidates or ()) if isinstance(c, dict))
        return Outcome(
            status="clarify",
            payload=payload,
            candidates=candidates,
            latency_ms=latency_ms,
            usage=usage,
            route=route,
        )
    # status == "refusal"
    reason = body.get("reason")
    return Outcome(
        status="refusal",
        reason=reason if isinstance(reason, str) else None,
        latency_ms=latency_ms,
        usage=usage,
        route=route,
    )


async def _call_one(
    client: httpx.AsyncClient,
    case: GoldenCase,
    semaphore: asyncio.Semaphore,
    *,
    now_year: int | None,
) -> Outcome:
    """One `/translate` call, semaphore-bounded, never raising.

    The semaphore is acquired for the *whole* call, request through parse --
    `concurrency` is a bound on in-flight HTTP calls, not just on how many
    requests may be sent before the first response comes back.
    """
    async with semaphore:
        started = time.monotonic()
        try:
            response = await client.post(
                "/translate",
                json={"question": case.question, "now_year": now_year},
                timeout=CALL_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # Covers both transport failures (timeout, connect error -- an
            # `httpx.RequestError`) and non-2xx statuses (`raise_for_status`'s
            # own `httpx.HTTPStatusError`, a subclass of the same base):
            # either way the HTTP call itself failed, exactly the condition
            # `Outcome.error` documents. Everything else on the `Outcome`
            # stays at its default (unset), per that contract.
            return Outcome(latency_ms=_elapsed_ms(started), error=str(exc))
        latency_ms = _elapsed_ms(started)
        try:
            return _to_outcome(response, latency_ms)
        except (ValueError, TypeError, KeyError, ValidationError) as exc:
            # A 2xx response this module cannot parse into any of the three
            # known shapes -- treated the same as a transport failure: there
            # is nothing here `score()` could give credit for either way.
            return Outcome(latency_ms=latency_ms, error=f"malformed response body: {exc}")


async def run_benchmark(
    cases: Sequence[GoldenCase],
    *,
    client: httpx.AsyncClient,
    concurrency: int,
    now_year: int | None = None,
) -> list[Outcome]:
    """Replay `cases` against `client`, `concurrency` calls in flight at
    once, and return one `Outcome` per case, **in `cases`' own order** --
    not completion order. `asyncio.gather` guarantees that ordering from the
    order its awaitables were passed, which is exactly why the task list
    below is built eagerly, one task per case, before anything is awaited.

    `client` is the caller's: base URL, auth header and any other
    per-request wiring belong on it, not here -- this function's only job is
    the fan-out and the per-call timeout/retry policy.
    """
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(_call_one(client, case, semaphore, now_year=now_year))
        for case in cases
    ]
    return list(await asyncio.gather(*tasks))


# --------------------------------------------------------------------------
# sampling: --canonical-only and --stratify
# --------------------------------------------------------------------------


def canonical_only(cases: Sequence[GoldenCase]) -> list[GoldenCase]:
    """Cases with no `paraphrase_of` -- the population the primary McNemar
    test itself runs over (`stats.py`'s module docstring), and `--canonical-
    only`'s own filter."""
    return [c for c in cases if c.paraphrase_of is None]


def stratified_sample(
    cases: Sequence[GoldenCase], n: int, *, seed: int, registry: Registry
) -> list[GoldenCase]:
    """A seeded sample of about `n` cases, drawn proportionally across
    **difficulty** strata (easy/medium/hard), keeping every paraphrase
    family whole -- a family is included or excluded as one unit, never
    split.

    **Judgement call** (see task-6-report.md): the brief asks for
    "proportionally across derived slices", but `golden.SLICES` overlap by
    design -- a case can be `join` and `coded-vocabulary` and
    `multi-condition` at once -- so they cannot back a *proportional
    allocation* without double-counting a case into more than one bucket's
    budget. `derive_difficulty` is the one derived axis in `golden.py` that
    actually partitions the set (every case gets exactly one of
    easy/medium/hard), so it stands in for "slices" here as the
    stratification key.

    A family's stratum is read off its **canonical** member (or, failing
    that, an arbitrary member of the family present in `cases` -- expected
    never to happen for a well-formed golden set, where a paraphrase's
    canonical case is always in the same file).

    The per-stratum budget is allocated by that stratum's share of the total
    *case* count (not family count) in `cases`, via the largest-remainder
    method so the budgets sum to exactly `n`; within a stratum, whole
    families are added (in a seed-shuffled order) until that stratum's
    budget is met or exceeded. Because families vary in size, the returned
    case count lands close to `n` rather than exactly at it -- the
    granularity is one family, not one case.

    Deterministic: identical `(cases, n, seed)` produce a byte-identical
    sample every time.
    """
    if n <= 0:
        return []

    families: dict[str, list[GoldenCase]] = {}
    family_order: list[str] = []
    for case in cases:
        fid = family(case)
        if fid not in families:
            families[fid] = []
            family_order.append(fid)
        families[fid].append(case)

    strata: dict[str, list[str]] = defaultdict(list)
    family_size: dict[str, int] = {}
    for fid in family_order:
        members = families[fid]
        canonical_case = next((c for c in members if c.id == fid), members[0])
        strata[derive_difficulty(canonical_case, registry)].append(fid)
        family_size[fid] = len(members)

    total_cases = sum(family_size.values())
    if total_cases == 0:
        return []

    stratum_names = sorted(strata)
    stratum_case_totals = {
        name: sum(family_size[fid] for fid in strata[name]) for name in stratum_names
    }
    raw_targets = {
        name: n * stratum_case_totals[name] / total_cases for name in stratum_names
    }
    targets = {name: int(raw_targets[name]) for name in stratum_names}
    remainder = n - sum(targets.values())
    # Largest fractional remainder first; ties broken by stratum name so the
    # allocation itself is deterministic even before the per-stratum family
    # shuffle below.
    by_fraction = sorted(
        stratum_names, key=lambda name: (-(raw_targets[name] - targets[name]), name)
    )
    for name in by_fraction[: max(remainder, 0)]:
        targets[name] += 1

    rng = random.Random(seed)
    chosen: set[str] = set()
    for name in stratum_names:
        family_ids = list(strata[name])
        rng.shuffle(family_ids)
        budget = targets[name]
        running = 0
        for fid in family_ids:
            if running >= budget:
                break
            chosen.add(fid)
            running += family_size[fid]

    return [case for case in cases if family(case) in chosen]
