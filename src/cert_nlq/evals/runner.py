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


def _safe_json(response: httpx.Response) -> dict | None:
    """The response body as a dict, or `None` when it is not JSON at all or
    not a JSON object -- never raises. Used to populate `Outcome.raw` on
    every path, including a non-2xx response that still carries a JSON
    error body.
    """
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _to_outcome(body: dict, latency_ms: float) -> Outcome:
    """One already-parsed `/translate` response body -> one `Outcome`,
    `raw=body` on every instance this returns (`Outcome.raw`'s own
    docstring: a later evaluator-version bump re-scores from the untouched
    wire body, not from this function's own translated snapshot of it).

    Raises `ValueError`/`TypeError`/`KeyError`/`pydantic.ValidationError` on
    anything this cannot make sense of, and the caller (`_call_one`) turns
    every one of those into an error `Outcome` -- still carrying `raw`,
    since the body was parsed successfully even if this function could not
    place it into one of the three known shapes -- rather than letting it
    propagate.
    """
    status = _STATUS_MAP.get(body.get("status"))
    if status is None:
        raise ValueError(f"unrecognised wire status {body.get('status')!r}")
    usage = tuple(u for u in (body.get("usage") or ()) if isinstance(u, dict))
    route = body.get("route") if isinstance(body.get("route"), dict) else None
    payload = body.get("payload") if isinstance(body.get("payload"), dict) else None

    if status == "ok":
        return Outcome(
            status="ok", payload=payload, latency_ms=latency_ms, usage=usage, route=route,
            raw=body,
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
            raw=body,
        )
    # status == "refusal"
    reason = body.get("reason")
    return Outcome(
        status="refusal",
        reason=reason if isinstance(reason, str) else None,
        latency_ms=latency_ms,
        usage=usage,
        route=route,
        raw=body,
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
        except httpx.HTTPError as exc:
            # A pure transport failure (timeout, connect error): no response
            # was ever received, so there is no wire body for `raw` either.
            return Outcome(latency_ms=_elapsed_ms(started), error=str(exc))
        raw_body = _safe_json(response)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # The wire answered, just not with 2xx -- whatever body it sent
            # (a JSON error payload, say) is preserved on `raw` even though
            # the call still counts as an error for scoring.
            return Outcome(latency_ms=_elapsed_ms(started), error=str(exc), raw=raw_body)
        latency_ms = _elapsed_ms(started)
        if raw_body is None:
            # A 2xx response whose body was not JSON, or not a JSON object
            # -- there is nothing here `score()` could give credit for.
            return Outcome(
                latency_ms=latency_ms, error="response body was not a JSON object"
            )
        try:
            return _to_outcome(raw_body, latency_ms)
        except (ValueError, TypeError, KeyError, ValidationError) as exc:
            # A 2xx, JSON-object response this module still cannot place
            # into any of the three known shapes -- treated the same as a
            # transport failure for scoring, but `raw` is kept: the body
            # was real, this function just could not translate it.
            return Outcome(
                latency_ms=latency_ms, error=f"malformed response body: {exc}", raw=raw_body
            )


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
    **(expectation kind, difficulty)** strata -- `("ok", "easy")`,
    `("clarify", "medium")`, `("refusal", "medium")`, and so on -- keeping
    every paraphrase family whole: a family is included or excluded as one
    unit, never split.

    **Judgement call** (see task-6-report.md). Two amendments to the
    brief's literal "proportionally across derived slices":

    1. `golden.SLICES` overlap by design -- a case can be `join` and
       `coded-vocabulary` and `multi-condition` at once -- so they cannot
       back a *proportional allocation* without double-counting a case into
       more than one bucket's budget. `derive_difficulty` is the one
       derived axis in `golden.py` that partitions the ok-case structural
       variety (easy/medium/hard) without overlap.
    2. Difficulty alone is not enough, because it is silent about
       `clarify`/`refusal` cases relative to `ok` ones in a way that starves
       exactly the population a small sample most needs to keep alive: at
       `--stratify 40` over a set where refusals split across three
       reasons, a difficulty-only draw proportions ~2.6 refusal cases
       across all three, landing most reasons at n=0 or n=1 -- refusal
       precision/recall and candidate recall become unestimable in
       precisely the affordable sample a reviewer would actually run.
       Crossing **expectation kind** (`case.expect.kind`: `"ok"`,
       `"clarify"`, `"refusal"`) with difficulty keeps every kind
       represented proportionally to its own share of the set, independent
       of how the ok cases happen to distribute across difficulty.

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

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    family_size: dict[str, int] = {}
    for fid in family_order:
        members = families[fid]
        canonical_case = next((c for c in members if c.id == fid), members[0])
        stratum_key = (canonical_case.expect.kind, derive_difficulty(canonical_case, registry))
        strata[stratum_key].append(fid)
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
