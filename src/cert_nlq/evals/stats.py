"""The paired-design statistics layer (Task 4b, owner design 2026-09-18).

Both models answer the same golden questions, so the comparison is paired
data, not two marginal accuracies -- every function here exploits that.
Nothing here re-derives correctness: every statistic reads `Row.e2e_correct`
(or another already-scored `Row` field) exactly as `scoring.score` wrote it,
and family membership is likewise never re-derived -- a `family_of`
callable, built by the caller from `golden.family`/`GoldenCase.paraphrase_of`,
is the only source of the family key any function here ever sees. A row is
its family's *canonical* member iff `family_of(row) == row.case_id` -- that
is exactly `golden.family`'s own definition (`paraphrase_of or case.id`)
restated in terms of what a bare `Row` can see, and every family-aware
function below relies on it rather than taking a second "is canonical" flag.

**Inference discipline (owner design 2026-09-18).** There is exactly ONE
confirmatory test in this eval: the primary McNemar comparison, OpenAI vs
self-hosted, over canonical cases only (`paraphrase_of is None`) --
"preregistered" operationally, in the sense that it is declared here, before
any run, and a run row's `golden_hash` pins the exact set it ran over.
`mcnemar_exact` itself does **not** filter to canonical cases -- the
*report* owns that filter (so this function stays pure and reusable), which
means calling it over the paraphrase-expanded set, or per slice, is possible
but is no longer the preregistered analysis; a caller doing that must
present the result as descriptive (an interval, with n), never as a second
p-value. No per-slice p-values are emitted by anything in this module. If
per-slice inference is ever wanted, it needs a Benjamini-Hochberg correction
applied at the report layer -- nothing here computes one, and nothing here
should be read as license to skip it.

Everything else in this module -- `cluster_bootstrap`, `paired_delta_ci`,
`family_accuracy`, `family_consistency`, `latency_summary` -- is
**descriptive**: an interval or a rate with its n, framed as where to look
next, never a p-value standing in for a second confirmatory test.

**Numerical conventions fixed by this module** (the open choices, and why,
are in `task-4b-report.md`):

* `wilson` only accepts confidence levels with a tabulated z (0.90, 0.95,
  0.99) -- there is deliberately no inverse-normal-CDF fallback; a level
  outside the table raises rather than silently approximating one.
* Bootstrap percentile intervals use linear interpolation between order
  statistics (numpy's default method, "R type 7").
* `latency_summary`'s p50/p95 use the same nearest-rank convention as
  `aggregate.py`'s own `_latency` (`ceil(p/100 * n)`-th smallest, 1-indexed,
  clamped into range).

**Degenerate inputs never raise and never fabricate a `0.0` or a `1.0` that
was not earned.** `n == 0` returns `None`-valued rates/intervals throughout;
the one deliberate exception is `mcnemar_exact` with zero discordant pairs,
which is genuinely `p = 1.0` (the two models disagreed nowhere, so nothing
in the data argues against the null) -- not a fallback, an honest answer.
"""
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

from .scoring import Row

# ---------------------------------------------------------------------------
# wilson()
# ---------------------------------------------------------------------------

#: z for a two-sided Wilson interval, at the confidence levels this module
#: supports. A small table, not a general inverse-normal-CDF: three levels
#: cover every use in this eval, and a table entry is auditable in a way a
#: numerical root-find is not (implementation note, beyond the brief).
_WILSON_Z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def wilson(
    successes: int, n: int, confidence: float = 0.95
) -> tuple[float | None, float | None]:
    """The Wilson score interval for `successes / n`.

    Valid only for independent Bernoulli observations -- canonical-set
    rates, or per-family-collapsed rates. On the paraphrase-expanded set,
    where paraphrase siblings are correlated, use `cluster_bootstrap`
    instead; Wilson's independence assumption does not survive families.

    `(None, None)` on `n == 0` -- there is no rate to bound, and reporting
    `(0.0, 1.0)` or any other placeholder would claim a measurement that
    was not made.
    """
    if n == 0:
        return None, None
    if not 0 <= successes <= n:
        raise ValueError(f"successes ({successes}) must be between 0 and n ({n})")
    if confidence not in _WILSON_Z:
        raise ValueError(
            f"unsupported confidence {confidence!r}; wilson() only "
            f"tabulates {sorted(_WILSON_Z)}"
        )
    z = _WILSON_Z[confidence]
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


# ---------------------------------------------------------------------------
# family_accuracy() / family_consistency()
# ---------------------------------------------------------------------------


def _bucket_families(
    rows: Sequence[Row], family_of: Callable[[Row], str]
) -> dict[str, dict[str, Any]]:
    """Group rows into `{family_id: {"canonical": Row | None, "paraphrases":
    [Row, ...]}}`. A row is canonical for its family iff `family_of(row) ==
    row.case_id` -- see the module docstring."""
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"canonical": None, "paraphrases": []}
    )
    for row in rows:
        fid = family_of(row)
        entry = buckets[fid]
        if row.case_id == fid:
            entry["canonical"] = row
        else:
            entry["paraphrases"].append(row)
    return buckets


def family_accuracy(
    rows: Sequence[Row], family_of: Callable[[Row], str]
) -> dict[str, Any]:
    """Fraction of families -- canonical + its confirmed paraphrases, over
    families with >=1 paraphrase present -- whose **every** member is
    `e2e_correct`. Owner definition: *does the model get this semantic
    question right across wording?*

    A family with paraphrases but no canonical row in `rows` cannot be
    judged -- "every member correct" needs the canonical member's own
    correctness, which is exactly what is missing. Its paraphrases are
    skipped, and counted in `skipped`, rather than silently dropped or
    scored as though the canonical answer were known.

    `rate` is `None` on zero eligible families -- never a fabricated `0.0`.
    Always read beside `family_consistency`: accuracy alone says nothing
    about *why* a family failed (see that function's docstring).
    """
    buckets = _bucket_families(rows, family_of)
    families = all_correct = skipped = 0
    for entry in buckets.values():
        paraphrases = entry["paraphrases"]
        if not paraphrases:
            continue
        canonical = entry["canonical"]
        if canonical is None:
            skipped += len(paraphrases)
            continue
        families += 1
        if canonical.e2e_correct and all(p.e2e_correct for p in paraphrases):
            all_correct += 1
    rate = all_correct / families if families else None
    return {
        "families": families,
        "all_correct": all_correct,
        "rate": rate,
        "skipped": skipped,
    }


def family_consistency(
    rows: Sequence[Row], family_of: Callable[[Row], str]
) -> dict[str, Any]:
    """Fraction of confirmed paraphrases whose `e2e_correct` **matches**
    their canonical member's. Owner definition: *does wording change the
    model's behaviour?*

    This is deliberately not a goodness measure by itself: a family that is
    wrong on the canonical case and wrong on every paraphrase scores a
    perfect 1.0 here -- consistently wrong, which is exactly why the report
    never prints this without `family_accuracy` beside it (a reader who
    sees only "100% consistent" and not "0% accurate" would draw the
    opposite conclusion from the truth).

    A paraphrase whose family's canonical row is absent from `rows` has
    nothing to compare against -- skipped, and counted in `skipped`.
    `rate` is `None` on zero paraphrases, never a fabricated `0.0`.
    """
    buckets = _bucket_families(rows, family_of)
    paraphrases = matching = skipped = 0
    for entry in buckets.values():
        members = entry["paraphrases"]
        if not members:
            continue
        canonical = entry["canonical"]
        if canonical is None:
            skipped += len(members)
            continue
        for member in members:
            paraphrases += 1
            if member.e2e_correct == canonical.e2e_correct:
                matching += 1
    rate = matching / paraphrases if paraphrases else None
    return {
        "paraphrases": paraphrases,
        "matching_canonical": matching,
        "rate": rate,
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# bootstrap machinery shared by cluster_bootstrap() / paired_delta_ci() /
# latency_summary()
# ---------------------------------------------------------------------------


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear interpolation between order statistics -- numpy's default
    method ("R type 7"), the interpolation convention this module's
    bootstrap intervals use throughout (implementation note, beyond the
    brief; see `task-4b-report.md`). `sorted_values` must already be sorted
    ascending and non-empty."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    idx = q * (n - 1)
    lo_idx = math.floor(idx)
    hi_idx = math.ceil(idx)
    if lo_idx == hi_idx:
        return sorted_values[int(idx)]
    frac = idx - lo_idx
    return sorted_values[lo_idx] + (sorted_values[hi_idx] - sorted_values[lo_idx]) * frac


def cluster_bootstrap(
    rows: Sequence[Row],
    statistic: Callable[[Sequence[Row]], float | None],
    family_of: Callable[[Row], str],
    n_resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float | None, float | None]:
    """Percentile-method cluster bootstrap: resample **families**, never
    individual rows -- the same count of families as observed, with
    replacement -- `n_resamples` times, and take the empirical
    `(1-confidence)/2` / `1-(1-confidence)/2` percentiles of `statistic`
    over the resamples.

    Paraphrase siblings are correlated (a model that cannot do nested logic
    fails all three together); resampling individual rows would treat that
    correlation as independent information and report a confidently-too-
    narrow interval. `statistic` receives each resample's rows (all of every
    sampled family's members, concatenated) and returns a float, or `None`
    if the resample cannot support the statistic (e.g. a zero denominator);
    `None` resamples are dropped before taking percentiles.

    Seeded with `random.Random(seed)`, so identical `(rows, seed)` produce
    a byte-identical interval every time -- the caller (the run report) is
    responsible for recording `seed` beside the interval it produced, since
    a different seed is a different, if very similar, answer.

    `(None, None)` when there are no families to resample, or when
    `statistic` returned `None` on every resample.
    """
    buckets: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        buckets[family_of(row)].append(row)
    family_ids = list(buckets)
    n_fam = len(family_ids)
    if n_fam == 0:
        return None, None
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n_resamples):
        sample_ids = [rng.choice(family_ids) for _ in range(n_fam)]
        resampled = [row for fid in sample_ids for row in buckets[fid]]
        value = statistic(resampled)
        if value is not None:
            values.append(value)
    if not values:
        return None, None
    values.sort()
    lo = _percentile(values, (1 - confidence) / 2)
    hi = _percentile(values, 1 - (1 - confidence) / 2)
    return lo, hi


def paired_delta_ci(
    rows_a: Sequence[Row],
    rows_b: Sequence[Row],
    metric: Callable[[Row], float],
    family_of: Callable[[Row], str],
    seed: int,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float | None, float | None, float | None]:
    """Cluster-bootstrap CI on the per-family paired difference of `metric`.

    Sign convention (pinned): `delta` is `mean(metric(row_a) -
    metric(row_b))`, so **A scoring higher than B on `metric` is a positive
    delta**, always, for any metric where higher is better (accuracy,
    exact-match rate, ...). A caller comparing a lower-is-better metric must
    negate `metric` (or read the sign as "positive = A worse") -- this
    function does not know which direction is good for an arbitrary metric.

    `rows_a` and `rows_b` must name the same set of `case_id`s -- the paired
    design this whole layer exploits -- asserted, not silently tolerated.
    Per-case differences are averaged within family first (one number per
    family), so a 5-paraphrase family and a 1-case family both count as one
    family in the resample, matching `cluster_bootstrap`'s own per-family
    weighting; `delta` is the mean of those per-family means.

    `(None, None, None)` when there are no families to resample (`rows_a`
    and `rows_b` both empty).
    """
    a_by_id = {row.case_id: row for row in rows_a}
    b_by_id = {row.case_id: row for row in rows_b}
    assert set(a_by_id) == set(b_by_id), (
        "paired_delta_ci requires rows_a and rows_b to name the same "
        f"case_ids; got {len(a_by_id)} vs {len(b_by_id)} distinct ids"
    )
    diffs_by_family: dict[str, list[float]] = defaultdict(list)
    for case_id, row_a in a_by_id.items():
        row_b = b_by_id[case_id]
        fid = family_of(row_a)
        diffs_by_family[fid].append(metric(row_a) - metric(row_b))
    family_ids = list(diffs_by_family)
    n_fam = len(family_ids)
    if n_fam == 0:
        return None, None, None
    family_means = {
        fid: statistics.fmean(diffs) for fid, diffs in diffs_by_family.items()
    }
    delta = statistics.fmean(family_means.values())
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n_resamples):
        sample_ids = [rng.choice(family_ids) for _ in range(n_fam)]
        values.append(statistics.fmean(family_means[fid] for fid in sample_ids))
    values.sort()
    lo = _percentile(values, (1 - confidence) / 2)
    hi = _percentile(values, 1 - (1 - confidence) / 2)
    return delta, lo, hi


# ---------------------------------------------------------------------------
# mcnemar_exact()
# ---------------------------------------------------------------------------


def mcnemar_exact(
    rows_a: Sequence[Row],
    rows_b: Sequence[Row],
    outcome: Callable[[Row], bool],
) -> dict[str, Any]:
    """Exact two-sided McNemar test over the discordant pairs.

    `p = min(1.0, 2 * P(X <= min(n01, n10)))`, `X ~ Binomial(n01 + n10,
    0.5)`, computed with `math.comb` -- not the chi-square approximation,
    which is exactly wrong where this eval expects to sit: on the order of
    a couple hundred canonical cases with two decent models, the discordant
    count is small, precisely where the approximation is worst.

    `n10` counts `case_id`s where `outcome(row_a)` is true and
    `outcome(row_b)` is false (A right, B wrong only); `n01` the reverse.
    `rows_a`/`rows_b` must be the same length and carry the same
    `case_id`s in the same order -- **asserted, loudly**: this function
    does not re-pair by id itself, because the common caller already has
    two aligned row lists straight off a per-case join, and a caller that
    does not have aligned lists has a bug worth failing on immediately
    rather than silently mis-pairing rows.

    Does **not** filter to canonical cases -- the report owns that filter
    (module docstring); this stays pure over whatever two aligned row
    lists it is given, canonical-only or not.

    Zero discordant pairs -> `p = 1.0`: the models disagreed nowhere, which
    is a genuine (if uninformative) answer, not a fallback.
    """
    assert len(rows_a) == len(rows_b), (
        f"mcnemar_exact requires aligned rows: got {len(rows_a)} rows_a vs "
        f"{len(rows_b)} rows_b"
    )
    assert [r.case_id for r in rows_a] == [r.case_id for r in rows_b], (
        "mcnemar_exact requires rows_a and rows_b to carry the same "
        "case_ids in the same order"
    )
    n10 = n01 = 0
    for row_a, row_b in zip(rows_a, rows_b, strict=True):
        oa, ob = bool(outcome(row_a)), bool(outcome(row_b))
        if oa and not ob:
            n10 += 1
        elif ob and not oa:
            n01 += 1
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    p = min(1.0, 2 * tail / (2**n))
    return {"n01": n01, "n10": n10, "p": p}


# ---------------------------------------------------------------------------
# latency_summary()
# ---------------------------------------------------------------------------


def _nearest_rank(sorted_values: Sequence[float], percentile: float) -> float:
    """Same convention as `aggregate.py`'s own `_nearest_rank`: the
    `ceil(p/100 * n)`-th smallest value, 1-indexed and clamped into range.
    `sorted_values` must already be sorted ascending and non-empty."""
    n = len(sorted_values)
    rank = min(max(math.ceil(percentile / 100 * n), 1), n)
    return sorted_values[rank - 1]


def _median_delta_ci(
    rows_a: Sequence[Row],
    rows_b: Sequence[Row],
    family_of: Callable[[Row], str],
    seed: int,
    n_resamples: int,
    confidence: float,
) -> tuple[float | None, float | None, float | None]:
    """Cluster bootstrap over per-case latency differences, statistic =
    **median** (not mean, unlike `paired_delta_ci`) -- latency is skewed,
    the same reason `latency_summary` reports p50/p95 rather than a mean
    elsewhere in this codebase (`aggregate.py`).

    Sign convention: `delta = median(latency_ms(row_a) - latency_ms(row_b))`
    over every paired case, so **positive means A is slower than B** -- the
    opposite polarity from `paired_delta_ci`'s "positive = A better", since
    lower latency is better. A reader must not assume "positive" means "A
    wins" here.

    Families are resampled (not individual cases): each resample draws
    `n_fam` families with replacement and pools *every* case-level diff
    from the sampled families (a family contributing more diffs than
    another counts proportionally more within a draw, but which families
    are drawn is still what carries the resampling variance).
    """
    a_by_id = {row.case_id: row for row in rows_a}
    b_by_id = {row.case_id: row for row in rows_b}
    assert set(a_by_id) == set(b_by_id), (
        "latency_summary's median_delta_ci requires rows and other to name "
        f"the same case_ids; got {len(a_by_id)} vs {len(b_by_id)} distinct ids"
    )
    diffs_by_family: dict[str, list[float]] = defaultdict(list)
    for case_id, row_a in a_by_id.items():
        row_b = b_by_id[case_id]
        diffs_by_family[family_of(row_a)].append(row_a.latency_ms - row_b.latency_ms)
    family_ids = list(diffs_by_family)
    n_fam = len(family_ids)
    if n_fam == 0:
        return None, None, None
    all_diffs = [d for diffs in diffs_by_family.values() for d in diffs]
    delta = statistics.median(all_diffs)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n_resamples):
        sample_ids = [rng.choice(family_ids) for _ in range(n_fam)]
        pooled = [d for fid in sample_ids for d in diffs_by_family[fid]]
        values.append(statistics.median(pooled))
    values.sort()
    lo = _percentile(values, (1 - confidence) / 2)
    hi = _percentile(values, 1 - (1 - confidence) / 2)
    return delta, lo, hi


def latency_summary(
    rows: Sequence[Row],
    *,
    other: Sequence[Row] | None = None,
    family_of: Callable[[Row], str] | None = None,
    seed: int | None = None,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """p50/p95 latency (nearest-rank, same convention as `aggregate.py`),
    plus `median_delta_ci` -- the confidence interval on the median of
    per-case latency differences against `other`, via the paired family
    bootstrap (see `_median_delta_ci`) -- whenever `other` (the comparison
    side of the pair) is supplied.

    `other`/`family_of`/`seed` are keyword-only and all default to `None`:
    called with just `rows`, this is a single-side p50/p95 summary and
    `median_delta_ci` is `None`; called with `other` (and then `family_of`
    and `seed` become required) it is the full paired comparison. This
    keeps the single-arg shape usable on its own while still letting the
    report get the delta CI out of one call (implementation note, beyond
    the brief's literal single-argument signature -- see `task-4b-report.md`).

    Every value is `None`, never a fabricated `0.0`, when `rows` is empty.
    """
    if not rows:
        return {"p50": None, "p95": None, "median_delta_ci": None}
    values = sorted(row.latency_ms for row in rows)
    p50 = _nearest_rank(values, 50)
    p95 = _nearest_rank(values, 95)
    median_delta_ci = None
    if other is not None:
        assert family_of is not None and seed is not None, (
            "latency_summary requires family_of and seed when other is given"
        )
        median_delta_ci = _median_delta_ci(
            rows, other, family_of, seed, n_resamples, confidence
        )
    return {"p50": p50, "p95": p95, "median_delta_ci": median_delta_ci}
