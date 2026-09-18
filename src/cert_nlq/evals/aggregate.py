"""Turn per-question `Row`s into the §11 metric block.

`aggregate(rows, cases, registry) -> Metrics` is the one entry point. It
never recomputes anything `scoring.score` already decided -- `e2e_correct`,
`exact_match`, `groups_recall_hit`, every field on `Row` is read verbatim --
and it never recomputes what `golden.derive_slices` / `derive_difficulty`
already decide either; both are *called* here, on the case a row joins to
by `case_id`, not duplicated.

**Every number is `{"value": ..., "n": ...}` or `None`.** `None` means "no
row backs this number" -- a slice with zero measurable rows for a metric
must not report `0.0`, which is a claim ("the model got zero of these
right") the data does not support. This rule has no exceptions in this
module; every helper below returns `None` rather than dividing by zero.

**Error rows are excluded from every model metric and counted once, at the
top.** `row.error is not None` means the HTTP call itself failed --
Task 3's `Row` scores that the same as a genuine wrong answer by design, so
an aggregator that does not filter it out silently folds infra flakiness
into the model's own error rate (the frozen decision from Task 3's review,
restated in this module's own tests). The filter happens once, before any
metric or slice bucketing, in `aggregate` itself -- every helper below
receives an already-filtered row list and does not know errors exist.

**The metric block is a pure function of `list[Row]`.** None of the §11
numbers need the case at all -- they read only `Row` fields -- so
`_metric_block` takes rows and nothing else, and is called once for the
whole run and once per bucket of every slicing axis (derived slice,
difficulty, source). Slices overlap by design (`golden.SLICES`'s own
docstring), so a row can land in more than one `by_slice` bucket; summing
`by_slice` counts across slices does not reproduce the total.

Two judgement calls extend past what the brief's file list literally
allows, both recorded in `task-4-report.md`:

1. **`Row.groups_recall_source`** (added to `scoring.py`, additive/
   backward-compatible) records which path produced `groups_recall_hit` --
   `"echo"` (the real measurement, Task 2b's route) or `"proxy"` (Task 3's
   lower-bound heuristic). Without it, `aggregate` would have no way to
   honour "report the two separately, never blended" -- the provenance was
   otherwise lost the moment `score()` returned a `Row`.
2. **Tokens per query report totals only.** `Row.tokens_in`/`tokens_out`
   are already the summed totals across every `usage` record in the
   `Outcome` (Task 3's own `score()`); nothing on `Row` preserves which
   provider call each usage record belonged to, so a genuine per-stage
   split is not derivable here. The brief is explicit that `Row` must not
   be extended for this one (a stage tag on `Row.tokens_in` would just be
   `sum()` again under a different name) -- the per-call detail lives in
   the store's raw outcome JSON, a Task 5 concern.
"""
import math
from collections.abc import Iterable, Sequence
from typing import Any

from ..registry.models import Registry
from .golden import GoldenCase, derive_difficulty, derive_slices
from .scoring import Row

#: One metric entry. `None` when unmeasured; otherwise `value` is the ratio
#: (or, for latency, the percentile) and `n` is the count backing it --
#: printed beside every number so a reader cannot mistake a 2-row slice for
#: a 200-row one.
Metric = dict[str, Any] | None

#: `aggregate`'s return: the ordered §11 block, plus `by_slice` /
#: `by_difficulty` / `by_source`, each a `name -> the same block` mapping.
Metrics = dict[str, Any]

#: Resolution rate needs a live user choosing a candidate or abandoning --
#: nothing a replayed golden set can produce. Always `None`, always with
#: this note, so a reader of the raw dict (not just the rendered report)
#: sees why the trio's middle number is empty rather than missing.
_RESOLUTION_RATE_NOTE = (
    "production metric, from NLEvent -- not measurable on a benchmark replay"
)

_TOKENS_NOTE = (
    "Row carries only per-run totals (tokens_in/tokens_out, already summed "
    "across every usage record by scoring.score()); the per-stage split "
    "needs the usage list itself, which lives in the store's raw outcome "
    "JSON, not on Row -- see this module's docstring."
)


# --------------------------------------------------------------------------
# generic ratio helpers -- every one returns `None` on a zero denominator
# --------------------------------------------------------------------------


def _bool_rate(values: Iterable[bool | None]) -> Metric:
    """`hits / n` over the values that are not `None`; `None` if none are."""
    present = [v for v in values if v is not None]
    n = len(present)
    if n == 0:
        return None
    return {"value": sum(1 for v in present if v) / n, "n": n}


def _rate_over(rows: Sequence[Row], predicate) -> Metric:
    """`hits / len(rows)` -- every row counts, not just the ones where the
    predicate is even applicable (clarification rate: a row that answered
    `ok` is a rate-lowering row, not an excluded one)."""
    n = len(rows)
    if n == 0:
        return None
    return {"value": sum(1 for r in rows if predicate(r)) / n, "n": n}


def _sum_ratio(pairs: Iterable[tuple[int | None, int | None]]) -> Metric:
    """`sum(numerators) / sum(denominators)`, skipping any pair with a
    `None` half -- the micro-average behind field/join P-R, operator
    accuracy, and value accuracy alike."""
    num = den = 0
    for numerator, denominator in pairs:
        if numerator is None or denominator is None:
            continue
        num += numerator
        den += denominator
    if den == 0:
        return None
    return {"value": num / den, "n": den}


# --------------------------------------------------------------------------
# §11 items 1-2: root accuracy, join precision/recall
# --------------------------------------------------------------------------


def _root_accuracy(rows: Sequence[Row]) -> Metric:
    return _bool_rate(r.root_correct for r in rows)


def _join_pr(rows: Sequence[Row]) -> tuple[Metric, Metric]:
    """Set-based, micro-averaged over every row that has a join answer to
    compare -- every `ok`-kind row, whether or not it actually names a
    join: a joinless row contributes `(tp=0, fp=0, fn=0)`, which adds
    nothing to either ratio, so it is harmless to include rather than
    re-deriving the `join` slice here to filter it out."""
    tp_fp: list[tuple[int, int]] = []
    tp_fn: list[tuple[int, int]] = []
    for row in rows:
        if row.joins_expected is None or row.joins_actual is None:
            continue
        expected, actual = set(row.joins_expected), set(row.joins_actual)
        tp = len(expected & actual)
        tp_fp.append((tp, tp + len(actual - expected)))
        tp_fn.append((tp, tp + len(expected - actual)))
    return _sum_ratio(tp_fp), _sum_ratio(tp_fn)


# --------------------------------------------------------------------------
# §11 item 3: group-selection recall, echoed and proxy reported separately
# --------------------------------------------------------------------------


def _groups_recall(rows: Sequence[Row], source: str) -> Metric:
    bucket = [r for r in rows if r.groups_recall_source == source]
    return _bool_rate(r.groups_recall_hit for r in bucket)


# --------------------------------------------------------------------------
# §11 item 4 (headline): field selection precision/recall
# --------------------------------------------------------------------------


def _field_pr(rows: Sequence[Row]) -> tuple[Metric, Metric]:
    """Micro over rows where `root_correct is True` -- an ok case that
    routed to the wrong root, or didn't route at all, contributes nothing
    to the headline number (it is already reflected in root accuracy).

    A root-correct, pure-aggregate row (no gold condition field at all,
    e.g. a bare `count(*)`) has `field_tp = field_fp = field_fn = 0` --
    `scoring.score` sets all three together, never one without the others
    -- so it contributes `(0, 0)` to both ratios: no phantom true positive,
    no phantom miss. `measured` already guarantees `field_tp is not None`,
    and `score()` guarantees `field_fp`/`field_fn` are not `None` whenever
    `field_tp` isn't, so no `or 0` coercion is needed here -- a future
    change that breaks that invariant should raise, not silently score a
    `None` as zero."""
    measured = [r for r in rows if r.root_correct is True and r.field_tp is not None]
    precision = _sum_ratio((r.field_tp, r.field_tp + r.field_fp) for r in measured)
    recall = _sum_ratio((r.field_tp, r.field_tp + r.field_fn) for r in measured)
    return precision, recall


# --------------------------------------------------------------------------
# §11 item 5: operator accuracy
# --------------------------------------------------------------------------


def _operator_accuracy(rows: Sequence[Row]) -> Metric:
    return _sum_ratio((r.ops_correct, r.ops_total) for r in rows)


# --------------------------------------------------------------------------
# §11 item 6: refusal precision/recall, per reason
# --------------------------------------------------------------------------


def _refusal_pr(rows: Sequence[Row]) -> dict[str, dict[str, Metric]]:
    """Precision's denominator is "refusal-returned" rows (actual reason ==
    this one); recall's is "refusal-expected" rows (gold reason == this
    one) -- exactly the brief's two row sets. Reasons are whatever appears
    in either column across `rows`, not the full closed taxonomy: a reason
    nobody expected and nobody returned in this run has nothing to report."""
    reasons = sorted(
        {r.refusal_reason_expected for r in rows if r.refusal_reason_expected is not None}
        | {r.refusal_reason_actual for r in rows if r.refusal_reason_actual is not None}
    )
    result: dict[str, dict[str, Metric]] = {}
    for reason in reasons:
        returned = [r for r in rows if r.refusal_reason_actual == reason]
        expected = [r for r in rows if r.refusal_reason_expected == reason]
        precision = (
            {"value": sum(1 for r in returned if r.refusal_reason_expected == reason)
             / len(returned), "n": len(returned)}
            if returned
            else None
        )
        recall = (
            {"value": sum(1 for r in expected if r.refusal_reason_actual == reason)
             / len(expected), "n": len(expected)}
            if expected
            else None
        )
        result[reason] = {"precision": precision, "recall": recall}
    return result


# --------------------------------------------------------------------------
# §11 item 7: exact-match rate; e2e_correct rate (the primary outcome)
# --------------------------------------------------------------------------


def _exact_match_rate(rows: Sequence[Row]) -> Metric:
    return _bool_rate(r.exact_match for r in rows)


def _e2e_correct_rate(rows: Sequence[Row]) -> Metric:
    """Over every row, not just `ok`-kind ones -- `e2e_correct` is defined
    per-kind already (`end_to_end_correct`) and is never `None`, so the
    denominator is the whole bucket."""
    return _rate_over(rows, lambda r: r.e2e_correct)


# --------------------------------------------------------------------------
# §11.0: the clarification trio, always rendered together
# --------------------------------------------------------------------------


def _clarification(rows: Sequence[Row]) -> dict[str, Any]:
    return {
        "clarification_rate": _rate_over(rows, lambda r: r.status_actual == "clarify"),
        "resolution_rate": {"value": None, "note": _RESOLUTION_RATE_NOTE},
        "candidate_recall": _bool_rate(r.candidate_hit for r in rows),
    }


# --------------------------------------------------------------------------
# §11 item 9: latency p50/p95, nearest-rank
# --------------------------------------------------------------------------


def _nearest_rank(sorted_values: Sequence[float], percentile: float) -> float:
    """Standard nearest-rank: the `ceil(p/100 * n)`-th smallest value,
    1-indexed and clamped into range. `sorted_values` must already be
    sorted ascending and non-empty."""
    n = len(sorted_values)
    rank = min(max(math.ceil(percentile / 100 * n), 1), n)
    return sorted_values[rank - 1]


def _latency(rows: Sequence[Row]) -> tuple[Metric, Metric]:
    if not rows:
        return None, None
    values = sorted(r.latency_ms for r in rows)
    n = len(values)
    p50 = {"value": _nearest_rank(values, 50), "n": n}
    p95 = {"value": _nearest_rank(values, 95), "n": n}
    return p50, p95


# --------------------------------------------------------------------------
# §11 item 10: tokens per query (totals only -- see module docstring)
# --------------------------------------------------------------------------


def _tokens_per_query(rows: Sequence[Row]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        tokens_in = tokens_out = None
    else:
        tokens_in = {"value": sum(r.tokens_in for r in rows) / n, "n": n}
        tokens_out = {"value": sum(r.tokens_out for r in rows) / n, "n": n}
    return {"tokens_in": tokens_in, "tokens_out": tokens_out, "note": _TOKENS_NOTE}


# --------------------------------------------------------------------------
# §11 item 11 (last, secondary): value-resolution accuracy
# --------------------------------------------------------------------------


def _value_accuracy(rows: Sequence[Row]) -> Metric:
    return _sum_ratio((r.values_correct, r.values_total) for r in rows)


# --------------------------------------------------------------------------
# the block, and the top-level entry point
# --------------------------------------------------------------------------


def _metric_block(rows: Sequence[Row]) -> Metrics:
    """The §11 block, in §11's exact order, for one bucket of (already
    error-filtered) rows. Pure in `rows` -- no case, no registry -- so it is
    the one function called both for the whole run and for every slice /
    difficulty / source bucket below."""
    join_precision, join_recall = _join_pr(rows)
    field_precision, field_recall = _field_pr(rows)
    latency_p50, latency_p95 = _latency(rows)
    return {
        "root_accuracy": _root_accuracy(rows),
        "join_precision": join_precision,
        "join_recall": join_recall,
        "groups_recall_echoed": _groups_recall(rows, "echo"),
        "groups_recall_proxy": _groups_recall(rows, "proxy"),
        "field_precision": field_precision,
        "field_recall": field_recall,
        "operator_accuracy": _operator_accuracy(rows),
        "refusal": _refusal_pr(rows),
        "exact_match_rate": _exact_match_rate(rows),
        "e2e_correct_rate": _e2e_correct_rate(rows),
        "clarification": _clarification(rows),
        "latency_p50": latency_p50,
        "latency_p95": latency_p95,
        "tokens_per_query": _tokens_per_query(rows),
        "value_accuracy": _value_accuracy(rows),
    }


def aggregate(
    rows: Sequence[Row], cases: Sequence[GoldenCase], registry: Registry
) -> Metrics:
    """Rows -> the §11 metric block, plus the same block per derived slice,
    per difficulty, and per source.

    Rows join to cases by `case_id`; a row whose `case_id` names no case in
    `cases` is dropped from every slice/difficulty/source bucket (it cannot
    be placed) but still counts in the overall block, and is counted, not
    silently lost, via `errors.unmatched_case_rows` -- the run happened even
    if this call's `cases` list is a stale or partial view of the golden set
    that produced it, and that must be visible rather than read as a
    quietly shrunk axis.

    Error rows (`row.error is not None`) are filtered out here, once, before
    any other computation -- see the module docstring.
    """
    total = len(rows)
    error_rows = [r for r in rows if r.error is not None]
    ok_rows = [r for r in rows if r.error is None]

    case_by_id = {case.id: case for case in cases}
    by_slice: dict[str, list[Row]] = {}
    by_difficulty: dict[str, list[Row]] = {}
    by_source: dict[str, list[Row]] = {}
    unmatched = 0
    for row in ok_rows:
        case = case_by_id.get(row.case_id)
        if case is None:
            # A stale or partial `cases` snapshot must not silently shrink
            # every per-axis n -- counted here so it is visible beside
            # `error_count`, never just absent from the by_slice/
            # by_difficulty/by_source tables with no trace.
            unmatched += 1
            continue
        for slice_name in derive_slices(case, registry):
            by_slice.setdefault(slice_name, []).append(row)
        by_difficulty.setdefault(derive_difficulty(case, registry), []).append(row)
        by_source.setdefault(case.source, []).append(row)

    metrics: Metrics = {
        "errors": {
            "error_count": len(error_rows),
            "error_rate": (len(error_rows) / total) if total else None,
            "n": total,
            #: Rows whose `case_id` matched no case in `cases` -- always `0`
            #: in a healthy run (a complete, current golden-set snapshot).
            #: Excluded from every by_slice/by_difficulty/by_source bucket
            #: for the same reason error rows are: axis n's + this count
            #: must reconcile with the overall (non-error) row count.
            "unmatched_case_rows": unmatched,
        },
        **_metric_block(ok_rows),
    }

    metrics["by_slice"] = {
        name: _metric_block(bucket) for name, bucket in sorted(by_slice.items())
    }
    metrics["by_difficulty"] = {
        name: _metric_block(bucket) for name, bucket in sorted(by_difficulty.items())
    }
    metrics["by_source"] = {
        name: _metric_block(bucket) for name, bucket in sorted(by_source.items())
    }
    return metrics
