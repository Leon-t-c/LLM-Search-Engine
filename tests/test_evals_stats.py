"""`stats.py`, the paired-design statistics layer.

Every reference value here is hand-computed in the docstring or the test
body, never taken from the module's own arithmetic -- that is the whole
point of "known answers" (task-4b-brief.md, Step 1). Rows are built with
`_row()`, the same minimal-`Row` style `test_evals_aggregate.py` uses;
family structure is supplied by hand-built `family_of` callables, since
`stats.py` never derives family membership itself (see its module
docstring: a row is canonical for its family iff `family_of(row) ==
row.case_id`).
"""
import random

import pytest

from cert_nlq.evals.scoring import Row
from cert_nlq.evals.stats import (
    _percentile,
    cluster_bootstrap,
    family_accuracy,
    family_consistency,
    latency_summary,
    mcnemar_exact,
    paired_delta_ci,
    wilson,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _row(case_id="c-1", *, e2e_correct=False, latency_ms=0.0, **overrides):
    fields = {
        "case_id": case_id,
        "tags": (),
        "status_expected": "ok",
        "status_actual": "ok",
        "root_correct": None,
        "joins_expected": None,
        "joins_actual": None,
        "groups_recall_hit": None,
        "groups_recall_source": None,
        "fields_expected": None,
        "fields_actual": None,
        "field_tp": None,
        "field_fp": None,
        "field_fn": None,
        "ops_correct": None,
        "ops_total": None,
        "values_correct": None,
        "values_total": None,
        "exact_match": None,
        "refusal_reason_expected": None,
        "refusal_reason_actual": None,
        "clarify_field_expected": None,
        "candidate_hit": None,
        "e2e_correct": e2e_correct,
        "latency_ms": latency_ms,
        "tokens_in": 0,
        "tokens_out": 0,
        "error": None,
    }
    fields.update(overrides)
    return Row(**fields)


def _family_map(canonical_id, paraphrase_ids):
    """A `case_id -> family_id` dict for one family: the canonical id maps
    to itself, every paraphrase id maps to the canonical id -- exactly
    `golden.family`'s own rule."""
    mapping = {canonical_id: canonical_id}
    for pid in paraphrase_ids:
        mapping[pid] = canonical_id
    return mapping


def _mapping_family_of(mapping):
    """A `family_of` callable backed by a `case_id -> family_id` dict."""

    def family_of(row):
        return mapping[row.case_id]

    return family_of


def _case_id_family_of(row):
    """`family_of` when every case is its own (single-member) family."""
    return row.case_id


def _split_family_of(row):
    """`family_of` for `case_id`s shaped `"<family>-<member>"`."""
    return row.case_id.split("-")[0]


def _single_family_of(row):
    """`family_of` that collapses every row into one family, `"c1"`."""
    return "c1"


def _e2e_outcome(row):
    return row.e2e_correct


def _e2e_metric(row):
    return 1.0 if row.e2e_correct else 0.0


# --------------------------------------------------------------------------
# wilson()
# --------------------------------------------------------------------------


def test_wilson_matches_published_reference():
    # 8/10 @ 95% -> approx (0.490, 0.943), hand-computed in task-4b-report.md.
    lo, hi = wilson(8, 10, confidence=0.95)
    assert round(lo, 3) == 0.490
    assert round(hi, 3) == 0.943


def test_wilson_n_zero_is_degenerate():
    assert wilson(0, 0) == (None, None)


def test_wilson_unsupported_confidence_raises():
    with pytest.raises(ValueError):
        wilson(8, 10, confidence=0.975)


def test_wilson_successes_out_of_range_raises():
    with pytest.raises(ValueError):
        wilson(11, 10)


def test_wilson_successes_out_of_range_with_n_zero_raises():
    # n == 0 alone is the legitimate degenerate case (-> (None, None)); a
    # positive successes count against a zero n is not a "no data" case,
    # it's an impossible argument, and must not be swallowed by the n == 0
    # early return.
    with pytest.raises(ValueError):
        wilson(5, 0)


# --------------------------------------------------------------------------
# mcnemar_exact()
# --------------------------------------------------------------------------


def _discordant_rows(n10, n01):
    """`n10` cases where A is right and B is wrong, then `n01` where B is
    right and A is wrong -- aligned, same case_ids, same order."""
    rows_a, rows_b = [], []
    idx = 0
    for _ in range(n10):
        rows_a.append(_row(f"c{idx}", e2e_correct=True))
        rows_b.append(_row(f"c{idx}", e2e_correct=False))
        idx += 1
    for _ in range(n01):
        rows_a.append(_row(f"c{idx}", e2e_correct=False))
        rows_b.append(_row(f"c{idx}", e2e_correct=True))
        idx += 1
    return rows_a, rows_b


def test_mcnemar_hand_computed_reference():
    # n01=2, n10=8 -> two-sided p = 2*(C(10,0)+C(10,1)+C(10,2))/2**10
    #               = 2*(1+10+45)/1024 = 112/1024 = 0.109375
    rows_a, rows_b = _discordant_rows(n10=8, n01=2)
    result = mcnemar_exact(rows_a, rows_b, outcome=_e2e_outcome)
    assert result["n10"] == 8
    assert result["n01"] == 2
    assert result["p"] == pytest.approx(0.109375)
    assert round(result["p"], 3) == 0.109


def test_mcnemar_zero_discordant_is_one():
    rows_a = [_row("c1", e2e_correct=True), _row("c2", e2e_correct=False)]
    rows_b = [_row("c1", e2e_correct=True), _row("c2", e2e_correct=False)]
    result = mcnemar_exact(rows_a, rows_b, outcome=_e2e_outcome)
    assert result == {"n01": 0, "n10": 0, "p": 1.0}


def test_mcnemar_empty_rows_is_one():
    result = mcnemar_exact([], [], outcome=_e2e_outcome)
    assert result == {"n01": 0, "n10": 0, "p": 1.0}


def test_mcnemar_mismatched_case_ids_raises_loudly():
    rows_a = [_row("c1", e2e_correct=True)]
    rows_b = [_row("c2", e2e_correct=True)]
    with pytest.raises(ValueError, match=r"c1.*c2"):
        mcnemar_exact(rows_a, rows_b, outcome=_e2e_outcome)


def test_mcnemar_mismatched_length_raises_loudly():
    rows_a = [_row("c1", e2e_correct=True), _row("c2", e2e_correct=True)]
    rows_b = [_row("c1", e2e_correct=True)]
    with pytest.raises(ValueError):
        mcnemar_exact(rows_a, rows_b, outcome=_e2e_outcome)


def test_mcnemar_does_not_filter_canonical():
    # A paraphrase-shaped case_id is scored exactly like any other -- the
    # function has no opinion on canonical-ness (the report owns that
    # filter, per the module docstring).
    rows_a = [_row("c1", e2e_correct=True), _row("c1-para", e2e_correct=True)]
    rows_b = [_row("c1", e2e_correct=False), _row("c1-para", e2e_correct=False)]
    result = mcnemar_exact(rows_a, rows_b, outcome=_e2e_outcome)
    assert result["n10"] == 2
    assert result["n01"] == 0


# --------------------------------------------------------------------------
# family_accuracy() / family_consistency()
# --------------------------------------------------------------------------


def test_family_all_pass_accuracy_1_consistency_1():
    mapping = _family_map("c1", ["c1-p1", "c1-p2"])
    rows = [
        _row("c1", e2e_correct=True),
        _row("c1-p1", e2e_correct=True),
        _row("c1-p2", e2e_correct=True),
    ]
    family_of = _mapping_family_of(mapping)

    acc = family_accuracy(rows, family_of)
    cons = family_consistency(rows, family_of)
    assert acc == {"families": 1, "all_correct": 1, "rate": 1.0, "skipped": 0}
    assert cons == {
        "paraphrases": 2,
        "matching_canonical": 2,
        "rate": 1.0,
        "skipped": 0,
    }


def test_family_all_fail_accuracy_0_consistency_1():
    # The consistently-wrong family: every member is e2e_correct=False, so
    # accuracy is 0 (nobody got the right answer) but consistency is a
    # perfect 1.0 (every paraphrase agrees with the canonical member) --
    # this is exactly why consistency is never printed alone: "100%
    # consistent" here means "wrong the same way every time", not "good".
    mapping = _family_map("c1", ["c1-p1", "c1-p2"])
    rows = [
        _row("c1", e2e_correct=False),
        _row("c1-p1", e2e_correct=False),
        _row("c1-p2", e2e_correct=False),
    ]
    family_of = _mapping_family_of(mapping)

    acc = family_accuracy(rows, family_of)
    cons = family_consistency(rows, family_of)
    assert acc == {"families": 1, "all_correct": 0, "rate": 0.0, "skipped": 0}
    assert cons == {
        "paraphrases": 2,
        "matching_canonical": 2,
        "rate": 1.0,
        "skipped": 0,
    }


def test_family_canonical_right_four_of_five_paraphrases_worked_example():
    # The brief's own worked example: canonical right, 4/5 paraphrases
    # right -> consistency 0.80.
    paraphrase_ids = [f"c1-p{i}" for i in range(5)]
    mapping = _family_map("c1", paraphrase_ids)
    rows = [_row("c1", e2e_correct=True)]
    rows += [_row(pid, e2e_correct=(i != 4)) for i, pid in enumerate(paraphrase_ids)]
    family_of = _mapping_family_of(mapping)

    cons = family_consistency(rows, family_of)
    assert cons["paraphrases"] == 5
    assert cons["matching_canonical"] == 4
    assert cons["rate"] == pytest.approx(0.80)

    # Not all five members are correct (one paraphrase is wrong), so this
    # family does not count toward family_accuracy.
    acc = family_accuracy(rows, family_of)
    assert acc == {"families": 1, "all_correct": 0, "rate": 0.0, "skipped": 0}


def test_family_canonical_wrong_paraphrases_right():
    mapping = _family_map("c1", ["c1-p1", "c1-p2"])
    rows = [
        _row("c1", e2e_correct=False),
        _row("c1-p1", e2e_correct=True),
        _row("c1-p2", e2e_correct=True),
    ]
    family_of = _mapping_family_of(mapping)

    acc = family_accuracy(rows, family_of)
    cons = family_consistency(rows, family_of)
    assert acc["rate"] == 0.0
    assert cons["rate"] == 0.0


def test_family_accuracy_consistency_zero_rows_are_none_not_zero():
    family_of = _case_id_family_of
    acc = family_accuracy([], family_of)
    cons = family_consistency([], family_of)
    assert acc == {"families": 0, "all_correct": 0, "rate": None, "skipped": 0}
    assert cons == {
        "paraphrases": 0,
        "matching_canonical": 0,
        "rate": None,
        "skipped": 0,
    }


def test_family_single_member_family_is_not_raised_and_not_counted():
    # A family with a canonical row and no paraphrases at all: not an
    # error, just not eligible for either metric.
    family_of = _case_id_family_of
    rows = [_row("c1", e2e_correct=True)]
    acc = family_accuracy(rows, family_of)
    cons = family_consistency(rows, family_of)
    assert acc["rate"] is None
    assert acc["families"] == 0
    assert cons["rate"] is None
    assert cons["paraphrases"] == 0


def test_family_missing_canonical_is_skipped_and_counted():
    mapping = _family_map("c1", ["c1-p1", "c1-p2"])
    rows = [
        _row("c1-p1", e2e_correct=True),
        _row("c1-p2", e2e_correct=False),
    ]
    family_of = _mapping_family_of(mapping)

    acc = family_accuracy(rows, family_of)
    cons = family_consistency(rows, family_of)
    assert acc == {"families": 0, "all_correct": 0, "rate": None, "skipped": 2}
    assert cons == {
        "paraphrases": 0,
        "matching_canonical": 0,
        "rate": None,
        "skipped": 2,
    }


# --------------------------------------------------------------------------
# cluster_bootstrap()
# --------------------------------------------------------------------------


def _rate_statistic(sample_rows):
    if not sample_rows:
        return None
    return sum(1 for r in sample_rows if r.e2e_correct) / len(sample_rows)


def test_cluster_bootstrap_same_seed_twice_is_byte_identical():
    mapping = _family_map("c1", ["c1-p1"])
    mapping.update(_family_map("c2", ["c2-p1"]))
    rows = [
        _row("c1", e2e_correct=True),
        _row("c1-p1", e2e_correct=True),
        _row("c2", e2e_correct=False),
        _row("c2-p1", e2e_correct=True),
    ]
    family_of = _mapping_family_of(mapping)

    first = cluster_bootstrap(rows, _rate_statistic, family_of, n_resamples=2000, seed=13)
    second = cluster_bootstrap(rows, _rate_statistic, family_of, n_resamples=2000, seed=13)
    assert first == second


def test_cluster_bootstrap_empty_rows_returns_none():
    assert cluster_bootstrap([], _rate_statistic, _case_id_family_of) == (None, None)


def test_cluster_bootstrap_single_family_is_degenerate_point_not_raised():
    rows = [_row("c1", e2e_correct=True), _row("c1-p1", e2e_correct=True)]
    family_of = _single_family_of
    lo, hi = cluster_bootstrap(rows, _rate_statistic, family_of, n_resamples=500, seed=1)
    # Only one family exists, so every resample is the same family drawn
    # once -- the interval collapses to the single observed value, not a
    # raise and not a fabricated 0.0.
    assert lo == hi == 1.0


def test_clustering_matters_family_interval_wider_than_case_level():
    # 10 families of 3 perfectly-correlated members each: five families are
    # entirely e2e_correct=True, five are entirely False. The true rate is
    # exactly 0.5, but the *effective* sample size for a rate over
    # correlated triples is 10 (families), not 30 (rows) -- a case-level
    # bootstrap that ignores the correlation treats it as 30 independent
    # draws and reports a narrower, overconfident interval.
    rows = []
    for family_idx in range(10):
        for member in range(3):
            rows.append(
                _row(f"f{family_idx}-{member}", e2e_correct=(family_idx < 5))
            )
    family_of = _split_family_of

    family_lo, family_hi = cluster_bootstrap(
        rows, _rate_statistic, family_of, n_resamples=4000, seed=7
    )

    # The naive case-level bootstrap: resample individual rows (ignoring
    # family membership) the same number of times, same percentile method.
    rng = random.Random(7)
    n = len(rows)
    naive_values = []
    for _ in range(4000):
        sample = [rng.choice(rows) for _ in range(n)]
        naive_values.append(_rate_statistic(sample))
    naive_values.sort()
    naive_lo = _percentile(naive_values, 0.025)
    naive_hi = _percentile(naive_values, 0.975)

    assert (family_hi - family_lo) > (naive_hi - naive_lo)


# --------------------------------------------------------------------------
# paired_delta_ci()
# --------------------------------------------------------------------------


def test_paired_delta_ci_sign_convention_a_better_is_positive():
    # Sign convention (pinned): delta = mean(metric(a) - metric(b)), so A
    # scoring higher than B is a positive delta.
    rows_a = [_row(f"c{i}", e2e_correct=True) for i in range(4)]
    rows_b = [
        _row("c0", e2e_correct=True),
        _row("c1", e2e_correct=True),
        _row("c2", e2e_correct=False),
        _row("c3", e2e_correct=False),
    ]
    metric = _e2e_metric
    family_of = _case_id_family_of

    delta, _, _ = paired_delta_ci(rows_a, rows_b, metric, family_of, seed=1, n_resamples=500)
    assert delta == pytest.approx(0.5)

    # Swapping the arguments flips the sign -- same magnitude, opposite
    # direction, because now B is "A" and does worse than the new "B".
    delta_swapped, _, _ = paired_delta_ci(
        rows_b, rows_a, metric, family_of, seed=1, n_resamples=500
    )
    assert delta_swapped == pytest.approx(-0.5)


def test_paired_delta_ci_mismatched_case_ids_raises_loudly():
    rows_a = [_row("c1", e2e_correct=True)]
    rows_b = [_row("c2", e2e_correct=True)]
    metric = _e2e_metric
    with pytest.raises(ValueError, match=r"c1.*c2"):
        paired_delta_ci(rows_a, rows_b, metric, _case_id_family_of, seed=1)


def test_paired_delta_ci_duplicate_case_id_raises_loudly():
    # A duplicated case_id would otherwise silently vanish into the dict
    # build (the second row winning, the first dropped with no warning) --
    # this must be caught before it can quietly change delta.
    rows_a = [
        _row("c1", e2e_correct=True),
        _row("c1", e2e_correct=False),
    ]
    rows_b = [
        _row("c1", e2e_correct=True),
        _row("c1", e2e_correct=True),
    ]
    with pytest.raises(ValueError, match="c1"):
        paired_delta_ci(rows_a, rows_b, _e2e_metric, _case_id_family_of, seed=1)


def test_paired_delta_ci_empty_rows_returns_none_triple():
    metric = _e2e_metric
    result = paired_delta_ci([], [], metric, _case_id_family_of, seed=1)
    assert result == (None, None, None)


def test_paired_delta_ci_zero_resamples_returns_delta_only():
    rows_a = [_row("c1", e2e_correct=True), _row("c2", e2e_correct=True)]
    rows_b = [_row("c1", e2e_correct=True), _row("c2", e2e_correct=False)]
    delta, lo, hi = paired_delta_ci(
        rows_a, rows_b, _e2e_metric, _case_id_family_of, seed=1, n_resamples=0
    )
    # The point estimate is still computable with zero resamples; the
    # interval is not (there is nothing to build it from), and this must
    # not raise IndexError trying anyway.
    assert delta == pytest.approx(0.5)
    assert lo is None
    assert hi is None


def test_paired_delta_ci_same_seed_twice_is_byte_identical():
    rows_a = [_row(f"c{i}", e2e_correct=(i % 2 == 0)) for i in range(6)]
    rows_b = [_row(f"c{i}", e2e_correct=(i % 3 == 0)) for i in range(6)]
    metric = _e2e_metric
    family_of = _case_id_family_of

    first = paired_delta_ci(rows_a, rows_b, metric, family_of, seed=99, n_resamples=1000)
    second = paired_delta_ci(rows_a, rows_b, metric, family_of, seed=99, n_resamples=1000)
    assert first == second


# --------------------------------------------------------------------------
# latency_summary()
# --------------------------------------------------------------------------


def test_latency_summary_p50_p95_nearest_rank():
    # Same nearest-rank convention as aggregate.py's own _latency: the
    # ceil(p/100*n)-th smallest value, 1-indexed. n=10, values 100..1000.
    rows = [_row(f"c{i}", latency_ms=float((i + 1) * 100)) for i in range(10)]
    summary = latency_summary(rows)
    assert summary["p50"] == 500.0  # ceil(0.5*10)=5th smallest
    assert summary["p95"] == 1000.0  # ceil(0.95*10)=10th smallest
    assert summary["median_delta_ci"] is None


def test_latency_summary_empty_rows_are_all_none():
    summary = latency_summary([])
    assert summary == {"p50": None, "p95": None, "median_delta_ci": None}


def test_latency_summary_median_delta_ci_sign_convention_a_slower_is_positive():
    # Sign convention: delta = median(latency_a - latency_b), so A being
    # slower than B is a positive delta -- the opposite polarity from
    # paired_delta_ci's "positive = A better", since lower latency is
    # better and this function does not flip the sign for that.
    rows_a = [
        _row("c0", latency_ms=100.0),
        _row("c1", latency_ms=110.0),
        _row("c2", latency_ms=120.0),
    ]
    rows_b = [
        _row("c0", latency_ms=90.0),
        _row("c1", latency_ms=95.0),
        _row("c2", latency_ms=100.0),
    ]
    family_of = _case_id_family_of
    summary = latency_summary(
        rows_a, other=rows_b, family_of=family_of, seed=5, n_resamples=500
    )
    delta, _, _ = summary["median_delta_ci"]
    assert delta == pytest.approx(15.0)  # median([10, 15, 20]) == 15

    delta_swapped, _, _ = latency_summary(
        rows_b, other=rows_a, family_of=family_of, seed=5, n_resamples=500
    )["median_delta_ci"]
    assert delta_swapped == pytest.approx(-15.0)


def test_latency_summary_requires_family_of_and_seed_with_other():
    rows_a = [_row("c0", latency_ms=100.0)]
    rows_b = [_row("c0", latency_ms=90.0)]
    with pytest.raises(AssertionError):
        latency_summary(rows_a, other=rows_b)


def test_latency_summary_median_delta_ci_duplicate_case_id_raises_loudly():
    rows_a = [
        _row("c0", latency_ms=100.0),
        _row("c0", latency_ms=200.0),
    ]
    rows_b = [
        _row("c0", latency_ms=90.0),
        _row("c0", latency_ms=90.0),
    ]
    with pytest.raises(ValueError, match="c0"):
        latency_summary(
            rows_a, other=rows_b, family_of=_case_id_family_of, seed=5
        )


def test_latency_summary_median_delta_ci_zero_resamples_returns_delta_only():
    rows_a = [_row("c0", latency_ms=110.0), _row("c1", latency_ms=100.0)]
    rows_b = [_row("c0", latency_ms=90.0), _row("c1", latency_ms=100.0)]
    summary = latency_summary(
        rows_a,
        other=rows_b,
        family_of=_case_id_family_of,
        seed=5,
        n_resamples=0,
    )
    delta, lo, hi = summary["median_delta_ci"]
    assert delta == pytest.approx(10.0)  # median([20, 0]) == 10
    assert lo is None
    assert hi is None
