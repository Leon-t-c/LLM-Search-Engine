"""`aggregate()`, exercised against the widget fixture only.

Rows are built directly with `_row()` -- known answers hand-picked so every
ratio in the §11 block can be checked against arithmetic done by hand in the
test itself, never against the module's own computation. Cases (where a
test needs one, for the slice/difficulty/source axes) are built with
`_ok_case()`/`_clarify_case()`/`_refusal_case()`, the same style
`test_evals_scoring.py` uses.
"""
from cert_nlq.evals.aggregate import aggregate
from cert_nlq.evals.golden import GoldenCase
from cert_nlq.evals.scoring import Row

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _row(case_id="c-1", **overrides):
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
        "e2e_correct": False,
        "latency_ms": 0.0,
        "tokens_in": 0,
        "tokens_out": 0,
        "error": None,
    }
    fields.update(overrides)
    return Row(**fields)


def _cond(field, op, value=None):
    node = {"field": field, "op": op}
    if value is not None:
        node["value"] = value
    return node


def _and(*children):
    return {"combinator": "AND", "children": list(children)}


def _ok_case(case_id, payload, *, source="hand-written", paraphrase_of=None, tags=()):
    return GoldenCase.model_validate({
        "id": case_id,
        "question": "a question",
        "tags": list(tags),
        "source": source,
        "paraphrase_of": paraphrase_of,
        "expect": {"kind": "ok", "payload": payload},
    })


def _clarify_case(case_id, field, *, source="hand-written"):
    return GoldenCase.model_validate({
        "id": case_id,
        "question": "a question",
        "source": source,
        "expect": {"kind": "clarify", "field": field},
    })


def _refusal_case(case_id, reason, *, source="hand-written"):
    return GoldenCase.model_validate({
        "id": case_id,
        "question": "a question",
        "source": source,
        "expect": {"kind": "refusal", "reason": reason},
    })


# --------------------------------------------------------------------------
# root accuracy
# --------------------------------------------------------------------------


def test_root_accuracy_is_mean_over_measured_rows(registry):
    rows = [
        _row("c1", root_correct=True),
        _row("c2", root_correct=True),
        _row("c3", root_correct=False),
        _row("c4", root_correct=None),  # not measured -- e.g. a clarify row
    ]
    metrics = aggregate(rows, [], registry)
    assert metrics["root_accuracy"] == {"value": 2 / 3, "n": 3}


def test_root_accuracy_is_none_with_no_measured_rows(registry):
    rows = [_row("c1", root_correct=None)]
    metrics = aggregate(rows, [], registry)
    assert metrics["root_accuracy"] is None


# --------------------------------------------------------------------------
# join precision/recall
# --------------------------------------------------------------------------


def test_join_precision_recall_are_set_based_and_micro_averaged(registry):
    rows = [
        # tp=1, fp=1 (predicted "shipment" extra), fn=0
        _row("c1", joins_expected=("vendor",), joins_actual=("vendor", "shipment")),
        # tp=1, fp=0, fn=1 (missed "shipment")
        _row("c2", joins_expected=("vendor", "shipment"), joins_actual=("vendor",)),
        # a joinless row: contributes (0, 0) to both ratios, harmlessly
        _row("c3", joins_expected=(), joins_actual=()),
        # not an ok-kind row -- excluded entirely
        _row("c4", joins_expected=None, joins_actual=None),
    ]
    metrics = aggregate(rows, [], registry)
    # sum tp=2, sum fp=1, sum fn=1
    assert metrics["join_precision"] == {"value": 2 / 3, "n": 3}
    assert metrics["join_recall"] == {"value": 2 / 3, "n": 3}


# --------------------------------------------------------------------------
# group-selection recall: echoed and proxy, never blended
# --------------------------------------------------------------------------


def test_groups_recall_echoed_and_proxy_are_reported_separately(registry):
    rows = [
        _row("c1", groups_recall_hit=True, groups_recall_source="echo"),
        _row("c2", groups_recall_hit=True, groups_recall_source="echo"),
        _row("c3", groups_recall_hit=False, groups_recall_source="echo"),
        _row("c4", groups_recall_hit=True, groups_recall_source="proxy"),
        _row("c5", groups_recall_hit=None, groups_recall_source=None),
    ]
    metrics = aggregate(rows, [], registry)
    assert metrics["groups_recall_echoed"] == {"value": 2 / 3, "n": 3}
    assert metrics["groups_recall_proxy"] == {"value": 1.0, "n": 1}
    # No blended third number anywhere in the block.
    assert "groups_recall" not in metrics
    assert "groups_recall_hit" not in metrics


def test_groups_recall_is_none_when_that_path_never_fired(registry):
    rows = [_row("c1", groups_recall_hit=True, groups_recall_source="echo")]
    metrics = aggregate(rows, [], registry)
    assert metrics["groups_recall_proxy"] is None


# --------------------------------------------------------------------------
# field selection precision/recall (the headline)
# --------------------------------------------------------------------------


def test_field_pr_is_micro_over_rows_with_correct_root(registry):
    rows = [
        _row("c1", root_correct=True, field_tp=3, field_fp=1, field_fn=2),
        _row("c2", root_correct=True, field_tp=2, field_fp=0, field_fn=0),
        # wrong root: excluded from the headline entirely, however its
        # field_tp/fp/fn happen to be populated.
        _row("c3", root_correct=False, field_tp=5, field_fp=5, field_fn=5),
        # not applicable to this case kind at all.
        _row("c4", root_correct=None, field_tp=None, field_fp=None, field_fn=None),
    ]
    metrics = aggregate(rows, [], registry)
    assert metrics["field_precision"] == {"value": 5 / 6, "n": 6}
    assert metrics["field_recall"] == {"value": 5 / 7, "n": 7}


def test_field_pr_pure_aggregate_row_contributes_zero_zero(registry):
    """A root-correct row with no gold condition field at all (a bare
    `count(*)`, say) scores field_tp=field_fp=field_fn=0 in `scoring.py`.
    It must contribute (0, 0) to the headline ratio -- no phantom true
    positive inflating the numerator, no phantom miss inflating the
    denominator -- so mixing it into an otherwise-perfect bucket leaves the
    ratio unchanged."""
    rows = [
        _row("c1", root_correct=True, field_tp=2, field_fp=0, field_fn=0),
        _row("c2", root_correct=True, field_tp=0, field_fp=0, field_fn=0),
    ]
    metrics = aggregate(rows, [], registry)
    assert metrics["field_precision"] == {"value": 1.0, "n": 2}
    assert metrics["field_recall"] == {"value": 1.0, "n": 2}


# --------------------------------------------------------------------------
# operator accuracy
# --------------------------------------------------------------------------


def test_operator_accuracy_sums_correct_over_total(registry):
    rows = [
        _row("c1", ops_correct=2, ops_total=3),
        _row("c2", ops_correct=1, ops_total=1),
        _row("c3", ops_correct=None, ops_total=None),
    ]
    metrics = aggregate(rows, [], registry)
    assert metrics["operator_accuracy"] == {"value": 3 / 4, "n": 4}


# --------------------------------------------------------------------------
# refusal precision/recall, per reason
# --------------------------------------------------------------------------


def test_refusal_precision_recall_per_reason(registry):
    rows = [
        # correctly refused for the right reason
        _row(
            "c1",
            refusal_reason_expected="not_a_query",
            refusal_reason_actual="not_a_query",
        ),
        # expected not_a_query, actually refused for a different reason
        _row(
            "c2",
            refusal_reason_expected="not_a_query",
            refusal_reason_actual="field_not_in_schema",
        ),
        # an ok-expected case that got refused (unexpectedly) as not_a_query
        _row("c3", refusal_reason_expected=None, refusal_reason_actual="not_a_query"),
    ]
    metrics = aggregate(rows, [], registry)
    refusal = metrics["refusal"]

    naq = refusal["not_a_query"]
    assert naq["precision"] == {"value": 1 / 2, "n": 2}  # returned: c1, c3
    assert naq["recall"] == {"value": 1 / 2, "n": 2}  # expected: c1, c2

    fnis = refusal["field_not_in_schema"]
    assert fnis["precision"] == {"value": 0.0, "n": 1}  # returned: c2, wrong
    assert fnis["recall"] is None  # never expected -- no data, not a 0


def test_refusal_block_is_empty_dict_with_no_refusal_rows(registry):
    metrics = aggregate([_row("c1")], [], registry)
    assert metrics["refusal"] == {}


# --------------------------------------------------------------------------
# exact-match rate; e2e_correct rate (the primary outcome)
# --------------------------------------------------------------------------


def test_exact_match_and_e2e_correct_rates(registry):
    rows = [
        _row("c1", exact_match=True, e2e_correct=True),
        _row("c2", exact_match=True, e2e_correct=True),
        _row("c3", exact_match=False, e2e_correct=False),
        # a clarify-kind row: exact_match not applicable, e2e_correct still is
        _row("c4", exact_match=None, e2e_correct=True),
    ]
    metrics = aggregate(rows, [], registry)
    assert metrics["exact_match_rate"] == {"value": 2 / 3, "n": 3}
    assert metrics["e2e_correct_rate"] == {"value": 3 / 4, "n": 4}


# --------------------------------------------------------------------------
# the clarification trio -- always rendered together
# --------------------------------------------------------------------------


def test_clarification_trio_values(registry):
    rows = [
        _row("c1", status_actual="clarify", candidate_hit=True),
        _row("c2", status_actual="clarify", candidate_hit=False),
        _row("c3", status_actual="ok", candidate_hit=None),
    ]
    metrics = aggregate(rows, [], registry)
    trio = metrics["clarification"]
    assert trio["clarification_rate"] == {"value": 2 / 3, "n": 3}
    assert trio["candidate_recall"] == {"value": 1 / 2, "n": 2}
    assert trio["resolution_rate"] == {
        "value": None,
        "note": "production metric, from NLEvent -- not measurable on a benchmark replay",
    }


def test_clarification_trio_keys_always_coexist_even_empty(registry):
    metrics = aggregate([], [], registry)
    trio = metrics["clarification"]
    assert set(trio) == {"clarification_rate", "resolution_rate", "candidate_recall"}
    assert trio["clarification_rate"] is None
    assert trio["candidate_recall"] is None
    assert trio["resolution_rate"]["value"] is None  # present, not omitted


# --------------------------------------------------------------------------
# latency p50/p95 -- nearest-rank, hand-computed
# --------------------------------------------------------------------------


def test_latency_percentiles_nearest_rank_on_ten_values(registry):
    # sorted: 1..10. p50 -> ceil(0.5*10)=5th smallest -> 5.
    # p95 -> ceil(0.95*10)=ceil(9.5)=10th smallest -> 10 (the max).
    values = [5, 1, 9, 3, 7, 2, 8, 4, 6, 10]
    rows = [_row(f"c{i}", latency_ms=float(v)) for i, v in enumerate(values)]
    metrics = aggregate(rows, [], registry)
    assert metrics["latency_p50"] == {"value": 5.0, "n": 10}
    assert metrics["latency_p95"] == {"value": 10.0, "n": 10}


def test_latency_is_none_with_no_rows(registry):
    metrics = aggregate([], [], registry)
    assert metrics["latency_p50"] is None
    assert metrics["latency_p95"] is None


# --------------------------------------------------------------------------
# tokens per query -- totals only, with the per-stage note
# --------------------------------------------------------------------------


def test_tokens_per_query_reports_totals_and_a_note(registry):
    rows = [
        _row("c1", tokens_in=100, tokens_out=10),
        _row("c2", tokens_in=200, tokens_out=20),
        _row("c3", tokens_in=300, tokens_out=30),
    ]
    metrics = aggregate(rows, [], registry)
    tokens = metrics["tokens_per_query"]
    assert tokens["tokens_in"] == {"value": 200.0, "n": 3}
    assert tokens["tokens_out"] == {"value": 20.0, "n": 3}
    assert "usage list" in tokens["note"] and "store's raw outcome JSON" in tokens["note"]


def test_tokens_per_query_is_none_with_no_rows_but_note_stays(registry):
    metrics = aggregate([], [], registry)
    tokens = metrics["tokens_per_query"]
    assert tokens["tokens_in"] is None
    assert tokens["tokens_out"] is None
    assert tokens["note"]


# --------------------------------------------------------------------------
# value-resolution accuracy -- last, secondary
# --------------------------------------------------------------------------


def test_value_accuracy(registry):
    rows = [
        _row("c1", values_correct=4, values_total=5),
        _row("c2", values_correct=1, values_total=1),
    ]
    metrics = aggregate(rows, [], registry)
    assert metrics["value_accuracy"] == {"value": 5 / 6, "n": 6}


def test_metric_block_key_order_matches_section_11(registry):
    metrics = aggregate([_row("c1")], [], registry)
    expected_order = [
        "errors",
        "root_accuracy",
        "join_precision",
        "join_recall",
        "groups_recall_echoed",
        "groups_recall_proxy",
        "field_precision",
        "field_recall",
        "operator_accuracy",
        "refusal",
        "exact_match_rate",
        "e2e_correct_rate",
        "clarification",
        "latency_p50",
        "latency_p95",
        "tokens_per_query",
        "value_accuracy",
        "by_slice",
        "by_difficulty",
        "by_source",
    ]
    assert list(metrics) == expected_order


# --------------------------------------------------------------------------
# error rows: excluded from every model metric, counted once at the top
# --------------------------------------------------------------------------


def test_error_rows_are_excluded_from_metrics_but_counted(registry):
    rows = [
        _row("c1", e2e_correct=True, root_correct=True, error=None),
        _row("c2", e2e_correct=True, root_correct=True, error=None),
        _row("c3", e2e_correct=False, root_correct=False, error=None),
        _row("c4", e2e_correct=False, root_correct=False, error="timeout"),
        _row("c5", e2e_correct=False, root_correct=False, error="502"),
    ]
    metrics = aggregate(rows, [], registry)
    # No `cases` passed here (this test is about the error filter, not the
    # case join) -- so all 3 non-error rows are, correctly, unmatched too.
    assert metrics["errors"] == {
        "error_count": 2,
        "error_rate": 2 / 5,
        "n": 5,
        "unmatched_case_rows": 3,
    }
    # Only the three non-error rows feed the accuracy numbers.
    assert metrics["root_accuracy"] == {"value": 2 / 3, "n": 3}
    assert metrics["e2e_correct_rate"] == {"value": 2 / 3, "n": 3}


def test_error_line_is_none_rate_with_no_rows_at_all(registry):
    metrics = aggregate([], [], registry)
    assert metrics["errors"] == {
        "error_count": 0,
        "error_rate": None,
        "n": 0,
        "unmatched_case_rows": 0,
    }


# --------------------------------------------------------------------------
# by_slice / by_difficulty / by_source: the same block, joined through cases
# --------------------------------------------------------------------------


def test_by_slice_carries_n_and_overlapping_cases_land_in_both_buckets(registry):
    # A single condition and a join together: both "multi-condition"-free
    # simple filter rules it out of "simple-filter" (it does other work),
    # but it is squarely "join". Add a second, plain single-condition case
    # for "simple-filter".
    join_payload = {
        "root": "widget",
        "where": _and(_cond("widget.price", ">", 100)),
        "join": ["shipment"],
        "columns": ["shipment.amount"],
    }
    simple_payload = {
        "root": "widget",
        "where": _and(_cond("widget.price", ">", 100)),
    }
    cases = [
        _ok_case("g1", join_payload),
        _ok_case("g2", simple_payload),
    ]
    rows = [
        _row("g1", root_correct=True),
        _row("g2", root_correct=True),
    ]
    metrics = aggregate(rows, cases, registry)

    assert metrics["by_slice"]["join"]["root_accuracy"] == {"value": 1.0, "n": 1}
    assert metrics["by_slice"]["simple-filter"]["root_accuracy"] == {
        "value": 1.0,
        "n": 1,
    }
    # g1 is not a simple-filter case (it also joins); g2 never joins, so the
    # two slices do not share a row despite both having n == 1 here.
    assert metrics["by_slice"]["simple-filter"]["root_accuracy"]["n"] == 1


def test_by_difficulty_and_by_source_partition_cleanly(registry):
    easy_payload = {"root": "widget", "where": _and(_cond("widget.price", ">", 100))}
    cases = [
        _ok_case("g1", easy_payload, source="hand-written"),
        _ok_case("g2", easy_payload, source="live-capture"),
    ]
    rows = [_row("g1", root_correct=True), _row("g2", root_correct=False)]
    metrics = aggregate(rows, cases, registry)

    assert metrics["by_difficulty"]["easy"]["root_accuracy"] == {"value": 0.5, "n": 2}
    assert metrics["by_source"]["hand-written"]["root_accuracy"] == {
        "value": 1.0,
        "n": 1,
    }
    assert metrics["by_source"]["live-capture"]["root_accuracy"] == {
        "value": 0.0,
        "n": 1,
    }


def test_empty_slice_metric_is_none_not_zero(registry):
    """A slice with rows present, but none measurable for a given metric,
    must report `None` -- not `0.0`, which would claim the model got zero
    of something it was never asked."""
    refusal_payload_case = _refusal_case("g1", "not_a_query")
    rows = [_row("g1", root_correct=True, ops_correct=None, ops_total=None)]
    metrics = aggregate(rows, [refusal_payload_case], registry)
    assert metrics["by_slice"]["refusal"]["operator_accuracy"] is None
    assert metrics["by_slice"]["refusal"]["field_precision"] is None


def test_row_with_unmatched_case_id_is_dropped_from_axes_not_from_totals(registry):
    rows = [_row("ghost", root_correct=True, e2e_correct=True)]
    metrics = aggregate(rows, [], registry)  # no cases at all
    assert metrics["root_accuracy"] == {"value": 1.0, "n": 1}
    assert metrics["by_slice"] == {}
    assert metrics["by_difficulty"] == {}
    assert metrics["by_source"] == {}


def test_unmatched_case_id_is_counted_loudly_not_silently_dropped(registry):
    """A stale/partial `cases` snapshot must not quietly shrink every
    per-axis n. One row joins cleanly; one row's case_id is bogus. The
    bogus row is counted in `unmatched_case_rows`, present in the overall
    block, and absent from every axis -- and the axis n's plus the
    unmatched count reconcile with the overall (non-error) row count."""
    payload = {"root": "widget", "where": _and(_cond("widget.price", ">", 100))}
    cases = [_ok_case("real", payload)]
    rows = [
        _row("real", root_correct=True, e2e_correct=True),
        _row("bogus-id", root_correct=False, e2e_correct=False),
    ]
    metrics = aggregate(rows, cases, registry)

    assert metrics["errors"]["unmatched_case_rows"] == 1
    # Still in the overall block: both rows feed root_accuracy.
    assert metrics["root_accuracy"] == {"value": 0.5, "n": 2}
    # Absent from every axis: only the matched row reaches any bucket.
    axis_n = metrics["by_slice"]["simple-filter"]["root_accuracy"]["n"]
    assert axis_n == 1
    assert metrics["by_source"]["hand-written"]["root_accuracy"]["n"] == 1
    # Reconciliation: axis n (matched) + unmatched == overall non-error n.
    assert axis_n + metrics["errors"]["unmatched_case_rows"] == 2


# --------------------------------------------------------------------------
# a paraphrase family's rows land in the same family (source) bucket
# --------------------------------------------------------------------------


def test_paraphrase_family_rows_join_through_cases_into_one_bucket(registry):
    payload = {"root": "widget", "where": _and(_cond("widget.price", ">", 100))}
    original = _ok_case("g1", payload, source="hand-written")
    paraphrase_1 = _ok_case("g1-p1", payload, source="paraphrase", paraphrase_of="g1")
    paraphrase_2 = _ok_case("g1-p2", payload, source="paraphrase", paraphrase_of="g1")
    cases = [original, paraphrase_1, paraphrase_2]
    rows = [
        _row("g1", root_correct=True),
        _row("g1-p1", root_correct=True),
        _row("g1-p2", root_correct=False),
    ]
    metrics = aggregate(rows, cases, registry)

    assert metrics["by_source"]["hand-written"]["root_accuracy"] == {
        "value": 1.0,
        "n": 1,
    }
    # Both paraphrases (different case_ids, same family via paraphrase_of)
    # land together in the "paraphrase" source bucket.
    assert metrics["by_source"]["paraphrase"]["root_accuracy"] == {
        "value": 0.5,
        "n": 2,
    }
