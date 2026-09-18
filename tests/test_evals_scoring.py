"""The scoring core, exercised against the widget fixture only.

Every case here builds a `GoldenCase` directly (skipping `load_golden`'s
registry cross-checks, which are Task 2's concern) and an `Outcome` by hand,
so each test isolates exactly the one scoring rule it is about.
"""
from cert_nlq.evals.golden import GoldenCase
from cert_nlq.evals.scoring import Outcome, canonical, score

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _ok_case(case_id, payload, tags=("selection",)):
    return GoldenCase.model_validate({
        "id": case_id,
        "question": "a question",
        "tags": list(tags),
        "expect": {"kind": "ok", "payload": payload},
    })


def _clarify_case(case_id, field, tags=("clarify",)):
    return GoldenCase.model_validate({
        "id": case_id,
        "question": "a question",
        "tags": list(tags),
        "expect": {"kind": "clarify", "field": field},
    })


def _refusal_case(case_id, reason, tags=None):
    return GoldenCase.model_validate({
        "id": case_id,
        "question": "a question",
        "tags": list(tags or [f"refusal:{reason}"]),
        "expect": {"kind": "refusal", "reason": reason},
    })


def _outcome(**overrides):
    return Outcome(**overrides)


def _cond(field, op, value=None, value2=None):
    node = {"field": field, "op": op}
    if value is not None:
        node["value"] = value
    if value2 is not None:
        node["value2"] = value2
    return node


def _and(*children):
    return {"combinator": "AND", "children": list(children)}


# --------------------------------------------------------------------------
# canonical(): key order, child-order commutativity, type coercion, aliases
# --------------------------------------------------------------------------


def test_canonical_ignores_key_order_and_and_or_child_order(registry):
    root = registry.root("widget")
    gold = {
        "root": "widget",
        "where": _and(
            _cond("widget.status", "=", "A"),
            _cond("widget.region", "=", "N"),
        ),
    }
    reordered = {
        "where": {
            "children": [
                {"op": "=", "field": "widget.region", "value": "N"},
                {"value": "A", "field": "widget.status", "op": "="},
            ],
            "combinator": "AND",
        },
        "root": "widget",
    }
    assert canonical(gold, root) == canonical(reordered, root)


def test_canonical_coerces_year_type_noise(registry):
    root = registry.root("widget")
    as_int = {"root": "widget", "where": _and(_cond("widget.year", "=", 2024))}
    as_str = {"root": "widget", "where": _and(_cond("widget.year", "=", "2024"))}
    assert canonical(as_int, root) == canonical(as_str, root)


def test_canonical_coerces_money_type_noise(registry):
    root = registry.root("widget")
    numeric = {"root": "widget", "where": _and(_cond("widget.price", ">", 1000))}
    formatted = {"root": "widget", "where": _and(_cond("widget.price", ">", "$1,000.00"))}
    assert canonical(numeric, root) == canonical(formatted, root)


def test_canonical_leaves_text_values_uncoerced(registry):
    """A text field's stored code is not numeric type noise -- '01' and 1
    must stay distinct, unlike the year/money cases above."""
    root = registry.root("widget")
    a = {"root": "widget", "where": _and(_cond("widget.status", "=", "01"))}
    b = {"root": "widget", "where": _and(_cond("widget.status", "=", 1))}
    assert canonical(a, root) != canonical(b, root)


def test_canonical_normalises_aggregate_aliases_through_having_and_sort(registry):
    root = registry.root("widget")
    gold = {
        "root": "widget",
        "aggregate": [
            {"fn": "count", "field": "*", "as": "n"},
            {"fn": "sum", "field": "widget.price", "as": "total"},
        ],
        "having": [{"agg": "total", "op": ">", "value": 100000}],
        "sort": {"agg": "n", "dir": "desc"},
    }
    renamed_and_reordered = {
        "root": "widget",
        "aggregate": [
            {"fn": "sum", "field": "widget.price", "as": "revenue"},
            {"fn": "count", "field": "*", "as": "cnt"},
        ],
        "having": [{"agg": "revenue", "op": ">", "value": 100000}],
        "sort": {"agg": "cnt", "dir": "desc"},
    }
    assert canonical(gold, root) == canonical(renamed_and_reordered, root)


def test_canonical_treats_absent_and_empty_containers_the_same(registry):
    root = registry.root("widget")
    explicit = {
        "root": "widget",
        "where": None,
        "columns": [],
        "join": [],
        "aggregate": [{"fn": "count", "field": "*", "as": "n"}],
        "having": [],
        "sort": None,
    }
    minimal = {
        "root": "widget",
        "aggregate": [{"fn": "count", "field": "*", "as": "n"}],
    }
    assert canonical(explicit, root) == canonical(minimal, root)


def test_canonical_treats_absent_and_null_value2_the_same(registry):
    root = registry.root("widget")
    explicit_null = {
        "root": "widget",
        "where": _and(_cond("widget.serial", "=", 5, value2=None)),
    }
    # _cond() omits value2 entirely when None, so this is the "absent" case.
    absent = {"root": "widget", "where": _and(_cond("widget.serial", "=", 5))}
    assert canonical(explicit_null, root) == canonical(absent, root)


# --------------------------------------------------------------------------
# score(): exact_match, including the columns asymmetry
# --------------------------------------------------------------------------


def test_gold_empty_columns_matches_actual_with_columns(registry):
    root = registry.root("widget")
    where = _and(_cond("widget.status", "=", "A"))
    case = _ok_case("g-cols-1", {"root": "widget", "where": where, "columns": []})
    outcome = _outcome(
        status="ok",
        payload={"root": "widget", "where": where, "columns": ["widget.region"]},
    )
    row = score(case, outcome, root)
    assert row.exact_match is True


def test_gold_named_columns_do_not_match_empty_actual_columns(registry):
    root = registry.root("widget")
    where = _and(_cond("widget.status", "=", "A"))
    case = _ok_case(
        "g-cols-2", {"root": "widget", "where": where, "columns": ["widget.region"]}
    )
    outcome = _outcome(status="ok", payload={"root": "widget", "where": where, "columns": []})
    row = score(case, outcome, root)
    assert row.exact_match is False


def test_exact_match_true_for_alias_renamed_equivalent_payload(registry):
    root = registry.root("widget")
    gold_payload = {
        "root": "widget",
        "aggregate": [{"fn": "count", "field": "*", "as": "n"}],
        "having": [{"agg": "n", "op": ">", "value": 5}],
    }
    actual_payload = {
        "root": "widget",
        "aggregate": [{"fn": "count", "field": "*", "as": "cnt"}],
        "having": [{"agg": "cnt", "op": ">", "value": 5}],
    }
    case = _ok_case("g-alias", gold_payload, tags=("aggregate",))
    outcome = _outcome(status="ok", payload=actual_payload)
    row = score(case, outcome, root)
    assert row.exact_match is True


# --------------------------------------------------------------------------
# score(): field precision/recall, conditioned on root_correct
# --------------------------------------------------------------------------


def test_field_precision_recall_when_root_correct(registry):
    root = registry.root("widget")
    gold_payload = {
        "root": "widget",
        "where": _and(
            _cond("widget.status", "=", "A"),
            _cond("widget.region", "=", "N"),
        ),
    }
    actual_payload = {
        "root": "widget",
        "where": _and(
            _cond("widget.status", "=", "A"),
            _cond("widget.price", ">", 100),
        ),
    }
    case = _ok_case("g-pr-1", gold_payload)
    outcome = _outcome(status="ok", payload=actual_payload)
    row = score(case, outcome, root)

    assert row.root_correct is True
    assert set(row.fields_expected) == {"widget.status", "widget.region"}
    assert set(row.fields_actual) == {"widget.status", "widget.price"}
    assert row.field_tp == 1  # widget.status
    assert row.field_fp == 1  # widget.price, extra
    assert row.field_fn == 1  # widget.region, missing
    # widget.status matched on both sides, same op:
    assert row.ops_total == 1
    assert row.ops_correct == 1
    assert row.values_total == 1
    assert row.values_correct == 1


def test_field_metrics_when_root_wrong_score_everything_as_missed(registry):
    root = registry.root("widget")
    gold_payload = {
        "root": "widget",
        "where": _and(
            _cond("widget.status", "=", "A"),
            _cond("widget.region", "=", "N"),
        ),
    }
    actual_payload = {
        "root": "vendor",
        "where": _and(_cond("vendor.name", "=", "Acme")),
    }
    case = _ok_case("g-pr-2", gold_payload)
    outcome = _outcome(status="ok", payload=actual_payload)
    row = score(case, outcome, root)

    assert row.root_correct is False
    assert row.field_tp == 0
    assert row.field_fn == 2
    assert row.field_fp == 1
    assert row.ops_total == 0
    assert row.values_total == 0
    assert row.exact_match is False


def test_operator_accuracy_conditioned_on_matched_field(registry):
    root = registry.root("widget")
    gold_payload = {"root": "widget", "where": _and(_cond("widget.year", ">", 2020))}
    actual_payload = {"root": "widget", "where": _and(_cond("widget.year", "<", 2020))}
    case = _ok_case("g-op-1", gold_payload)
    outcome = _outcome(status="ok", payload=actual_payload)
    row = score(case, outcome, root)

    assert row.root_correct is True
    assert row.field_tp == 1  # widget.year present on both sides
    assert row.ops_total == 1
    assert row.ops_correct == 0  # ">" vs "<"
    # No (field, op) pair is shared, so nothing to score a value on:
    assert row.values_total == 0
    assert row.values_correct == 0


# --------------------------------------------------------------------------
# score(): refusal, clarify, and failure paths
# --------------------------------------------------------------------------


def test_refusal_expected_and_returned_scores_reason_only(registry):
    root = registry.root("widget")
    case = _refusal_case("g-ref-1", "not_a_query")
    outcome = _outcome(status="refusal", reason="not_a_query")
    row = score(case, outcome, root)

    assert row.refusal_reason_expected == "not_a_query"
    assert row.refusal_reason_actual == "not_a_query"
    assert row.root_correct is None
    assert row.exact_match is None
    assert row.field_tp is None
    assert row.candidate_hit is None


def test_expected_ok_got_refusal_scores_zero_fields_without_raising(registry):
    root = registry.root("widget")
    gold_payload = {
        "root": "widget",
        "where": _and(
            _cond("widget.status", "=", "A"),
            _cond("widget.region", "=", "N"),
        ),
    }
    case = _ok_case("g-refused", gold_payload)
    outcome = _outcome(status="refusal", reason="field_not_in_schema")
    row = score(case, outcome, root)

    assert row.root_correct is False
    assert row.field_tp == 0
    assert row.field_fn == 2
    assert row.field_fp == 0
    assert row.exact_match is False
    assert row.refusal_reason_actual == "field_not_in_schema"


def test_expected_refusal_got_ok_leaves_ok_only_fields_not_applicable(registry):
    root = registry.root("widget")
    case = _refusal_case("g-should-refuse", "not_a_query")
    outcome = _outcome(
        status="ok",
        payload={"root": "widget", "where": _and(_cond("widget.status", "=", "A"))},
    )
    row = score(case, outcome, root)

    assert row.refusal_reason_expected == "not_a_query"
    assert row.refusal_reason_actual is None
    assert row.root_correct is None
    assert row.exact_match is None
    assert row.field_tp is None


def test_clarify_candidate_hit(registry):
    root = registry.root("widget")
    case = _clarify_case("g-clarify-1", "widget.status")
    outcome = _outcome(
        status="clarify",
        candidates=(
            {"field": "widget.status", "op": "=", "value": "A", "label": "Status"},
            {"field": "widget.status", "op": "=", "value": "X", "label": "Status"},
        ),
    )
    row = score(case, outcome, root)
    assert row.clarify_field_expected == "widget.status"
    assert row.candidate_hit is True


def test_clarify_candidate_miss(registry):
    root = registry.root("widget")
    case = _clarify_case("g-clarify-2", "widget.status")
    outcome = _outcome(
        status="clarify",
        candidates=({"field": "widget.region", "op": "=", "value": "N", "label": "Region"},),
    )
    row = score(case, outcome, root)
    assert row.candidate_hit is False


def test_error_outcome_produces_a_row_not_an_exception(registry):
    root = registry.root("widget")
    gold_payload = {"root": "widget", "where": _and(_cond("widget.status", "=", "A"))}
    case = _ok_case("g-error", gold_payload)
    outcome = _outcome(error="502 Bad Gateway", latency_ms=1234.0)
    row = score(case, outcome, root)

    assert row.error == "502 Bad Gateway"
    assert row.status_actual is None
    assert row.root_correct is False
    assert row.field_tp == 0
    assert row.field_fn == 1
    assert row.exact_match is False
    assert row.tokens_in == 0
    assert row.tokens_out == 0
    assert row.latency_ms == 1234.0


def test_tokens_in_sums_uncached_cached_and_cache_write_excluding_reasoning(registry):
    root = registry.root("widget")
    gold_payload = {"root": "widget", "where": _and(_cond("widget.status", "=", "A"))}
    case = _ok_case("g-tokens", gold_payload)
    usage = (
        {
            "uncached_input_tokens": 100,
            "cached_input_tokens": 20,
            "cache_write_tokens": 5,
            "output_tokens": 50,
            "reasoning_tokens": 30,
        },
        {"uncached_input_tokens": 10, "output_tokens": 5},
    )
    outcome = _outcome(status="ok", payload=gold_payload, usage=usage)
    row = score(case, outcome, root)

    assert row.tokens_in == 135
    assert row.tokens_out == 55


# --------------------------------------------------------------------------
# score(): joins and groups_recall_hit
# --------------------------------------------------------------------------


def test_joins_expected_and_actual_are_recorded_sorted(registry):
    root = registry.root("widget")
    gold_payload = {
        "root": "widget",
        "join": ["vendor", "shipment"],
        "where": _and(_cond("shipment.amount", ">", 100)),
    }
    actual_payload = {
        "root": "widget",
        "join": ["shipment"],
        "where": _and(_cond("shipment.amount", ">", 100)),
    }
    case = _ok_case("g-join-1", gold_payload, tags=("join",))
    outcome = _outcome(status="ok", payload=actual_payload)
    row = score(case, outcome, root)

    assert row.joins_expected == ("shipment", "vendor")
    assert row.joins_actual == ("shipment",)


def test_groups_recall_hit_true_when_field_shares_group_with_an_actual_field(registry):
    root = registry.root("widget")
    # widget.status and widget.certified are both "Lifecycle" fields.
    gold_payload = {"root": "widget", "where": _and(_cond("widget.status", "=", "A"))}
    actual_payload = {"root": "widget", "where": _and({"field": "widget.certified", "op": "is_true"})}
    case = _ok_case("g-grp-1", gold_payload)
    outcome = _outcome(status="ok", payload=actual_payload)
    row = score(case, outcome, root)
    assert row.groups_recall_hit is True


def test_groups_recall_hit_false_when_no_actual_field_shares_the_group(registry):
    root = registry.root("widget")
    gold_payload = {"root": "widget", "where": _and(_cond("widget.status", "=", "A"))}
    actual_payload = {"root": "widget", "where": _and(_cond("widget.price", ">", 10))}
    case = _ok_case("g-grp-2", gold_payload)
    outcome = _outcome(status="ok", payload=actual_payload)
    row = score(case, outcome, root)
    assert row.groups_recall_hit is False


def test_groups_recall_hit_none_when_gold_has_no_condition_fields(registry):
    root = registry.root("widget")
    gold_payload = {"root": "widget", "aggregate": [{"fn": "count", "field": "*", "as": "n"}]}
    actual_payload = {"root": "widget", "aggregate": [{"fn": "count", "field": "*", "as": "n"}]}
    case = _ok_case("g-grp-3", gold_payload, tags=("aggregate",))
    outcome = _outcome(status="ok", payload=actual_payload)
    row = score(case, outcome, root)
    assert row.groups_recall_hit is None
