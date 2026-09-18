"""The scoring core, exercised against the widget fixture only.

Every case here builds a `GoldenCase` directly (skipping `load_golden`'s
registry cross-checks, which are Task 2's concern) and an `Outcome` by hand,
so each test isolates exactly the one scoring rule it is about.
"""
import pytest
from pydantic import ValidationError

from cert_nlq.evals.golden import GoldenCase
from cert_nlq.evals.scoring import (
    Outcome,
    canonical,
    end_to_end_correct,
    score,
)

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _ok_case(case_id, payload, tags=()):
    return GoldenCase.model_validate({
        "id": case_id,
        "question": "a question",
        "tags": list(tags),
        "source": "hand-written",
        "expect": {"kind": "ok", "payload": payload},
    })


def _clarify_case(case_id, field, tags=()):
    return GoldenCase.model_validate({
        "id": case_id,
        "question": "a question",
        "tags": list(tags),
        "source": "hand-written",
        "expect": {"kind": "clarify", "field": field},
    })


def _refusal_case(case_id, reason, tags=()):
    return GoldenCase.model_validate({
        "id": case_id,
        "question": "a question",
        "tags": list(tags),
        "source": "hand-written",
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
    case = _ok_case("g-alias", gold_payload)
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
    case = _ok_case("g-join-1", gold_payload)
    outcome = _outcome(status="ok", payload=actual_payload)
    row = score(case, outcome, root)

    assert row.joins_expected == ("shipment", "vendor")
    assert row.joins_actual == ("shipment",)


def test_groups_recall_hit_true_when_field_shares_group_with_an_actual_field(registry):
    root = registry.root("widget")
    # widget.status and widget.certified are both "Lifecycle" fields.
    gold_payload = {"root": "widget", "where": _and(_cond("widget.status", "=", "A"))}
    actual_payload = {"root": "widget", "where": _and(
        {"field": "widget.certified", "op": "is_true"})}
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
    case = _ok_case("g-grp-3", gold_payload)
    outcome = _outcome(status="ok", payload=actual_payload)
    row = score(case, outcome, root)
    assert row.groups_recall_hit is None


def test_groups_recall_hit_covers_aggregate_only_gold_fields(registry):
    """§11 says gold *fields*, not gold condition fields -- an aggregate-only
    gold case naming a real field (not `*`) must not be silently excluded
    from the denominator just because it has no `where`."""
    root = registry.root("widget")
    gold_payload = {
        "root": "widget",
        "aggregate": [{"fn": "sum", "field": "widget.price", "as": "total"}],
    }
    # widget.shipped shares widget.price's "Commercial" group.
    actual_payload = {"root": "widget", "where": _and(
        _cond("widget.shipped", ">", "2024-01-01"))}
    case = _ok_case("g-grp-4", gold_payload)
    outcome = _outcome(status="ok", payload=actual_payload)
    row = score(case, outcome, root)
    assert row.groups_recall_hit is True


# --------------------------------------------------------------------------
# fix-up round: alias-key robustness, Outcome vocabulary, exact_match gating,
# value multisets
# --------------------------------------------------------------------------


def test_canonical_accepts_the_alias_spelling_a_plain_model_dump_would_emit(registry):
    """`Aggregate.alias` serialises as `"as"` only with `by_alias=True`; a
    runner doing a plain `model_dump()` emits `"alias"` instead. Both must
    normalise the same way, including through a having reference."""
    root = registry.root("widget")
    as_spelled = {
        "root": "widget",
        "aggregate": [{"fn": "count", "field": "*", "as": "n"}],
        "having": [{"agg": "n", "op": ">", "value": 5}],
    }
    alias_spelled = {
        "root": "widget",
        "aggregate": [{"fn": "count", "field": "*", "alias": "cnt"}],
        "having": [{"agg": "cnt", "op": ">", "value": 5}],
    }
    assert canonical(as_spelled, root) == canonical(alias_spelled, root)


def test_canonical_prefers_as_when_both_spellings_present_and_agree(registry):
    root = registry.root("widget")
    both_agreeing = {
        "root": "widget",
        "aggregate": [{"fn": "count", "field": "*", "as": "n", "alias": "n"}],
        "having": [{"agg": "n", "op": ">", "value": 5}],
    }
    as_only = {
        "root": "widget",
        "aggregate": [{"fn": "count", "field": "*", "as": "n"}],
        "having": [{"agg": "n", "op": ">", "value": 5}],
    }
    assert canonical(both_agreeing, root) == canonical(as_only, root)


def test_canonical_treats_conflicting_alias_spellings_as_garbage_not_a_guess(registry):
    root = registry.root("widget")
    conflicting = {
        "root": "widget",
        "aggregate": [{"fn": "count", "field": "*", "as": "n", "alias": "m"}],
    }
    # Self-consistent: the exact same garbage compares equal to itself.
    assert canonical(conflicting, root) == canonical(conflicting, root)
    # But it must not spuriously match a clean, differently-shaped aggregate.
    clean = {"root": "widget", "aggregate": [{"fn": "count", "field": "*", "as": "n"}]}
    assert canonical(conflicting, root) != canonical(clean, root)


def test_outcome_rejects_the_wires_own_status_vocabulary():
    """`"refused"`/`"needs_clarification"` are the wire's spellings, not
    `Outcome`'s -- constructing one with them must fail loudly rather than
    silently reading as a wrong-forever refusal-reason comparison."""
    with pytest.raises(ValidationError):
        Outcome(status="refused")
    with pytest.raises(ValidationError):
        Outcome(status="needs_clarification")


def test_exact_match_is_false_when_a_clarify_payload_coincidentally_canonicalises_equal(
    registry,
):
    root = registry.root("widget")
    where = _and(_cond("widget.status", "=", "A"))
    gold_payload = {"root": "widget", "where": where}
    case = _ok_case("g-clarify-payload", gold_payload)
    # The service's pruned needs_clarification payload happens to match gold
    # canonically, but the status is "clarify", not "ok".
    outcome = _outcome(status="clarify", payload={"root": "widget", "where": where})
    row = score(case, outcome, root)
    assert row.exact_match is False
    assert row.field_tp == 0


def test_values_correct_treats_repeated_field_op_as_a_multiset(registry):
    """`price > 100 OR price > 500` on both sides, values swapped in
    writing order, must still score as a full value match."""
    root = registry.root("widget")
    gold_payload = {
        "root": "widget",
        "where": {
            "combinator": "OR",
            "children": [
                _cond("widget.price", ">", 100),
                _cond("widget.price", ">", 500),
            ],
        },
    }
    actual_payload = {
        "root": "widget",
        "where": {
            "combinator": "OR",
            "children": [
                _cond("widget.price", ">", 500),
                _cond("widget.price", ">", 100),
            ],
        },
    }
    case = _ok_case("g-multiset", gold_payload)
    outcome = _outcome(status="ok", payload=actual_payload)
    row = score(case, outcome, root)
    assert row.values_total == 1  # one matched (field, op) pair
    assert row.values_correct == 1


def test_values_correct_catches_a_dropped_duplicate(registry):
    """Same (field, op) pair, but the actual side dropped one of the two
    values -- the multiset compare must catch that as incorrect."""
    root = registry.root("widget")
    gold_payload = {
        "root": "widget",
        "where": {
            "combinator": "OR",
            "children": [
                _cond("widget.price", ">", 100),
                _cond("widget.price", ">", 500),
            ],
        },
    }
    actual_payload = {
        "root": "widget",
        "where": {
            "combinator": "OR",
            "children": [
                _cond("widget.price", ">", 100),
                _cond("widget.price", ">", 100),
            ],
        },
    }
    case = _ok_case("g-multiset-2", gold_payload)
    outcome = _outcome(status="ok", payload=actual_payload)
    row = score(case, outcome, root)
    assert row.values_total == 1
    assert row.values_correct == 0


# --------------------------------------------------------------------------
# the echoed route: group-selection recall, measured rather than proxied
# --------------------------------------------------------------------------


def _route(root="widget", joins=(), groups=()):
    return {"root": root, "joins": list(joins), "groups": list(groups)}


def test_groups_recall_reads_the_echoed_route_when_there_is_one(registry):
    """With an echo this is the real metric: did stage 1 select the group the
    gold field lives in? The payload is beside the point -- here it names a
    different field from the same group, which the proxy would have called a
    hit for the wrong reason."""
    root = registry.root("widget")
    gold = {"root": "widget", "where": _and(_cond("widget.price", ">", 100))}
    actual = {"root": "widget", "where": _and(_cond("widget.shipped", ">", "2024-01-01"))}
    hit = score(
        _ok_case("g-1", gold),
        _outcome(status="ok", payload=actual, route=_route(groups=["Commercial"])),
        root,
    )
    miss = score(
        _ok_case("g-1", gold),
        _outcome(status="ok", payload=actual, route=_route(groups=["Lifecycle"])),
        root,
    )
    assert hit.groups_recall_hit is True
    assert miss.groups_recall_hit is False


def test_an_identity_group_is_not_required_of_the_route(registry):
    """Identity groups are in scope whatever stage 1 picks, so a router that
    does not name one has not missed anything."""
    root = registry.root("widget")
    gold = {"root": "widget", "where": _and(_cond("widget.region", "=", "N"))}
    row = score(
        _ok_case("g-2", gold),
        _outcome(status="ok", payload=gold, route=_route(groups=["Lifecycle"])),
        root,
    )
    assert row.groups_recall_hit is True


def test_without_a_route_the_documented_proxy_still_runs(registry):
    """No echo (an older deploy, or a response that never routed): the
    lower-bound proxy from Task 3 is what remains, unchanged."""
    root = registry.root("widget")
    gold = {"root": "widget", "where": _and(_cond("widget.price", ">", 100))}
    actual = {"root": "widget", "where": _and(_cond("widget.shipped", ">", "2024-01-01"))}
    row = score(_ok_case("g-3", gold), _outcome(status="ok", payload=actual), root)
    # Same group, different field: the proxy's optimistic hit.
    assert row.groups_recall_hit is True


def test_a_malformed_route_falls_back_to_the_proxy_without_raising(registry):
    """A garbled echo is an unknown route, not an empty one. Scoring it as
    "routed to nothing" would report a perfect service as missing every
    group, so the proxy takes over.

    (`Outcome.route` is typed `dict | None`, so a non-dict never gets this
    far -- pydantic refuses it at construction. What is left to tolerate is
    a dict whose `groups` is missing or the wrong shape.)"""
    root = registry.root("widget")
    gold = {"root": "widget", "where": _and(_cond("widget.price", ">", 100))}
    for bad in ({}, {"groups": "Commercial"}, {"groups": None}, {"root": "widget"}):
        row = score(
            _ok_case("g-4", gold),
            _outcome(status="ok", payload=gold, route=bad),
            root,
        )
        assert row.groups_recall_hit is True, bad


def test_a_route_carrying_no_groups_at_all_is_a_miss_not_a_fallback(registry):
    """An empty list is a real answer -- stage 1 said it placed nothing --
    and it is wrong whenever the gold answer needed a group."""
    root = registry.root("widget")
    gold = {"root": "widget", "where": _and(_cond("widget.price", ">", 100))}
    row = score(
        _ok_case("g-5", gold),
        _outcome(status="ok", payload=gold, route=_route(groups=[])),
        root,
    )
    assert row.groups_recall_hit is False


def test_groups_recall_source_is_echo_when_the_route_has_usable_groups(registry):
    """Task 4's aggregator reports echoed and proxied group recall
    separately; the row must say which path produced its hit."""
    root = registry.root("widget")
    gold = {"root": "widget", "where": _and(_cond("widget.price", ">", 100))}
    row = score(
        _ok_case("g-6", gold),
        _outcome(status="ok", payload=gold, route=_route(groups=["Commercial"])),
        root,
    )
    assert row.groups_recall_source == "echo"


def test_groups_recall_source_is_proxy_without_a_usable_route(registry):
    root = registry.root("widget")
    gold = {"root": "widget", "where": _and(_cond("widget.price", ">", 100))}
    for bad_route in (None, {}, {"groups": "Commercial"}):
        row = score(
            _ok_case("g-7", gold),
            _outcome(status="ok", payload=gold, route=bad_route),
            root,
        )
        assert row.groups_recall_source == "proxy", bad_route


def test_groups_recall_source_is_none_when_the_hit_is_unmeasured(registry):
    """No gold condition/aggregate/group_by/sort field at all -- the same
    case `groups_recall_hit` itself returns `None` for."""
    root = registry.root("widget")
    gold = {"root": "widget", "aggregate": [{"fn": "count", "field": "*", "as": "n"}]}
    row = score(
        _ok_case("g-8", gold),
        _outcome(status="ok", payload=gold, route=_route(groups=["Commercial"])),
        root,
    )
    assert row.groups_recall_hit is None
    assert row.groups_recall_source is None


def test_groups_recall_source_is_none_off_the_ok_path(registry):
    """Only `kind == "ok"` rows ever compute `groups_recall_hit`; a clarify
    or refusal row must not pick up a stray source either."""
    root = registry.root("widget")
    case = _clarify_case("g-9", "widget.status")
    row = score(case, _outcome(status="clarify", route=_route(root="widget")), root)
    assert row.groups_recall_hit is None
    assert row.groups_recall_source is None


# --------------------------------------------------------------------------
# end_to_end_correct(): the frozen primary outcome
# --------------------------------------------------------------------------


def test_e2e_correct_for_an_ok_case_needs_status_ok_and_exact_match(registry):
    root = registry.root("widget")
    gold = {"root": "widget", "where": _and(_cond("widget.status", "=", "A"))}
    right = score(_ok_case("g-6", gold), _outcome(status="ok", payload=gold), root)
    assert right.exact_match is True
    assert right.e2e_correct is True

    other = {"root": "widget", "where": _and(_cond("widget.status", "=", "X"))}
    wrong = score(_ok_case("g-6", gold), _outcome(status="ok", payload=other), root)
    assert wrong.e2e_correct is False


def test_an_ok_case_answered_with_a_clarification_is_incorrect(registry):
    """The asymmetry the definition exists to pin: the caller was asked a
    question instead of being given the answer, and the pruned payload
    canonicalising equal to gold does not change that."""
    root = registry.root("widget")
    gold = {"root": "widget", "where": _and(_cond("widget.status", "=", "A"))}
    row = score(
        _ok_case("g-7", gold),
        _outcome(status="clarify", payload=gold,
                 candidates=({"field": "widget.status", "value": "A"},)),
        root,
    )
    assert row.e2e_correct is False


def test_e2e_correct_for_a_clarify_case_needs_the_right_field(registry):
    root = registry.root("widget")
    case = _clarify_case("g-8", "widget.status")
    right = score(
        case,
        _outcome(status="clarify",
                 candidates=({"field": "widget.status", "value": "A"},)),
        root,
    )
    wrong_field = score(
        case,
        _outcome(status="clarify",
                 candidates=({"field": "widget.region", "value": "N"},)),
        root,
    )
    assert right.e2e_correct is True
    assert wrong_field.e2e_correct is False


def test_e2e_correct_for_a_refusal_case_needs_the_right_reason(registry):
    root = registry.root("widget")
    case = _refusal_case("g-9", "not_a_query")
    right = score(case, _outcome(status="refusal", reason="not_a_query"), root)
    wrong = score(
        case, _outcome(status="refusal", reason="field_not_in_schema"), root
    )
    silent = score(case, _outcome(status="refusal"), root)
    assert right.e2e_correct is True
    assert wrong.e2e_correct is False
    assert silent.e2e_correct is False


def test_a_transport_failure_is_never_end_to_end_correct(registry):
    root = registry.root("widget")
    gold = {"root": "widget", "where": _and(_cond("widget.status", "=", "A"))}
    row = score(_ok_case("g-10", gold), _outcome(error="timeout"), root)
    assert row.e2e_correct is False


def test_end_to_end_correct_is_one_function_every_branch_reads(registry):
    """Called directly, with the diagnostics passed in: the row's primary
    outcome and its diagnostics can never disagree about the same run."""
    gold = {"root": "widget", "where": _and(_cond("widget.status", "=", "A"))}
    ok_case = _ok_case("g-11", gold)
    assert end_to_end_correct(
        ok_case, _outcome(status="ok"), exact_match=True, candidate_hit=None
    ) is True
    assert end_to_end_correct(
        ok_case, _outcome(status="ok"), exact_match=False, candidate_hit=None
    ) is False
    clarify = _clarify_case("g-12", "widget.status")
    assert end_to_end_correct(
        clarify, _outcome(status="clarify"), exact_match=None, candidate_hit=True
    ) is True
    assert end_to_end_correct(
        clarify, _outcome(status="ok"), exact_match=None, candidate_hit=True
    ) is False
    refusal = _refusal_case("g-13", "not_a_query")
    assert end_to_end_correct(
        refusal,
        _outcome(status="refusal", reason="not_a_query"),
        exact_match=None,
        candidate_hit=None,
    ) is True


def test_gold_root_scores_the_route_on_a_clarify_case(registry):
    """The stage-1 metric covers the whole set now: a clarify case carries no
    payload to read a root off, so `gold_root` plus the echo answers it."""
    root = registry.root("widget")
    case = GoldenCase.model_validate({
        "id": "g-r1", "question": "widgets in a funny state", "tags": [],
        "source": "hand-written", "gold_root": "widget",
        "expect": {"kind": "clarify", "field": "widget.status"},
    })
    right = score(
        case, _outcome(status="clarify", route=_route(root="widget")), root
    )
    wrong = score(
        case, _outcome(status="clarify", route=_route(root="vendor")), root
    )
    assert right.root_correct is True
    assert wrong.root_correct is False


def test_gold_root_scores_the_route_on_a_refusal_case(registry):
    root = registry.root("widget")
    case = GoldenCase.model_validate({
        "id": "g-r2", "question": "widgets nobody stocks", "tags": [],
        "source": "hand-written", "gold_root": "widget",
        "expect": {"kind": "refusal", "reason": "field_not_in_schema"},
    })
    row = score(
        case,
        _outcome(status="refusal", reason="field_not_in_schema",
                 route=_route(root="widget")),
        root,
    )
    assert row.root_correct is True


def test_a_case_with_no_gold_root_or_no_echo_is_not_measured(registry):
    """`None` is "not measurable", not "wrong". A `not_a_query` refusal has
    no right root -- "hello" routes nowhere -- and a response that never
    routed has nothing to compare against."""
    root = registry.root("widget")
    rootless = _refusal_case("g-r3", "not_a_query")
    anchored = GoldenCase.model_validate({
        "id": "g-r4", "question": "widgets in a funny state", "tags": [],
        "source": "hand-written", "gold_root": "widget",
        "expect": {"kind": "clarify", "field": "widget.status"},
    })
    assert score(
        rootless, _outcome(status="refusal", reason="not_a_query",
                           route=_route(root="widget")), root
    ).root_correct is None
    assert score(anchored, _outcome(status="clarify"), root).root_correct is None
    assert score(
        anchored, _outcome(status="clarify", route={"groups": []}), root
    ).root_correct is None


def test_a_route_whose_groups_are_not_strings_falls_back_to_the_proxy(registry):
    """`{"groups": [1, 2]}` is an unreadable echo, not "routed to nothing".
    Filtering the non-strings out would score a miss against a service that
    may well have routed correctly."""
    root = registry.root("widget")
    gold = {"root": "widget", "where": _and(_cond("widget.price", ">", 100))}
    row = score(
        _ok_case("g-r5", gold),
        _outcome(status="ok", payload=gold, route={"root": "widget",
                                                   "groups": [1, 2]}),
        root,
    )
    assert row.groups_recall_hit is True
