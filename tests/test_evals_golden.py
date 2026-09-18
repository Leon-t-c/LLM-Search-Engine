"""The golden-set loader, exercised against the widget fixture only.

The seed set the harness actually replays is written against the live
registry and is not a test input: a test that needed it would fail whenever
the host's schema moved, which is a fact about the host and not about this
loader. Everything here is the fixture, like every other suite in the repo.
"""
import json

import pytest
from pydantic import ValidationError

from cert_nlq.evals.golden import (
    SLICES,
    ExpectClarify,
    ExpectOk,
    ExpectRefusal,
    GoldenCase,
    GoldenError,
    derive_difficulty,
    derive_slices,
    family,
    load_golden,
)

OK_PAYLOAD = {
    "root": "widget",
    "where": {
        "combinator": "AND",
        "children": [{"field": "widget.status", "op": "=", "value": "A"}],
    },
}


def _case(case_id="g-001", **overrides):
    case = {
        "id": case_id,
        "question": "active widgets",
        "tags": [],
        "source": "hand-written",
        "expect": {"kind": "ok", "payload": OK_PAYLOAD},
    }
    case.update(overrides)
    return case


def _write(tmp_path, *cases, name="golden.jsonl"):
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8"
    )
    return path


def test_a_valid_ok_case_loads(tmp_path, registry):
    cases = load_golden(_write(tmp_path, _case()), registry)
    assert [c.id for c in cases] == ["g-001"]
    assert isinstance(cases[0].expect, ExpectOk)
    assert cases[0].tags == ()
    assert cases[0].expect.payload["root"] == "widget"


def test_a_case_is_frozen(tmp_path, registry):
    case = load_golden(_write(tmp_path, _case()), registry)[0]
    with pytest.raises(ValidationError):
        case.question = "something else"


def test_blank_lines_are_skipped(tmp_path, registry):
    path = tmp_path / "golden.jsonl"
    path.write_text(f"\n{json.dumps(_case())}\n\n", encoding="utf-8")
    assert len(load_golden(path, registry)) == 1


def test_clarify_and_refusal_cases_load(tmp_path, registry):
    cases = load_golden(
        _write(
            tmp_path,
            _case("g-001", question="widgets in a funny state",
                  expect={"kind": "clarify", "field": "widget.status"}),
            _case("g-002", expect={"kind": "refusal", "reason": "not_a_query"}),
        ),
        registry,
    )
    assert isinstance(cases[0].expect, ExpectClarify)
    assert cases[0].expect.field == "widget.status"
    assert isinstance(cases[1].expect, ExpectRefusal)
    assert cases[1].expect.reason == "not_a_query"


def test_a_gold_payload_naming_an_unknown_field_is_rejected(tmp_path, registry):
    bad = {
        "root": "widget",
        "where": {
            "combinator": "AND",
            "children": [{"field": "widget.nope", "op": "=", "value": "A"}],
        },
    }
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(tmp_path, _case("g-042", expect={"kind": "ok", "payload": bad})),
            registry,
        )
    assert "g-042" in str(exc.value)
    assert "unknown field" in str(exc.value)


def test_a_gold_payload_with_an_illegal_operator_is_rejected(tmp_path, registry):
    bad = {
        "root": "widget",
        "where": {
            "combinator": "AND",
            "children": [{"field": "widget.status", "op": ">", "value": "A"}],
        },
    }
    with pytest.raises(GoldenError, match="g-007"):
        load_golden(
            _write(tmp_path, _case("g-007", expect={"kind": "ok", "payload": bad})),
            registry,
        )


def test_a_gold_payload_naming_an_unknown_root_is_rejected(tmp_path, registry):
    bad = dict(OK_PAYLOAD, root="sprocket")
    with pytest.raises(GoldenError, match="unknown root"):
        load_golden(
            _write(tmp_path, _case("g-009", expect={"kind": "ok", "payload": bad})),
            registry,
        )


def test_duplicate_ids_are_rejected(tmp_path, registry):
    with pytest.raises(GoldenError, match="duplicate id"):
        load_golden(_write(tmp_path, _case("g-001"), _case("g-001")), registry)


def test_an_unknown_refusal_reason_is_rejected(tmp_path, registry):
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-050", expect={"kind": "refusal", "reason": "too_hard"}),
            ),
            registry,
        )
    assert "g-050" in str(exc.value)
    assert "too_hard" in str(exc.value)


def test_an_unknown_clarification_field_is_rejected(tmp_path, registry):
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-051", expect={"kind": "clarify", "field": "widget.nope"}),
            ),
            registry,
        )
    assert "g-051" in str(exc.value)


def test_the_aux_tags_load_and_a_retired_slice_tag_does_not(tmp_path, registry):
    """`tags` is a closed set of hand tags now; slices are derived.

    The retired names are rejected rather than ignored, so a file half-way
    through the migration fails loudly instead of quietly losing the tagging
    it thought it had.
    """
    cases = load_golden(
        _write(tmp_path, _case("g-060", tags=["review", "regression"])), registry
    )
    assert cases[0].tags == ("review", "regression")
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(tmp_path, _case("g-061", tags=["selection", "aggregate"])),
            registry,
        )
    message = str(exc.value)
    assert "g-061" in message
    assert "aggregate" in message and "selection" in message


def test_every_problem_is_reported_not_just_the_first(tmp_path, registry):
    """Four broken cases, four sentences -- fixing them one run at a time is
    exactly the loop this loader exists to avoid."""
    bad_field = {
        "root": "widget",
        "where": {
            "combinator": "AND",
            "children": [{"field": "widget.nope", "op": "=", "value": "A"}],
        },
    }
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-001", expect={"kind": "ok", "payload": bad_field}),
                _case("g-002", expect={"kind": "refusal", "reason": "nope"}),
                _case("g-003", expect={"kind": "clarify", "field": "widget.nope"}),
                _case("g-004"),
                _case("g-004"),
            ),
            registry,
        )
    message = str(exc.value)
    for case_id in ("g-001", "g-002", "g-003", "g-004"):
        assert case_id in message
    # Four: one bad field, one invented refusal reason, one unknown
    # clarification field, one duplicate id. (It was five while a slice tag
    # could disagree with the expectation it was filed under; that check is
    # gone with the tags themselves.)
    assert len(exc.value.problems) == 4


def test_a_malformed_line_is_reported_with_its_line_number(tmp_path, registry):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps(_case()) + "\n{not json\n", encoding="utf-8"
    )
    with pytest.raises(GoldenError, match="line 2"):
        load_golden(path, registry)


def test_a_case_missing_its_expectation_is_reported_by_id(tmp_path, registry):
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps({"id": "g-070", "question": "hi", "source": "hand-written"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenError, match="g-070"):
        load_golden(path, registry)


def test_a_case_can_carry_a_note(tmp_path, registry):
    """A note is for an expectation whose *correctness* is surprising -- it
    travels on the case, so editing the expectation back to the obvious answer
    means deleting the sentence explaining why the obvious answer is wrong."""
    cases = load_golden(
        _write(tmp_path, _case(note="the obvious reading is wrong because ...")),
        registry,
    )
    assert cases[0].note == "the obvious reading is wrong because ..."
    assert load_golden(_write(tmp_path, _case()), registry)[0].note is None


def test_a_stray_key_on_a_case_is_rejected(tmp_path, registry):
    """A mistyped `notes` must not vanish. Three cases in the seed set carry a
    `note` whose whole job is to defend a counter-intuitive expectation; a typo
    that silently drops one leaves the expectation looking like a mistake."""
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(tmp_path, _case("g-080", **{"notes": "typo'd key"})), registry
        )
    assert "g-080" in str(exc.value)
    assert "notes" in str(exc.value)


def test_a_stray_key_on_an_expectation_is_rejected(tmp_path, registry):
    """`field` left behind on a case edited from clarify to ok used to load,
    silently dropped, still reading as though it said something."""
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-081", expect={"kind": "ok", "payload": OK_PAYLOAD,
                                       "field": "widget.status"}),
            ),
            registry,
        )
    assert "g-081" in str(exc.value)
    assert "field" in str(exc.value)


def test_a_clarification_field_with_no_vocabulary_is_rejected(tmp_path, registry):
    """needs_clarification is only reachable through a coded field: an
    unresolvable value on a field with nothing to offer back is refused, by
    design. A gold case naming `widget.serial` asserts an unreachable outcome
    and would make candidate recall a constant."""
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-082", expect={"kind": "clarify", "field": "widget.serial"}),
            ),
            registry,
        )
    message = str(exc.value)
    assert "g-082" in message
    assert "widget.serial" in message
    assert "vocabulary" in message


def test_the_unknown_clarification_field_message_says_what_is_wrong(tmp_path, registry):
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-083", expect={"kind": "clarify", "field": "widget.nope"}),
            ),
            registry,
        )
    message = str(exc.value)
    assert "g-083" in message
    assert "unknown clarification field" in message
    assert "widget.nope" in message


def test_a_duplicate_id_still_reports_that_cases_other_defects(tmp_path, registry):
    """The duplicate is a problem with the file; whatever else is wrong with
    the case is a problem with the case. Reporting only the first costs the
    reader another whole run to find the second."""
    bad = {
        "root": "widget",
        "where": {
            "combinator": "AND",
            "children": [{"field": "widget.nope", "op": "=", "value": "A"}],
        },
    }
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-084"),
                _case("g-084", expect={"kind": "ok", "payload": bad}),
            ),
            registry,
        )
    message = str(exc.value)
    assert "duplicate id" in message
    assert "unknown field" in message


def test_a_case_must_say_where_it_came_from(tmp_path, registry):
    """`source` is required, with no default.

    Provenance is what nobody remembers six weeks later, and the primary
    analysis turns on it: it excludes paraphrases, which it can only do if
    every case says what it is.
    """
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps({
            "id": "g-087",
            "question": "active widgets",
            "tags": [],
            "expect": {"kind": "ok", "payload": OK_PAYLOAD},
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenError) as exc:
        load_golden(path, registry)
    message = str(exc.value)
    assert "g-087" in message
    assert "source" in message


def test_a_stray_key_in_a_gold_payload_is_rejected(tmp_path, registry):
    """`Payload` ignores extras -- the service must stay tolerant on its wire.
    A gold payload is the opposite: `groupby` would validate perfectly and then
    never match anything, reporting a typo in the ruler as a translation miss."""
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-088", expect={"kind": "ok", "payload": {
                    **OK_PAYLOAD, "groupby": ["widget.region"]}}),
            ),
            registry,
        )
    message = str(exc.value)
    assert "g-088" in message
    assert "groupby" in message


def test_one_case_with_two_defects_reports_both(tmp_path, registry):
    bad = {
        "root": "widget",
        "groupby": ["widget.region"],
        "where": {
            "combinator": "AND",
            "children": [{"field": "widget.nope", "op": "=", "value": "A"}],
        },
    }
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(tmp_path, _case("g-089", expect={"kind": "ok", "payload": bad})),
            registry,
        )
    assert len(exc.value.problems) == 2
    assert all(p.startswith("g-089: ") for p in exc.value.problems)


# --------------------------------------------------------------------------
# derived slices and difficulty
# --------------------------------------------------------------------------


def _built(**overrides) -> GoldenCase:
    """A `GoldenCase` built directly, skipping the file and the registry
    cross-checks -- these tests are about the derivations, not the loader."""
    return GoldenCase.model_validate(_case(**overrides))


def _ok(payload, **overrides) -> GoldenCase:
    return _built(expect={"kind": "ok", "payload": payload}, **overrides)


def _cond(field, op, value=None):
    node = {"field": field, "op": op}
    if value is not None:
        node["value"] = value
    return node


def _where(*children):
    return {"combinator": "AND", "children": list(children)}


def test_a_lone_condition_is_a_simple_filter(registry):
    case = _ok({"root": "widget", "where": _where(_cond("widget.serial", "=", 7))})
    assert derive_slices(case, registry) == {"simple-filter"}
    assert derive_difficulty(case, registry) == "easy"


def test_two_conditions_are_multi_condition_not_simple(registry):
    case = _ok({
        "root": "widget",
        "where": _where(
            _cond("widget.serial", "=", 7), _cond("widget.year", "=", 2024)
        ),
    })
    assert derive_slices(case, registry) == {"multi-condition"}
    assert derive_difficulty(case, registry) == "medium"


def test_a_lone_condition_under_a_group_by_is_not_a_simple_filter(registry):
    """One condition "and nothing else" means nothing else. A breakdown
    question wearing a single filter is not the easy slice."""
    case = _ok({
        "root": "widget",
        "where": _where(_cond("widget.year", "=", 2024)),
        "aggregate": [{"fn": "count", "field": "*", "as": "total"}],
        "group_by": ["widget.region"],
    })
    slices = derive_slices(case, registry)
    assert "simple-filter" not in slices
    assert slices == {"aggregation", "group-by"}


def test_nested_logic_is_derived_from_where_depth_and_is_hard(registry):
    case = _ok({
        "root": "widget",
        "where": _where(
            _cond("widget.year", "=", 2024),
            {"combinator": "OR", "children": [
                _cond("widget.certified", "is_true"),
                _cond("widget.status", "=", "A"),
            ]},
        ),
    })
    slices = derive_slices(case, registry)
    assert "nested-logic" in slices
    assert "multi-condition" in slices
    assert derive_difficulty(case, registry) == "hard"


def test_a_flat_group_is_not_nested_logic(registry):
    case = _ok({
        "root": "widget",
        "where": _where(
            _cond("widget.year", "=", 2024), _cond("widget.status", "=", "A")
        ),
    })
    assert "nested-logic" not in derive_slices(case, registry)


def test_in_op_and_coded_vocabulary_are_derived_from_the_conditions(registry):
    case = _ok({
        "root": "widget",
        "where": _where(_cond("widget.status", "in", ["A", "X"])),
    })
    assert derive_slices(case, registry) == {
        "simple-filter", "in-op", "coded-vocabulary"
    }


def test_a_vocabulary_less_field_is_not_coded_vocabulary(registry):
    case = _ok({"root": "widget", "where": _where(_cond("widget.price", ">", 10))})
    assert "coded-vocabulary" not in derive_slices(case, registry)


def test_one_condition_plus_a_join_is_a_join_case_and_only_medium(registry):
    """The rubric's own edge: easy needs no join, and one join is not two."""
    case = _ok({
        "root": "widget",
        "where": _where(_cond("shipment.amount", ">", 100)),
        "join": ["shipment"],
    })
    assert "join" in derive_slices(case, registry)
    assert "simple-filter" not in derive_slices(case, registry)
    assert derive_difficulty(case, registry) == "medium"


def test_a_join_is_derived_from_the_fields_even_when_the_slot_is_empty(registry):
    case = _ok({
        "root": "widget",
        "where": _where(_cond("shipment.amount", ">", 100)),
    })
    assert "join" in derive_slices(case, registry)


def test_aggregation_group_by_and_having_are_each_their_own_slice(registry):
    case = _ok({
        "root": "widget",
        "where": _where(_cond("widget.year", "=", 2024)),
        "aggregate": [{"fn": "count", "field": "*", "as": "total"}],
        "group_by": ["widget.region"],
        "having": [{"agg": "total", "op": ">", "value": 10}],
    })
    assert derive_slices(case, registry) >= {"aggregation", "group-by", "having"}


def test_a_mixed_grain_aggregate_is_hard(registry):
    """Aggregating a joined table's field per a root field: two row-grains in
    one answer, and the join fans out."""
    case = _ok({
        "root": "widget",
        "where": _where(_cond("widget.year", "=", 2024)),
        "join": ["shipment"],
        "aggregate": [{"fn": "avg", "field": "shipment.amount", "as": "average"}],
        "group_by": ["widget.region"],
    })
    assert derive_difficulty(case, registry) == "hard"


def test_a_single_grain_aggregate_is_medium(registry):
    case = _ok({
        "root": "widget",
        "where": _where(_cond("widget.year", "=", 2024)),
        "aggregate": [{"fn": "avg", "field": "widget.price", "as": "average"}],
        "group_by": ["widget.region"],
    })
    assert derive_difficulty(case, registry) == "medium"


def test_ambiguous_wording_makes_a_case_hard_whatever_its_shape(registry):
    case = _ok(
        {"root": "widget", "where": _where(_cond("widget.serial", "=", 7))},
        tags=["ambiguous-wording"],
    )
    assert derive_difficulty(case, registry) == "hard"


def test_clarify_and_refusal_slices_carry_no_structural_slice(registry):
    clarify = _built(expect={"kind": "clarify", "field": "widget.status"})
    refusal = _built(expect={"kind": "refusal", "reason": "field_not_in_schema"})
    assert derive_slices(clarify, registry) == {"clarification"}
    assert derive_slices(refusal, registry) == {"refusal"}
    assert derive_difficulty(clarify, registry) == "medium"
    assert derive_difficulty(refusal, registry) == "medium"


def test_every_derived_slice_name_is_declared(registry):
    case = _ok({
        "root": "widget",
        "where": _where(
            _cond("widget.status", "in", ["A", "X"]),
            {"combinator": "OR", "children": [
                _cond("shipment.amount", ">", 100),
                _cond("widget.price", ">", 10),
            ]},
        ),
        "join": ["shipment"],
        "aggregate": [{"fn": "count", "field": "*", "as": "total"}],
        "group_by": ["widget.region"],
        "having": [{"agg": "total", "op": ">", "value": 1}],
    })
    assert derive_slices(case, registry) <= SLICES


def test_difficulty_override_wins_and_needs_a_note(registry):
    overridden = _ok(
        {"root": "widget", "where": _where(_cond("widget.serial", "=", 7))},
        difficulty_override="hard",
        note="the serial format is the whole difficulty here",
    )
    assert derive_difficulty(overridden, registry) == "hard"
    with pytest.raises(ValidationError, match="note"):
        _ok(
            {"root": "widget", "where": _where(_cond("widget.serial", "=", 7))},
            difficulty_override="hard",
        )


# --------------------------------------------------------------------------
# gold_root, paraphrase_of, and the literal-preservation guard
# --------------------------------------------------------------------------


def test_gold_root_is_rejected_on_an_ok_case_that_disagrees(tmp_path, registry):
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(tmp_path, _case("g-090", gold_root="vendor")), registry
        )
    message = str(exc.value)
    assert "g-090" in message
    assert "gold_root" in message


def test_gold_root_may_restate_an_ok_payloads_own_root(tmp_path, registry):
    cases = load_golden(
        _write(tmp_path, _case("g-091", gold_root="widget")), registry
    )
    assert cases[0].gold_root == "widget"


def test_gold_root_records_the_route_a_clarify_case_expects(tmp_path, registry):
    cases = load_golden(
        _write(
            tmp_path,
            _case("g-092", gold_root="widget", question="widgets in a funny state",
                  expect={"kind": "clarify", "field": "widget.status"}),
        ),
        registry,
    )
    assert cases[0].gold_root == "widget"


def test_an_unknown_gold_root_is_rejected(tmp_path, registry):
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-093", gold_root="sprocket",
                      expect={"kind": "refusal", "reason": "not_a_query"}),
            ),
            registry,
        )
    assert "g-093" in str(exc.value)
    assert "unknown gold_root" in str(exc.value)


def _paraphrase(case_id, of, question, **overrides):
    return _case(
        case_id,
        question=question,
        tags=["review"],
        source="paraphrase",
        paraphrase_of=of,
        **overrides,
    )


def test_a_paraphrase_loads_and_reports_its_family(tmp_path, registry):
    cases = load_golden(
        _write(
            tmp_path,
            _case("g-100", question="widgets active in 2024 over $500"),
            _paraphrase("g-100-p1", "g-100", "active 2024 widgets above $500"),
        ),
        registry,
    )
    assert [family(c) for c in cases] == ["g-100", "g-100"]


def test_paraphrase_of_must_name_a_case_in_the_file(tmp_path, registry):
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(tmp_path, _paraphrase("g-101-p1", "g-999", "active widgets")),
            registry,
        )
    assert "g-101-p1" in str(exc.value)
    assert "g-999" in str(exc.value)


def test_a_paraphrase_of_a_paraphrase_is_rejected(tmp_path, registry):
    """One level only: a chain makes the family depend on which link you
    walked, and the cluster bootstrap resamples families."""
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-102", question="active widgets in 2024"),
                _paraphrase("g-102-p1", "g-102", "2024 widgets, active"),
                _paraphrase("g-102-p2", "g-102-p1", "active widgets, 2024"),
            ),
            registry,
        )
    message = str(exc.value)
    assert "g-102-p2" in message
    assert "itself a paraphrase" in message


def test_a_paraphrase_that_drops_a_number_is_rejected(tmp_path, registry):
    """The classic boundary drift: the rewording reads fine and asks a
    different question."""
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-103", question="widgets shipped after 2024"),
                _paraphrase("g-103-p1", "g-103", "widgets shipped last year"),
            ),
            registry,
        )
    message = str(exc.value)
    assert "g-103-p1" in message
    assert "2024" in message


def test_a_paraphrase_that_drops_a_quoted_code_is_rejected(tmp_path, registry):
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-104", question="widgets for client C-102"),
                _paraphrase("g-104-p1", "g-104", "that client's widgets"),
            ),
            registry,
        )
    assert "g-104-p1" in str(exc.value)


def test_a_rewording_that_keeps_every_literal_is_fine(tmp_path, registry):
    cases = load_golden(
        _write(
            tmp_path,
            _case("g-105", question="widgets for client C-102 shipped after 2024"),
            _paraphrase(
                "g-105-p1", "g-105", "widgets shipped after 2024 for C-102"
            ),
        ),
        registry,
    )
    assert len(cases) == 2


def test_source_paraphrase_and_paraphrase_of_must_go_together(tmp_path, registry):
    with pytest.raises(ValidationError, match="paraphrase"):
        GoldenCase.model_validate(_case("g-106", source="paraphrase"))
    with pytest.raises(ValidationError, match="paraphrase"):
        GoldenCase.model_validate(_case("g-107", paraphrase_of="g-106"))


def test_a_comma_after_a_number_is_punctuation_not_part_of_it(tmp_path, registry):
    """"...in 2024, most common first" reworded to "...in 2024 with the most
    common first" keeps the number. Matching greedily through the comma would
    demand the paraphrase keep the punctuation too, and fail a rewording that
    only moved the clause."""
    cases = load_golden(
        _write(
            tmp_path,
            _case("g-110", question="widget counts in 2024, most common first"),
            _paraphrase(
                "g-110-p1", "g-110", "most common first: widget counts in 2024"
            ),
        ),
        registry,
    )
    assert len(cases) == 2


def test_a_grouped_amounts_commas_are_part_of_the_number(tmp_path, registry):
    """The other half of the same rule: `$5,000,000` is one token, and a
    paraphrase that rounds it to "5 million" has changed the query."""
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-111", question="widgets over $5,000,000"),
                _paraphrase("g-111-p1", "g-111", "widgets over 5 million"),
            ),
            registry,
        )
    assert "5,000,000" in str(exc.value)


def test_a_not_a_query_refusal_is_easy_and_other_reasons_are_not(registry):
    """Owner ruling, 2026-09-18: a greeting is not a medium task. The rule
    sits outside the structural reading because no structure expresses it --
    and it stops short of the other reasons, where refusing a question about
    a table that nearly exists is genuinely not easy."""
    greeting = _built(expect={"kind": "refusal", "reason": "not_a_query"})
    missing = _built(expect={"kind": "refusal", "reason": "field_not_in_schema"})
    assert derive_difficulty(greeting, registry) == "easy"
    assert derive_difficulty(missing, registry) == "medium"


def test_an_ambiguous_not_a_query_case_is_still_hard(registry):
    """The hard checks run first: a hand tag beats the reason shortcut."""
    case = _built(
        tags=["ambiguous-wording"],
        expect={"kind": "refusal", "reason": "not_a_query"},
    )
    assert derive_difficulty(case, registry) == "hard"


def test_a_paraphrase_whose_expectation_differs_is_rejected(tmp_path, registry):
    """A rewording that changes the answer is not a paraphrase. It belongs in
    the set as a case of its own, with its own id and its own gold."""
    other = {
        "root": "widget",
        "where": {"combinator": "AND",
                  "children": [{"field": "widget.status", "op": "=", "value": "X"}]},
    }
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-120", question="active widgets in 2024"),
                _paraphrase("g-120-p1", "g-120", "2024 widgets, active",
                            expect={"kind": "ok", "payload": other}),
            ),
            registry,
        )
    message = str(exc.value)
    assert "g-120-p1" in message and "g-120" in message
    assert "expectation differs" in message


def test_a_paraphrase_may_not_change_the_kind_or_the_field(tmp_path, registry):
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-121", question="widgets in a state",
                      expect={"kind": "clarify", "field": "widget.status"}),
                _paraphrase("g-121-p1", "g-121", "a state of widgets",
                            expect={"kind": "clarify", "field": "widget.region"}),
            ),
            registry,
        )
    assert "expectation differs" in str(exc.value)


def test_an_int_gold_value_does_not_match_a_float_one(tmp_path, registry):
    """Value equality would call `2024` and `2024.0` the same answer; the
    gold conventions pin years as ints and money as floats, so the comparison
    is over the rendered JSON instead."""
    as_float = {
        "root": "widget",
        "where": {"combinator": "AND",
                  "children": [{"field": "widget.year", "op": "=", "value": 2024.0}]},
    }
    as_int = {
        "root": "widget",
        "where": {"combinator": "AND",
                  "children": [{"field": "widget.year", "op": "=", "value": 2024}]},
    }
    with pytest.raises(GoldenError, match="expectation differs"):
        load_golden(
            _write(
                tmp_path,
                _case("g-122", question="widgets from 2024",
                      expect={"kind": "ok", "payload": as_int}),
                _paraphrase("g-122-p1", "g-122", "2024 widgets",
                            expect={"kind": "ok", "payload": as_float}),
            ),
            registry,
        )


def test_a_clarify_case_whose_word_resolves_is_rejected(tmp_path, registry):
    """A gold `clarify` asserts that a value in the question does NOT map
    onto the field's vocabulary. If it does, the service correctly answers
    `ok` and this case scores it a failure for ever -- unreachable gold, the
    same class as a clarify case with nothing left after pruning."""
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-130", question="active widgets in the north",
                      expect={"kind": "clarify", "field": "widget.status"}),
            ),
            registry,
        )
    message = str(exc.value)
    assert "g-130" in message
    assert "active" in message
    assert "'A'" in message


def test_a_two_word_vocabulary_synonym_is_caught_too(tmp_path, registry):
    """`in service` is a synonym of `A`, and it is two words. Checking only
    single words would miss exactly the phrases a vocabulary spells out."""
    with pytest.raises(GoldenError, match="in service"):
        load_golden(
            _write(
                tmp_path,
                _case("g-131", question="widgets in service last year",
                      expect={"kind": "clarify", "field": "widget.status"}),
            ),
            registry,
        )


def test_a_clarify_case_with_a_genuinely_unresolvable_word_loads(tmp_path, registry):
    """Exact matches only, no fuzz: "mothballed" is not `Retired`, and a
    false block here would stop the whole file loading over a case that is
    perfectly good. Nor does the indefinite article count as the code `A`."""
    cases = load_golden(
        _write(
            tmp_path,
            _case("g-132", question="mothballed widgets from a 2024 batch",
                  expect={"kind": "clarify", "field": "widget.status"}),
        ),
        registry,
    )
    assert cases[0].expect.field == "widget.status"
