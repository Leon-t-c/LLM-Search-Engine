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
    ExpectClarify,
    ExpectOk,
    ExpectRefusal,
    GoldenError,
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
        "tags": ["selection"],
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
    assert cases[0].tags == ("selection",)
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
            _case("g-001", tags=["clarify"],
                  expect={"kind": "clarify", "field": "widget.status"}),
            _case("g-002", tags=["refusal:not_a_query"],
                  expect={"kind": "refusal", "reason": "not_a_query"}),
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
                _case("g-050", tags=["refusal:too_hard"],
                      expect={"kind": "refusal", "reason": "too_hard"}),
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
                _case("g-051", tags=["clarify"],
                      expect={"kind": "clarify", "field": "widget.nope"}),
            ),
            registry,
        )
    assert "g-051" in str(exc.value)


def test_a_case_needs_exactly_one_slice_tag(tmp_path, registry):
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-060", tags=["regression"]),
                _case("g-061", tags=["selection", "aggregate"]),
            ),
            registry,
        )
    assert "g-060" in str(exc.value)
    assert "g-061" in str(exc.value)


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
                _case("g-002", tags=["refusal:not_a_query"],
                      expect={"kind": "refusal", "reason": "nope"}),
                _case("g-003", tags=["clarify"],
                      expect={"kind": "clarify", "field": "widget.nope"}),
                _case("g-004"),
                _case("g-004"),
            ),
            registry,
        )
    message = str(exc.value)
    for case_id in ("g-001", "g-002", "g-003", "g-004"):
        assert case_id in message
    # Five, not four: g-002's invented reason is wrong twice over -- the reason
    # itself is not in the closed set, and the slice tag it was filed under
    # claims a different one. Two defects, two sentences.
    assert len(exc.value.problems) == 5


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
        json.dumps({"id": "g-070", "question": "hi", "tags": ["selection"]}) + "\n",
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
                _case("g-082", tags=["clarify"],
                      expect={"kind": "clarify", "field": "widget.serial"}),
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
                _case("g-083", tags=["clarify"],
                      expect={"kind": "clarify", "field": "widget.nope"}),
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


def test_a_slice_tag_must_agree_with_the_expectation_kind(tmp_path, registry):
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-085", tags=["aggregate"],
                      expect={"kind": "refusal", "reason": "not_a_query"}),
                _case("g-086", tags=["clarify"]),
            ),
            registry,
        )
    message = str(exc.value)
    assert "g-085" in message and "'aggregate'" in message and "'refusal'" in message
    assert "g-086" in message and "'clarify'" in message and "'ok'" in message


def test_a_refusal_tag_must_name_the_expected_reason(tmp_path, registry):
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-087", tags=["refusal:not_a_query"],
                      expect={"kind": "refusal", "reason": "field_not_in_schema"}),
            ),
            registry,
        )
    message = str(exc.value)
    assert "g-087" in message
    assert "refusal:not_a_query" in message
    assert "field_not_in_schema" in message


def test_a_stray_key_in_a_gold_payload_is_rejected(tmp_path, registry):
    """`Payload` ignores extras -- the service must stay tolerant on its wire.
    A gold payload is the opposite: `groupby` would validate perfectly and then
    never match anything, reporting a typo in the ruler as a translation miss."""
    with pytest.raises(GoldenError) as exc:
        load_golden(
            _write(
                tmp_path,
                _case("g-088", tags=["aggregate"], expect={"kind": "ok", "payload": {
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
