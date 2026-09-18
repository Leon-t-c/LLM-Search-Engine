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
        json.dumps({"id": "g-070", "question": "hi", "tags": ["selection"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenError, match="g-070"):
        load_golden(path, registry)
