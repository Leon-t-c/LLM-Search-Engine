import pytest
from pydantic import ValidationError

from cert_nlq.ir.response import Flag, NeedsClarification, Ok, Refused
from cert_nlq.translate.refusals import RefusalReason

PAYLOAD = {"root": "widget", "where": {"combinator": "AND",
           "children": [{"field": "widget.status", "op": "=", "value": "A"}]}}


def test_ok_serialises_with_a_status_discriminator():
    body = Ok(payload=PAYLOAD, registry_version="sha256:fixture-v1").model_dump()
    assert body["status"] == "ok"
    assert body["payload"]["root"] == "widget"
    assert body["flags"] == ()


def test_ok_carries_flags_for_shaky_conditions():
    ok = Ok(
        payload=PAYLOAD,
        registry_version="v",
        flags=[Flag(condition_index=0, phrase="problem widgets", confidence=0.4)],
    )
    assert ok.flags[0].phrase == "problem widgets"


@pytest.mark.parametrize("bad", [1.4, -0.1])
def test_confidence_outside_zero_to_one_is_rejected(bad):
    """Both bounds — an earlier draft exercised only the upper one."""
    with pytest.raises(ValidationError):
        Flag(condition_index=0, phrase="x", confidence=bad)


def test_needs_clarification_requires_candidates():
    """An open 'please be more specific' is never an acceptable response."""
    with pytest.raises(ValidationError, match="at least one candidate"):
        NeedsClarification(
            payload=PAYLOAD,
            unresolved={"phrase": "problem widgets",
                        "question": "What counts as a problem widget?",
                        "candidates": []},
            registry_version="v",
        )


def test_needs_clarification_with_candidates_validates():
    response = NeedsClarification(
        payload=PAYLOAD,
        unresolved={"phrase": "problem widgets",
                    "question": "What counts as a problem widget?",
                    "candidates": [
                        {"field": "widget.status", "op": "=", "value": "X",
                         "label": "Status"}]},
        registry_version="v",
    )
    assert response.status == "needs_clarification"
    assert response.unresolved.candidates[0].label == "Status"


def test_refused_reasons_are_a_closed_set():
    refused = Refused(reason=RefusalReason.NOT_A_QUERY, detail="not a data question")
    assert refused.status == "refused"
    with pytest.raises(ValidationError):
        Refused(reason="i_dont_feel_like_it", detail="x")


def test_ambiguous_is_not_a_refusal_reason():
    """Ambiguity is answerable; it routes to needs_clarification instead."""
    assert "ambiguous" not in {r.value for r in RefusalReason}


def test_needs_aggregation_is_not_a_refusal_reason():
    """Aggregation is a core requirement carried by the IR, not an edge case."""
    assert "needs_aggregation" not in {r.value for r in RefusalReason}
