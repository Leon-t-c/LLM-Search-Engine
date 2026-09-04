import pytest
from pydantic import ValidationError

from cert_nlq.ir.payload import Aggregate, Payload, Sort


def test_minimal_payload_round_trips():
    payload = Payload.model_validate(
        {"root": "widget", "combinator": "AND",
         "conditions": [{"field": "widget.status", "op": "=", "value": "A"}]}
    )
    assert payload.root == "widget"
    assert payload.conditions[0].value == "A"
    assert payload.join == ()
    assert payload.sort is None


def test_aggregate_uses_as_on_the_wire_and_alias_in_python():
    agg = Aggregate.model_validate({"fn": "sum", "field": "shipment.amount", "as": "total"})
    assert agg.alias == "total"
    assert agg.model_dump(by_alias=True)["as"] == "total"


def test_aggregate_function_must_be_enumerated():
    with pytest.raises(ValidationError):
        Aggregate.model_validate({"fn": "stddev", "field": "widget.price", "as": "s"})


def test_combinator_must_be_and_or_or():
    with pytest.raises(ValidationError):
        Payload.model_validate({"root": "widget", "combinator": "XOR", "conditions": []})


def test_sort_requires_exactly_one_of_field_or_agg():
    with pytest.raises(ValidationError, match="exactly one"):
        Sort.model_validate({"field": "widget.price", "agg": "total", "dir": "asc"})
    with pytest.raises(ValidationError, match="exactly one"):
        Sort.model_validate({"dir": "asc"})


def test_sort_direction_is_constrained():
    with pytest.raises(ValidationError):
        Sort.model_validate({"field": "widget.price", "dir": "sideways"})


def test_full_aggregate_payload_parses():
    payload = Payload.model_validate({
        "root": "widget",
        "combinator": "AND",
        "conditions": [{"field": "widget.year", "op": "=", "value": 2024}],
        "join": ["shipment"],
        "aggregate": [{"fn": "sum", "field": "shipment.amount", "as": "total"},
                      {"fn": "count", "field": "*", "as": "n"}],
        "group_by": ["widget.region"],
        "having": [{"agg": "total", "op": ">", "value": 100000}],
        "sort": {"agg": "total", "dir": "desc"},
    })
    assert [a.alias for a in payload.aggregate] == ["total", "n"]
    assert payload.having[0].value == 100000
    assert payload.sort.agg == "total"
