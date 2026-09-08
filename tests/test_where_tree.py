import json

import pytest
from pydantic import ValidationError

from cert_nlq.ir.payload import Condition, Group, Payload, iter_conditions
from cert_nlq.ir.schema import MAX_GROUP_DEPTH, build_payload_model, json_schema_for
from cert_nlq.ir.validate import PayloadError, validate_payload


def _leaf(field="widget.status", op="=", value="A"):
    return {"field": field, "op": op, "value": value}


def test_a_group_of_conditions_parses():
    p = Payload.model_validate(
        {"root": "widget",
         "where": {"combinator": "AND", "children": [_leaf(), _leaf("widget.year", "=", 2024)]}}
    )
    assert p.where.combinator == "AND"
    assert [type(c).__name__ for c in p.where.children] == ["Condition", "Condition"]


def test_the_shape_that_used_to_be_refused_now_parses():
    """`(A or B) and C` — the question the old format could not express."""
    p = Payload.model_validate({"root": "widget", "where": {
        "combinator": "AND", "children": [
            {"combinator": "OR", "children": [_leaf(value="A"), _leaf(value="X")]},
            _leaf("widget.year", "=", 2024),
        ]}})
    inner = p.where.children[0]
    assert isinstance(inner, Group) and inner.combinator == "OR"
    assert len(list(iter_conditions(p.where))) == 3


def test_iter_conditions_reaches_every_leaf_at_depth():
    p = Payload.model_validate({"root": "widget", "where": {
        "combinator": "AND", "children": [
            {"combinator": "OR", "children": [
                {"combinator": "AND", "children": [_leaf(), _leaf("widget.year", "=", 2024)]},
                _leaf("widget.price", ">", 1),
            ]},
            _leaf("widget.region", "=", "N"),
        ]}})
    assert sorted(c.field for c in iter_conditions(p.where)) == [
        "widget.price", "widget.region", "widget.status", "widget.year",
    ]


def test_an_empty_group_is_rejected():
    with pytest.raises(ValidationError, match="at least one child"):
        Group.model_validate({"combinator": "AND", "children": []})


def test_in_takes_a_list_of_values():
    c = Condition.model_validate(
        {"field": "widget.status", "op": "in", "value": ["A", "X"]}
    )
    assert c.value == ("A", "X")


def test_a_payload_with_no_where_and_no_aggregate_is_rejected(registry):
    with pytest.raises(PayloadError, match="at least one condition"):
        validate_payload({"root": "widget"}, registry)


def test_validation_reaches_conditions_inside_groups(registry):
    """A bad leaf must be caught however deeply it is nested."""
    with pytest.raises(PayloadError, match="operator '>' not allowed"):
        validate_payload({"root": "widget", "where": {
            "combinator": "AND", "children": [
                {"combinator": "OR", "children": [
                    _leaf(), {"field": "widget.status", "op": ">", "value": "A"}]},
            ]}}, registry)


def test_in_requires_a_non_empty_list(registry):
    with pytest.raises(PayloadError, match="'in' needs at least one value"):
        validate_payload({"root": "widget", "where": {"combinator": "AND", "children": [
            {"field": "widget.status", "op": "in", "value": []}]}}, registry)


def test_a_scalar_operator_rejects_a_list(registry):
    with pytest.raises(PayloadError, match="takes a single value"):
        validate_payload({"root": "widget", "where": {"combinator": "AND", "children": [
            {"field": "widget.status", "op": "=", "value": ["A", "X"]}]}}, registry)


def test_depth_beyond_the_bound_is_rejected(registry):
    """The schema cannot offer it, but a hand-written payload can arrive."""
    node = _leaf()
    for _ in range(MAX_GROUP_DEPTH + 1):
        node = {"combinator": "AND", "children": [node]}
    with pytest.raises(PayloadError, match="nested more than"):
        validate_payload({"root": "widget", "where": node}, registry)


def test_the_generated_schema_is_finite_not_recursive(registry):
    """A recursive schema hits depth limits in some strict output modes."""
    schema = json_schema_for(registry.root("widget"))
    defs = schema.get("$defs", {})

    def refs(node, out):
        if isinstance(node, dict):
            if "$ref" in node:
                out.add(node["$ref"].split("/")[-1])
            for v in node.values():
                refs(v, out)
        elif isinstance(node, list):
            for v in node:
                refs(v, out)
        return out

    for name, body in defs.items():
        seen, stack = set(), list(refs(body, set()))
        while stack:
            n = stack.pop()
            assert n != name, f"{name} is self-referential"
            if n in seen:
                continue
            seen.add(n)
            stack += list(refs(defs.get(n, {}), set()))


def test_the_generated_model_accepts_a_nested_where(registry):
    model = build_payload_model(registry.root("widget"))
    model.model_validate({"root": "widget", "where": {
        "combinator": "AND", "children": [
            {"combinator": "OR", "children": [_leaf(value="A"), _leaf(value="X")]},
            _leaf("widget.year", "=", 2024)]}})


def test_the_generated_model_still_closes_the_vocabulary_inside_a_group(registry):
    """Nesting must not become a hole in the closed-vocabulary guarantee."""
    model = build_payload_model(registry.root("widget"))
    with pytest.raises(ValidationError):
        model.model_validate({"root": "widget", "where": {
            "combinator": "AND", "children": [
                {"combinator": "OR", "children": [
                    {"field": "widget.invented", "op": "=", "value": "A"}]}]}})


def test_values_resolve_inside_groups(registry):
    from cert_nlq.translate.values import resolve_values

    p = Payload.model_validate({"root": "widget", "where": {
        "combinator": "AND", "children": [
            {"combinator": "OR", "children": [
                {"field": "widget.status", "op": "=", "value": "retired"}]},
            {"field": "widget.year", "op": "=", "value": "last year"}]}})
    out, unresolved = resolve_values(p, registry.root("widget"), now_year=2026)
    got = {c.field: c.value for c in iter_conditions(out.where)}
    assert got == {"widget.status": "X", "widget.year": 2025}
    assert unresolved == []


def test_in_values_resolve_elementwise(registry):
    from cert_nlq.translate.values import resolve_values

    p = Payload.model_validate({"root": "widget", "where": {"combinator": "AND", "children": [
        {"field": "widget.status", "op": "in", "value": ["retired", "in service"]}]}})
    out, unresolved = resolve_values(p, registry.root("widget"), now_year=2026)
    assert list(iter_conditions(out.where))[0].value == ("X", "A")
    assert unresolved == []


def test_an_unresolvable_element_is_reported_not_dropped(registry):
    from cert_nlq.translate.values import Unresolvable, resolve_values

    p = Payload.model_validate({"root": "widget", "where": {"combinator": "AND", "children": [
        {"field": "widget.status", "op": "in", "value": ["retired", "banana"]}]}})
    out, unresolved = resolve_values(p, registry.root("widget"), now_year=2026)
    assert unresolved == [Unresolvable(field="widget.status", op="in", value="banana")]
    assert list(iter_conditions(out.where))[0].value == ("retired", "banana")


def test_looks_nested_is_gone():
    """The heuristic and its refusal reason both go: the shape is expressible."""
    import cert_nlq.translate.refusals as r

    assert not hasattr(r, "looks_nested")
    assert "needs_nested_logic" not in {x.value for x in r.RefusalReason}
