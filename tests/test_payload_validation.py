import pytest

from cert_nlq.ir.validate import PayloadError, validate_payload


def _ok(**overrides):
    conditions = overrides.pop(
        "conditions", [{"field": "widget.status", "op": "=", "value": "A"}]
    )
    payload = {"root": "widget"}
    if conditions:
        payload["where"] = {"combinator": "AND", "children": conditions}
    payload.update(overrides)
    return payload


def test_a_good_payload_validates(registry):
    result = validate_payload(_ok(), registry)
    assert result.where.children[0].field == "widget.status"


def test_unknown_root_is_rejected(registry):
    with pytest.raises(PayloadError, match="unknown root"):
        validate_payload(_ok(root="sprocket"), registry)


def test_unknown_field_is_rejected(registry):
    with pytest.raises(PayloadError, match="unknown field"):
        validate_payload(
            _ok(conditions=[{"field": "widget.nope", "op": "=", "value": "A"}]), registry
        )


def test_operator_not_allowed_for_the_field_is_rejected(registry):
    """`>` is legal on numbers and absent from a text field's operator list."""
    with pytest.raises(PayloadError, match="operator '>' not allowed"):
        validate_payload(
            _ok(conditions=[{"field": "widget.status", "op": ">", "value": "A"}]), registry
        )


def test_a_value_operator_requires_a_value(registry):
    with pytest.raises(PayloadError, match="requires a value"):
        validate_payload(
            _ok(conditions=[{"field": "widget.status", "op": "="}]), registry
        )


def test_between_requires_a_second_value(registry):
    with pytest.raises(PayloadError, match="requires 'value2'"):
        validate_payload(
            _ok(conditions=[{"field": "widget.year", "op": "between", "value": 2023}]),
            registry,
        )


def test_a_non_between_operator_must_not_carry_value2(registry):
    with pytest.raises(PayloadError, match="takes no 'value2'"):
        validate_payload(
            _ok(conditions=[{"field": "widget.year", "op": "=", "value": 2023,
                             "value2": 2025}]),
            registry,
        )


def test_a_blank_operator_must_not_carry_a_value(registry):
    with pytest.raises(PayloadError, match="takes no value"):
        validate_payload(
            _ok(conditions=[{"field": "widget.status", "op": "is_blank", "value": "A"}]),
            registry,
        )


def test_a_joined_field_without_the_join_is_rejected(registry):
    with pytest.raises(PayloadError, match="requires join"):
        validate_payload(
            _ok(conditions=[{"field": "shipment.amount", "op": ">", "value": 5}]),
            registry,
        )


def test_a_joined_field_with_the_join_is_accepted(registry):
    result = validate_payload(
        _ok(join=["shipment"],
            conditions=[{"field": "shipment.amount", "op": ">", "value": 5}]),
        registry,
    )
    assert result.join == ("shipment",)


def test_unenumerated_join_is_rejected(registry):
    with pytest.raises(PayloadError, match="join 'court' is not available"):
        validate_payload(_ok(join=["court"]), registry)


def test_sum_over_a_text_field_is_rejected(registry):
    with pytest.raises(PayloadError, match="cannot sum"):
        validate_payload(
            _ok(aggregate=[{"fn": "sum", "field": "widget.status", "as": "t"}]), registry
        )


def test_sum_over_money_is_accepted(registry):
    result = validate_payload(
        _ok(aggregate=[{"fn": "sum", "field": "widget.price", "as": "total"}],
            group_by=["widget.region"]),
        registry,
    )
    assert result.aggregate[0].alias == "total"


def test_star_is_only_valid_for_count(registry):
    with pytest.raises(PayloadError, match="only 'count' accepts"):
        validate_payload(
            _ok(aggregate=[{"fn": "sum", "field": "*", "as": "t"}]), registry
        )


def test_duplicate_aggregate_aliases_are_rejected(registry):
    with pytest.raises(PayloadError, match="duplicate aggregate alias"):
        validate_payload(
            _ok(aggregate=[{"fn": "count", "field": "*", "as": "n"},
                           {"fn": "max", "field": "widget.price", "as": "n"}]),
            registry,
        )


def test_having_must_reference_a_declared_alias(registry):
    with pytest.raises(PayloadError, match="having references unknown alias"):
        validate_payload(
            _ok(aggregate=[{"fn": "count", "field": "*", "as": "n"}],
                having=[{"agg": "total", "op": ">", "value": 1}]),
            registry,
        )


def test_sort_by_alias_must_reference_a_declared_alias(registry):
    with pytest.raises(PayloadError, match="sort references unknown alias"):
        validate_payload(_ok(sort={"agg": "total", "dir": "desc"}), registry)


def test_sort_by_field_must_be_a_known_field(registry):
    with pytest.raises(PayloadError, match="unknown sort field"):
        validate_payload(_ok(sort={"field": "widget.nope", "dir": "asc"}), registry)


def test_columns_must_be_known_fields(registry):
    with pytest.raises(PayloadError, match="unknown column"):
        validate_payload(_ok(columns=["widget.nope"]), registry)


def test_group_by_must_be_a_known_field(registry):
    with pytest.raises(PayloadError, match="unknown group_by field"):
        validate_payload(
            _ok(aggregate=[{"fn": "count", "field": "*", "as": "n"}],
                group_by=["widget.nope"]),
            registry,
        )


def test_iequals_is_accepted_where_the_registry_offers_it(registry):
    """Case-insensitive equality for dirty coded columns on a case-inconsistent column.

    The translator treats operators as opaque registry strings, so `iequals`
    needs no special handling here — this test exists to prove that, and to
    fail loudly if someone adds an operator allow-list later.
    """
    result = validate_payload(
        _ok(conditions=[{"field": "widget.region", "op": "iequals", "value": "N"}]),
        registry,
    )
    assert result.where.children[0].op == "iequals"


def test_iequals_is_rejected_where_the_registry_does_not_offer_it(registry):
    with pytest.raises(PayloadError, match="operator 'iequals' not allowed"):
        validate_payload(
            _ok(conditions=[{"field": "widget.price", "op": "iequals", "value": 1}]),
            registry,
        )


def test_a_payload_with_no_conditions_and_no_aggregate_is_rejected(registry):
    """An unfiltered scan of a whole table is never the intent."""
    with pytest.raises(PayloadError, match="at least one condition"):
        validate_payload(_ok(conditions=[]), registry)


def test_all_problems_are_reported_together(registry):
    with pytest.raises(PayloadError) as exc:
        validate_payload(_ok(join=["court"], columns=["widget.nope"]), registry)
    assert len(exc.value.problems) == 2


@pytest.mark.parametrize(
    "slot, extra",
    [
        ("column", {"columns": ["shipment.amount"]}),
        ("aggregate field",
         {"aggregate": [{"fn": "sum", "field": "shipment.amount", "as": "t"}]}),
        ("group_by field",
         {"aggregate": [{"fn": "count", "field": "*", "as": "n"}],
          "group_by": ["shipment.amount"]}),
        ("sort field", {"sort": {"field": "shipment.amount", "dir": "desc"}}),
    ],
)
def test_every_slot_says_requires_join_not_unknown(registry, slot, extra):
    """A joinable field must never be reported as nonexistent, in any slot.

    The problem text is fed back to the model verbatim as retry guidance, so
    "add the join" and "pick a different field" send the correction down
    different paths. `conditions` got this right from the start; the other
    four slots did not.
    """
    with pytest.raises(PayloadError) as exc:
        validate_payload(_ok(**extra), registry)
    joined = [p for p in exc.value.problems if "shipment.amount" in p]
    assert joined, f"no problem mentioned the field for slot {slot!r}"
    assert "requires join 'shipment'" in joined[0], joined[0]
    assert "unknown" not in joined[0], joined[0]


def test_a_genuinely_absent_field_still_says_unknown(registry):
    """The counterpart: don't turn every miss into a join suggestion."""
    with pytest.raises(PayloadError, match="unknown column 'nosuch.field'"):
        validate_payload(_ok(columns=["nosuch.field"]), registry)


def test_avg_over_a_text_field_is_rejected(registry):
    """`sum` was covered; `avg` shares the branch and was not."""
    with pytest.raises(PayloadError, match="cannot avg"):
        validate_payload(
            _ok(aggregate=[{"fn": "avg", "field": "widget.status", "as": "a"}]),
            registry,
        )


def test_an_aggregate_over_an_absent_field_is_rejected(registry):
    with pytest.raises(PayloadError, match="unknown aggregate field"):
        validate_payload(
            _ok(aggregate=[{"fn": "sum", "field": "nosuch.field", "as": "t"}]),
            registry,
        )
