import pytest
from pydantic import ValidationError

from cert_nlq.ir.schema import build_payload_model, fields_in_scope, json_schema_for


def test_empty_groups_means_every_group(registry):
    root = registry.root("widget")
    keys = {f.key for f in fields_in_scope(root, groups=(), joins=())}
    # shipment.amount belongs to a joined table, so it is out of scope
    # until the join is requested.
    assert keys == {
        "widget.region", "widget.serial", "widget.year",
        "widget.status", "widget.price", "widget.shipped", "widget.certified",
    }


def test_named_groups_narrow_the_scope(registry):
    root = registry.root("widget")
    keys = {f.key for f in fields_in_scope(root, groups=("Lifecycle",), joins=())}
    assert keys == {
        "widget.status", "widget.certified",
        # the identity group is always included so a router miss stays recoverable
        "widget.region", "widget.serial", "widget.year",
    }


def test_the_identity_group_comes_from_the_registry_not_a_constant(registry):
    """Group names are root-prefixed, so no constant could work across roots."""
    root = registry.root("widget")
    moved = root.model_copy(update={"groups": tuple(
        g.model_copy(update={"identity": g.name == "Lifecycle"}) for g in root.groups
    )})
    keys = {f.key for f in fields_in_scope(moved, groups=("Commercial",))}
    assert keys == {
        "widget.price", "widget.shipped",          # the selected group
        "widget.status", "widget.certified",       # now the identity group
    }


def test_scope_preserves_registry_field_order(registry):
    root = registry.root("widget")
    scoped = [f.key for f in fields_in_scope(root, groups=("Commercial",))]
    declared = [f.key for f in root.fields if f.key in set(scoped)]
    assert scoped == declared


def test_a_joined_table_brings_its_fields_into_scope(registry):
    root = registry.root("widget")
    keys = {f.key for f in fields_in_scope(root, groups=("Commercial",), joins=("shipment",))}
    assert "shipment.amount" in keys


def test_a_join_no_selected_group_covers_falls_back_to_all_its_fields(registry):
    """Otherwise the join is requested and contributes nothing."""
    root = registry.root("widget")
    keys = {f.key for f in fields_in_scope(root, groups=("Lifecycle",), joins=("shipment",))}
    assert "shipment.amount" in keys


def test_generated_model_accepts_an_in_scope_field(registry):
    model = build_payload_model(registry.root("widget"), groups=("Lifecycle",))
    parsed = model.model_validate(
        {"root": "widget", "combinator": "AND",
         "conditions": [{"field": "widget.status", "op": "=", "value": "A"}]}
    )
    assert parsed.conditions[0].field == "widget.status"


def test_generated_model_rejects_an_out_of_scope_field(registry):
    model = build_payload_model(registry.root("widget"), groups=("Lifecycle",))
    with pytest.raises(ValidationError):
        model.model_validate(
            {"root": "widget", "combinator": "AND",
             "conditions": [{"field": "widget.price", "op": ">", "value": 1}]}
        )


def test_generated_model_rejects_a_fabricated_field(registry):
    model = build_payload_model(registry.root("widget"))
    with pytest.raises(ValidationError):
        model.model_validate(
            {"root": "widget", "combinator": "AND",
             "conditions": [{"field": "widget.invented", "op": "=", "value": "x"}]}
        )


def test_generated_model_pins_root_to_a_single_value(registry):
    model = build_payload_model(registry.root("widget"))
    with pytest.raises(ValidationError):
        model.model_validate({"root": "vendor", "combinator": "AND", "conditions": []})


def test_join_choices_come_from_the_registry(registry):
    model = build_payload_model(registry.root("widget"))
    model.model_validate(
        {"root": "widget", "combinator": "AND", "conditions": [], "join": ["shipment"]}
    )
    with pytest.raises(ValidationError):
        model.model_validate(
            {"root": "widget", "combinator": "AND", "conditions": [],
             "join": ["arbitrary_table"]}
        )


def test_count_star_is_an_allowed_aggregate_target(registry):
    model = build_payload_model(registry.root("widget"))
    parsed = model.model_validate(
        {"root": "widget", "combinator": "AND", "conditions": [],
         "aggregate": [{"fn": "count", "field": "*", "as": "n"}]}
    )
    assert parsed.aggregate[0].field == "*"


def test_json_schema_enumerates_fields_and_omits_out_of_scope_ones(registry):
    schema = json_schema_for(registry.root("widget"), groups=("Lifecycle",))
    body = repr(schema)
    assert "widget.status" in body
    assert "widget.price" not in body


def test_a_root_with_no_joins_still_generates(registry):
    """Literal[()] is illegal, so an empty join list needs a sentinel."""
    model = build_payload_model(registry.root("vendor"))
    parsed = model.model_validate(
        {"root": "vendor", "combinator": "AND",
         "conditions": [{"field": "vendor.name", "op": "contains", "value": "x"}]}
    )
    assert parsed.join == ()
    with pytest.raises(ValidationError):
        model.model_validate(
            {"root": "vendor", "combinator": "AND", "conditions": [], "join": ["widget"]}
        )


def test_an_unknown_group_still_leaves_the_identity_group_in_scope(registry):
    """A stage-1 group miss must degrade, never strand every field."""
    keys = {f.key for f in fields_in_scope(registry.root("widget"),
                                           groups=("Nonexistent Group",))}
    assert keys == {"widget.region", "widget.serial", "widget.year"}


def test_a_genuinely_empty_slice_is_a_programming_error(registry):
    empty = registry.root("widget").model_copy(update={"fields": ()})
    with pytest.raises(ValueError, match="no fields in scope"):
        build_payload_model(empty)
