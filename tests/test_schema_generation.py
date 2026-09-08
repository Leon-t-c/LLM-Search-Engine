import json

import pytest
from pydantic import ValidationError

from cert_nlq.ir.schema import (
    ScopeError,
    build_payload_model,
    fields_in_scope,
    json_schema_for,
)


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
        {"root": "widget", "where": {"combinator": "AND",
         "children": [{"field": "widget.status", "op": "=", "value": "A"}]}}
    )
    assert parsed.where.children[0].field == "widget.status"


def test_generated_model_rejects_an_out_of_scope_field(registry):
    model = build_payload_model(registry.root("widget"), groups=("Lifecycle",))
    with pytest.raises(ValidationError):
        model.model_validate(
            {"root": "widget", "where": {"combinator": "AND",
             "children": [{"field": "widget.price", "op": ">", "value": 1}]}}
        )


def test_generated_model_rejects_a_fabricated_field(registry):
    model = build_payload_model(registry.root("widget"))
    with pytest.raises(ValidationError):
        model.model_validate(
            {"root": "widget", "where": {"combinator": "AND",
             "children": [{"field": "widget.invented", "op": "=", "value": "x"}]}}
        )


def test_generated_model_pins_root_to_a_single_value(registry):
    model = build_payload_model(registry.root("widget"))
    with pytest.raises(ValidationError):
        model.model_validate({"root": "vendor"})


def test_join_choices_come_from_the_registry(registry):
    model = build_payload_model(registry.root("widget"))
    model.model_validate({"root": "widget", "join": ["shipment"]})
    with pytest.raises(ValidationError):
        model.model_validate({"root": "widget", "join": ["arbitrary_table"]})


def test_count_star_is_an_allowed_aggregate_target(registry):
    model = build_payload_model(registry.root("widget"))
    parsed = model.model_validate(
        {"root": "widget",
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
        {"root": "vendor", "where": {"combinator": "AND",
         "children": [{"field": "vendor.name", "op": "contains", "value": "x"}]}}
    )
    assert parsed.join == ()
    with pytest.raises(ValidationError):
        model.model_validate({"root": "vendor", "join": ["widget"]})


def test_an_unknown_group_still_leaves_the_identity_group_in_scope(registry):
    """A stage-1 group miss must degrade, never strand every field."""
    keys = {f.key for f in fields_in_scope(registry.root("widget"),
                                           groups=("Nonexistent Group",))}
    assert keys == {"widget.region", "widget.serial", "widget.year"}


def test_a_genuinely_empty_slice_is_a_programming_error(registry):
    empty = registry.root("widget").model_copy(update={"fields": ()})
    with pytest.raises(ValueError, match="no fields in scope"):
        build_payload_model(empty)


def test_a_genuinely_empty_slice_raises_a_scope_error(registry):
    """A distinct type, so the HTTP layer can tell this apart from a bug."""
    empty = registry.root("widget").model_copy(update={"fields": ()})
    assert issubclass(ScopeError, ValueError)
    with pytest.raises(ScopeError):
        build_payload_model(empty)


def test_the_field_key_enum_is_not_repeated(registry):
    """Every duplicate copy is tokens paid on every single call."""
    schema = json_schema_for(registry.root("widget"))
    body = json.dumps(schema)
    # Count the enum lists that contain a representative key.
    assert body.count('"widget.shipped"') == 2, (
        "expected one copy in the shared field enum and one in the "
        "aggregate enum (which also allows '*')"
    )


def test_deduplication_preserves_validation(registry):
    """The $ref rewrite must not loosen what the schema accepts."""
    model = build_payload_model(registry.root("widget"))
    schema = json_schema_for(registry.root("widget"))
    assert "$defs" in schema
    # the model itself is unchanged — only its serialised schema is rewritten
    model.model_validate(
        {"root": "widget", "where": {"combinator": "AND",
         "children": [{"field": "widget.price", "op": ">", "value": 1}]},
         "columns": ["widget.price"], "group_by": ["widget.region"]}
    )


def test_dedupe_enums_is_idempotent(registry):
    from cert_nlq.ir.schema import dedupe_enums

    once = json_schema_for(registry.root("widget"))
    assert dedupe_enums(once) == once


def test_sort_field_is_vocabulary_closed(registry):
    """`sort.field`'s domain is the same as `columns`; it must be closed too."""
    model = build_payload_model(registry.root("widget"))
    model.model_validate(
        {"root": "widget", "sort": {"field": "widget.price", "dir": "desc"}}
    )
    with pytest.raises(ValidationError):
        model.model_validate(
            {"root": "widget", "sort": {"field": "widget.invented", "dir": "desc"}}
        )


def test_sort_still_requires_exactly_one_target(registry):
    """Narrowing `field` by subclassing must not drop the inherited xor rule."""
    model = build_payload_model(registry.root("widget"))
    with pytest.raises(ValidationError):
        model.model_validate(
            {"root": "widget",
             "sort": {"field": "widget.price", "agg": "total", "dir": "asc"}}
        )


def test_having_op_is_closed_but_agg_is_not(registry):
    """`op` is a fixed comparison set; `agg` names a same-payload alias.

    The alias is not knowable when this schema is built, so it stays an open
    string and validation resolves it. The operator has no such excuse.
    """
    model = build_payload_model(registry.root("widget"))
    model.model_validate(
        {"root": "widget",
         "aggregate": [{"fn": "count", "field": "*", "as": "n"}],
         "having": [{"agg": "n", "op": ">", "value": 1}]}
    )
    with pytest.raises(ValidationError):
        model.model_validate(
            {"root": "widget",
             "having": [{"agg": "n", "op": "not_an_operator", "value": 1}]}
        )


def test_an_operator_outside_the_slice_is_rejected(registry):
    """The op union is closed, even though it is not scoped per field."""
    model = build_payload_model(registry.root("widget"))
    with pytest.raises(ValidationError):
        model.model_validate(
            {"root": "widget", "where": {"combinator": "AND",
             "children": [{"field": "widget.status", "op": "made_up", "value": "A"}]}}
        )


def test_a_hoisted_ref_carries_no_sibling_keys(registry):
    """Bare `$ref` — some providers' strict modes reject siblings beside it."""
    schema = json_schema_for(registry.root("widget"))
    refs = []

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                refs.append(node)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    assert refs, "expected at least one hoisted enum"
    offenders = [r for r in refs if set(r) != {"$ref"}]
    assert offenders == [], f"$ref with siblings: {offenders}"


def _walk(node):
    """Every dict in the schema, including the root."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def _is_object(node):
    return node.get("type") == "object" or isinstance(node.get("properties"), dict)


def test_the_generated_schema_meets_the_strict_subset(registry):
    """Both providers reject a schema that leaves an object open.

    No stub can catch this: a fake client accepts any dict, so the failure
    only ever appears against the live API, after the routing call has
    already been paid for.
    """
    schema = json_schema_for(registry.root("widget"))
    nodes = list(_walk(schema))
    objects = [n for n in nodes if _is_object(n)]
    # Otherwise a walker that visited nothing would satisfy every `== []`
    # assertion below.
    assert objects, "expected the walk to find objects to check"

    open_objects = [n for n in objects if n.get("additionalProperties") is not False]
    assert open_objects == [], f"objects left open: {len(open_objects)}"

    for node in objects:
        properties = list(node.get("properties") or ())
        assert list(node.get("required") or ()) == properties, (
            f"required does not cover properties: {node.get('required')} "
            f"vs {properties}"
        )

    assert [n for n in nodes if "default" in n] == []
    assert [n for n in nodes if "title" in n] == []


def test_closing_the_schema_does_not_change_what_is_accepted(registry):
    """Validation runs off the models, so the wire shape must not matter.

    Every key stated, absences stated explicitly — the only payload shape the
    closed schema now permits — still validates.
    """
    from cert_nlq.ir.validate import validate_payload

    payload = validate_payload(
        {
            "root": "widget",
            "where": None,
            "columns": [],
            "join": [],
            "aggregate": [{"fn": "count", "field": "*", "as": "n"}],
            "group_by": [],
            "having": [],
            "sort": None,
        },
        registry,
    )
    assert payload.columns == ()
    assert payload.where is None


def test_the_strict_pass_leaves_every_hoisted_ref_bare(registry):
    """Closing the objects must not add a sibling beside a `$ref`."""
    schema = json_schema_for(registry.root("widget"))
    nodes = list(_walk(schema))
    assert [n for n in nodes if "$ref" in n], "expected at least one hoisted enum"
    offenders = [n for n in nodes if "$ref" in n and set(n) != {"$ref"}]
    assert offenders == []


def test_a_fixed_value_is_expressed_as_a_single_value_enum(registry):
    """One vendor's normaliser knows `enum` and does not know `const`.

    An unrecognised keyword is not an error there — it is demoted into the
    description, which silently unconstrains the slot. `enum` means the same
    thing for one value and both vendors support it, so nothing is lost.
    """
    for name in ("widget", "vendor"):
        schema = json_schema_for(registry.root(name))
        nodes = list(_walk(schema))
        assert [n for n in nodes if "const" in n] == []
        pinned = [n for n in nodes if isinstance(n.get("enum"), list)
                  and len(n["enum"]) == 1]
        assert pinned, "expected the single-value slot to stay constrained"
        assert all(n.get("type") == "string" for n in pinned), (
            "the sibling type must survive the rewrite"
        )


def test_the_single_valued_slot_names_exactly_the_one_legal_value(registry):
    schema = json_schema_for(registry.root("widget"))
    assert schema["properties"]["root"] == {"enum": ["widget"], "type": "string"}


def test_the_strict_pass_treats_a_property_map_as_data_not_as_a_schema():
    """A field must survive whatever it happens to be called.

    Walking every dict value makes a property's *name* collide with a schema
    keyword: one called `title` is dropped and one called `properties` has
    the closing keywords injected into it. Nothing in the registry hits this
    today; a generator that silently drops a field by name is the wrong thing
    to leave lying around.
    """
    from cert_nlq.ir.schema import strictify

    result = strictify({
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "default": {"type": "string"},
            "properties": {"type": "string"},
        },
    })
    assert set(result["properties"]) == {"title", "default", "properties"}
    assert result["properties"]["properties"] == {"type": "string"}
    assert result["required"] == ["title", "default", "properties"]


def test_the_strict_pass_still_strips_the_keywords_where_they_are_keywords():
    from cert_nlq.ir.schema import strictify

    result = strictify({
        "type": "object",
        "title": "Decoration",
        "properties": {"a": {"type": "string", "title": "A", "default": "x"}},
    })
    assert "title" not in result
    assert result["properties"]["a"] == {"type": "string"}


def test_the_strict_pass_reaches_every_schema_position():
    from cert_nlq.ir.schema import strictify

    inner = {"type": "object", "title": "drop me",
             "properties": {"a": {"type": "string"}}}
    result = strictify({
        "$defs": {"D": dict(inner)},
        "type": "object",
        "properties": {
            "one": {"type": "array", "items": dict(inner)},
            "two": {"anyOf": [dict(inner), {"type": "null"}]},
            "three": {"oneOf": [dict(inner)]},
            "four": {"allOf": [dict(inner)]},
            "five": {"type": "array", "prefixItems": [dict(inner)]},
        },
    })
    reached = [
        result["$defs"]["D"],
        result["properties"]["one"]["items"],
        result["properties"]["two"]["anyOf"][0],
        result["properties"]["three"]["oneOf"][0],
        result["properties"]["four"]["allOf"][0],
        result["properties"]["five"]["prefixItems"][0],
    ]
    assert len(reached) == 6
    for node in reached:
        assert "title" not in node
        assert node["additionalProperties"] is False
        assert node["required"] == ["a"]


def test_an_object_with_no_properties_gets_no_required_list():
    """An empty `required` is not what either vendor's own pass emits."""
    from cert_nlq.ir.schema import strictify

    result = strictify({"type": "object"})
    assert result["additionalProperties"] is False
    assert "required" not in result


def test_the_strict_pass_leaves_a_bare_ref_bare():
    from cert_nlq.ir.schema import strictify

    assert strictify({"$ref": "#/$defs/X"}) == {"$ref": "#/$defs/X"}
