import pytest
from pydantic import ValidationError

from cert_nlq.registry.models import FieldSpec, JoinSpec, Registry


def test_fixture_parses(registry):
    assert registry.version == "sha256:fixture-v1"
    assert [r.root for r in registry.roots] == ["widget", "vendor"]


def test_root_lookup(registry):
    assert registry.root("widget").label == "Widgets"
    assert registry.root("nope") is None


def test_fields_by_key(registry):
    widget = registry.root("widget")
    assert widget.fields_by_key["widget.status"].label == "Status"


def test_group_names_follow_registry_order_not_alphabetical(registry):
    """The host app lists sections the way its tabs do; preserve that."""
    assert registry.root("widget").group_names() == [
        "Location & Identity", "Lifecycle", "Commercial", "Aftercare",
    ]


def test_identity_groups_are_marked_by_the_registry(registry):
    """No hard-coded group name — each root declares its own."""
    assert registry.root("widget").identity_groups() == {"Location & Identity"}
    assert registry.root("vendor").identity_groups() == {"Location & Identity"}


def test_group_lookup_carries_the_description(registry):
    group = registry.root("widget").group("Lifecycle")
    assert "retired" in group.description
    assert group.identity is False
    assert registry.root("widget").group("absent") is None


def test_fields_in_group_are_sorted_by_label(registry):
    labels = [f.label for f in registry.root("widget").fields_in_group("Commercial")]
    assert labels == sorted(labels)
    assert "Price" in labels


def test_an_undeclared_group_is_appended_not_rejected(registry_json):
    """A cosmetic group mismatch must not take the whole feature down.

    `group_names` reports it, `group` has no metadata for it. This mirrors the
    host app, which also shows an unlisted group after the ordered ones.
    """
    widget = registry_json["roots"][0]
    widget["fields"].append(
        {"key": "widget.note", "table": "widget", "column": "note",
         "label": "Note", "type": "text", "group": "Undeclared",
         "operators": ["contains"], "vocabulary": []}
    )
    root = Registry.model_validate(registry_json).root("widget")
    assert root.group_names() == [
        "Location & Identity", "Lifecycle", "Commercial", "Aftercare",
        "Undeclared",
    ]
    assert root.group("Undeclared") is None


def test_join_lookup_normalises_pairs(registry):
    join = registry.root("widget").join("shipment")
    assert join.cardinality == "one-to-many"
    assert join.on == (("region", "region"), ("serial", "serial"), ("year", "shipyear"))
    assert registry.root("widget").join("absent") is None


def test_bare_join_column_expands_to_a_pair():
    """The spec allowed a bare shared name; we normalise it to a pair."""
    join = JoinSpec.model_validate(
        {"table": "t", "label": "T", "cardinality": "one-to-one", "on": ["year"]}
    )
    assert join.on == (("year", "year"),)


def test_unknown_field_type_is_rejected():
    with pytest.raises(ValidationError):
        FieldSpec.model_validate(
            {"key": "a.b", "table": "a", "column": "b", "label": "B",
             "type": "blob", "group": "G", "operators": ["="]}
        )


def test_a_field_with_no_operators_is_rejected():
    with pytest.raises(ValidationError):
        FieldSpec.model_validate(
            {"key": "a.b", "table": "a", "column": "b", "label": "B",
             "type": "text", "group": "G", "operators": []}
        )


def test_unknown_cardinality_is_rejected():
    with pytest.raises(ValidationError):
        JoinSpec.model_validate(
            {"table": "t", "label": "T", "cardinality": "sometimes", "on": ["x"]}
        )


def test_duplicate_root_names_are_rejected(registry_json):
    registry_json["roots"].append(registry_json["roots"][0])
    with pytest.raises(ValidationError):
        Registry.model_validate(registry_json)
