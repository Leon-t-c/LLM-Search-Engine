import pytest

from cert_nlq.translate.provider import FakeProvider, ProviderError
from cert_nlq.translate.router import RoutingError, route, route_schema


def test_route_returns_the_models_choice(registry):
    provider = FakeProvider([
        {"root": "widget", "joins": ["shipment"], "groups": ["Commercial"]}
    ])
    result = route("total shipment value by region", registry, provider)
    assert (result.root, result.joins, result.groups) == (
        "widget", ("shipment",), ("Commercial",)
    )


def test_route_prompt_lists_roots_joins_and_groups(registry):
    provider = FakeProvider([{"root": "widget", "joins": [], "groups": []}])
    route("anything", registry, provider)
    schema = repr(provider.calls[0]["schema"])
    assert "widget" in schema and "vendor" in schema
    assert "shipment" in schema
    assert "Lifecycle" in schema
    system = provider.calls[0]["system"]
    assert "Widgets" in system, "the prompt must carry human labels, not just keys"


def test_route_prompt_describes_each_group_by_its_contents(registry):
    """A group name may be an opaque UI abbreviation.

    The name alone is not enough to route on, so the prompt carries the
    registry description plus a sample of the group's actual field labels.
    """
    provider = FakeProvider([{"root": "widget", "joins": [], "groups": []}])
    route("anything", registry, provider)
    system = provider.calls[0]["system"]
    assert "in service, retired" in system, "the description must appear"
    assert "Shipped On" in system, "example field labels must appear"


def test_route_prompt_lists_groups_in_registry_order(registry):
    provider = FakeProvider([{"root": "widget", "joins": [], "groups": []}])
    route("anything", registry, provider)
    system = provider.calls[0]["system"]
    positions = [system.index(name) for name in
                 ("Location & Identity", "Lifecycle", "Commercial")]
    assert positions == sorted(positions)


def test_the_field_label_sample_is_capped(registry):
    """A large group must not paste every one of its labels into the call."""
    from cert_nlq.translate.router import MAX_SAMPLE_LABELS, sample_labels

    root = registry.root("widget")
    status = root.fields_by_key["widget.status"]
    padded = root.model_copy(update={"fields": root.fields + tuple(
        status.model_copy(update={"key": f"widget.x{i}", "column": f"x{i}",
                                  "label": f"Extra {i:02d}", "group": "Lifecycle"})
        for i in range(12)
    )})
    labels = sample_labels(padded, "Lifecycle")
    assert MAX_SAMPLE_LABELS == 8
    assert len(labels) == MAX_SAMPLE_LABELS
    assert labels == sorted(labels), "sampled labels come back in label order"


def test_route_rejects_an_unknown_root(registry):
    provider = FakeProvider([{"root": "sprocket", "joins": [], "groups": []}])
    with pytest.raises(RoutingError, match="unknown root"):
        route("anything", registry, provider)


def test_route_drops_a_join_not_available_on_the_root(registry):
    """A hallucinated join is dropped, not refused — the query still runs."""
    provider = FakeProvider([
        {"root": "widget", "joins": ["shipment", "cogwheel"], "groups": []}
    ])
    assert route("anything", registry, provider).joins == ("shipment",)


def test_route_drops_an_unknown_group(registry):
    provider = FakeProvider([
        {"root": "widget", "joins": [], "groups": ["Lifecycle", "Astrology"]}
    ])
    assert route("anything", registry, provider).groups == ("Lifecycle",)


def test_empty_groups_are_preserved_as_all_groups(registry):
    provider = FakeProvider([{"root": "widget", "joins": [], "groups": []}])
    assert route("anything", registry, provider).groups == ()


def test_a_malformed_router_response_raises_routing_error(registry):
    provider = FakeProvider([{"joins": [], "groups": []}])
    with pytest.raises(RoutingError, match="malformed"):
        route("anything", registry, provider)


def test_provider_failure_propagates(registry):
    provider = FakeProvider([ProviderError("upstream down")])
    with pytest.raises(ProviderError):
        route("anything", registry, provider)


def test_route_schema_constrains_root_to_the_registry(registry):
    schema = route_schema(registry)
    assert schema["properties"]["root"]["enum"] == ["widget", "vendor"]
    assert schema["additionalProperties"] is False


def test_a_single_root_registry_routes_on_groups_alone(registry_json):
    """The single-root shape: one root, no joins, groups doing all the work.

    Root selection is trivially correct here, so group selection is the only
    stage-1 signal that means anything — and it must still work.
    """
    from cert_nlq.registry.models import Registry

    solo = Registry.model_validate(
        {**registry_json, "roots": [
            {**registry_json["roots"][0], "joins": []}
        ]}
    )
    provider = FakeProvider([
        {"root": "widget", "joins": [], "groups": ["Lifecycle"]}
    ])
    result = route("retired widgets", solo, provider)
    assert (result.root, result.joins, result.groups) == (
        "widget", (), ("Lifecycle",)
    )
    schema = route_schema(solo)
    assert schema["properties"]["root"]["enum"] == ["widget"]
    # No joins to offer, so the enum falls back to the empty-string sentinel
    # rather than producing an illegal empty enum.
    assert schema["properties"]["joins"]["items"]["enum"] == [""]


def test_the_sample_spreads_across_the_group_rather_than_taking_a_prefix(registry):
    """Labels arrive sorted, so the first eight are an alphabetical accident.

    A prefix makes the examples argue against the description: a group whose
    early labels all share a prefix looks, to the model, like a group about
    only that. The sample has to reach the far end of the range.
    """
    from cert_nlq.translate.router import MAX_SAMPLE_LABELS, sample_labels

    root = registry.root("widget")
    status = root.fields_by_key["widget.status"]
    # Two clearly separated runs: everything a prefix would reach, and
    # everything it would not.
    padded = root.model_copy(update={"fields": tuple(
        status.model_copy(update={"key": f"widget.a{i}", "column": f"a{i}",
                                  "label": f"Aaa {i:02d}", "group": "Lifecycle"})
        for i in range(20)
    ) + tuple(
        status.model_copy(update={"key": f"widget.z{i}", "column": f"z{i}",
                                  "label": f"Zzz {i:02d}", "group": "Lifecycle"})
        for i in range(20)
    )})

    labels = sample_labels(padded, "Lifecycle")

    assert len(labels) == MAX_SAMPLE_LABELS
    assert any(lab.startswith("Zzz") for lab in labels), (
        "a prefix sample would show only the 'Aaa' half and the model would "
        "never learn the group holds anything else"
    )
    assert any(lab.startswith("Aaa") for lab in labels)
    assert labels == sorted(labels), "still in label order"


def test_a_small_group_is_shown_whole(registry):
    """Nothing is dropped or reordered when everything fits."""
    from cert_nlq.translate.router import sample_labels

    root = registry.root("widget")
    labels = sample_labels(root, "Commercial")
    assert labels == [f.label for f in root.fields_in_group("Commercial")]


def test_the_sample_is_identical_across_calls(registry):
    """It is part of a cached prompt prefix, so it must be byte-stable."""
    from cert_nlq.translate.router import sample_labels

    root = registry.root("widget")
    assert sample_labels(root, "Lifecycle") == sample_labels(root, "Lifecycle")
