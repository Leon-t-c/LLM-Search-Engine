import pytest

from cert_nlq.ir.payload import iter_conditions
from cert_nlq.translate.provider import FakeProvider, ProviderError
from cert_nlq.translate.refusals import RefusalReason
from cert_nlq.translate.translator import translate

ROUTE = {"root": "widget", "joins": [], "groups": ["Lifecycle"]}


def test_a_clean_question_produces_ok(registry):
    provider = FakeProvider([
        ROUTE,
        {"root": "widget", "where": {"combinator": "AND",
          "children": [{"field": "widget.status", "op": "=", "value": "Active"}]}},
    ])
    response = translate("active widgets", registry, provider, now_year=2026)
    assert response.status == "ok"
    first = next(iter_conditions(response.payload.where))
    assert first.value == "A", "vocabulary must resolve"
    assert response.registry_version == "sha256:fixture-v1"


def test_an_unresolved_value_becomes_a_clarification_with_candidates(registry):
    provider = FakeProvider([
        ROUTE,
        {"root": "widget", "where": {"combinator": "AND",
          "children": [{"field": "widget.region", "op": "=", "value": "N"},
                        {"field": "widget.status", "op": "=", "value": "problematic"}]}},
    ])
    response = translate("problematic widgets up north", registry, provider, now_year=2026)
    assert response.status == "needs_clarification"
    assert response.unresolved.phrase == "problematic"
    # the resolved condition still populates the builder
    assert [c.field for c in iter_conditions(response.payload.where)] == [
        "widget.region"
    ]
    # candidates are the field's own codes, labelled and never paraphrased
    assert {c.value for c in response.unresolved.candidates} == {"A", "X"}
    assert response.unresolved.candidates[0].label == "Status"


def test_a_schema_invalid_response_retries_once_then_succeeds(registry):
    provider = FakeProvider([
        ROUTE,
        {"root": "widget", "where": {"combinator": "AND",
          "children": [{"field": "widget.invented", "op": "=", "value": "A"}]}},
        {"root": "widget", "where": {"combinator": "AND",
          "children": [{"field": "widget.status", "op": "=", "value": "X"}]}},
    ])
    response = translate("retired widgets", registry, provider, now_year=2026)
    assert response.status == "ok"
    assert len(provider.calls) == 3
    assert "failed validation" in provider.calls[2]["question"]


def test_two_schema_invalid_responses_refuse(registry):
    bad = {"root": "widget", "where": {"combinator": "AND",
            "children": [{"field": "widget.invented", "op": "=", "value": "A"}]}}
    provider = FakeProvider([ROUTE, bad, bad])
    response = translate("nonsense", registry, provider, now_year=2026)
    assert response.status == "refused"
    assert response.reason == RefusalReason.FIELD_NOT_IN_SCHEMA


def test_a_routing_failure_refuses_as_not_a_query(registry):
    provider = FakeProvider([{"root": "sprocket", "joins": [], "groups": []}])
    response = translate("how do I file a form?", registry, provider, now_year=2026)
    assert response.status == "refused"
    assert response.reason == RefusalReason.NOT_A_QUERY


def test_provider_errors_are_not_swallowed(registry):
    """The API layer decides how to degrade; the translator must not hide it."""
    provider = FakeProvider([ProviderError("rate limited")])
    with pytest.raises(ProviderError):
        translate("active widgets", registry, provider, now_year=2026)


def test_stage_two_prompt_only_carries_the_narrowed_slice(registry):
    provider = FakeProvider([
        ROUTE,
        {"root": "widget", "where": {"combinator": "AND",
          "children": [{"field": "widget.status", "op": "=", "value": "A"}]}},
    ])
    translate("active widgets", registry, provider, now_year=2026)
    schema = repr(provider.calls[1]["schema"])
    assert "widget.status" in schema
    assert "widget.price" not in schema, "the Commercial group was not routed to"


def test_relative_year_resolves_against_the_supplied_year(registry):
    provider = FakeProvider([
        {"root": "widget", "joins": [], "groups": ["Location & Identity"]},
        {"root": "widget", "where": {"combinator": "AND",
          "children": [{"field": "widget.year", "op": "=", "value": "last year"}]}},
    ])
    response = translate("widgets from last year", registry, provider, now_year=2026)
    assert next(iter_conditions(response.payload.where)).value == 2025


def test_an_unresolvable_sole_condition_refuses_rather_than_clarifies(registry):
    """With nothing left to populate the builder, a clarification is empty."""
    provider = FakeProvider([
        ROUTE,
        {"root": "widget", "where": {"combinator": "AND",
          "children": [{"field": "widget.status", "op": "=", "value": "problematic"}]}},
    ])
    response = translate("problematic widgets", registry, provider, now_year=2026)
    assert response.status == "refused"
    assert response.reason == RefusalReason.FIELD_NOT_IN_SCHEMA
