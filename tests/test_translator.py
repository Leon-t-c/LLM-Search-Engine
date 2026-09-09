import pytest

from cert_nlq.ir.payload import iter_conditions
from cert_nlq.translate.provider import FakeProvider, ProviderError
from cert_nlq.translate.refusals import RefusalReason
from cert_nlq.translate.translator import TRANSLATOR_SYSTEM, translate

ROUTE = {"root": "widget", "joins": [], "groups": ["Lifecycle"]}


def test_the_prompt_offers_in_as_the_flat_alternative():
    """A same-field disjunction has a flat form; the prompt must say so.

    Prompt text is the input to everything this project measures, and no
    behavioural test reaches it. This is a tripwire, not a proof.
    """
    assert "`in`" in TRANSLATOR_SYSTEM


@pytest.mark.parametrize(
    "slot", ["columns", "join", "aggregate", "group_by", "having", "sort"]
)
def test_the_prompt_names_every_slot_the_payload_offers(slot):
    """A slot the prompt never mentions is a retry the golden set pays for.

    The schema offers all of these; a prompt describing only conditions
    scores the resulting miss as a model failure.
    """
    assert f"`{slot}`" in TRANSLATOR_SYSTEM


def test_the_prompt_sends_counting_to_the_aggregate_slot():
    """The validator carries aggregation, so counting must not be a filter."""
    assert "count" in TRANSLATOR_SYSTEM
    assert "the caller counts the rows" not in TRANSLATOR_SYSTEM


def test_the_prompt_states_the_rule_the_validator_enforces():
    """An empty payload is rejected downstream; say so before it is built."""
    assert "at least one condition" in TRANSLATOR_SYSTEM


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


def test_a_candidate_keeps_the_operator_it_was_asked_under(registry):
    """An `!=` question must not come back as an `=` suggestion.

    Offering the inverse of what was asked is worse than offering nothing:
    it reads as a helpful correction and means the opposite.
    """
    provider = FakeProvider([
        ROUTE,
        {"root": "widget", "where": {"combinator": "AND",
          "children": [{"field": "widget.region", "op": "=", "value": "N"},
                        {"field": "widget.status", "op": "!=", "value": "problematic"}]}},
    ])
    response = translate(
        "widgets up north that are not problematic", registry, provider, now_year=2026
    )
    assert response.status == "needs_clarification"
    assert response.unresolved.candidates
    assert all(c.op == "!=" for c in response.unresolved.candidates)


def test_an_unresolvable_sole_condition_refuses_rather_than_clarifies(registry):
    """With nothing left to populate the builder, a clarification is empty.

    The reason names the value, not the field: `widget.status` is in the
    schema, and saying otherwise sends a reader after the wrong bug.
    """
    provider = FakeProvider([
        ROUTE,
        {"root": "widget", "where": {"combinator": "AND",
          "children": [{"field": "widget.status", "op": "=", "value": "problematic"}]}},
    ])
    response = translate("problematic widgets", registry, provider, now_year=2026)
    assert response.status == "refused"
    assert response.reason == RefusalReason.VALUE_NOT_UNDERSTOOD


def test_the_routers_joins_are_carried_into_the_payload(registry):
    """A joined field must not cost a retry the router already paid for.

    The router decides the joins and the schema then offers that table's
    fields, but nothing asks the model to restate the decision. A payload
    that omits it is not an error, and treating it as one burns the single
    retry and ends in a refusal that names the wrong cause.
    """
    provider = FakeProvider([
        {"root": "widget", "joins": ["shipment"], "groups": ["Commercial"]},
        {"root": "widget", "where": {"combinator": "AND",
          "children": [{"field": "shipment.amount", "op": ">", "value": "$500"}]}},
    ])
    response = translate("widgets shipped for over $500", registry, provider, now_year=2026)
    assert response.status == "ok"
    assert len(provider.calls) == 2, "no retry should have been needed"
    assert response.payload.join == ("shipment",)


def test_a_join_the_model_supplied_survives_the_seeding(registry):
    """Seeding is a union: the model's own join is kept and still checked."""
    provider = FakeProvider([
        {"root": "widget", "joins": ["shipment"], "groups": ["Commercial"]},
        {"root": "widget", "join": ["vendor", "shipment"],
         "where": {"combinator": "AND",
          "children": [{"field": "shipment.amount", "op": ">", "value": 500}]}},
    ])
    response = translate("widgets shipped for over 500", registry, provider, now_year=2026)
    assert response.status == "ok"
    assert response.payload.join == ("vendor", "shipment")


def test_a_join_shaped_failure_does_not_ask_for_a_different_field(registry):
    """"Pick another field" is the opposite of the fix when the field is fine."""
    provider = FakeProvider([
        {"root": "widget", "joins": [], "groups": ["Commercial"]},
        {"root": "widget", "where": {"combinator": "AND",
          "children": [{"field": "shipment.amount", "op": ">", "value": 500}]}},
        {"root": "widget", "join": ["shipment"], "where": {"combinator": "AND",
          "children": [{"field": "shipment.amount", "op": ">", "value": 500}]}},
    ])
    response = translate("widgets shipped for over 500", registry, provider, now_year=2026)
    assert response.status == "ok"
    retry = provider.calls[2]["question"]
    assert "requires join" in retry
    assert "`join`" in retry
    assert "only the offered fields" not in retry


def test_an_ordinary_failure_still_asks_for_the_offered_fields(registry):
    provider = FakeProvider([
        ROUTE,
        {"root": "widget", "where": {"combinator": "AND",
          "children": [{"field": "widget.invented", "op": "=", "value": "A"}]}},
        {"root": "widget", "where": {"combinator": "AND",
          "children": [{"field": "widget.status", "op": "=", "value": "X"}]}},
    ])
    translate("retired widgets", registry, provider, now_year=2026)
    assert "only the offered fields" in provider.calls[2]["question"]


BOTH_GROUPS = {"root": "widget", "joins": [],
               "groups": ["Location & Identity", "Lifecycle"]}


def test_every_unresolved_value_is_pruned_not_just_the_clarified_one(registry):
    """A clarification payload states facts; an unresolved value is not one.

    It is documented as the seed for the caller's builder, so a value that
    failed to resolve shipping inside it shows the user a bogus filter as
    though it were valid, and compiles to a query matching nothing.
    """
    provider = FakeProvider([
        BOTH_GROUPS,
        {"root": "widget", "where": {"combinator": "AND", "children": [
            {"field": "widget.serial", "op": "=", "value": 7},
            {"field": "widget.status", "op": "=", "value": "problematic"},
            {"field": "widget.region", "op": "=", "value": "middle earth"},
        ]}},
    ])
    response = translate("odd widgets", registry, provider, now_year=2026)
    assert response.status == "needs_clarification"
    assert response.unresolved.phrase == "problematic"
    kept = list(iter_conditions(response.payload.where))
    assert [c.field for c in kept] == ["widget.serial"]
    assert all(c.value not in ("problematic", "middle earth") for c in kept)


def test_nothing_survives_the_pruning_so_it_refuses(registry):
    """Two unresolvable conditions and nothing else leave an empty builder."""
    provider = FakeProvider([
        BOTH_GROUPS,
        {"root": "widget", "where": {"combinator": "AND", "children": [
            {"field": "widget.status", "op": "=", "value": "problematic"},
            {"field": "widget.region", "op": "=", "value": "middle earth"},
        ]}},
    ])
    response = translate("odd widgets", registry, provider, now_year=2026)
    assert response.status == "refused"
    assert response.reason == RefusalReason.VALUE_NOT_UNDERSTOOD


def test_a_value_with_no_vocabulary_to_offer_refuses_as_not_understood(registry):
    """An unparseable amount is not a field missing from the schema.

    There is no closed vocabulary to offer as candidates, so there is nothing
    to clarify with and it refuses — but the reason has to say what actually
    happened, or every value failure looks like a schema failure.
    """
    provider = FakeProvider([
        {"root": "widget", "joins": [],
         "groups": ["Commercial", "Location & Identity"]},
        {"root": "widget", "where": {"combinator": "AND", "children": [
            {"field": "widget.serial", "op": "=", "value": 7},
            {"field": "widget.price", "op": ">", "value": "a fair whack"},
        ]}},
    ])
    response = translate("pricey widget 7", registry, provider, now_year=2026)
    assert response.status == "refused"
    assert response.reason == RefusalReason.VALUE_NOT_UNDERSTOOD


def test_a_validation_failure_still_refuses_as_field_not_in_schema(registry):
    """The reason stays put for the case it actually describes."""
    bad = {"root": "widget", "where": {"combinator": "AND",
            "children": [{"field": "widget.invented", "op": "=", "value": "A"}]}}
    provider = FakeProvider([ROUTE, bad, bad])
    response = translate("nonsense", registry, provider, now_year=2026)
    assert response.reason == RefusalReason.FIELD_NOT_IN_SCHEMA


def test_the_join_hint_is_chosen_by_the_kind_of_failure_not_its_wording(
    registry, monkeypatch
):
    """Reword the message and the hint must still be the join one.

    The hint used to be picked by searching the formatted message for a
    phrase, which quietly made that phrase part of the contract: reword it —
    an ordinary, encouraged thing to do to text a model reads — and the
    branch stops firing with nothing failing to say so. This test rewords it
    on purpose.
    """
    from cert_nlq.ir import validate as validate_module
    from cert_nlq.translate.provider import FakeProvider

    real = validate_module._resolve_field

    def reworded(key, available, all_fields, label):
        spec, problem = real(key, available, all_fields, label)
        if getattr(problem, "kind", None) == validate_module.PROBLEM_UNJOINED:
            problem = validate_module.Problem(
                validate_module.PROBLEM_UNJOINED,
                f"{key!r} lives on a table you have not asked for",
            )
        return spec, problem

    monkeypatch.setattr(validate_module, "_resolve_field", reworded)

    provider = FakeProvider([
        {"root": "widget", "joins": [], "groups": ["Commercial"]},
        {"root": "widget", "where": {"combinator": "AND",
          "children": [{"field": "shipment.amount", "op": ">", "value": 500}]}},
        {"root": "widget", "join": ["shipment"], "where": {"combinator": "AND",
          "children": [{"field": "shipment.amount", "op": ">", "value": 500}]}},
    ])
    translate("widgets shipped for over 500", registry, provider, now_year=2026)
    retry = provider.calls[2]["question"]
    assert "lives on a table you have not asked for" in retry
    assert "`join`" in retry, "the join hint, chosen without reading the message"
    assert "only the offered fields" not in retry


def test_a_failure_of_another_kind_still_gets_the_general_hint(registry):
    """The counterpart: the structural branch must not fire on everything."""
    from cert_nlq.ir.validate import PROBLEM_UNJOINED, PayloadError

    error = PayloadError(["unknown field 'widget.invented'"])
    assert PROBLEM_UNJOINED not in error.kinds
