from cert_nlq.ir.payload import Payload
from cert_nlq.translate.values import (
    resolve_money,
    resolve_relative_year,
    resolve_values,
    resolve_vocabulary,
)


def _spec(registry, key):
    return registry.root("widget").fields_by_key[key]


def test_vocabulary_matches_a_meaning(registry):
    assert resolve_vocabulary(_spec(registry, "widget.status"), "Active") == "A"


def test_vocabulary_matches_a_synonym_case_insensitively(registry):
    assert resolve_vocabulary(_spec(registry, "widget.status"), "IN SERVICE") == "A"


def test_vocabulary_matches_a_code_already_supplied(registry):
    assert resolve_vocabulary(_spec(registry, "widget.status"), "X") == "X"


def test_vocabulary_returns_none_when_nothing_matches(registry):
    assert resolve_vocabulary(_spec(registry, "widget.status"), "banana") is None


def test_vocabulary_is_none_for_a_field_without_one(registry):
    assert resolve_vocabulary(_spec(registry, "widget.price"), "anything") is None


def test_relative_year_uses_the_calendar_year_not_the_data():
    """MAX(year) in the real data is corrupt; never derive 'latest' from it."""
    assert resolve_relative_year("this year", 2026) == 2026
    assert resolve_relative_year("last year", 2026) == 2025
    assert resolve_relative_year("this tax year", 2026) == 2026
    assert resolve_relative_year("2024", 2026) == 2024
    assert resolve_relative_year("a while ago", 2026) is None


def test_money_strips_currency_formatting():
    assert resolve_money("$50,000") == 50000.0
    assert resolve_money("50000") == 50000.0
    assert resolve_money("1.5") == 1.5
    assert resolve_money("lots") is None


def test_resolve_values_rewrites_a_vocabulary_condition(registry):
    payload = Payload.model_validate(
        {"root": "widget", "combinator": "AND",
         "conditions": [{"field": "widget.status", "op": "=", "value": "retired"}]}
    )
    resolved, unresolved = resolve_values(payload, registry.root("widget"), now_year=2026)
    assert resolved.conditions[0].value == "X"
    assert unresolved == []


def test_resolve_values_rewrites_a_relative_year(registry):
    payload = Payload.model_validate(
        {"root": "widget", "combinator": "AND",
         "conditions": [{"field": "widget.year", "op": "=", "value": "last year"}]}
    )
    resolved, _ = resolve_values(payload, registry.root("widget"), now_year=2026)
    assert resolved.conditions[0].value == 2025


def test_resolve_values_rewrites_money_including_value2(registry):
    payload = Payload.model_validate(
        {"root": "widget", "combinator": "AND",
         "conditions": [{"field": "widget.price", "op": "between",
                         "value": "$10,000", "value2": "$20,000"}]}
    )
    resolved, _ = resolve_values(payload, registry.root("widget"), now_year=2026)
    assert (resolved.conditions[0].value, resolved.conditions[0].value2) == (10000.0, 20000.0)


def test_placeholders_pass_through_untouched(registry):
    """The host re-injects the real name and resolves it against its data."""
    payload = Payload.model_validate(
        {"root": "vendor", "combinator": "AND",
         "conditions": [{"field": "vendor.name", "op": "contains", "value": "<NAME_1>"}]}
    )
    resolved, unresolved = resolve_values(payload, registry.root("vendor"), now_year=2026)
    assert resolved.conditions[0].value == "<NAME_1>"
    assert unresolved == []


def test_an_unmatched_vocabulary_value_is_reported_not_guessed(registry):
    payload = Payload.model_validate(
        {"root": "widget", "combinator": "AND",
         "conditions": [{"field": "widget.status", "op": "=", "value": "banana"}]}
    )
    resolved, unresolved = resolve_values(payload, registry.root("widget"), now_year=2026)
    assert unresolved == ["widget.status=banana"]
    assert resolved.conditions[0].value == "banana"


def test_valueless_operators_are_left_alone(registry):
    payload = Payload.model_validate(
        {"root": "widget", "combinator": "AND",
         "conditions": [{"field": "widget.certified", "op": "is_true"}]}
    )
    resolved, unresolved = resolve_values(payload, registry.root("widget"), now_year=2026)
    assert resolved.conditions[0].value is None
    assert unresolved == []


def test_a_field_missing_from_the_root_is_passed_through(registry):
    """Validation already rejected it; resolution must not crash on it."""
    payload = Payload.model_validate(
        {"root": "widget", "combinator": "AND",
         "conditions": [{"field": "unknown.field", "op": "=", "value": "z"}]}
    )
    resolved, unresolved = resolve_values(payload, registry.root("widget"), now_year=2026)
    assert resolved.conditions[0].value == "z"
    assert unresolved == []
