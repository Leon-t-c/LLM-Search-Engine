import pytest

from cert_nlq.ir.payload import Payload
from cert_nlq.registry.models import VocabEntry
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
        {"root": "widget", "where": {"combinator": "AND",
         "children": [{"field": "widget.status", "op": "=", "value": "retired"}]}}
    )
    resolved, unresolved = resolve_values(payload, registry.root("widget"), now_year=2026)
    assert resolved.where.children[0].value == "X"
    assert unresolved == []


def test_resolve_values_rewrites_a_relative_year(registry):
    payload = Payload.model_validate(
        {"root": "widget", "where": {"combinator": "AND",
         "children": [{"field": "widget.year", "op": "=", "value": "last year"}]}}
    )
    resolved, _ = resolve_values(payload, registry.root("widget"), now_year=2026)
    assert resolved.where.children[0].value == 2025


def test_resolve_values_rewrites_money_including_value2(registry):
    payload = Payload.model_validate(
        {"root": "widget", "where": {"combinator": "AND",
         "children": [{"field": "widget.price", "op": "between",
                       "value": "$10,000", "value2": "$20,000"}]}}
    )
    resolved, _ = resolve_values(payload, registry.root("widget"), now_year=2026)
    assert (resolved.where.children[0].value, resolved.where.children[0].value2) == (
        10000.0, 20000.0)


def test_placeholders_pass_through_untouched(registry):
    """The host re-injects the real name and resolves it against its data."""
    payload = Payload.model_validate(
        {"root": "vendor", "where": {"combinator": "AND",
         "children": [{"field": "vendor.name", "op": "contains", "value": "<NAME_1>"}]}}
    )
    resolved, unresolved = resolve_values(payload, registry.root("vendor"), now_year=2026)
    assert resolved.where.children[0].value == "<NAME_1>"
    assert unresolved == []


def test_an_unmatched_vocabulary_value_is_reported_not_guessed(registry):
    payload = Payload.model_validate(
        {"root": "widget", "where": {"combinator": "AND",
         "children": [{"field": "widget.status", "op": "=", "value": "banana"}]}}
    )
    resolved, unresolved = resolve_values(payload, registry.root("widget"), now_year=2026)
    assert unresolved == ["widget.status=banana"]
    assert resolved.where.children[0].value == "banana"


def test_valueless_operators_are_left_alone(registry):
    payload = Payload.model_validate(
        {"root": "widget", "where": {"combinator": "AND",
         "children": [{"field": "widget.certified", "op": "is_true"}]}}
    )
    resolved, unresolved = resolve_values(payload, registry.root("widget"), now_year=2026)
    assert resolved.where.children[0].value is None
    assert unresolved == []


def test_a_field_missing_from_the_root_is_passed_through(registry):
    """Validation already rejected it; resolution must not crash on it."""
    payload = Payload.model_validate(
        {"root": "widget", "where": {"combinator": "AND",
         "children": [{"field": "unknown.field", "op": "=", "value": "z"}]}}
    )
    resolved, unresolved = resolve_values(payload, registry.root("widget"), now_year=2026)
    assert resolved.where.children[0].value == "z"
    assert unresolved == []


def test_money_accepts_a_negative_amount():
    """Credits and reversals are legitimate here — this is deliberate."""
    assert resolve_money("-$500") == -500.0


@pytest.mark.parametrize(
    "raw", ["nan", "NaN", "  nan  ", "inf", "-inf", "Infinity", "1e400", "-1e400"]
)
def test_money_rejects_non_finite_values(raw):
    """`float()` parses these without raising, and overflows 1e400 to inf.

    A non-finite amount that is reported as *resolved* reaches the query as a
    value matching nothing — the silent wrong answer this module exists to
    prevent. The overflow case is what makes it reachable: a model emitting a
    large number in scientific notation needs no exotic input to trigger it.
    """
    assert resolve_money(raw) is None


def test_a_non_finite_money_value_is_reported_unresolved(registry):
    """End to end: the rejection must surface in `unresolved`, not vanish."""
    payload = Payload.model_validate(
        {"root": "widget", "where": {"combinator": "AND",
         "children": [{"field": "widget.price", "op": ">", "value": "1e400"}]}}
    )
    resolved, unresolved = resolve_values(payload, registry.root("widget"), now_year=2026)
    assert unresolved == ["widget.price=1e400"]
    assert resolved.where.children[0].value == "1e400", "left as written, not coerced"


def test_a_code_beats_another_entrys_synonym(registry):
    """Precedence matters only under collision, so build one deliberately."""
    spec = registry.root("widget").fields_by_key["widget.status"].model_copy(
        update={"vocabulary": (
            VocabEntry(code="A", meaning="Active", synonyms=()),
            VocabEntry(code="B", meaning="Retired", synonyms=("A",)),
        )}
    )
    assert resolve_vocabulary(spec, "A") == "A", "the code must win, not B's synonym"


def test_a_vocabulary_beats_the_type_parser(registry):
    """A field may declare both a vocabulary and a numeric type.

    The registry does not forbid it, so the dispatch order is reachable. The
    vocabulary wins: an explicit entry the field's owner defined outranks a
    generic parse.
    """
    spec = registry.root("widget").fields_by_key["widget.price"].model_copy(
        update={"vocabulary": (VocabEntry(code="Z", meaning="unpriced", synonyms=()),)}
    )
    assert resolve_vocabulary(spec, "unpriced") == "Z"
    payload = Payload.model_validate(
        {"root": "widget", "where": {"combinator": "AND",
         "children": [{"field": "widget.price", "op": "=", "value": "unpriced"}]}}
    )
    root = registry.root("widget")
    patched = root.model_copy(update={"fields": tuple(
        spec if f.key == "widget.price" else f for f in root.fields
    )})
    resolved, unresolved = resolve_values(payload, patched, now_year=2026)
    assert resolved.where.children[0].value == "Z"
    assert unresolved == []
