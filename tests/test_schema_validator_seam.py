"""Check the generated schema and `validate_payload` against each other.

Both sides are covered exhaustively on their own — one suite asserts what the
generated schema constrains, another asserts what validation rejects — and
until this module existed the *seam* between them was covered nowhere. That
is the expensive gap. The design claim the whole service rests on is that a
structured-output model cannot emit a payload the next stage refuses; what
was actually true was the weaker claim that it cannot emit an unknown *word*.
Anything the schema permits and validation refuses costs a retry, and the
translator allows exactly one before it gives up — so two of them in a single
answer is a refusal for a question the service could have answered.

The technique here is a table of hand-written payloads, not a property-testing
library. A generator would buy coverage of value combinations that neither
side inspects, and cost the reader the ability to see at a glance which shapes
are claimed to round-trip. **The point is that the two sides are checked
against each other at all**, not that the checking is clever.

Each table entry is asserted twice: the constrained model accepts it (so the
row cannot drift into describing a payload the schema would never offer) and
`validate_payload` accepts it too. The rows walk every slot, all three
nesting levels, and the fully-explicit form the closed schema forces — every
key stated, every absence stated as `null` or `[]`.

Written after a review found four disagreements by reading the two files side
by side. It would have caught all of them:

* a group with no children — permitted, because the array-length keyword that
  would forbid it is not in the strict subset, and refused downstream as a
  raw parse error rather than a collected problem;
* the placeholder the schema emits in `join` when a root has no joins at all,
  which validation then rejected as a table that does not exist;
* an absent filter with no aggregate beside it, whose message named only one
  of the two ways to satisfy it;
* a `having` comparison operator, which the schema closes to six values and
  validation did not check at all — the one case running the other way, where
  the gate the host trusts was the looser of the two.

And a fifth the review did not list: a `sort` object stating both of its two
targets as `null`, which the closed schema offers as a second way to say "no
sort" and which parsing then refused.
"""
import pytest

from cert_nlq.ir.schema import HAVING_OPS, NO_JOIN, build_payload_model
from cert_nlq.ir.validate import PayloadError, validate_payload
from cert_nlq.translate.translator import _normalise


def _explicit(**overrides) -> dict:
    """A payload in the only shape the closed schema permits.

    Every property is required there — that is what "closed" means — so a
    model has to state each absence rather than omit the key. Any row that
    quietly omitted one would be testing a shape the model can never send.
    """
    payload = {
        "root": "widget",
        "where": None,
        "columns": [],
        "join": [],
        "aggregate": [],
        "group_by": [],
        "having": [],
        "sort": None,
    }
    payload.update(overrides)
    return payload


def _condition(field="widget.status", op="=", value="A", value2=None) -> dict:
    return {"field": field, "op": op, "value": value, "value2": value2}


def _group(*children, combinator="AND") -> dict:
    return {"combinator": combinator, "children": list(children)}


#: Payloads the constrained schema offers and validation must accept. Named,
#: because a bare list of dicts tells a failing run nothing about what broke.
ROUND_TRIPS = {
    "a filter and nothing else": _explicit(where=_group(_condition())),
    "the fully explicit absent-everything form": _explicit(
        aggregate=[{"fn": "count", "field": "*", "as": "n"}]
    ),
    "an operator that takes no value, stated as two nulls": _explicit(
        where=_group(_condition("widget.certified", "is_true", None))
    ),
    "a list operator": _explicit(
        where=_group(_condition("widget.status", "in", ["A", "X"]))
    ),
    "a range operator carrying its second value": _explicit(
        where=_group(_condition("widget.year", "between", 2020, 2024))
    ),
    "two levels of grouping": _explicit(
        where=_group(
            _condition("widget.region", "=", "N"),
            _group(
                _condition("widget.status", "=", "A"),
                _condition("widget.status", "=", "X"),
                combinator="OR",
            ),
        )
    ),
    "three levels of grouping, the deepest the schema offers": _explicit(
        where=_group(
            _condition("widget.region", "=", "N"),
            _group(
                _condition("widget.year", ">", 2020),
                _group(
                    _condition("widget.status", "=", "A"),
                    _condition("widget.status", "=", "X"),
                    combinator="OR",
                ),
            ),
        )
    ),
    "columns beside a filter": _explicit(
        where=_group(_condition()), columns=["widget.region", "widget.price"]
    ),
    "an aggregate broken down per category, ordered and filtered": _explicit(
        aggregate=[{"fn": "sum", "field": "widget.price", "as": "total"}],
        group_by=["widget.region"],
        having=[{"agg": "total", "op": ">", "value": 100}],
        sort={"field": None, "agg": "total", "dir": "desc"},
    ),
    "a sort on a field, with the other target stated as null": _explicit(
        where=_group(_condition()),
        sort={"field": "widget.price", "agg": None, "dir": "asc"},
    ),
    "a join, with the joined table's field used": _explicit(
        where=_group(_condition("shipment.amount", ">", 1000)),
        join=["shipment"],
        columns=["shipment.amount"],
    ),
}


@pytest.mark.parametrize("name", sorted(ROUND_TRIPS))
def test_what_the_schema_offers_validation_accepts(registry, name):
    """One row of the table, through both gates in turn."""
    payload = ROUND_TRIPS[name]
    model = build_payload_model(registry.root("widget"), joins=("shipment",))
    # Asserted first: a row the constrained model rejects is a row describing
    # a payload no model could send, and proves nothing about the seam.
    model.model_validate(payload)
    validate_payload(payload, registry)


@pytest.mark.parametrize("name", sorted(ROUND_TRIPS))
def test_every_row_states_every_slot(name):
    """Guards the table itself: an omitted key is not a shape the model sends."""
    assert set(ROUND_TRIPS[name]) == {
        "root", "where", "columns", "join",
        "aggregate", "group_by", "having", "sort",
    }


def test_the_table_reaches_the_deepest_nesting_the_schema_allows(registry):
    """Otherwise the depth rows could all quietly shrink to one level."""
    def depth(node, level=1):
        if not isinstance(node, dict) or "children" not in node:
            return level - 1
        return max((depth(c, level + 1) for c in node["children"]), default=level)

    from cert_nlq.ir.schema import MAX_GROUP_DEPTH

    deepest = max(depth(p["where"]) for p in ROUND_TRIPS.values())
    assert deepest == MAX_GROUP_DEPTH


# --------------------------------------------------------------------------
# The disagreements. Each is legal under the schema and was not accepted
# downstream; each is now reconciled before validation rather than bounced
# back to the model as its one retry.
# --------------------------------------------------------------------------


def test_an_empty_group_is_normalised_away_not_bounced_back(registry):
    """The strict subset has no way to say "at least one child"."""
    raw = _explicit(
        where=_group(_condition(), _group()),
        aggregate=[{"fn": "count", "field": "*", "as": "n"}],
    )
    # The constrained model itself permits it, which is the whole problem:
    # the schema handed to the provider offers this shape.
    build_payload_model(registry.root("widget")).model_validate(raw)
    validate_payload(_normalise(raw), registry)


def test_a_group_emptied_all_the_way_down_leaves_no_filter(registry):
    """Nesting does not rescue an empty group — the prune is recursive."""
    raw = _explicit(where=_group(_group(_group())))
    assert _normalise(raw)["where"] is None


def test_an_emptied_filter_with_nothing_beside_it_still_fails_clearly(registry):
    """Normalising must not turn an empty filter into a silent whole-table scan."""
    raw = _normalise(_explicit(where=_group()))
    with pytest.raises(PayloadError) as exc:
        validate_payload(raw, registry)
    assert exc.value.problems == ["a payload needs at least one condition or one aggregate"]


def test_a_surviving_group_keeps_its_real_children(registry):
    """The prune drops the empty ones and nothing else."""
    raw = _explicit(
        where=_group(_condition("widget.region", "=", "N"), _group(), _condition())
    )
    kept = _normalise(raw)["where"]["children"]
    assert [c["field"] for c in kept] == ["widget.region", "widget.status"]


def test_the_no_joins_placeholder_is_filtered_out_before_validation(registry):
    """A root with no joins still has to offer the slot *something*."""
    model = build_payload_model(registry.root("vendor"))
    raw = _explicit(
        root="vendor",
        where=_group(_condition("vendor.name", "contains", "x")),
        join=[NO_JOIN],
    )
    model.model_validate(raw)
    validate_payload(_normalise(raw), registry)


def test_the_placeholder_is_the_only_join_a_join_less_root_offers(registry):
    """If this ever stops being true the filter above is papering over a bug."""
    from cert_nlq.ir.schema import json_schema_for

    schema = json_schema_for(registry.root("vendor"))
    assert schema["properties"]["join"]["items"]["enum"] == [NO_JOIN]


def test_filtering_the_placeholder_leaves_a_real_join_alone(registry):
    raw = _explicit(where=_group(_condition()), join=["shipment", NO_JOIN])
    assert _normalise(raw)["join"] == ["shipment"]


def test_an_absent_filter_with_no_aggregate_names_both_ways_out(registry):
    """The message is fed back verbatim; naming one of two remedies halves it."""
    with pytest.raises(PayloadError) as exc:
        validate_payload(_explicit(columns=["widget.region"]), registry)
    assert exc.value.problems == ["a payload needs at least one condition or one aggregate"]


def test_the_prompt_names_both_ways_out_too(registry):
    from cert_nlq.translate.translator import TRANSLATOR_SYSTEM

    assert "at least one condition or one aggregate" in TRANSLATOR_SYSTEM


def test_the_prompt_rules_out_the_shape_the_schema_cannot(registry):
    """The empty group is normalised away; the prompt should still discourage it."""
    from cert_nlq.translate.translator import TRANSLATOR_SYSTEM

    assert "at least one child" in TRANSLATOR_SYSTEM


@pytest.mark.parametrize("op", HAVING_OPS)
def test_every_comparison_the_schema_offers_having_is_accepted(registry, op):
    validate_payload(
        _explicit(
            aggregate=[{"fn": "count", "field": "*", "as": "n"}],
            having=[{"agg": "n", "op": op, "value": 1}],
        ),
        registry,
    )


def test_a_having_operator_outside_that_set_is_rejected(registry):
    """The one case that ran the other way: the schema was the tighter gate.

    Validation is what the caller re-checks against, so a comparison the
    schema forbids must not slip through it — a payload assembled by anything
    other than a structured-output call would carry it straight past.
    """
    with pytest.raises(PayloadError, match="not a comparison"):
        validate_payload(
            _explicit(
                aggregate=[{"fn": "count", "field": "*", "as": "n"}],
                having=[{"agg": "n", "op": "contains", "value": 1}],
            ),
            registry,
        )


def test_the_two_sides_read_the_comparison_set_from_one_place():
    """Two lists that say the same thing today are two lists that can drift."""
    from cert_nlq.ir import validate as validate_module

    assert validate_module.HAVING_OPS is HAVING_OPS


def test_a_sort_stating_neither_target_means_no_sort(registry):
    """The closed schema offers two ways to say "no sort"; both must work.

    `sort` is nullable, so `null` says it. But `sort` is also required and its
    two targets are both nullable, so an object with each of them null says it
    just as plainly — and that form was refused.
    """
    raw = _explicit(
        where=_group(_condition()), sort={"field": None, "agg": None, "dir": "asc"}
    )
    validate_payload(_normalise(raw), registry)


def test_the_schema_really_does_offer_that_form(registry):
    """Without this the row above is only asserting the normaliser's own shape."""
    schema = build_payload_model(registry.root("widget")).model_json_schema()
    sort = schema["$defs"]["WidgetSort"]
    for target in ("field", "agg"):
        assert {"type": "null"} in sort["properties"][target]["anyOf"]


def test_a_sort_naming_both_targets_is_still_a_real_error(registry):
    """Normalising the empty case must not start swallowing the ambiguous one."""
    from pydantic import ValidationError

    raw = _explicit(
        where=_group(_condition()),
        sort={"field": "widget.price", "agg": "total", "dir": "asc"},
    )
    with pytest.raises(ValidationError):
        validate_payload(_normalise(raw), registry)


def test_normalising_leaves_a_payload_with_nothing_to_fix_untouched(registry):
    """It is a reconciliation pass, not a rewrite pass."""
    for payload in ROUND_TRIPS.values():
        assert _normalise(payload) == payload


def test_normalising_leaves_a_shape_it_does_not_recognise_alone(registry):
    """Model output is not trusted to be well-formed; the real error must survive."""
    junk = {"root": "widget", "where": "not a group", "join": "not a list", "sort": 7}
    assert _normalise(junk) == junk
    assert _normalise([1, 2, 3]) == [1, 2, 3]


# --------------------------------------------------------------------------
# The reason any of this matters: the retry budget. One retry, then a
# refusal — so a disagreement is not a cosmetic mismatch, it is an answerable
# question going unanswered. These drive the whole translator and assert the
# provider was called exactly twice: once to route, once to translate. A
# third call means the disagreement is still costing the retry.
# --------------------------------------------------------------------------

_ROUTE = {"root": "widget", "joins": [], "groups": ["Lifecycle"]}


def test_an_empty_group_does_not_cost_the_retry(registry):
    from cert_nlq.translate.provider import FakeProvider
    from cert_nlq.translate.translator import translate

    provider = FakeProvider([
        _ROUTE,
        _explicit(
            where=_group(_condition("widget.status", "=", "Active"), _group()),
        ),
    ])
    response = translate("active widgets", registry, provider, now_year=2026)
    assert response.status == "ok"
    assert len(provider.calls) == 2


def test_the_no_joins_placeholder_does_not_cost_the_retry(registry):
    from cert_nlq.translate.provider import FakeProvider
    from cert_nlq.translate.translator import translate

    provider = FakeProvider([
        {"root": "vendor", "joins": [], "groups": []},
        _explicit(
            root="vendor",
            where=_group(_condition("vendor.name", "contains", "acme")),
            join=[NO_JOIN],
        ),
    ])
    response = translate("vendors called acme", registry, provider, now_year=2026)
    assert response.status == "ok"
    assert response.payload.join == ()
    assert len(provider.calls) == 2


def test_a_sort_naming_neither_target_does_not_cost_the_retry(registry):
    from cert_nlq.translate.provider import FakeProvider
    from cert_nlq.translate.translator import translate

    provider = FakeProvider([
        _ROUTE,
        _explicit(
            where=_group(_condition("widget.status", "=", "Active")),
            sort={"field": None, "agg": None, "dir": "asc"},
        ),
    ])
    response = translate("active widgets", registry, provider, now_year=2026)
    assert response.status == "ok"
    assert response.payload.sort is None
    assert len(provider.calls) == 2


def test_two_disagreements_in_one_answer_would_have_cost_the_whole_budget(registry):
    """The compounding case, spelled out: one retry covers at most one round.

    Not a demonstration of a bug that still exists — all three shapes above
    are reconciled now. It pins the cost of ever letting one back in.
    """
    from cert_nlq.translate.translator import _MAX_ATTEMPTS

    assert _MAX_ATTEMPTS == 2
