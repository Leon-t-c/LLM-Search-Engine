"""Per-question scoring: (golden case, service outcome) -> one flat `Row`.

Every function here is pure. `score` never raises -- a malformed or entirely
absent service response (a timeout, a 502) must still produce a row, because
the alternative is an eval run that stops at the first flaky call instead of
recording it as one.

`Outcome` is this module's own vocabulary, not the wire's -- see its
docstring for the mapping a runner must apply before constructing one. Making
that vocabulary a `Literal` means the wrong one (the wire's own strings)
cannot be constructed in the first place, rather than silently comparing
false forever.
"""
import json
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..registry.models import FieldSpec, RootSpec
from ..translate.values import resolve_money
from .golden import GoldenCase, iter_condition_dicts

#: Aggregate token-count keys a usage record may carry (`translate.provider`).
#: The input keys are disjoint by construction -- every adapter converts into
#: this non-overlapping form -- so summing them is meaningful even across a
#: run mixing vendors. `reasoning_tokens` is deliberately excluded: it is a
#: breakdown of `output_tokens`, not a sibling of it.
_INPUT_USAGE_KEYS = ("uncached_input_tokens", "cached_input_tokens", "cache_write_tokens")


class Outcome(BaseModel):
    """What one replayed question got back, flattened for scoring.

    `status` is the golden set's vocabulary (`GoldenCase.expect.kind`), not
    the wire's (`ir.response`) -- a runner must translate before
    constructing one:

    | wire (`ir.response`)              | `Outcome.status` |
    |------------------------------------|------------------|
    | `Ok.status` (`"ok"`)               | `"ok"`           |
    | `NeedsClarification.status`        | `"clarify"`      |
    | (`"needs_clarification"`)          |                  |
    | `Refused.status` (`"refused"`)     | `"refusal"`      |
    | transport failure (timeout, 502)   | `None`, with `error` set |

    This is a `Literal`, not a bare `str`, so passing the wire's own
    `"refused"` raises at construction instead of silently comparing false
    against `status_actual == "refusal"` forever -- the trap being guarded
    against is a runner that skips the translation and gets a scoring run
    that reports 0% refusal accuracy with no exception anywhere.

    `error` set means the HTTP call itself failed (a timeout, a 502) --
    everything else is `None`/empty in that case, and `score` must still
    produce a `Row` rather than raise. `payload` and `candidates` are plain
    dicts, not `ir` models: a malformed service response must not crash
    scoring, only cost that case its credit.
    """

    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "clarify", "refusal"] | None = None
    payload: dict | None = None
    reason: str | None = None
    candidates: tuple[dict, ...] = ()
    latency_ms: float = 0.0
    usage: tuple[dict, ...] = ()
    #: The stage-1 decision the service echoed back: `{"root", "joins",
    #: "groups"}`. A plain dict, not a `Route`: it is untrusted wire data,
    #: and a malformed one must cost this row its *route* measurement, not
    #: crash the run. `None` means the response carried no echo at all --
    #: an older service, or a path that refused before routing -- and
    #: `_groups_recall_hit` falls back to its documented proxy there.
    route: dict | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Row:
    """One flat measurement record. `None` means "not applicable to this
    case's kind", not "zero" -- a clarify case has no `field_tp`, an ok case
    always has one (possibly `0`).

    A row built from an `error` `Outcome` (the HTTP call itself failed) is
    not distinguished from a genuine model miss anywhere in this record --
    by design, it scores `root_correct=False`, `field_fn=len(gold fields)`,
    `exact_match=False`, and so on, the same as a real wrong answer would.
    That is deliberate here (the alternative is a third value on every
    boolean/count field, for a condition -- infra flakiness -- that a
    caller can already detect for free): `error` is carried on the row
    specifically so a caller can filter `error is not None` out before
    computing accuracy, and report those rows separately. An aggregator
    that skips this filter will silently fold transport failures into the
    model's own error rate.
    """

    case_id: str
    tags: tuple[str, ...]
    status_expected: str
    status_actual: str | None
    root_correct: bool | None
    joins_expected: tuple[str, ...] | None
    joins_actual: tuple[str, ...] | None
    groups_recall_hit: bool | None
    fields_expected: tuple[str, ...] | None
    fields_actual: tuple[str, ...] | None
    field_tp: int | None
    field_fp: int | None
    field_fn: int | None
    ops_correct: int | None
    ops_total: int | None
    values_correct: int | None
    values_total: int | None
    exact_match: bool | None
    refusal_reason_expected: str | None
    refusal_reason_actual: str | None
    clarify_field_expected: str | None
    candidate_hit: bool | None
    #: The primary outcome, frozen by `end_to_end_correct` and **stored**
    #: rather than recomputed: a later change to the definition must show up
    #: as a new `EVALUATOR_VERSION` beside old rows, never as old rows
    #: quietly meaning something new.
    e2e_correct: bool
    latency_ms: float
    tokens_in: int
    tokens_out: int
    error: str | None


# --------------------------------------------------------------------------
# canonical()
# --------------------------------------------------------------------------


def canonical(payload: dict, root: RootSpec) -> str:
    """A recursively key-sorted, type-normalised string form of `payload`.

    Two payloads that mean the same query canonicalise to the same string.
    Three normalisations happen here, each because a difference that carries
    no meaning must not read as a mismatch:

    1. **Values are coerced by the registry field's type.** `"2024"` and
       `2024` are the same year; `"$1,000"` and `1000.0` are the same money
       amount. Text, date and bool fields are compared as written -- there is
       no ambiguity there to normalise away.
    2. **Aggregate aliases are erased.** An alias is the model's arbitrary
       invention, not part of the question's meaning. Aggregates are sorted
       into a canonical order (by `(fn, field)`) and each one's alias is
       rewritten to a positional token (`agg0`, `agg1`, ...); every
       `having.agg` and `sort.agg` reference is rewritten through the same
       mapping. Two payloads identical up to alias spelling and aggregate
       order compare equal. Both wire spellings of the alias are accepted on
       an aggregate dict -- `"as"` (`Aggregate`'s wire alias) and `"alias"`
       (its Python attribute name, which a runner's plain `model_dump()`
       without `by_alias=True` would emit) -- preferring `"as"` when both are
       present and agree. Present-and-different is treated as garbage this
       function refuses to guess between: that aggregate is compared by its
       literal `as`/`alias` values instead of being folded into the
       positional-token scheme.
    3. **`AND`/`OR` children are sorted by their own canonical form.** A
       combinator is commutative -- `A AND B` is `B AND A` -- so order at
       that level is not semantic, and comparing children positionally would
       fail two payloads that differ only in which condition the model wrote
       first. Sorting each group's children by their canonical string makes
       the comparison order-independent at every level, recursively.

    Absence and emptiness are folded together throughout: `join: []` and no
    `join` key at all canonicalise the same way, and so do an explicit
    `value2: null` and a `value2` key that is simply missing -- both are
    "there is no second value", not two different facts.

    `columns` is *not* specially handled here -- the gold-set convention that
    an empty `columns` means "unspecified, don't compare" is a property of
    which payload is the gold one, which this function does not know. That
    logic lives in the caller (`score`'s exact-match check), not here.

    Known gaps, accepted rather than engineered around (see the report): an
    `in` list's element order is compared literally, not as a set, even
    though the operator is set-membership; and two aggregates that
    legitimately share the same `(fn, field)` but sit at different list
    positions in the gold and actual payloads can, in the presence of a
    `having`/`sort` reference to one of them, canonicalise differently
    depending on which physical aggregate the tie-break happens to land on.
    """
    payload = payload if isinstance(payload, dict) else {}
    agg_items, alias_map = _canon_aggregate_list(payload.get("aggregate") or ())
    struct = {
        "root": payload.get("root"),
        "where": _canon_node(payload.get("where"), root),
        "columns": sorted(_strs(payload.get("columns"))),
        "join": sorted(_strs(payload.get("join"))),
        "aggregate": agg_items,
        "group_by": sorted(_strs(payload.get("group_by"))),
        "having": _canon_having(payload.get("having"), alias_map),
        "sort": _canon_sort(payload.get("sort"), alias_map),
    }
    return json.dumps(struct, sort_keys=True, default=str)


def _strs(value) -> list[str]:
    return [v for v in (value or ()) if isinstance(v, str)]


def _canon_node(node, root: RootSpec):
    """One `where` node (a condition or a group), recursively canonicalised."""
    if not isinstance(node, dict):
        return None
    if "children" in node:
        children = [_canon_node(c, root) for c in (node.get("children") or ())]
        children = [c for c in children if c is not None]
        children.sort(key=lambda c: json.dumps(c, sort_keys=True))
        return {"combinator": node.get("combinator"), "children": children}
    if "field" not in node:
        return None
    field = node.get("field")
    spec = root.fields_by_key.get(field) if isinstance(field, str) else None
    return {
        "field": field,
        "op": node.get("op"),
        "value": _coerce_value(node.get("value"), spec),
        "value2": _coerce_value(node.get("value2"), spec),
    }


#: The tolerant `where` walker, shared with the gold model rather than
#: reimplemented here -- an actual payload off the wire and a gold payload
#: off a file are the same shape, and two copies of this walk would drift.
_iter_condition_dicts = iter_condition_dicts


def _coerce_value(value, spec: FieldSpec | None):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_coerce_scalar(v, spec) for v in value]
    return _coerce_scalar(value, spec)


def _coerce_scalar(value, spec: FieldSpec | None):
    field_type = spec.type if spec is not None else None
    if field_type in ("number", "year"):
        return _as_number(value)
    if field_type == "money":
        return _as_money(value)
    return value


def _as_number(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _as_money(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    amount = resolve_money(value)
    return amount if amount is not None else value


def _aggregate_alias(item: dict) -> tuple[str | None, bool]:
    """The alias an aggregate dict names, tolerant of both wire spellings.

    `Aggregate.alias` serialises as `"as"` (`model_dump(by_alias=True)`) but
    a runner doing a plain `model_dump()` emits the Python attribute name,
    `"alias"`, instead. Accepting only one spelling would leave the alias
    map empty for the other, silently failing every alias/having/sort
    rewrite for that payload with no exception anywhere -- so both are read
    here, preferring `"as"` when both are present and agree.

    Returns `(alias, is_garbage)`: `is_garbage` is true only when both keys
    are present and disagree, which this function has no principled way to
    resolve -- the caller falls back to comparing that aggregate literally
    rather than guessing.
    """
    as_value, alias_value = item.get("as"), item.get("alias")
    as_ok = isinstance(as_value, str)
    alias_ok = isinstance(alias_value, str)
    if as_ok and alias_ok:
        return (as_value, False) if as_value == alias_value else (None, True)
    if as_ok:
        return as_value, False
    if alias_ok:
        return alias_value, False
    return None, False


def _canon_aggregate_list(agg_list):
    """Sort aggregates into a canonical `(fn, field)` order and map each
    alias to a positional token `agg0`, `agg1`, ... in that order.
    """
    items = [a for a in (agg_list or ()) if isinstance(a, dict)]
    items = sorted(items, key=lambda a: (str(a.get("fn")), str(a.get("field"))))
    alias_map: dict[str, str] = {}
    canon = []
    for index, item in enumerate(items):
        base = {"fn": item.get("fn"), "field": item.get("field")}
        alias, is_garbage = _aggregate_alias(item)
        if is_garbage:
            canon.append({**base, "as": item.get("as"), "alias": item.get("alias")})
            continue
        token = f"agg{index}"
        if alias is not None:
            alias_map[alias] = token
        canon.append({**base, "as": token})
    return canon, alias_map


def _canon_having(having_list, alias_map: dict[str, str]):
    items = []
    for entry in having_list or ():
        if not isinstance(entry, dict):
            continue
        agg = entry.get("agg")
        items.append({
            "agg": alias_map.get(agg, agg),
            "op": entry.get("op"),
            # A having value is always an aggregate result, i.e. always
            # numeric -- there is no registry field to look a type up on
            # (it's keyed by alias, not by field), so this coerces loosely
            # rather than by a `FieldSpec`.
            "value": _loose_number(entry.get("value")),
        })
    items.sort(key=lambda h: json.dumps(h, sort_keys=True))
    return items


def _canon_sort(sort, alias_map: dict[str, str]):
    if not isinstance(sort, dict):
        return None
    field, agg = sort.get("field"), sort.get("agg")
    if field is None and agg is None:
        return None
    return {
        "field": field,
        "agg": alias_map.get(agg, agg) if agg is not None else None,
        "dir": sort.get("dir", "asc"),
    }


def _loose_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _exact_match(gold_payload: dict, actual_payload: dict | None, root: RootSpec) -> bool:
    """Canonical-equal, with `columns` dropped from *both* sides when the
    gold payload names none -- an empty gold `columns` means "the defaults,
    I don't care", not "an empty list is the right answer" (see the report).
    """
    if not isinstance(actual_payload, dict):
        return False
    gold, actual = dict(gold_payload), dict(actual_payload)
    if not gold.get("columns"):
        gold.pop("columns", None)
        actual.pop("columns", None)
    return canonical(gold, root) == canonical(actual, root)


# --------------------------------------------------------------------------
# condition-field bookkeeping
# --------------------------------------------------------------------------


def _condition_field_keys(payload: dict | None) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    return {
        c["field"]
        for c in _iter_condition_dicts(payload.get("where"))
        if isinstance(c.get("field"), str)
    }


def _field_ops(payload: dict | None) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    if not isinstance(payload, dict):
        return result
    for c in _iter_condition_dicts(payload.get("where")):
        field, op = c.get("field"), c.get("op")
        if isinstance(field, str) and isinstance(op, str):
            result.setdefault(field, set()).add(op)
    return result


def _field_op_values(
    payload: dict | None, root: RootSpec
) -> dict[tuple[str, str], list[tuple]]:
    """Every `(field, op)` pair's coerced `(value, value2)` tuples.

    A list per pair, not a single value: the same field can carry the same
    operator more than once (`price > 100 OR price > 500`), and collapsing
    to the last-written condition would silently drop one of them from
    value scoring. `score` compares these as multisets (`Counter`), so
    `[100.0, 500.0]` matches regardless of which condition was written
    first on either side.
    """
    result: dict[tuple[str, str], list[tuple]] = {}
    if not isinstance(payload, dict):
        return result
    for c in _iter_condition_dicts(payload.get("where")):
        field, op = c.get("field"), c.get("op")
        if not (isinstance(field, str) and isinstance(op, str)):
            continue
        spec = root.fields_by_key.get(field)
        value = _coerce_value(c.get("value"), spec)
        value2 = _coerce_value(c.get("value2"), spec)
        pair = (
            tuple(value) if isinstance(value, list) else value,
            tuple(value2) if isinstance(value2, list) else value2,
        )
        result.setdefault((field, op), []).append(pair)
    return result


def _all_touched_fields(payload: dict | None) -> set[str]:
    """Every field key the payload names anywhere -- conditions, aggregates,
    columns, group-by, sort. Used only by the `groups_recall_hit` proxy
    below, where the question is "did the route reach this field's group at
    all", not "is this a condition".
    """
    if not isinstance(payload, dict):
        return set()
    fields = set(_condition_field_keys(payload))
    fields.update(
        a.get("field")
        for a in (payload.get("aggregate") or ())
        if isinstance(a, dict) and isinstance(a.get("field"), str) and a.get("field") != "*"
    )
    fields.update(_strs(payload.get("columns")))
    fields.update(_strs(payload.get("group_by")))
    sort = payload.get("sort")
    if isinstance(sort, dict) and isinstance(sort.get("field"), str):
        fields.add(sort["field"])
    return fields


def _routed_groups(route: dict | None) -> frozenset[str] | None:
    """The groups an echoed route names, or `None` if there is no usable echo.

    Every shape that is not "a dict with a list of strings under `groups`"
    returns `None`, which sends the caller back to the proxy. Deliberately
    not an exception and deliberately not an empty set: a garbled echo is
    an unknown route, and scoring it as "routed to nothing" would report a
    perfect service as missing every group.
    """
    if not isinstance(route, dict):
        return None
    groups = route.get("groups")
    if not isinstance(groups, (list, tuple)):
        return None
    return frozenset(g for g in groups if isinstance(g, str))


def _gold_groups(gold_fields: set[str], root: RootSpec) -> frozenset[str]:
    """The registry groups a correct route had to select to reach these fields.

    Identity groups are dropped: they are in scope whatever stage 1 chooses,
    so requiring the router to name one would mark a correct route wrong for
    not saying something it never had to say.
    """
    identity = root.identity_groups()
    by_key = root.fields_by_key
    return frozenset(
        by_key[f].group
        for f in gold_fields
        if f in by_key and by_key[f].group and by_key[f].group not in identity
    )


def _groups_recall_hit(
    gold_fields: set[str],
    actual_payload: dict | None,
    root: RootSpec,
    route: dict | None = None,
) -> bool | None:
    """Did stage 1 select every group the gold answer needed?

    **With a route echo this is the real measurement**, not a proxy: the
    service reports the groups it chose, and the question is simply whether
    every group a gold field lives in (identity groups excepted -- they are
    always in scope) is among them. A group the router missed is a group
    stage 2 was never shown, so the field was unreachable whatever the
    payload happened to say.

    **Without one it falls back to the lower-bound proxy below**, which is
    what this was before the echo existed. Both paths are kept because a run
    against an older deploy, or a response that refused before it ever
    routed, has no echo to read -- and a metric that vanished in those cases
    would silently change what a mixed run's average meant.

    The proxy, for the no-echo path: the service does not say which groups
    it routed to -- only the payload it produced -- so this cannot measure
    group selection directly. Instead:
    a gold field (every field the gold payload names at all -- conditions,
    aggregates, `group_by`, `sort`, not just conditions) counts as reached if
    it appears anywhere in the actual payload, *or* if some field the actual
    payload does mention sits in the same registry group. That second clause
    is the proxy, and it is optimistic on purpose: it cannot tell "the group
    was selected but this field was dropped" apart from "the group was never
    selected and another field from a different, coincidentally-matching,
    group happened to be used" -- it only ever produces a hit or a documented
    non-hit, never proof. Treat a proxy `True` as "consistent with a correct
    route", not as "the route was verified correct".

    Returns `None` when there is nothing to derive it from: the gold payload
    names no field at all (an aggregate-only `count(*)` with no `where`,
    `group_by`, or `sort` field).
    """
    if not gold_fields:
        return None
    routed = _routed_groups(route)
    if routed is not None:
        return _gold_groups(gold_fields, root) <= routed
    actual_fields = _all_touched_fields(actual_payload)
    actual_groups = {
        root.fields_by_key[f].group for f in actual_fields if f in root.fields_by_key
    }
    for field in gold_fields:
        if field in actual_fields:
            continue
        spec = root.fields_by_key.get(field)
        if spec is not None and spec.group and spec.group in actual_groups:
            continue
        return False
    return True


# --------------------------------------------------------------------------
# end_to_end_correct(): the one frozen definition of "right answer"
# --------------------------------------------------------------------------


def end_to_end_correct(
    case: GoldenCase,
    outcome: Outcome,
    *,
    exact_match: bool | None,
    candidate_hit: bool | None,
) -> bool:
    """Was this one question answered correctly? (Owner definition, 2026-09-18.)

    Every consumer -- the primary McNemar test, the family metrics, the
    headline accuracy -- reads this one function, through the `e2e_correct`
    it writes onto each row. One definition, in one place, chosen before the
    results were seen.

    Per gold kind:

    * **ok** -- the service returned `ok` *and* the payload exactly matches
      under the canonicalisation already in force. Phase 3's
      semantic-equivalence judge will only ever *upgrade* this, and it does
      so by bumping `EVALUATOR_VERSION`: a new definition is a new version,
      never a quiet edit, so no model can be favoured by a rule written
      after its results were in.
    * **clarify** -- the service asked for clarification *and* asked about
      the right field. Asking about the wrong field is not correct
      behaviour, however well-formed the question. Whether the pruned
      remainder of the payload was right stays diagnostic, outside the
      primary outcome.
    * **refusal** -- the service refused *and* gave the expected reason.

    The status gates are stated here even where the diagnostic they guard
    is already gated the same way (`exact_match` is computed only for an
    `ok` status): the asymmetries are the point of the definition, and they
    must not depend on a gate living somewhere else. An ok-case that came
    back as a clarification whose pruned payload happens to canonicalise
    equal to gold is **incorrect** -- the caller was asked a question
    instead of being given the answer.

    `exact_match` and `candidate_hit` are passed in rather than recomputed
    so that the row's diagnostics and its primary outcome can never
    disagree about the same run.
    """
    kind = case.expect.kind
    if kind == "ok":
        return outcome.status == "ok" and bool(exact_match)
    if kind == "clarify":
        return outcome.status == "clarify" and bool(candidate_hit)
    return outcome.status == "refusal" and outcome.reason == case.expect.reason


# --------------------------------------------------------------------------
# score()
# --------------------------------------------------------------------------


def score(case: GoldenCase, outcome: Outcome, root: RootSpec) -> Row:
    """Turn one (golden case, service outcome) pair into one `Row`.

    Never raises. A row field that does not apply to this case's `expect`
    kind -- field precision/recall on a clarify case, a clarify field on an
    ok case -- is `None` rather than `0` or a guess, so a downstream consumer
    can tell "not measured here" from "measured, and zero".
    """
    kind = case.expect.kind
    status_actual = outcome.status
    actual_payload = outcome.payload if isinstance(outcome.payload, dict) else None

    root_correct = joins_expected = joins_actual = None
    groups_recall_hit = None
    fields_expected = fields_actual = None
    field_tp = field_fp = field_fn = None
    ops_correct = ops_total = None
    values_correct = values_total = None
    exact_match = None
    refusal_reason_expected = None
    clarify_field_expected = None
    candidate_hit = None

    if kind == "ok":
        gold_payload = case.expect.payload
        root_correct = (
            status_actual == "ok"
            and actual_payload is not None
            and actual_payload.get("root") == gold_payload.get("root")
        )

        joins_expected = tuple(sorted(_strs(gold_payload.get("join"))))
        joins_actual = tuple(sorted(_strs((actual_payload or {}).get("join"))))

        gold_fields = _condition_field_keys(gold_payload)
        actual_fields = _condition_field_keys(actual_payload)
        fields_expected = tuple(sorted(gold_fields))
        fields_actual = tuple(sorted(actual_fields))

        if root_correct:
            field_tp = len(gold_fields & actual_fields)
            field_fp = len(actual_fields - gold_fields)
            field_fn = len(gold_fields - actual_fields)
        else:
            # An expected-ok case that got a refusal, a clarification, a
            # wrong root, or nothing at all: no credit, and every gold field
            # is missed outright, per the brief's own worked example.
            field_tp = 0
            field_fp = len(actual_fields)
            field_fn = len(gold_fields)

        gold_ops = _field_ops(gold_payload)
        actual_ops = _field_ops(actual_payload)
        matched_fields = set(gold_ops) & set(actual_ops) if root_correct else set()
        ops_total = len(matched_fields)
        ops_correct = sum(1 for f in matched_fields if gold_ops[f] == actual_ops[f])

        gold_vals = _field_op_values(gold_payload, root)
        actual_vals = _field_op_values(actual_payload, root)
        matched_pairs = set(gold_vals) & set(actual_vals) if root_correct else set()
        values_total = len(matched_pairs)
        values_correct = sum(
            1
            for p in matched_pairs
            if Counter(gold_vals[p]) == Counter(actual_vals[p])
        )

        # Gated on status_actual == "ok" (not just root_correct or a bare
        # canonical compare): a clarify response's pruned payload can
        # canonicalise identically to gold by coincidence, and that must not
        # read as an exact match sitting beside field_tp=0.
        exact_match = status_actual == "ok" and _exact_match(
            gold_payload, actual_payload, root
        )
        groups_recall_hit = _groups_recall_hit(
            _all_touched_fields(gold_payload), actual_payload, root, outcome.route
        )

    elif kind == "clarify":
        clarify_field_expected = case.expect.field
        candidate_field_keys = {
            c.get("field") for c in outcome.candidates if isinstance(c, dict)
        }
        candidate_hit = clarify_field_expected in candidate_field_keys

    else:  # kind == "refusal"
        refusal_reason_expected = case.expect.reason

    # Recorded whenever the service actually refused, regardless of what was
    # expected -- an ok or clarify case that came back refused is exactly
    # the interesting failure this makes visible.
    refusal_reason_actual = outcome.reason if status_actual == "refusal" else None

    tokens_in = sum(
        sum(int(u.get(key) or 0) for key in _INPUT_USAGE_KEYS) for u in outcome.usage
    )
    tokens_out = sum(int(u.get("output_tokens") or 0) for u in outcome.usage)

    return Row(
        case_id=case.id,
        tags=case.tags,
        status_expected=kind,
        status_actual=status_actual,
        root_correct=root_correct,
        joins_expected=joins_expected,
        joins_actual=joins_actual,
        groups_recall_hit=groups_recall_hit,
        fields_expected=fields_expected,
        fields_actual=fields_actual,
        field_tp=field_tp,
        field_fp=field_fp,
        field_fn=field_fn,
        ops_correct=ops_correct,
        ops_total=ops_total,
        values_correct=values_correct,
        values_total=values_total,
        exact_match=exact_match,
        refusal_reason_expected=refusal_reason_expected,
        refusal_reason_actual=refusal_reason_actual,
        clarify_field_expected=clarify_field_expected,
        candidate_hit=candidate_hit,
        e2e_correct=end_to_end_correct(
            case, outcome, exact_match=exact_match, candidate_hit=candidate_hit
        ),
        latency_ms=outcome.latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        error=outcome.error,
    )
