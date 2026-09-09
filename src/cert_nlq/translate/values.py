"""Deterministic value resolution, registry and static data only.

Structure and values are separated deliberately: the model picks fields, and
values are resolved by lookup here, so a value error is fixed with a table
rather than by prompt-tuning, and the two are measured independently.

Anything needing a database lookup is left exactly as written, for the host to
resolve — this service has no data access.
"""
import math
import re

from pydantic import BaseModel, ConfigDict

from ..ir.payload import NO_VALUE_OPS, Group, JsonScalar, Payload
from ..registry.models import FieldSpec, RootSpec

#: Entity placeholders the host substituted before the question was sent.
PLACEHOLDER_PATTERN = re.compile(r"^<[A-Z]+_\d+>$")

_RELATIVE_YEARS = {
    "this year": 0, "this tax year": 0, "current year": 0, "current tax year": 0,
    "last year": -1, "previous year": -1, "prior year": -1, "last tax year": -1,
    "next year": 1,
}
_MONEY_STRIP = re.compile(r"[$,\s]")
_YEAR_LITERAL = re.compile(r"(19|20)\d{2}")


class Unresolvable(BaseModel):
    """One value that did not resolve, with the operator it appeared under.

    The operator travels with the value because a candidate offered back
    under the wrong one inverts the question: a phrase that failed under
    `!=` must not come back as an `=` suggestion.
    """

    model_config = ConfigDict(frozen=True)

    field: str
    op: str
    value: JsonScalar


def resolve_vocabulary(spec: FieldSpec, raw) -> str | None:
    """Map a word the user typed onto the code they would have typed.

    Matches an existing code first, then a meaning, then a synonym — all
    case-folded. The precedence order is load-bearing: under collision (two
    entries sharing a case-folded synonym), the first entry wins without
    signalling the ambiguity — a registry data-quality problem to fix at source.

    Returns None when nothing matches; a guessed status code is a query that
    runs cleanly and returns the wrong rows.
    """
    if not spec.vocabulary:
        return None
    needle = str(raw).strip().casefold()
    for entry in spec.vocabulary:
        if entry.code.casefold() == needle:
            return entry.code
    for entry in spec.vocabulary:
        if entry.meaning.casefold() == needle:
            return entry.code
    for entry in spec.vocabulary:
        if any(s.casefold() == needle for s in entry.synonyms):
            return entry.code
    return None


def resolve_relative_year(raw, now_year: int) -> int | None:
    """Resolve a year phrase from the calendar, never from MAX(year).

    A stored maximum is a fact about the rows, not about the calendar. A
    question asking for "this year" means the calendar, so that is what it
    is answered from.
    """
    text = str(raw).strip().casefold()
    if text in _RELATIVE_YEARS:
        return now_year + _RELATIVE_YEARS[text]
    if _YEAR_LITERAL.fullmatch(text):
        return int(text)
    return None


def resolve_money(raw) -> float | None:
    """Strip currency formatting and parse, or return None.

    The finiteness check is required, not defensive. `float()` accepts
    "nan", "inf", "-inf" and "Infinity" without raising, and silently
    overflows "1e400" to infinity — so without it a nonsensical amount is
    reported as *resolved* and written into the payload, where it reaches the
    query as a value that matches nothing. That is precisely the
    silent-wrong-answer failure this module exists to prevent, and it is the
    overflow case rather than a literal "nan" that makes it reachable.

    A negative amount is accepted deliberately — credits and reversals are
    legitimate values in this position.
    """
    try:
        amount = float(_MONEY_STRIP.sub("", str(raw)))
    except (TypeError, ValueError):
        return None
    return amount if math.isfinite(amount) else None


def _resolve_one(spec: FieldSpec, raw, now_year: int):
    """Return (value, resolved?) for a single value slot."""
    if raw is None:
        return None, True
    if isinstance(raw, str) and PLACEHOLDER_PATTERN.match(raw.strip()):
        return raw, True
    if spec.vocabulary:
        code = resolve_vocabulary(spec, raw)
        return (code, True) if code is not None else (raw, False)
    if spec.type == "year":
        year = resolve_relative_year(raw, now_year)
        return (year, True) if year is not None else (raw, False)
    if spec.type == "money":
        amount = resolve_money(raw)
        return (amount, True) if amount is not None else (raw, False)
    return raw, True


def _resolve_node(node, by_key: dict, now_year: int, unresolved: list[Unresolvable]):
    """Resolve one leaf, or recurse into a group's children.

    A list value (`in`) is resolved element by element: each unresolved
    element is reported individually, and the list is left exactly as
    written if any element fails, rather than half-rewritten.
    """
    if isinstance(node, Group):
        return node.model_copy(update={"children": tuple(
            _resolve_node(c, by_key, now_year, unresolved) for c in node.children
        )})
    spec = by_key.get(node.field)
    if spec is None or node.op in NO_VALUE_OPS:
        return node
    if isinstance(node.value, tuple):
        parts = [_resolve_one(spec, v, now_year) for v in node.value]
        unresolved.extend(
            Unresolvable(field=node.field, op="in", value=raw)
            for raw, (_, ok) in zip(node.value, parts, strict=True) if not ok
        )
        if all(ok for _, ok in parts):
            return node.model_copy(update={"value": tuple(v for v, _ in parts)})
        return node
    value, ok = _resolve_one(spec, node.value, now_year)
    value2, ok2 = _resolve_one(spec, node.value2, now_year)
    if not ok:
        unresolved.append(Unresolvable(field=node.field, op=node.op, value=node.value))
    if not ok2:
        unresolved.append(Unresolvable(field=node.field, op=node.op, value=node.value2))
    return node.model_copy(update={"value": value, "value2": value2})


def resolve_values(
    payload: Payload, root: RootSpec, now_year: int
) -> tuple[Payload, list[Unresolvable]]:
    """Rewrite resolvable values in place; report the ones that did not resolve.

    An unresolved value is returned as written rather than guessed, so the
    caller can flag the condition or ask (the confidence ladder).
    """
    by_key = root.fields_by_key
    unresolved: list[Unresolvable] = []
    where = (
        _resolve_node(payload.where, by_key, now_year, unresolved)
        if payload.where is not None
        else None
    )
    return payload.model_copy(update={"where": where}), unresolved
