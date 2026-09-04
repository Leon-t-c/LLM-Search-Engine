"""Deterministic value resolution, registry and static data only.

Structure and values are separated deliberately: the model picks fields, and
values are resolved by lookup here, so a value error is fixed with a table
rather than by prompt-tuning, and the two are measured independently.

Anything needing a database lookup is left exactly as written, for the host to
resolve — this service has no data access.
"""
import math
import re

from ..ir.payload import NO_VALUE_OPS, Payload
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

    Eleven rows in the production data carry impossible years (up to 6465), so
    anything derived from the data's maximum is poisoned.
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


def resolve_values(
    payload: Payload, root: RootSpec, now_year: int
) -> tuple[Payload, list[str]]:
    """Rewrite resolvable values in place; report the ones that did not resolve.

    An unresolved value is returned as written rather than guessed, so the
    caller can flag the condition or ask (the confidence ladder).
    """
    by_key = root.fields_by_key
    conditions = []
    unresolved: list[str] = []
    for cond in payload.conditions:
        spec = by_key.get(cond.field)
        if spec is None or cond.op in NO_VALUE_OPS:
            conditions.append(cond)
            continue
        value, ok = _resolve_one(spec, cond.value, now_year)
        value2, ok2 = _resolve_one(spec, cond.value2, now_year)
        if not ok:
            unresolved.append(f"{cond.field}={cond.value}")
        if not ok2:
            unresolved.append(f"{cond.field}={cond.value2}")
        conditions.append(cond.model_copy(update={"value": value, "value2": value2}))
    return payload.model_copy(update={"conditions": tuple(conditions)}), unresolved
