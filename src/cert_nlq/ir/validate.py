"""Semantic validation of a payload against the registry.

The generated schema (schema.py) constrains *vocabulary*. This module enforces
the rules a Literal union cannot express: operator legality per field,
aggregate type rules, alias references, and join requirements.

Every problem is collected before raising, so a caller sees the whole picture
rather than the first failure.
"""
from ..registry.models import Registry, RootSpec
from .payload import NO_VALUE_OPS, Payload

NUMERIC_AGG_TYPES = frozenset({"number", "money"})
_NEEDS_TWO_VALUES = "between"


class PayloadError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def validate_payload(raw: dict | Payload, registry: Registry) -> Payload:
    payload = raw if isinstance(raw, Payload) else Payload.model_validate(raw)
    root = registry.root(payload.root)
    if root is None:
        raise PayloadError([f"unknown root {payload.root!r}"])

    problems: list[str] = []
    problems += _check_joins(payload, root)
    available = _available_fields(payload, root)
    problems += _check_conditions(payload, root, available)
    problems += _check_columns(payload, available)
    aliases, agg_problems = _check_aggregates(payload, available)
    problems += agg_problems
    problems += _check_having_and_sort(payload, available, aliases)
    if not payload.conditions and not payload.aggregate:
        problems.append("a payload needs at least one condition")
    if problems:
        raise PayloadError(problems)
    return payload


def _check_joins(payload: Payload, root: RootSpec) -> list[str]:
    return [
        f"join {table!r} is not available on {root.root!r}"
        for table in payload.join
        if root.join(table) is None
    ]


def _available_fields(payload: Payload, root: RootSpec) -> dict:
    """Field specs reachable given the requested joins."""
    tables = {root.root} | set(payload.join)
    return {f.key: f for f in root.fields if f.table in tables}


def _check_conditions(payload: Payload, root: RootSpec, available: dict) -> list[str]:
    problems = []
    all_fields = root.fields_by_key
    for cond in payload.conditions:
        spec = available.get(cond.field)
        if spec is None:
            other = all_fields.get(cond.field)
            if other is not None:
                problems.append(f"{cond.field!r} requires join {other.table!r}")
            else:
                problems.append(f"unknown field {cond.field!r}")
            continue
        if cond.op not in spec.operators:
            problems.append(f"operator {cond.op!r} not allowed for {spec.label!r}")
            continue
        if cond.op in NO_VALUE_OPS:
            if cond.value is not None or cond.value2 is not None:
                problems.append(f"{cond.op!r} on {spec.label!r} takes no value")
            continue
        if cond.value is None or cond.value == "":
            problems.append(f"{spec.label!r} requires a value")
        if cond.op == _NEEDS_TWO_VALUES and cond.value2 is None:
            problems.append(f"{spec.label!r} with 'between' requires 'value2'")
        if cond.op != _NEEDS_TWO_VALUES and cond.value2 is not None:
            problems.append(f"{spec.label!r} with {cond.op!r} takes no 'value2'")
    return problems


def _check_columns(payload: Payload, available: dict) -> list[str]:
    return [f"unknown column {key!r}" for key in payload.columns if key not in available]


def _check_aggregates(payload: Payload, available: dict) -> tuple[set[str], list[str]]:
    problems: list[str] = []
    aliases: set[str] = set()
    for agg in payload.aggregate:
        if agg.alias in aliases:
            problems.append(f"duplicate aggregate alias {agg.alias!r}")
        aliases.add(agg.alias)
        if agg.field == "*":
            if agg.fn != "count":
                problems.append("only 'count' accepts '*' as a target")
            continue
        spec = available.get(agg.field)
        if spec is None:
            problems.append(f"unknown aggregate field {agg.field!r}")
            continue
        if agg.fn in ("sum", "avg") and spec.type not in NUMERIC_AGG_TYPES:
            problems.append(f"cannot {agg.fn} {spec.label!r}: type is {spec.type!r}")
    problems += [
        f"unknown group_by field {key!r}"
        for key in payload.group_by
        if key not in available
    ]
    return aliases, problems


def _check_having_and_sort(
    payload: Payload, available: dict, aliases: set[str]
) -> list[str]:
    problems = [
        f"having references unknown alias {clause.agg!r}"
        for clause in payload.having
        if clause.agg not in aliases
    ]
    sort = payload.sort
    if sort is None:
        return problems
    if sort.agg is not None and sort.agg not in aliases:
        problems.append(f"sort references unknown alias {sort.agg!r}")
    if sort.field is not None and sort.field not in available:
        problems.append(f"unknown sort field {sort.field!r}")
    return problems
