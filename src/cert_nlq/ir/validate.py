"""Semantic validation of a payload against the registry.

The generated schema (schema.py) constrains *vocabulary*. This module enforces
the rules a Literal union cannot express: operator legality per field,
aggregate type rules, alias references, and join requirements.

Every problem is collected before raising, so a caller sees the whole picture
rather than the first failure.
"""
from ..registry.models import Registry, RootSpec
from .payload import LIST_OPS, NO_VALUE_OPS, Condition, Payload, iter_conditions
from .schema import HAVING_OPS, MAX_GROUP_DEPTH

NUMERIC_AGG_TYPES = frozenset({"number", "money"})
_NEEDS_TWO_VALUES = "between"


def _depth(node, level: int = 1) -> int:
    if isinstance(node, Condition) or node is None:
        return level - 1
    return max((_depth(c, level + 1) for c in node.children), default=level)


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
    problems += _check_columns(payload, root, available)
    aliases, agg_problems = _check_aggregates(payload, root, available)
    problems += agg_problems
    problems += _check_having_and_sort(payload, root, available, aliases)
    if payload.where is None and not payload.aggregate:
        problems.append(
            "a payload needs at least one condition or one aggregate"
        )
    if payload.where is not None and _depth(payload.where) > MAX_GROUP_DEPTH:
        problems.append(f"conditions are nested more than {MAX_GROUP_DEPTH} levels deep")
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


def _resolve_field(key: str, available: dict, all_fields: dict, label: str):
    """Look a key up, distinguishing "needs a join" from "does not exist".

    Returns `(spec, problem)` — exactly one of which is None.

    The distinction is load-bearing, not cosmetic: the problem text is fed
    back to the model verbatim as retry guidance, and "add the join" and "pick
    a different field" send a correction down entirely different paths. Telling
    a caller a field does not exist when it does, and merely needs its table
    joined, wastes the one retry and ends in a refusal.
    """
    spec = available.get(key)
    if spec is not None:
        return spec, None
    elsewhere = all_fields.get(key)
    if elsewhere is not None:
        return None, f"{key!r} requires join {elsewhere.table!r}"
    return None, f"unknown {label} {key!r}"


def _check_conditions(payload: Payload, root: RootSpec, available: dict) -> list[str]:
    problems = []
    all_fields = root.fields_by_key
    for cond in iter_conditions(payload.where):
        spec, problem = _resolve_field(cond.field, available, all_fields, "field")
        if problem is not None:
            problems.append(problem)
            continue
        if cond.op not in spec.operators:
            problems.append(f"operator {cond.op!r} not allowed for {spec.label!r}")
            continue
        if cond.op in NO_VALUE_OPS:
            if cond.value is not None or cond.value2 is not None:
                problems.append(f"{cond.op!r} on {spec.label!r} takes no value")
            continue
        is_list = isinstance(cond.value, tuple)
        if cond.op in LIST_OPS:
            if not is_list or not cond.value:
                problems.append("'in' needs at least one value")
                continue
        elif is_list:
            problems.append(f"{spec.label!r} takes a single value")
            continue
        if cond.value is None or cond.value == "":
            problems.append(f"{spec.label!r} requires a value")
        if cond.op == _NEEDS_TWO_VALUES and cond.value2 is None:
            problems.append(f"{spec.label!r} with 'between' requires 'value2'")
        if cond.op != _NEEDS_TWO_VALUES and cond.value2 is not None:
            problems.append(f"{spec.label!r} with {cond.op!r} takes no 'value2'")
    return problems


def _check_columns(payload: Payload, root: RootSpec, available: dict) -> list[str]:
    all_fields = root.fields_by_key
    problems = []
    for key in payload.columns:
        _, problem = _resolve_field(key, available, all_fields, "column")
        if problem is not None:
            problems.append(problem)
    return problems


def _check_aggregates(
    payload: Payload, root: RootSpec, available: dict
) -> tuple[set[str], list[str]]:
    all_fields = root.fields_by_key
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
        spec, problem = _resolve_field(agg.field, available, all_fields, "aggregate field")
        if problem is not None:
            problems.append(problem)
            continue
        if agg.fn in ("sum", "avg") and spec.type not in NUMERIC_AGG_TYPES:
            problems.append(f"cannot {agg.fn} {spec.label!r}: type is {spec.type!r}")
    for key in payload.group_by:
        _, problem = _resolve_field(key, available, all_fields, "group_by field")
        if problem is not None:
            problems.append(problem)
    return aliases, problems


def _check_having_and_sort(
    payload: Payload, root: RootSpec, available: dict, aliases: set[str]
) -> list[str]:
    problems: list[str] = []
    for clause in payload.having:
        if clause.agg not in aliases:
            problems.append(f"having references unknown alias {clause.agg!r}")
        # The generated schema closes this slot to the same six comparisons,
        # so a structured-output call cannot reach here — but this function is
        # the gate the caller re-checks against, and it is the only one a
        # payload assembled by anything else passes through. Read from the
        # schema's own constant: two lists saying the same thing today are two
        # lists that can drift.
        if clause.op not in HAVING_OPS:
            problems.append(f"having operator {clause.op!r} is not a comparison")
    sort = payload.sort
    if sort is None:
        return problems
    if sort.agg is not None and sort.agg not in aliases:
        problems.append(f"sort references unknown alias {sort.agg!r}")
    if sort.field is not None:
        _, problem = _resolve_field(sort.field, available, root.fields_by_key, "sort field")
        if problem is not None:
            problems.append(problem)
    return problems
