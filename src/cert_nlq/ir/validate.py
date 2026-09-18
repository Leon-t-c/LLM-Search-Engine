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


#: What kind of thing went wrong, for a caller that has to *branch* on the
#: failure rather than show it. Only the kinds something branches on are
#: named; everything else is `OTHER`.
PROBLEM_UNJOINED = "unjoined"
PROBLEM_OTHER = "other"


class Problem(str):
    """One validation problem: the sentence, plus what kind of problem it is.

    A `str` subclass rather than a record, so every existing consumer — the
    join into one message, the text fed back to the model as retry guidance,
    the equality checks in the suite — keeps working untouched, while a caller
    that must pick a different course of action for a different failure has
    something stable to test.

    The alternative was a substring search on the formatted sentence, which is
    what the retry hint used to do. That makes the wording of a message load
    bearing without saying so: reword it and the branch silently stops firing,
    with nothing failing to say so.
    """

    def __new__(cls, kind: str, message: str) -> "Problem":
        problem = super().__new__(cls, message)
        problem.kind = kind
        return problem


class PayloadError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))

    @property
    def kinds(self) -> frozenset[str]:
        """The distinct problem kinds collected here.

        Tolerates a plain `str` in `problems`: this type is constructible from
        anywhere, and a caller that hand-built one should get "other" rather
        than an AttributeError.
        """
        return frozenset(
            getattr(problem, "kind", PROBLEM_OTHER) for problem in self.problems
        )


def validate_payload(
    raw: dict | Payload, registry: Registry, expected_root: str | None = None
) -> Payload:
    """Check a payload against the registry, and against the root it was for.

    `expected_root` is the root the caller narrowed the schema to. Pass it
    whenever you have it. The generated schema pins `root` to a single value,
    so a payload naming a different one is already outside what the model was
    offered — but this module is the gate the host trusts, and a gate that
    only asks "does this root exist somewhere?" would let a well-formed answer
    about the wrong table through, with the routing decision silently
    discarded and nothing anywhere reporting a problem. Left optional so a
    caller validating a payload it did not route (a stored one, a replayed
    one) is not forced to invent an answer.
    """
    payload = raw if isinstance(raw, Payload) else Payload.model_validate(raw)
    root = registry.root(payload.root)
    if root is None:
        raise PayloadError([f"unknown root {payload.root!r}"])
    if expected_root is not None and payload.root != expected_root:
        raise PayloadError(
            [f"payload is for root {payload.root!r}, not {expected_root!r}"]
        )

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
    """Field specs reachable given the requested joins.

    Only a table the registry declares as a join has to be asked for. A
    root spans every other table its fields sit on, because the host joins
    those unconditionally -- checking against the root's own table name
    would refuse more than half of a real registry.
    """
    joinable = {j.table for j in root.joins}
    allowed = set(payload.join)
    return {f.key: f for f in root.fields
            if f.table not in joinable or f.table in allowed}


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
        return None, Problem(
            PROBLEM_UNJOINED, f"{key!r} requires join {elsewhere.table!r}"
        )
    return None, Problem(PROBLEM_OTHER, f"unknown {label} {key!r}")


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
