"""Stage 2 plus orchestration: question in, one of three responses out."""
from datetime import date

from pydantic import ValidationError

from ..ir.payload import Condition, Group, Payload
from ..ir.response import (
    Candidate,
    NeedsClarification,
    Ok,
    Refused,
    TranslateResponse,
    Unresolved,
)
from ..ir.schema import json_schema_for
from ..ir.validate import PayloadError, validate_payload
from ..registry.models import Registry, RootSpec
from .provider import Provider
from .refusals import RefusalReason
from .router import Route, RoutingError, route
from .values import Unresolvable, resolve_values

TRANSLATOR_SYSTEM = (
    "Translate the question into the given JSON payload shape.\n"
    "Use only the fields offered. Never invent a field, an operator or a table.\n"
    "Conditions form a tree of AND/OR groups, nested as deep as the shape "
    "offers. Nest a group only when the question genuinely needs mixed logic, "
    "such as (A or B) and C; keep it flat otherwise.\n"
    "Prefer the `in` operator over a group when one field takes several "
    "values: `status in [A, B]` rather than a group of two equalities.\n"
    "For a value from a coded field, write the meaning in plain words; a later "
    "step maps it to the stored code. Do not guess a code.\n"
    "Leave a name or address exactly as written, placeholders included.\n"
    "Counting questions are ordinary filters: the caller counts the rows."
)

#: One retry with the validation error appended, then refuse.
_MAX_ATTEMPTS = 2


def translate(
    question: str,
    registry: Registry,
    provider: Provider,
    now_year: int | None = None,
) -> TranslateResponse:
    try:
        chosen = route(question, registry, provider)
    except RoutingError:
        return Refused(
            reason=RefusalReason.NOT_A_QUERY,
            detail="This does not resolve to a question about the data.",
        )

    root = registry.root(chosen.root)
    payload, problem = _translate_stage_two(question, registry, root, chosen, provider)
    if payload is None:
        return Refused(reason=RefusalReason.FIELD_NOT_IN_SCHEMA, detail=problem)

    resolved, unresolved = resolve_values(payload, root, now_year or date.today().year)
    if unresolved:
        return _clarify(resolved, root, unresolved[0], registry.version)
    return Ok(payload=resolved, registry_version=registry.version)


def _translate_stage_two(
    question: str,
    registry: Registry,
    root: RootSpec,
    chosen: Route,
    provider: Provider,
) -> tuple[Payload | None, str]:
    """Call the model, retrying once with the validation error appended."""
    schema = json_schema_for(root, chosen.groups, chosen.joins)
    asked = question
    problem = ""
    for _ in range(_MAX_ATTEMPTS):
        raw = provider.complete(
            system=TRANSLATOR_SYSTEM,
            question=asked,
            schema=schema,
            schema_name=f"{root.root}Payload",
        )
        try:
            return validate_payload(raw, registry), ""
        except (PayloadError, ValidationError) as exc:
            problem = str(exc)
            asked = (
                f"{question}\n\n"
                f"Your previous answer failed validation: {problem}\n"
                "Correct it using only the offered fields and operators."
            )
    return None, problem


def _without_field(node, key: str):
    """Drop every condition on `key`, collapsing groups that empty out.

    Pruning a leaf out of a tree is not the same as filtering a list. A group
    left with no children is invalid, so it has to go too; a group left with
    exactly one child is just that child and is collapsed rather than left as
    a pointless wrapper. If everything goes, the caller gets None and there is
    nothing left to show.
    """
    if node is None:
        return None
    if isinstance(node, Condition):
        return None if node.field == key else node
    kept = tuple(
        k for k in (_without_field(c, key) for c in node.children) if k is not None
    )
    if not kept:
        return None
    if len(kept) == 1:
        return kept[0]
    return node.model_copy(update={"children": kept})


def _as_group(node) -> Group:
    """`Payload.where` is a group, so a lone surviving condition needs wrapping."""
    return node if isinstance(node, Group) else Group(combinator="AND", children=(node,))


def _clarify(
    payload: Payload, root: RootSpec, unresolved: Unresolvable, version: str
) -> TranslateResponse:
    """Drop the unresolved condition and offer the field's own codes.

    Candidates carry the registry's field label with the raw stored value —
    `Status = X`, never a paraphrase. Users read these codes fluently. Each
    candidate is offered under `unresolved.op`, the operator the phrase was
    actually asked under — for a list (`in`) operator that means each
    candidate offers one legal element, not a whole list.
    """
    key, phrase = unresolved.field, str(unresolved.value)
    spec = root.fields_by_key.get(key)
    kept = _without_field(payload.where, key)
    vocabulary = spec.vocabulary if spec is not None else ()
    if not vocabulary or kept is None:
        return Refused(
            reason=RefusalReason.FIELD_NOT_IN_SCHEMA,
            detail=f"Could not work out what {phrase!r} means here.",
        )
    candidates = tuple(
        Candidate(
            field=key,
            op=unresolved.op,
            value=entry.code,
            label=spec.label,
            confidence=round(1.0 / len(vocabulary), 2),
        )
        for entry in vocabulary
    )
    return NeedsClarification(
        payload=payload.model_copy(update={"where": _as_group(kept)}),
        unresolved=Unresolved(
            phrase=phrase,
            question=f"Which {spec.label} did you mean by {phrase!r}?",
            candidates=candidates,
        ),
        registry_version=version,
    )
