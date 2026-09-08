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

#: Retry guidance, picked by the shape of the failure. A field that merely
#: needs its table joined *is* in the schema, so the general sentence — use
#: only the offered fields — points the model away from the fix.
_RETRY_DEFAULT = "Correct it using only the offered fields and operators."
_RETRY_JOIN = (
    "The fields you used are offered. Name the table each one comes from in "
    "`join` and leave the fields as they are."
)

#: The marker validation emits for a field that exists but is unjoined.
_UNJOINED = "requires join"


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
        return _clarify(resolved, root, unresolved, registry.version)
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
            proposed = Payload.model_validate(raw)
            return validate_payload(_with_router_joins(proposed, chosen), registry), ""
        except (PayloadError, ValidationError) as exc:
            problem = str(exc)
            hint = _RETRY_JOIN if _UNJOINED in problem else _RETRY_DEFAULT
            asked = (
                f"{question}\n\n"
                f"Your previous answer failed validation: {problem}\n"
                f"{hint}"
            )
    return None, problem


def _with_router_joins(payload: Payload, chosen: Route) -> Payload:
    """Restate on the payload the joins stage 1 already chose.

    The router decides which tables the question needs and the schema then
    offers their fields, but nothing asks the model to repeat that decision —
    so a payload using a joined field without naming the join is not a model
    error. Failing it wastes the one retry and then refuses with a reason that
    is untrue: the field is in the schema.

    Union rather than overwrite, so a join the model added is kept and still
    checked against the registry, and the router's are appended in its own
    order — nothing here depends on iteration order.

    Left alone when the model ignored the schema's single-valued `root`: the
    router's joins belong to another entity then, and adding them would
    replace one honest failure with a confusing one.
    """
    if payload.root != chosen.root:
        return payload
    extra = tuple(t for t in chosen.joins if t not in payload.join)
    if not extra:
        return payload
    return payload.model_copy(update={"join": payload.join + extra})


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
    payload: Payload, root: RootSpec, unresolved: list[Unresolvable], version: str
) -> TranslateResponse:
    """Ask about the first unresolved value; ship none of them.

    Every unresolved field is pruned, not just the one being asked about. The
    payload that goes back is documented as the seed for the caller's builder,
    so anything left in it is a claim that it resolved — a second bogus value
    riding along shows the user an invalid filter as though it were valid, and
    compiles to a query matching nothing. Only the first is asked about, but
    the rest are simply not asserted.

    Candidates carry the registry's field label with the raw stored value —
    `Status = X`, never a paraphrase. Users read these codes fluently. Each
    candidate is offered under the operator the phrase was actually asked
    under — for a list (`in`) operator that means each
    candidate offers one legal element, not a whole list.
    """
    asked_about = unresolved[0]
    key, phrase = asked_about.field, str(asked_about.value)
    spec = root.fields_by_key.get(key)
    kept = payload.where
    # dict.fromkeys, not a set: one pass per distinct field, in the order the
    # values were reported, so the surviving tree does not depend on iteration
    # order. Pruning is idempotent per field, hence the dedupe.
    for unresolved_key in dict.fromkeys(u.field for u in unresolved):
        kept = _without_field(kept, unresolved_key)
    vocabulary = spec.vocabulary if spec is not None else ()
    if not vocabulary or kept is None:
        return Refused(
            reason=RefusalReason.FIELD_NOT_IN_SCHEMA,
            detail=f"Could not work out what {phrase!r} means here.",
        )
    candidates = tuple(
        Candidate(
            field=key,
            op=asked_about.op,
            value=entry.code,
            label=spec.label,
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
