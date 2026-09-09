"""Stage 2 plus orchestration: question in, one of three responses out."""
from datetime import UTC, datetime

from pydantic import ValidationError

from ..ir.payload import Condition, Group, Payload, iter_conditions
from ..ir.response import (
    Candidate,
    NeedsClarification,
    Ok,
    Refused,
    TranslateResponse,
    Unresolved,
)
from ..ir.schema import NO_JOIN, json_schema_for
from ..ir.validate import PROBLEM_UNJOINED, PayloadError, validate_payload
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
    "A group needs at least one child. Leave a group out rather than sending "
    "an empty one.\n"
    "Prefer the `in` operator over a group when one field takes several "
    "values: `status in [A, B]` rather than a group of two equalities.\n"
    "For a value from a coded field, write the meaning in plain words; a later "
    "step maps it to the stored code. Do not guess a code.\n"
    "Leave a name or address exactly as written, placeholders included.\n"
    "Counting, totalling and averaging belong in `aggregate`, never in a "
    "filter: how many is `aggregate: [{fn: count, field: *, as: total}]`.\n"
    "The remaining slots: `columns` names the fields to return, `group_by` "
    "breaks an aggregate down per category, `having` filters on an aggregate "
    "by its `as` name, `sort` orders by one field or one aggregate, and "
    "`join` names a related table the fields you used come from.\n"
    "Every payload needs at least one condition or one aggregate."
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

    resolved, unresolved = resolve_values(payload, root, now_year or _this_year())
    if unresolved:
        return _clarify(resolved, root, unresolved, registry.version)
    return Ok(payload=resolved, registry_version=registry.version)


def _this_year() -> int:
    """The fallback for a caller that did not say what year it is.

    UTC rather than the server's local date. "This year" resolved off a local
    clock gives two callers in different zones different answers for a few
    hours around every new year, and which answer you get depends on where
    the process happens to be deployed. A caller who cares sends `now_year`;
    a caller who does not gets one answer, the same everywhere.
    """
    return datetime.now(UTC).year


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
            proposed = Payload.model_validate(_normalise(raw))
            return validate_payload(
                _with_router_joins(proposed, chosen), registry, expected_root=root.root
            ), ""
        except (PayloadError, ValidationError) as exc:
            problem = str(exc)
            # Branch on the *kind* of failure, never on the wording of the
            # sentence: the sentence is written for the model to read and is
            # free to change, and a substring match on it would revert this
            # choice silently the first time someone improved it.
            unjoined = isinstance(exc, PayloadError) and PROBLEM_UNJOINED in exc.kinds
            hint = _RETRY_JOIN if unjoined else _RETRY_DEFAULT
            asked = (
                f"{question}\n\n"
                f"Your previous answer failed validation: {problem}\n"
                f"{hint}"
            )
    return None, problem


def _drop_empty_groups(node):
    """Remove every group with no children, innermost first.

    A dict without a `children` key is a leaf condition and is returned as it
    came. Anything that is not the expected shape is also returned untouched:
    this runs on raw provider output, and a malformed answer must still reach
    parsing and fail there with a message about what is actually wrong.
    """
    if not isinstance(node, dict) or "children" not in node:
        return node
    children = node.get("children")
    if not isinstance(children, (list, tuple)):
        return node
    kept = [k for k in (_drop_empty_groups(c) for c in children) if k is not None]
    if not kept:
        return None
    return {**node, "children": kept}


def _normalise(raw):
    """Reconcile the shapes the schema offers with the ones validation accepts.

    Three payloads are legal under the constrained schema and were refused
    downstream. Every one of them costs the same thing: a model that followed
    the schema exactly spends the single retry on a shape the schema itself
    put in front of it, and a second such shape in the same answer is a
    refusal for a question that had a perfectly good answer. So each is
    rewritten here into the thing it plainly means, rather than bounced back.

    * A group with no children. The strict subset both vendors accept has no
      array-length keyword, so "at least one child" cannot be stated in the
      schema at all — tightening it is not an option. An empty group carries
      no meaning, so it goes, the way an emptied one does in `_without_field`.
      The prompt discourages it too; this is the backstop.
    * The no-joins placeholder. A root with no joins still has to offer the
      required `join` slot something — see `NO_JOIN` — and it means "no
      joins", not a table by that name. Filtered out, so following the schema
      exactly is not punished.
    * A `sort` object naming neither of its two targets. `sort` is nullable
      *and* required, and both of its targets are nullable, so the closed
      schema offers two ways to say "no sort" and only one of them parsed.
      An object saying nothing about what to sort by is no sort.

    An object naming *both* sort targets is left alone: that is genuinely
    ambiguous, not a shape the schema pushed the model into, and it should
    fail and be retried.
    """
    if not isinstance(raw, dict):
        return raw
    out = dict(raw)
    join = out.get("join")
    if isinstance(join, (list, tuple)):
        out["join"] = [table for table in join if table != NO_JOIN]
    if "where" in out:
        out["where"] = _drop_empty_groups(out["where"])
    sort = out.get("sort")
    if isinstance(sort, dict) and sort.get("field") is None and sort.get("agg") is None:
        out["sort"] = None
    return out


def _with_router_joins(payload: Payload, chosen: Route) -> Payload:
    """Restate on the payload the joins stage 1 already chose.

    The router decides which tables the question needs and the schema then
    offers their fields, but nothing asks the model to repeat that decision —
    so a payload using a joined field without naming the join is not a model
    error. Failing it wastes the one retry and then refuses with a reason that
    is untrue: the field is in the schema.

    Only tables the payload actually references are added. Stage 1 is meant to
    be allowed to over-select — an over-broad *slice* costs nothing, because
    it only widens what the schema offers. A join is different: it changes the
    rows. Adding a one-to-many table nothing references multiplies the root's
    rows, so a count returns the number of related records instead of the
    number of things asked about — a wrong answer that looks entirely right.
    Restating a decision is safe; acting on one nobody used is not.

    Union rather than overwrite, so a join the model added is kept and still
    checked against the registry, and the router's are appended in its own
    order — nothing here depends on iteration order.

    Left alone when the model ignored the schema's single-valued `root`: the
    router's joins belong to another entity then, and adding them would
    replace one honest failure with a confusing one.
    """
    if payload.root != chosen.root:
        return payload
    used = _tables_referenced(payload)
    extra = tuple(
        t for t in chosen.joins if t in used and t not in payload.join
    )
    if not extra:
        return payload
    return payload.model_copy(update={"join": payload.join + extra})


def _tables_referenced(payload: Payload) -> set[str]:
    """Every table named by a field key anywhere in the payload.

    Keys are `table.column`, so the prefix is the table. Read from every slot
    that can carry a field, not just the conditions: a joined column can be
    selected, grouped by, aggregated or sorted on without ever appearing in a
    filter.

    `having` is not read here: it references an aggregate by alias, and that
    aggregate's own field is already counted.
    """
    keys = [c.field for c in iter_conditions(payload.where)]
    keys += list(payload.columns) + list(payload.group_by)
    keys += [a.field for a in payload.aggregate]
    if payload.sort is not None and payload.sort.field:
        keys.append(payload.sort.field)
    return {key.split(".", 1)[0] for key in keys if "." in key}


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
        # Not FIELD_NOT_IN_SCHEMA: the field is in the schema and the payload
        # validated. Either it has no closed vocabulary to offer as
        # candidates, or pruning the unresolved values left nothing for the
        # caller's builder — both are the value failing, not the field.
        return Refused(
            reason=RefusalReason.VALUE_NOT_UNDERSTOOD,
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
