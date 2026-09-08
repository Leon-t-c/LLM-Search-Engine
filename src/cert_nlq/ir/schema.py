"""Generate a constrained Pydantic model from a narrowed registry slice.

The model's `field`, `columns`, `group_by`, `sort.field`, `join` and `root`
slots are Literal unions drawn from the registry, so a structured-output
model physically cannot name a field, join or root that does not exist. The
vocabulary is closed this way; the field/operator *pairing* is not — `op` is
the union across the whole slice, so a number field can still be paired with
a text operator, and that combination is what validation, not the schema,
rejects.
"""
import json
from typing import Literal, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from ..registry.models import FieldSpec, RootSpec
from .payload import Having, Sort

#: Stands in for "no joins available" — `Literal[()]` is not constructible.
_NO_JOIN = "__none__"

#: Comparison operators legal against an aggregate in `having`. Fixed, and
#: independent of the registry — an aggregate is always a number.
HAVING_OPS = ("=", "!=", ">", "<", ">=", "<=")

class ScopeError(ValueError):
    """A root offers no fields to translate against.

    Subclasses ValueError so existing callers and tests are unaffected; it
    exists so the HTTP layer can tell a registry-shaped failure apart from a
    bug in this service.
    """


#: Levels of grouping the generated schema offers. Finite by construction —
#: Group1 holds conditions only, GroupN holds conditions or Group(N-1) — so
#: nothing is self-referential and no provider depth limit applies. Measured:
#: two extra levels cost about 600 bytes on a 110-field slice.
MAX_GROUP_DEPTH = 3


def fields_in_scope(
    root: RootSpec,
    groups: Sequence[str] = (),
    joins: Sequence[str] = (),
) -> list[FieldSpec]:
    """Fields the model may reference: the root's own, plus joined tables'.

    An empty `groups` means every group. A field belonging to another table is
    in scope only when that table is among `joins`. The root's identity groups
    are always kept, so a stage-1 group miss narrows the slice without
    stranding the key columns.

    Registry field order is preserved, which keeps the generated Literal — and
    therefore the JSON schema handed to the model — stable across calls.
    """
    tables = {root.root} | set(joins)
    candidates = [f for f in root.fields if f.table in tables]
    if not groups:
        return candidates

    wanted = set(groups) | root.identity_groups()
    kept = [f for f in candidates if f.group in wanted]
    # A joined table that no selected group covers would contribute nothing,
    # which makes the join pointless — fall back to all of that table's fields.
    covered = {f.table for f in kept}
    return [f for f in candidates if f.group in wanted or f.table not in covered]


def build_payload_model(
    root: RootSpec,
    groups: Sequence[str] = (),
    joins: Sequence[str] = (),
) -> type[BaseModel]:
    scoped = fields_in_scope(root, groups, joins)
    if not scoped:
        raise ScopeError(f"no fields in scope for root {root.root!r}")

    keys = tuple(f.key for f in scoped)
    ops = tuple(sorted({op for f in scoped for op in f.operators}))
    join_names = tuple(j.table for j in root.joins) or (_NO_JOIN,)
    title = root.root.replace("_", " ").title().replace(" ", "")

    condition = create_model(
        f"{title}Condition",
        __config__=ConfigDict(frozen=True),
        field=(Literal[keys], ...),
        op=(Literal[ops], ...),
        value=(str | int | float | bool | tuple[str | int | float | bool, ...] | None, None),
        value2=(str | int | float | bool | None, None),
    )

    # A finite chain rather than a recursive definition — Group1 holds
    # conditions only, GroupN holds conditions or Group(N-1) — so nothing in
    # the generated schema is self-referential.
    node = condition
    for level in range(1, MAX_GROUP_DEPTH + 1):
        node = create_model(
            f"{title}Group{level}",
            __config__=ConfigDict(frozen=True),
            combinator=(Literal["AND", "OR"], ...),
            children=(
                tuple[condition if level == 1 else Union[condition, node], ...],
                ...,
            ),
        )
    where = node

    aggregate = create_model(
        f"{title}Aggregate",
        __config__=ConfigDict(frozen=True, populate_by_name=True),
        fn=(Literal["count", "sum", "avg", "min", "max"], ...),
        field=(Literal[keys + ("*",)], ...),
        alias=(str, Field(alias="as")),
    )
    # `sort.field` and `having.op` are narrowed by subclassing rather than
    # redeclaring, so the xor rule on Sort and the shape of Having stay
    # single-source in payload.py.
    #
    # `having.agg` deliberately stays an unconstrained `str`: it names an alias
    # declared elsewhere in the same payload, which is not knowable when this
    # schema is built. Validation resolves it against the declared aliases.
    sort = create_model(f"{title}Sort", __base__=Sort,
                        field=(Literal[keys] | None, None))
    having = create_model(f"{title}Having", __base__=Having,
                          op=(Literal[HAVING_OPS], ...))

    return create_model(
        f"{title}Payload",
        __config__=ConfigDict(frozen=True, populate_by_name=True),
        root=(Literal[(root.root,)], ...),
        where=(where | None, None),
        columns=(tuple[Literal[keys], ...], ()),
        join=(tuple[Literal[join_names], ...], ()),
        aggregate=(tuple[aggregate, ...], ()),
        group_by=(tuple[Literal[keys], ...], ()),
        having=(tuple[having, ...], ()),
        sort=(sort | None, None),
    )


def dedupe_enums(schema: dict) -> dict:
    """Hoist each repeated `enum` list into `$defs` and `$ref` it.

    The field-key enum appears in several slots of the payload schema, and the
    whole schema is sent on every translation call. Pydantic inlines `Literal`
    enums, so this is a post-processing pass rather than something the model
    definition can express.

    Only lists that occur more than once are hoisted; a schema with no
    repetition comes back unchanged, which makes the function idempotent.
    """
    counts: dict[str, int] = {}

    def tally(node):
        if isinstance(node, dict):
            values = node.get("enum")
            if isinstance(values, list) and len(values) > 1:
                counts[json.dumps(values)] = counts.get(json.dumps(values), 0) + 1
            for value in node.values():
                tally(value)
        elif isinstance(node, list):
            for value in node:
                tally(value)

    tally(schema)
    repeated = [key for key, n in counts.items() if n > 1]
    if not repeated:
        return schema

    names = {key: f"Enum{i}" for i, key in enumerate(sorted(repeated))}

    def rewrite(node):
        if isinstance(node, dict):
            values = node.get("enum")
            key = json.dumps(values) if isinstance(values, list) else None
            if key in names:
                # Emit a BARE `$ref` — drop every sibling key rather than
                # carrying `title`/`description` alongside it. JSON Schema
                # 2020-12 permits siblings, but some providers' strict
                # structured-output modes reject them, and the siblings here
                # are purely informational. Dropping them also shrinks the
                # schema further.
                return {"$ref": f"#/$defs/{names[key]}"}
            return {k: rewrite(v) for k, v in node.items()}
        if isinstance(node, list):
            return [rewrite(v) for v in node]
        return node

    result = rewrite(schema)
    defs = dict(result.get("$defs") or {})
    for key, name in names.items():
        values = json.loads(key)
        defs[name] = {"type": "string", "enum": values}
    result["$defs"] = defs
    return result


def strictify(node):
    """Rewrite a generated schema into the subset the providers accept.

    Both vendors converge on the same demands: every object closed, every
    property required, and no keyword the validator does not understand.
    Optional fields survive as an explicit null in the type union, which is
    the documented way to say "may be absent" in a closed schema — the model
    then has to state the absence rather than omit the key.

    Done here rather than in each adapter so the two cannot drift, and so the
    shape can be asserted offline. Nothing else consumes this output:
    payload validation runs off the models, not the schema.
    """
    if isinstance(node, list):
        return [strictify(child) for child in node]
    if not isinstance(node, dict):
        return node

    # `default` goes because a closed schema states every key, so a default
    # can never apply; `title` goes because it is decoration the model never
    # reads and pays for on every call — a small slice carries dozens. A bare
    # `$ref` stays bare: nothing below adds a key to a node that declares
    # neither `type: object` nor `properties`.
    rewritten = {
        key: strictify(value)
        for key, value in node.items()
        if key not in ("default", "title")
    }
    if "const" in rewritten:
        # One vendor's normaliser handles a fixed keyword list, and `const`
        # is not on it. An unrecognised keyword there is not rejected — it is
        # folded into the description string, which leaves the slot
        # unconstrained while looking constrained. A single-value `enum` says
        # the same thing, is on both vendors' lists, and keeps any sibling
        # `type`.
        rewritten["enum"] = [rewritten.pop("const")]
    properties = rewritten.get("properties")
    if rewritten.get("type") != "object" and not isinstance(properties, dict):
        return rewritten
    rewritten["additionalProperties"] = False
    rewritten["required"] = list(properties or {})
    return rewritten


def json_schema_for(
    root: RootSpec,
    groups: Sequence[str] = (),
    joins: Sequence[str] = (),
) -> dict:
    """The constrained schema to hand to a structured-output call.

    Enum lists are deduplicated: the same field-key list otherwise ships
    several times in every call, and the schema is the bulk of the prompt.
    The result is then narrowed to the subset the providers accept — see
    `strictify` — so no adapter has to remember to do it.
    """
    model = build_payload_model(root, groups, joins)
    return strictify(dedupe_enums(model.model_json_schema(by_alias=True)))
