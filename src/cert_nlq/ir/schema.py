"""Generate a constrained Pydantic model from a narrowed registry slice.

The model's `field`, `op`, `join` and `fn` slots are Literal unions drawn from
the registry, so a structured-output model physically cannot name a field that
does not exist or a join we did not enumerate. This is the property that
replaces the SQL parser and security validator a text-to-SQL design needs.
"""
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, create_model

from ..registry.models import FieldSpec, RootSpec
from .payload import Having, Sort

#: Stands in for "no joins available" — `Literal[()]` is not constructible.
_NO_JOIN = "__none__"


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
        raise ValueError(f"no fields in scope for root {root.root!r}")

    keys = tuple(f.key for f in scoped)
    ops = tuple(sorted({op for f in scoped for op in f.operators}))
    join_names = tuple(j.table for j in root.joins) or (_NO_JOIN,)
    title = root.root.replace("_", " ").title().replace(" ", "")

    condition = create_model(
        f"{title}Condition",
        __config__=ConfigDict(frozen=True),
        field=(Literal[keys], ...),
        op=(Literal[ops], ...),
        value=(str | int | float | bool | None, None),
        value2=(str | int | float | bool | None, None),
    )
    aggregate = create_model(
        f"{title}Aggregate",
        __config__=ConfigDict(frozen=True, populate_by_name=True),
        fn=(Literal["count", "sum", "avg", "min", "max"], ...),
        field=(Literal[keys + ("*",)], ...),
        alias=(str, Field(alias="as")),
    )
    return create_model(
        f"{title}Payload",
        __config__=ConfigDict(frozen=True, populate_by_name=True),
        root=(Literal[(root.root,)], ...),
        combinator=(Literal["AND", "OR"], "AND"),
        conditions=(tuple[condition, ...], ()),
        columns=(tuple[Literal[keys], ...], ()),
        join=(tuple[Literal[join_names], ...], ()),
        aggregate=(tuple[aggregate, ...], ()),
        group_by=(tuple[Literal[keys], ...], ()),
        having=(tuple[Having, ...], ()),
        sort=(Sort | None, None),
    )


def json_schema_for(
    root: RootSpec,
    groups: Sequence[str] = (),
    joins: Sequence[str] = (),
) -> dict:
    """The constrained schema to hand to a structured-output call."""
    return build_payload_model(root, groups, joins).model_json_schema(by_alias=True)
