"""The IR: exactly what the host application's query compiler accepts.

Structural shape only. Whether a field exists, and whether an operator is
legal for it, is decided against the registry in `validate.py` — these models
cannot know, because the vocabulary arrives at runtime.
"""
from typing import Iterator, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

JsonScalar = str | int | float | bool

AGG_FNS = frozenset({"count", "sum", "avg", "min", "max"})

#: Operators that take no value at all.
NO_VALUE_OPS = frozenset({"is_blank", "is_not_blank", "is_true", "is_false"})

#: Operators taking a list rather than a scalar.
LIST_OPS = frozenset({"in"})


class Condition(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    op: str
    value: JsonScalar | tuple[JsonScalar, ...] | None = None
    value2: JsonScalar | None = None


class Group(BaseModel):
    """A combinator over child nodes, each a condition or another group.

    Recursive here on purpose: this model is hand-written, validates a payload
    from any source, and a forward reference costs nothing. The *generated*
    schema in schema.py is a finite chain instead — a recursive JSON Schema
    runs into depth and size limits in some strict structured-output modes.
    """

    model_config = ConfigDict(frozen=True)

    combinator: Literal["AND", "OR"]
    children: tuple[Union[Condition, "Group"], ...]

    @model_validator(mode="after")
    def _not_empty(self) -> "Group":
        if not self.children:
            raise ValueError("a group needs at least one child")
        return self


def iter_conditions(node: Union[Condition, Group, None]) -> Iterator[Condition]:
    """Every leaf condition, in order. The single traversal all consumers use.

    Two independent tree walks is two places to get depth or ordering wrong.
    """
    if node is None:
        return
    if isinstance(node, Condition):
        yield node
        return
    for child in node.children:
        yield from iter_conditions(child)


class Aggregate(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    fn: Literal["count", "sum", "avg", "min", "max"]
    field: str
    alias: str = Field(alias="as")


class Having(BaseModel):
    model_config = ConfigDict(frozen=True)

    agg: str
    op: str
    value: JsonScalar


class Sort(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str | None = None
    agg: str | None = None
    dir: Literal["asc", "desc"] = "asc"

    @model_validator(mode="after")
    def _one_target(self) -> "Sort":
        if (self.field is None) == (self.agg is None):
            raise ValueError("sort needs exactly one of 'field' or 'agg'")
        return self


class Payload(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    root: str
    where: Group | None = None
    columns: tuple[str, ...] = ()
    join: tuple[str, ...] = ()
    aggregate: tuple[Aggregate, ...] = ()
    group_by: tuple[str, ...] = ()
    having: tuple[Having, ...] = ()
    sort: Sort | None = None
