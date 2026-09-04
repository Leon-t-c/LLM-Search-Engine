"""The IR: exactly what the host application's query compiler accepts.

Structural shape only. Whether a field exists, and whether an operator is
legal for it, is decided against the registry in `validate.py` — these models
cannot know, because the vocabulary arrives at runtime.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

JsonScalar = str | int | float | bool

AGG_FNS = frozenset({"count", "sum", "avg", "min", "max"})

#: Operators that take no value at all.
NO_VALUE_OPS = frozenset({"is_blank", "is_not_blank", "is_true", "is_false"})


class Condition(BaseModel):
    model_config = ConfigDict(frozen=True)

    field: str
    op: str
    value: JsonScalar | None = None
    value2: JsonScalar | None = None


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
    combinator: Literal["AND", "OR"] = "AND"
    conditions: tuple[Condition, ...] = ()
    columns: tuple[str, ...] = ()
    join: tuple[str, ...] = ()
    aggregate: tuple[Aggregate, ...] = ()
    group_by: tuple[str, ...] = ()
    having: tuple[Having, ...] = ()
    sort: Sort | None = None
