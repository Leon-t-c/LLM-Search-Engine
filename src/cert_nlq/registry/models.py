"""Pydantic models for the registry document served by the host application.

The registry is the ONLY source of field vocabulary. Field keys are treated as
opaque strings — `table` and `column` are carried explicitly, so this service
never parses a key and works whether or not keys are table-qualified.
"""
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

FIELD_TYPES = frozenset({"text", "number", "money", "date", "year", "bool"})
CARDINALITIES = frozenset({"one-to-one", "one-to-many", "many-to-one"})


class VocabEntry(BaseModel):
    """A categorical code and what it means. Model-facing only, never displayed."""

    model_config = ConfigDict(frozen=True)

    code: str
    meaning: str
    synonyms: tuple[str, ...] = ()


class FieldSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    table: str
    column: str
    label: str
    type: str
    group: str = ""
    operators: tuple[str, ...]
    vocabulary: tuple[VocabEntry, ...] = ()

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        if value not in FIELD_TYPES:
            raise ValueError(f"unknown field type {value!r}")
        return value

    @field_validator("operators")
    @classmethod
    def _at_least_one_operator(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("a field with no operators cannot be queried")
        return value


class JoinSpec(BaseModel):
    """An enumerated join. The model picks the table; the condition is ours."""

    model_config = ConfigDict(frozen=True)

    table: str
    label: str
    cardinality: str
    on: tuple[tuple[str, str], ...]

    @field_validator("cardinality")
    @classmethod
    def _known_cardinality(cls, value: str) -> str:
        if value not in CARDINALITIES:
            raise ValueError(f"unknown cardinality {value!r}")
        return value

    @field_validator("on", mode="before")
    @classmethod
    def _normalise_pairs(cls, value):
        """Accept "year" or ["year", "shipyear"]; store pairs either way."""
        pairs = []
        for item in value or ():
            if isinstance(item, str):
                pairs.append((item, item))
            else:
                left, right = item
                pairs.append((left, right))
        if not pairs:
            raise ValueError("a join needs at least one column pair")
        return tuple(pairs)


class GroupSpec(BaseModel):
    """One picker section.

    `description` exists for the router prompt: a host application's group
    names are UI section labels — often abbreviated, and not always tied to
    the fields they contain — which a model cannot reliably interpret from
    the name alone.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    #: Always in scope, whatever stage 1 selects. Carries the key columns.
    identity: bool = False


class RootSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    root: str
    label: str
    key: tuple[str, ...]
    groups: tuple[GroupSpec, ...] = ()
    fields: tuple[FieldSpec, ...]
    joins: tuple[JoinSpec, ...] = ()

    @model_validator(mode="after")
    def _keys_unique(self) -> "RootSpec":
        keys = [f.key for f in self.fields]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate field keys on root {self.root!r}")
        return self

    @property
    def fields_by_key(self) -> dict[str, FieldSpec]:
        return {f.key: f for f in self.fields}

    def group(self, name: str) -> GroupSpec | None:
        for spec in self.groups:
            if spec.name == name:
                return spec
        return None

    def group_names(self) -> list[str]:
        """Declared groups in registry order, then any undeclared ones.

        Mirrors the host app: a group missing from its ordered list still shows
        up, after the ordered ones, so a new field can never vanish. An
        undeclared group is deliberately NOT a parse error — that would take
        the feature down over a cosmetic mismatch.
        """
        declared = [g.name for g in self.groups]
        seen = set(declared)
        extra = sorted(
            {f.group for f in self.fields if f.group and f.group not in seen}
        )
        return declared + extra

    def identity_groups(self) -> set[str]:
        return {g.name for g in self.groups if g.identity}

    def fields_in_group(self, name: str) -> list[FieldSpec]:
        return sorted(
            (f for f in self.fields if f.group == name), key=lambda f: f.label
        )

    def join(self, table: str) -> JoinSpec | None:
        for spec in self.joins:
            if spec.table == table:
                return spec
        return None


class Registry(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str
    roots: tuple[RootSpec, ...]

    @model_validator(mode="after")
    def _roots_unique(self) -> "Registry":
        names = [r.root for r in self.roots]
        if len(names) != len(set(names)):
            raise ValueError("duplicate root names")
        return self

    def root(self, name: str) -> RootSpec | None:
        for spec in self.roots:
            if spec.root == name:
                return spec
        return None
