"""Stage 1: narrow the registry structurally before translating.

This is schema retrieval done by the schema's own shape rather than by
similarity — no embeddings, no vector index, no top-k threshold to tune. It is
also separately measurable: a stage-1 miss is diagnosable on its own, which a
single blended call would hide.

Groups matter as much as roots here. One root can carry several hundred fields,
so routing to a root alone barely narrows anything.

The group names come from the host app's tab layout, so they are section
abbreviations rather than descriptions. Routing on the name alone would be
guesswork, so each group is presented with its registry description and a
sample of its real field labels.
"""
from pydantic import BaseModel, ConfigDict, ValidationError

from ..registry.models import Registry, RootSpec
from .provider import Provider

#: Field labels shown per group in the routing prompt. A group can hold dozens
#: of fields; pasting all of them into every call is not worth the tokens.
MAX_SAMPLE_LABELS = 8

#: Names the stage-1 schema. Providers may key cost decisions off it — a
#: routing call is classification and does not need the translator's model.
ROUTE_SCHEMA_NAME = "Route"

ROUTER_SYSTEM = (
    "You route a data question to one entity, the related tables it needs, and "
    "the field groups it touches.\n"
    "Choose exactly one root: the entity the answer has one row per.\n"
    "Choose joins only when the question needs a field from that table.\n"
    "Choose the narrowest set of field groups that covers the question. Group "
    "names are section labels from the application's own screens, so judge a "
    "group by its description and example fields, not by its name. Leave "
    "groups empty only when the question is too broad to place.\n"
    "You are not writing the query. Name the slice; another step translates."
)


class RoutingError(RuntimeError):
    """The router returned something unusable."""


class Route(BaseModel):
    model_config = ConfigDict(frozen=True)

    root: str
    joins: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()


def sample_labels(root: RootSpec, group: str) -> list[str]:
    """Up to MAX_SAMPLE_LABELS field labels from `group`, in label order."""
    return [f.label for f in root.fields_in_group(group)][:MAX_SAMPLE_LABELS]


def route_schema(registry: Registry) -> dict:
    """A hand-built JSON Schema — the shape is fixed, only the enums vary."""
    roots = [r.root for r in registry.roots]
    joins = sorted({j.table for r in registry.roots for j in r.joins})
    groups = sorted({g for r in registry.roots for g in r.group_names()})
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["root", "joins", "groups"],
        "properties": {
            "root": {"type": "string", "enum": roots},
            "joins": {"type": "array",
                      "items": {"type": "string", "enum": joins or [""]}},
            "groups": {"type": "array",
                       "items": {"type": "string", "enum": groups or [""]}},
        },
    }


def _describe_groups(root: RootSpec) -> str:
    """One line per group, in registry order: name, description, examples."""
    lines = []
    for name in root.group_names():
        spec = root.group(name)
        parts = [f"      - {name}"]
        if spec is not None and spec.description:
            parts.append(spec.description)
        labels = sample_labels(root, name)
        if labels:
            parts.append("e.g. " + ", ".join(labels))
        lines.append(" — ".join(parts))
    return "\n".join(lines)


def _describe(registry: Registry) -> str:
    blocks = []
    for spec in registry.roots:
        joins = ", ".join(f"{j.table} ({j.label})" for j in spec.joins) or "none"
        blocks.append(
            f"- {spec.root} — {spec.label}\n"
            f"    joins:  {joins}\n"
            f"    groups:\n{_describe_groups(spec)}"
        )
    return "\n".join(blocks)


def route(question: str, registry: Registry, provider: Provider) -> Route:
    """Pick a root, its joins, and its field groups.

    A hallucinated join or group is dropped rather than refused: an over-broad
    slice still answers the question, while refusing would not.
    """
    raw = provider.complete(
        system=f"{ROUTER_SYSTEM}\n\nAvailable entities:\n{_describe(registry)}",
        question=question,
        schema=route_schema(registry),
        schema_name=ROUTE_SCHEMA_NAME,
    )
    try:
        proposed = Route.model_validate(raw)
    except ValidationError as exc:
        raise RoutingError(f"router response is malformed: {exc}") from exc

    root = registry.root(proposed.root)
    if root is None:
        raise RoutingError(f"router chose unknown root {proposed.root!r}")

    available_joins = {j.table for j in root.joins}
    available_groups = set(root.group_names())
    return Route(
        root=root.root,
        joins=tuple(t for t in proposed.joins if t in available_joins),
        groups=tuple(g for g in proposed.groups if g in available_groups),
    )
