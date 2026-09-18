"""The golden set: the questions, and what the service is supposed to say.

A gold case is a question paired with one expectation, and the expectation is
the *whole* answer the harness scores against -- not a hint, not a partial
one. Three shapes, because the service has three: `ok` carries the payload it
should return **after value resolution** (stored codes, not the words a user
typed), `clarify` names the field whose vocabulary should come back as
candidates, and `refusal` names the closed reason.

Two rules run this module, and both exist because a wrong gold case is worse
than a missing one -- it does not fail, it silently reports the wrong number
for every run afterwards:

1. An ok-payload is validated through `ir.validate.validate_payload`, the same
   gate the service itself must pass. A gold payload naming a field that does
   not exist, or an operator the field does not allow, could never be produced
   by a correct service, so scoring against it measures nothing.
2. Every problem in the file is collected and reported together. A loader that
   raises on the first one turns a review pass over fifty-odd hand-written
   cases into fifty-odd runs.
"""
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..ir.validate import PayloadError, validate_payload
from ..registry.models import Registry
from ..translate.refusals import RefusalReason

#: The per-slice axis. Exactly one of these tags per case, so a report broken
#: down by slice partitions the set rather than double-counting it. A refusal
#: slice is named per reason -- "refusals are 80% correct" over four unrelated
#: reasons hides which one is broken.
SLICE_TAGS = frozenset(
    {"selection", "values", "aggregate", "join", "clarify"}
    | {f"refusal:{reason.value}" for reason in RefusalReason}
)


class ExpectOk(BaseModel):
    """The payload the service should return, resolved values and all.

    Held as a `dict` rather than a `Payload` on purpose. The harness scores
    exact match against what the service returned, and the comparison is a
    property of the gold file's own text: parsing into the model first would
    fill in defaults and normalise shapes, so a gold file that quietly said
    something other than what its author wrote would compare equal anyway.
    Validation happens separately, in `load_golden`, where its failures can be
    reported by case id.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["ok"] = "ok"
    payload: dict


class ExpectClarify(BaseModel):
    """`needs_clarification`, on this field.

    The field, not the candidate list: the candidates are the field's whole
    stored vocabulary, which the registry already decides. What the gold case
    asserts is that the service asked about the *right* field -- that is the
    numerator of candidate recall, and the part a translation can get wrong.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["clarify"] = "clarify"
    field: str


class ExpectRefusal(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["refusal"] = "refusal"
    reason: str


Expectation = Annotated[
    ExpectOk | ExpectClarify | ExpectRefusal, Field(discriminator="kind")
]


class GoldenCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    tags: tuple[str, ...] = ()
    expect: Expectation


class GoldenError(ValueError):
    """Everything wrong with a golden file, in one exception."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def load_golden(path: str | Path, registry: Registry) -> list[GoldenCase]:
    """Read a JSONL golden file and check every case against the registry.

    One case per line; blank lines and `#` comment lines are skipped so the
    file can carry section headings for the human who reviews it.

    Raises `GoldenError` listing every problem found, each prefixed with the
    case id it belongs to -- or with the line number, when the case is too
    broken to have a usable id.
    """
    problems: list[str] = []
    cases: list[GoldenCase] = []
    first_seen: dict[str, int] = {}

    for lineno, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            problems.append(f"line {lineno}: not valid JSON ({exc.msg})")
            continue
        if not isinstance(raw, dict):
            problems.append(f"line {lineno}: expected one JSON object per line")
            continue
        label = raw.get("id") if isinstance(raw.get("id"), str) else f"line {lineno}"
        try:
            case = GoldenCase.model_validate(raw)
        except ValidationError as exc:
            problems.append(f"{label}: {_summarise(exc)}")
            continue
        if case.id in first_seen:
            problems.append(
                f"{case.id}: duplicate id (first seen on line {first_seen[case.id]})"
            )
            continue
        first_seen[case.id] = lineno
        problems += _check_tags(case)
        problems += _check_expectation(case, registry)
        cases.append(case)

    if problems:
        raise GoldenError(problems)
    return cases


def _summarise(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<case>'}: {err['msg']}"
        for err in exc.errors()
    )


def _check_tags(case: GoldenCase) -> list[str]:
    slices = sorted(set(case.tags) & SLICE_TAGS)
    if len(slices) == 1:
        return []
    if not slices:
        return [f"{case.id}: no slice tag (one of {', '.join(sorted(SLICE_TAGS))})"]
    return [f"{case.id}: more than one slice tag {slices}"]


def _check_expectation(case: GoldenCase, registry: Registry) -> list[str]:
    expect = case.expect
    if isinstance(expect, ExpectRefusal):
        if expect.reason not in {reason.value for reason in RefusalReason}:
            return [f"{case.id}: unknown refusal reason {expect.reason!r}"]
        return []
    if isinstance(expect, ExpectClarify):
        # Any root: a clarification names a field, and which root the question
        # routes to is the service's decision, not the gold case's.
        if not any(
            expect.field in root.fields_by_key for root in registry.roots
        ):
            return [f"{case.id}: unknown clarification field {expect.field!r}"]
        return []
    try:
        validate_payload(expect.payload, registry)
    except PayloadError as exc:
        return [f"{case.id}: {problem}" for problem in exc.problems]
    except ValidationError as exc:
        return [f"{case.id}: {_summarise(exc)}"]
    return []
