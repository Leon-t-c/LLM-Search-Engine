"""The three response shapes: ok, needs_clarification, refused."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..translate.refusals import RefusalReason
from .payload import Payload

# No confidence thresholds here on purpose (decided 2026-09-04).
# The demo has three outcomes: ok, needs_clarification (a value failed to
# resolve — deterministic), refused. Nothing scores a guess, so a threshold
# would be two numbers we could not defend. Phase 2 measures the
# coverage-versus-caution curve and sets them against data.
#
# `Flag` and `Ok.flags` stay in the envelope: they cost nothing, they document
# where the middle rung will go, and they keep the response shape stable when
# it arrives. They are simply never populated yet.


class Candidate(BaseModel):
    """One legal value for a field, offered unranked.

    Candidates are the field's complete stored vocabulary — nothing here
    scores which one the phrase actually resembles, so there is no ordering
    to read into the sequence.
    """

    model_config = ConfigDict(frozen=True)

    field: str
    op: str
    value: str | int | float | bool | None = None
    label: str


class Unresolved(BaseModel):
    model_config = ConfigDict(frozen=True)

    phrase: str
    question: str
    candidates: tuple[Candidate, ...]

    @model_validator(mode="after")
    def _must_be_answerable(self) -> "Unresolved":
        if not self.candidates:
            raise ValueError(
                "a clarification needs at least one candidate; an open "
                "'please be more specific' tells the user nothing actionable"
            )
        return self


class Flag(BaseModel):
    model_config = ConfigDict(frozen=True)

    condition_index: int
    phrase: str
    confidence: float = Field(ge=0.0, le=1.0)


class Ok(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"
    payload: Payload
    registry_version: str
    flags: tuple[Flag, ...] = ()


class NeedsClarification(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["needs_clarification"] = "needs_clarification"
    payload: Payload
    unresolved: Unresolved
    registry_version: str


class Refused(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["refused"] = "refused"
    reason: RefusalReason
    detail: str


TranslateResponse = Ok | NeedsClarification | Refused
