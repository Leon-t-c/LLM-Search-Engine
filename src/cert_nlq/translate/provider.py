"""The single seam through which every LLM call passes.

Keeping this narrow is what makes the pipeline testable without a network or an
API key, and what makes the provider decision a one-file change
rather than a rewrite.
"""
from typing import Protocol, Sequence, runtime_checkable


class ProviderError(RuntimeError):
    """The provider failed, or returned something unusable."""


@runtime_checkable
class Provider(Protocol):
    def complete(
        self, system: str, question: str, schema: dict, schema_name: str
    ) -> dict:
        """Return a dict conforming to `schema`, or raise ProviderError."""
        ...


class FakeProvider:
    """A scripted provider for tests. Never makes a network call."""

    def __init__(self, responses: Sequence[dict | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(
        self, system: str, question: str, schema: dict, schema_name: str
    ) -> dict:
        self.calls.append(
            {"system": system, "question": question,
             "schema": schema, "schema_name": schema_name}
        )
        assert self._responses, (
            "FakeProvider ran out of scripted responses after "
            f"{len(self.calls)} call(s)"
        )
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt
