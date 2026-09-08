"""The single seam through which every LLM call passes.

Keeping this narrow is what makes the pipeline testable without a network or an
API key, and what makes the provider decision a one-file change
rather than a rewrite.
"""
from typing import Protocol, Sequence, runtime_checkable


class ProviderError(RuntimeError):
    """The provider failed, or returned something unusable."""


#: The token-count keys every adapter reports, and the property that makes a
#: mixed run addable: the input keys do not overlap. The vendors disagree on
#: this — one reports an input total that *includes* the cached reads, the
#: other reports counters that sit beside each other. Summing one key across
#: a golden set answered by both is only meaningful in the non-overlapping
#: form, so that is the form every adapter converts into.
#:
#: `reasoning_tokens` is the one key that is a breakdown rather than a
#: sibling: it is part of `output_tokens` on both vendors, so it prices the
#: output but must never be added to it.
USAGE_TOKEN_KEYS = (
    "uncached_input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
)


def token_count(source, name: str) -> int:
    """One count off a usage object, or 0 when the vendor did not send it.

    Every breakdown field is optional on both vendors' responses, and an
    absent one means none of that kind rather than an unknown quantity.
    Reading usage must also never fail a call that has already succeeded,
    so a half-formed object degrades to zero rather than raising.
    """
    value = getattr(source, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


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
