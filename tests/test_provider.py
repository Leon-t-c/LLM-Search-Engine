import pytest

from cert_nlq.translate.provider import FakeProvider, ProviderError


def test_fake_returns_scripted_responses_in_order():
    provider = FakeProvider([{"a": 1}, {"b": 2}])
    assert provider.complete("sys", "q", {}, "S") == {"a": 1}
    assert provider.complete("sys", "q", {}, "S") == {"b": 2}


def test_fake_records_what_it_was_asked():
    provider = FakeProvider([{"a": 1}])
    provider.complete("be terse", "how many widgets", {"type": "object"}, "Route")
    assert provider.calls == [
        {"system": "be terse", "question": "how many widgets",
         "schema": {"type": "object"}, "schema_name": "Route"}
    ]


def test_fake_raises_a_scripted_exception():
    provider = FakeProvider([ProviderError("rate limited")])
    with pytest.raises(ProviderError, match="rate limited"):
        provider.complete("sys", "q", {}, "S")


def test_fake_running_out_of_responses_is_a_test_bug_not_a_silent_pass():
    provider = FakeProvider([{"a": 1}])
    provider.complete("sys", "q", {}, "S")
    with pytest.raises(AssertionError, match="ran out of scripted responses"):
        provider.complete("sys", "q", {}, "S")
