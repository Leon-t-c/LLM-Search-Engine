from cert_nlq.config import Settings, get_settings


def test_settings_read_the_prefixed_environment(monkeypatch):
    monkeypatch.setenv("CERT_NLQ_REGISTRY_URL", "https://host.example")
    monkeypatch.setenv("CERT_NLQ_REGISTRY_TOKEN", "tok")
    settings = Settings()
    assert settings.registry_url == "https://host.example"
    assert settings.registry_token == "tok"


def test_settings_default_to_empty_rather_than_raising():
    """A missing setting must not explode at import time."""
    assert Settings(_env_file=None).registry_url == ""


def test_get_settings_is_cached_and_clearable(monkeypatch):
    monkeypatch.setenv("CERT_NLQ_MODEL", "first")
    get_settings.cache_clear()
    try:
        assert get_settings().model == "first"
        monkeypatch.setenv("CERT_NLQ_MODEL", "second")
        assert get_settings().model == "first", "should still be the cached one"
        get_settings.cache_clear()
        assert get_settings().model == "second"
    finally:
        get_settings.cache_clear()


def test_the_provider_setting_is_closed(monkeypatch):
    """A near miss must fail at boot, not deeper in with a vaguer message."""
    import pytest
    from pydantic import ValidationError

    for value in ("OpenAI", "claude ", "gemini"):
        monkeypatch.setenv("CERT_NLQ_PROVIDER", value)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)


def test_the_provider_setting_accepts_both_backends(monkeypatch):
    for value in ("openai", "claude"):
        monkeypatch.setenv("CERT_NLQ_PROVIDER", value)
        assert Settings(_env_file=None).provider == value
