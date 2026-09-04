import httpx
import pytest
import respx

from cert_nlq.registry.client import RegistryClient, RegistryUnavailable

URL = "https://cert.example/query/registry/"


@respx.mock
def test_fetch_parses_and_caches_by_version(registry_json):
    route = respx.get(URL).mock(
        return_value=httpx.Response(200, json=registry_json)
    )
    client = RegistryClient("https://cert.example", token="t")

    first = client.fetch()
    second = client.fetch()

    assert first.version == "sha256:fixture-v1"
    assert second is first, "a second fetch must reuse the cached registry"
    assert route.call_count == 1


@respx.mock
def test_force_refetches_and_replaces_the_cache(registry_json):
    bumped = {**registry_json, "version": "sha256:fixture-v2"}
    respx.get(URL).mock(
        side_effect=[
            httpx.Response(200, json=registry_json),
            httpx.Response(200, json=bumped),
        ]
    )
    client = RegistryClient("https://cert.example", token="t")

    assert client.fetch().version == "sha256:fixture-v1"
    assert client.fetch(force=True).version == "sha256:fixture-v2"
    assert client.cached.version == "sha256:fixture-v2"


@respx.mock
def test_token_is_sent_as_a_bearer_header(registry_json):
    route = respx.get(URL).mock(
        return_value=httpx.Response(200, json=registry_json)
    )
    RegistryClient("https://cert.example", token="secret-token").fetch()
    assert route.calls[0].request.headers["authorization"] == "Bearer secret-token"


@respx.mock
def test_trailing_slash_on_base_url_does_not_double_up(registry_json):
    route = respx.get(URL).mock(
        return_value=httpx.Response(200, json=registry_json)
    )
    RegistryClient("https://cert.example/", token="t").fetch()
    assert route.call_count == 1


@respx.mock
def test_http_error_raises_registry_unavailable():
    respx.get(URL).mock(return_value=httpx.Response(503, text="down"))
    with pytest.raises(RegistryUnavailable, match="503"):
        RegistryClient("https://cert.example", token="t").fetch()


@respx.mock
def test_transport_error_raises_registry_unavailable():
    respx.get(URL).mock(side_effect=httpx.ConnectError("no route"))
    with pytest.raises(RegistryUnavailable):
        RegistryClient("https://cert.example", token="t").fetch()


@respx.mock
def test_malformed_registry_raises_registry_unavailable():
    respx.get(URL).mock(return_value=httpx.Response(200, json={"version": "v"}))
    with pytest.raises(RegistryUnavailable, match="malformed"):
        RegistryClient("https://cert.example", token="t").fetch()


@respx.mock
def test_a_failed_refetch_leaves_the_previous_cache_intact(registry_json):
    respx.get(URL).mock(
        side_effect=[
            httpx.Response(200, json=registry_json),
            httpx.Response(500, text="boom"),
        ]
    )
    client = RegistryClient("https://cert.example", token="t")
    client.fetch()
    with pytest.raises(RegistryUnavailable):
        client.fetch(force=True)
    assert client.cached.version == "sha256:fixture-v1"
