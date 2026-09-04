"""Fetch the registry from the host application and cache it by version."""
import httpx
from pydantic import ValidationError

from .models import Registry

REGISTRY_PATH = "/query/registry/"


class RegistryUnavailable(RuntimeError):
    """The registry could not be fetched or could not be parsed."""


class RegistryClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        client: httpx.Client | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._url = base_url.rstrip("/") + REGISTRY_PATH
        self._token = token
        self._client = client
        self._timeout = timeout
        self.cached: Registry | None = None

    def fetch(self, force: bool = False) -> Registry:
        """Return the cached registry, fetching it if absent or forced.

        A failed forced refetch raises and leaves the previous cache in place,
        so registry skew degrades to a stale vocabulary rather than an outage.
        The host re-validates every payload regardless.
        """
        if self.cached is not None and not force:
            return self.cached
        registry = self._get()
        self.cached = registry
        return registry

    def _get(self) -> Registry:
        headers = {"Authorization": f"Bearer {self._token}"}
        try:
            if self._client is not None:
                response = self._client.get(self._url, headers=headers)
            else:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.get(self._url, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RegistryUnavailable(
                f"registry returned {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RegistryUnavailable(f"registry unreachable: {exc}") from exc

        # `response.json()` raises json.JSONDecodeError — a ValueError, NOT an
        # httpx.HTTPError — so it needs its own guard. Without it a non-JSON
        # 200 body escapes as an unhandled decode error and the API layer
        # returns 500 instead of the 503 it has a handler for.
        try:
            body = response.json()
        except ValueError as exc:
            raise RegistryUnavailable(
                f"registry response was not JSON: {exc}"
            ) from exc

        try:
            return Registry.model_validate(body)
        except ValidationError as exc:
            raise RegistryUnavailable(f"registry document is malformed: {exc}") from exc
