import json
from pathlib import Path

import pytest

from cert_nlq.registry.models import Registry

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def registry_json() -> dict:
    return json.loads((FIXTURES / "registry.json").read_text(encoding="utf-8"))


@pytest.fixture
def registry(registry_json) -> Registry:
    return Registry.model_validate(registry_json)
