"""Fixtures shared by real-provider integration tests.

These tests are opt-in: they require actual API credentials
and make real (paid) network calls. They never run in the
default CI job; only in a dedicated pre-release workflow.
"""

import os

import pytest

# python-dotenv is a dev-only dependency
#   see `tests/integration/requirements.txt`
# never required at runtime by the published package
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture()
def openai_api_key() -> str:
    return _require_env("OPENAI_API_KEY")


@pytest.fixture()
def openai_model_id() -> str:
    return os.environ.get("OPENAI_TEST_MODEL", "gpt-5.6-luna")


@pytest.fixture()
def openrouter_base_url() -> str:
    return os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


@pytest.fixture()
def openrouter_api_key() -> str:
    return _require_env("OPENROUTER_API_KEY")


@pytest.fixture()
def openrouter_model_id() -> str:
    return os.environ.get("OPENROUTER_TEST_MODEL", "openai/gpt-oss-20b:free")


@pytest.fixture()
def anthropic_api_key() -> str:
    return _require_env("ANTHROPIC_API_KEY")


@pytest.fixture()
def anthropic_model_id() -> str:
    return os.environ.get("ANTHROPIC_TEST_MODEL", "claude-haiku-4-5-20251001")


@pytest.fixture()
def google_api_key() -> str:
    return _require_env("GOOGLE_API_KEY")


@pytest.fixture()
def google_model_id() -> str:
    return os.environ.get("GOOGLE_TEST_MODEL", "gemini-3.5-flash")


@pytest.fixture()
def deepseek_api_key() -> str:
    return _require_env("DEEPSEEK_API_KEY")


@pytest.fixture()
def deepseek_model_id() -> str:
    return os.environ.get("DEEPSEEK_TEST_MODEL", "deepseek-v4-flash")


@pytest.fixture()
def mistral_api_key() -> str:
    return _require_env("MISTRAL_API_KEY")


@pytest.fixture()
def mistral_model_id() -> str:
    return os.environ.get("MISTRAL_TEST_MODEL", "ministral-3b-latest")


def _require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value or not value.strip():
        pytest.skip(f"{var_name} not set, skipping real-provider test")
    return value.strip()
