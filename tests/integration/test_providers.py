"""Real, minimal, "ping" style calls against actual provider SDKs.

These are not correctness tests: the unit test suite already covers
extraction and pricing logic at 100% coverage, deterministically, without
network calls or provider credentials. This file exists purely for anyone
who wants the extra peace of mind of seeing a genuine provider response
recognized end-to-end, not as a gate on correctness.

Accordingly, these tests are opt-in and never run in CI (see the dedicated
workflow_dispatch-only workflow). They require real, funded accounts for
whichever providers you want to check, and are expected to be adapted to
your own setup: comment out what you don't use, add credentials only for
the providers you actually have accounts with, skip the rest. There's no
obligation to keep every provider green here.

Uses the cheapest available model per provider and a one-token prompt to
keep any real spend negligible.

Prompt caching (both reads and Anthropic's cache-write tiers) is
exercised only at the unit-test level (deterministic SimpleNamespace
fixtures). Forcing a real cache hit or cache write here would require
padding requests to each provider's minimum cacheable size and is
best-effort on the provider side (not guaranteed per-request), which
would add flakiness without adding real coverage of tokenomicon's own
logic. The registered test plans below deliberately omit
cache_write_5m_per_million / cache_write_1h_per_million: since these
"ping" calls never trigger a real cache write, the write token counts
stay at 0 and tribute() never needs those rates.
"""

from decimal import Decimal

import pytest

from tokenomicon.api import augur
from tokenomicon.config import Config
from tokenomicon.pricing import PricingPlan


@pytest.fixture()
def cfg(monkeypatch) -> Config:
    test_config = Config()
    test_config.register(
        PricingPlan(
            model="openai-test-model",
            input_per_million=Decimal("0.20"),
            output_per_million=Decimal("1.20"),
            cached_input_per_million=Decimal("0.02"),
        )
    )
    test_config.register(
        PricingPlan(
            model="anthropic-test-model",
            input_per_million=Decimal("1.00"),
            output_per_million=Decimal("5.00"),
        )
    )
    test_config.register(
        PricingPlan(
            model="google-test-model",
            input_per_million=Decimal("1.50"),
            output_per_million=Decimal("9.00"),
        )
    )
    test_config.register(
        PricingPlan(
            model="deepseek-test-model",
            input_per_million=Decimal("0.14"),
            output_per_million=Decimal("0.28"),
            cached_input_per_million=Decimal("0.014"),
        )
    )
    test_config.register(
        PricingPlan(
            model="mistral-test-model",
            input_per_million=Decimal("0.04"),
            output_per_million=Decimal("0.04"),
        )
    )
    monkeypatch.setattr("tokenomicon.api.config", test_config)
    return test_config


class TestOpenAI:
    def test_openai_call_extracts_usage(
        self, openai_api_key, openai_model_id, cfg
    ) -> None:
        openai = pytest.importorskip("openai")
        client = openai.OpenAI(api_key=openai_api_key)

        @augur(model="openai-test-model")
        def call():
            return client.chat.completions.create(
                model=openai_model_id,
                messages=[{"role": "user", "content": "hi"}],
            )

        outcome = call()

        assert outcome.tribute is not None
        assert outcome.tribute > Decimal(0)
        assert outcome.input_tokens is not None and outcome.input_tokens > 0
        assert outcome.output_tokens is not None and outcome.output_tokens > 0
        assert outcome.cached_tokens is not None and outcome.cached_tokens >= 0
        assert (
            outcome.cache_write_5m_tokens is not None
            and outcome.cache_write_5m_tokens >= 0
        )
        assert (
            outcome.cache_write_1h_tokens is not None
            and outcome.cache_write_1h_tokens >= 0
        )


@pytest.mark.skip(reason="Placeholder test alternative for OpenAI calls")
class TestOpenAIViaOpenRouter:
    def test_openai_via_openrouter_call_extracts_usage(
        self, openrouter_base_url, openrouter_api_key, openrouter_model_id, cfg
    ) -> None:
        openai = pytest.importorskip("openai")
        client = openai.OpenAI(
            base_url=openrouter_base_url,
            api_key=openrouter_api_key,
        )

        @augur(model="openai-test-model")
        def call():
            return client.chat.completions.create(
                model=openrouter_model_id,
                messages=[{"role": "user", "content": "hi"}],
            )

        outcome = call()

        assert outcome.tribute is not None
        assert outcome.tribute > Decimal(0)
        assert outcome.input_tokens is not None and outcome.input_tokens > 0
        assert outcome.output_tokens is not None and outcome.output_tokens > 0
        assert outcome.cached_tokens is not None and outcome.cached_tokens >= 0
        assert (
            outcome.cache_write_5m_tokens is not None
            and outcome.cache_write_5m_tokens >= 0
        )
        assert (
            outcome.cache_write_1h_tokens is not None
            and outcome.cache_write_1h_tokens >= 0
        )


class TestAnthropic:
    def test_anthropic_call_extracts_usage(
        self, anthropic_api_key, anthropic_model_id, cfg
    ) -> None:
        anthropic = pytest.importorskip("anthropic")
        client = anthropic.Anthropic(api_key=anthropic_api_key)

        @augur(model="anthropic-test-model")
        def call():
            return client.messages.create(
                model=anthropic_model_id,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )

        outcome = call()

        assert outcome.tribute is not None
        assert outcome.tribute > Decimal(0)
        assert outcome.input_tokens is not None and outcome.input_tokens > 0
        assert outcome.output_tokens is not None and outcome.output_tokens > 0
        assert outcome.cached_tokens is not None and outcome.cached_tokens >= 0
        assert (
            outcome.cache_write_5m_tokens is not None
            and outcome.cache_write_5m_tokens >= 0
        )
        assert (
            outcome.cache_write_1h_tokens is not None
            and outcome.cache_write_1h_tokens >= 0
        )


class TestGoogle:
    def test_google_call_extracts_usage(
        self, google_api_key, google_model_id, cfg
    ) -> None:
        genai = pytest.importorskip("google.genai")
        client = genai.Client(api_key=google_api_key)

        @augur(model="google-test-model")
        def call():
            return client.models.generate_content(
                model=google_model_id,
                contents="hi",
            )

        outcome = call()

        assert outcome.tribute is not None
        assert outcome.tribute > Decimal(0)
        assert outcome.input_tokens is not None and outcome.input_tokens > 0
        assert outcome.output_tokens is not None and outcome.output_tokens > 0
        assert outcome.cached_tokens is not None and outcome.cached_tokens >= 0
        assert (
            outcome.cache_write_5m_tokens is not None
            and outcome.cache_write_5m_tokens >= 0
        )
        assert (
            outcome.cache_write_1h_tokens is not None
            and outcome.cache_write_1h_tokens >= 0
        )


@pytest.mark.skip(reason="Requires a funded DeepSeek account; adjust as needed")
class TestDeepSeek:
    def test_deepseek_call_extracts_usage(
        self, deepseek_api_key, deepseek_model_id, cfg
    ) -> None:
        openai = pytest.importorskip("openai")
        client = openai.OpenAI(
            api_key=deepseek_api_key, base_url="https://api.deepseek.com"
        )

        @augur(model="deepseek-test-model")
        def call():
            return client.chat.completions.create(
                model=deepseek_model_id,
                messages=[{"role": "user", "content": "hi"}],
            )

        outcome = call()

        assert outcome.tribute is not None
        assert outcome.tribute > Decimal(0)
        assert outcome.input_tokens is not None and outcome.input_tokens > 0
        assert outcome.output_tokens is not None and outcome.output_tokens > 0
        assert outcome.cached_tokens is not None and outcome.cached_tokens >= 0
        assert (
            outcome.cache_write_5m_tokens is not None
            and outcome.cache_write_5m_tokens >= 0
        )
        assert (
            outcome.cache_write_1h_tokens is not None
            and outcome.cache_write_1h_tokens >= 0
        )


class TestMistral:
    def test_mistral_call_extracts_usage(
        self, mistral_api_key, mistral_model_id, cfg
    ) -> None:
        pytest.importorskip("mistralai")
        from mistralai.client import Mistral

        client = Mistral(api_key=mistral_api_key)

        @augur(model="mistral-test-model")
        def call():
            return client.chat.complete(
                model=mistral_model_id,
                messages=[{"role": "user", "content": "hi"}],
            )

        outcome = call()

        assert outcome.tribute is not None
        assert outcome.tribute > Decimal(0)
        assert outcome.input_tokens is not None and outcome.input_tokens > 0
        assert outcome.output_tokens is not None and outcome.output_tokens > 0
        assert outcome.cached_tokens is not None and outcome.cached_tokens >= 0
        assert (
            outcome.cache_write_5m_tokens is not None
            and outcome.cache_write_5m_tokens >= 0
        )
        assert (
            outcome.cache_write_1h_tokens is not None
            and outcome.cache_write_1h_tokens >= 0
        )
