"""Real, minimal, "ping" style calls against actual provider SDKs.

Confirms tokenomicon's extractors recognize genuine response shapes,
not just hand-built SimpleNamespace fixtures. Uses the cheapest
available model per provider and a one-token prompt to keep any real
spend negligible.
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
            input_per_million=Decimal("0.75"),
            output_per_million=Decimal("4.50"),
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
        assert outcome.input_tokens is not None and outcome.input_tokens > 0
        assert outcome.output_tokens is not None and outcome.output_tokens > 0


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
        assert outcome.input_tokens is not None and outcome.input_tokens > 0
        assert outcome.output_tokens is not None and outcome.output_tokens > 0


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
        assert outcome.tribute > Decimal("0")


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
        assert outcome.tribute > Decimal("0")
