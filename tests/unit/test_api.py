from decimal import Decimal
from types import SimpleNamespace

import pytest

from tokenomicon.api import CallResult, augur
from tokenomicon.config import Config
from tokenomicon.exceptions import ModelNotConfiguredError, TokenExtractionWarning
from tokenomicon.pricing import PricingPlan


@pytest.fixture()
def cfg(monkeypatch) -> Config:
    """Config pre-loaded with a known model.

    This is patched in as the module-level singleton so augur decorator
    (which reads from tokenomicon.api.config) sees it.
    """

    test_config = Config()
    test_config.register(
        PricingPlan(
            model="claude-sonnet-5",
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
        )
    )
    monkeypatch.setattr("tokenomicon.api.config", test_config)
    return test_config


class TestAugurAutomaticExtraction:
    def test_returns_call_result_with_tribute(self, cfg: Config) -> None:
        @augur(model="claude-sonnet-5")
        def call_llm() -> SimpleNamespace:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=1_000_000)
            )

        outcome = call_llm()

        assert isinstance(outcome, CallResult)
        assert outcome.tribute == Decimal("18.00")
        assert outcome.input_tokens == 1_000_000
        assert outcome.output_tokens == 1_000_000
        assert outcome.currency == "USD"

    def test_original_response_is_preserved_untouched(self, cfg: Config) -> None:
        original = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
            choices=["hello"],
        )

        @augur(model="claude-sonnet-5")
        def call_llm() -> SimpleNamespace:
            return original

        outcome = call_llm()

        assert outcome.result is original
        assert outcome.result.choices == ["hello"]

    def test_passes_through_args_and_kwargs(self, cfg: Config) -> None:
        @augur(model="claude-sonnet-5")
        def call_llm(prompt: str, *, temperature: float = 0.0) -> SimpleNamespace:
            assert prompt == "hello"
            assert temperature == 0.7
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=10, output_tokens=5)
            )

        outcome = call_llm("hello", temperature=0.7)
        assert outcome.tribute is not None


class TestAugurManualTokensFallback:
    def test_uses_manual_tokens_when_extraction_fails(self, cfg: Config) -> None:
        @augur(
            model="claude-sonnet-5",
            manual_tokens=lambda r: (r["prompt_len"], r["gen_len"]),
        )
        def call_local() -> dict:
            return {"prompt_len": 200, "gen_len": 100}

        outcome = call_local()

        assert outcome.tribute == Decimal("3.00") * Decimal(200) / Decimal(
            1_000_000
        ) + Decimal("15.00") * Decimal(100) / Decimal(1_000_000)
        assert outcome.input_tokens == 200
        assert outcome.output_tokens == 100

    def test_manual_tokens_not_called_when_automatic_extraction_succeeds(
        self, cfg: Config
    ) -> None:
        # If the response is already OpenAI/Anthropic-shaped, manual_tokens
        # must not override a successful automatic extraction.
        manual_tokens_called = False

        def manual_tokens(_response: object) -> tuple[int, int]:
            nonlocal manual_tokens_called
            manual_tokens_called = True
            return (999, 999)

        @augur(model="claude-sonnet-5", manual_tokens=manual_tokens)
        def call_llm() -> SimpleNamespace:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=100, output_tokens=50)
            )

        outcome = call_llm()

        assert manual_tokens_called is False
        assert outcome.input_tokens == 100


class TestAugurUnrecognizedResponse:
    def test_warns_and_returns_none_tribute_when_no_extractor_and_no_manual_tokens(
        self, cfg: Config
    ) -> None:
        @augur(model="claude-sonnet-5")
        def call_unknown() -> dict:
            return {"unexpected": "shape"}

        with pytest.warns(TokenExtractionWarning, match="claude-sonnet-5"):
            outcome = call_unknown()

        assert outcome.tribute is None
        assert outcome.input_tokens is None
        assert outcome.output_tokens is None
        assert outcome.currency is None
        assert outcome.result == {"unexpected": "shape"}


class TestAugurModelResolution:
    def test_raises_when_model_not_registered(self, cfg: Config) -> None:
        @augur(model="nonexistent-model")
        def call_llm() -> SimpleNamespace:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=100, output_tokens=50)
            )

        with pytest.raises(ModelNotConfiguredError):
            call_llm()

    def test_llm_call_still_executes_even_if_model_unregistered(
        self, cfg: Config
    ) -> None:
        # Documents current behavior: the wrapped function runs (and is
        # "paid for") before the model lookup happens, so a misconfigured
        # model does not prevent the underlying call from firing.
        call_count = 0

        @augur(model="nonexistent-model")
        def call_llm() -> SimpleNamespace:
            nonlocal call_count
            call_count += 1
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=100, output_tokens=50)
            )

        with pytest.raises(ModelNotConfiguredError):
            call_llm()

        assert call_count == 1


class TestAugurCachedTokens:
    def test_cached_tokens_included_in_call_result(self, cfg: Config) -> None:
        @augur(model="claude-sonnet-5")
        def call_llm() -> SimpleNamespace:
            return SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=100, output_tokens=50, cache_read_input_tokens=200
                )
            )

        outcome = call_llm()

        assert outcome.cached_tokens == 200
        assert outcome.input_tokens == 100
        assert outcome.output_tokens == 50

    def test_cached_tokens_billed_at_discounted_rate(self, monkeypatch) -> None:
        cfg = Config()
        cfg.register(
            PricingPlan(
                model="claude-sonnet-5",
                input_per_million=Decimal("3.00"),
                output_per_million=Decimal("15.00"),
                cached_input_per_million=Decimal("0.30"),
            )
        )
        monkeypatch.setattr("tokenomicon.api.config", cfg)

        @augur(model="claude-sonnet-5")
        def call_llm() -> SimpleNamespace:
            return SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=1_000_000,
                    output_tokens=1_000_000,
                    cache_read_input_tokens=1_000_000,
                )
            )

        outcome = call_llm()

        assert outcome.tribute == Decimal("3.00") + Decimal("0.30") + Decimal("15.00")

    def test_cached_tokens_default_to_zero_when_not_reported(self, cfg: Config) -> None:
        @augur(model="claude-sonnet-5")
        def call_llm() -> SimpleNamespace:
            return SimpleNamespace(
                usage=SimpleNamespace(input_tokens=100, output_tokens=50)
            )

        outcome = call_llm()

        assert outcome.cached_tokens == 0

    def test_cached_tokens_zero_when_manual_tokens_used(self, cfg: Config) -> None:
        # manual_tokens fallback does not support cache accounting yet.
        @augur(
            model="claude-sonnet-5",
            manual_tokens=lambda r: (r["prompt_len"], r["gen_len"]),
        )
        def call_local() -> dict:
            return {"prompt_len": 200, "gen_len": 100}

        outcome = call_local()

        assert outcome.cached_tokens == 0

    def test_cached_tokens_none_when_extraction_fails(self, cfg: Config) -> None:
        @augur(model="claude-sonnet-5")
        def call_unknown() -> dict:
            return {"unexpected": "shape"}

        with pytest.warns(TokenExtractionWarning):
            outcome = call_unknown()

        assert outcome.cached_tokens is None
