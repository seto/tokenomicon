from decimal import Decimal

import pytest

from tokenomicon.exceptions import (
    CachePricingNotConfiguredError,
    InvalidCurrencyError,
    InvalidPricingError,
    NegativeTokenCountError,
)
from tokenomicon.pricing import PricingPlan


class TestPricingPlanValidation:
    def test_valid_plan_with_default_currency(self) -> None:
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
        )
        assert plan.currency == "USD"

    def test_valid_plan_with_explicit_currency(self) -> None:
        plan = PricingPlan(
            model="mistral-large",
            input_per_million=Decimal("2.00"),
            output_per_million=Decimal("6.00"),
            currency="EUR",
        )
        assert plan.currency == "EUR"

    def test_rejects_invalid_currency_code(self) -> None:
        with pytest.raises(InvalidCurrencyError):
            PricingPlan(
                model="gpt-5-mini",
                input_per_million=Decimal("0.25"),
                output_per_million=Decimal("2.00"),
                currency="FOO_NOT_REAL",
            )

    def test_rejects_lowercase_currency_code(self) -> None:
        # ISO 4217 codes are stored uppercase; lowercase must not silently pass.
        with pytest.raises(InvalidCurrencyError):
            PricingPlan(
                model="gpt-5-mini",
                input_per_million=Decimal("0.25"),
                output_per_million=Decimal("2.00"),
                currency="usd",
            )

    def test_rejects_negative_input_rate(self) -> None:
        with pytest.raises(InvalidPricingError):
            PricingPlan(
                model="gpt-5-mini",
                input_per_million=Decimal("-0.25"),
                output_per_million=Decimal("2.00"),
            )

    def test_rejects_negative_output_rate(self) -> None:
        with pytest.raises(InvalidPricingError):
            PricingPlan(
                model="gpt-5-mini",
                input_per_million=Decimal("0.25"),
                output_per_million=Decimal("-2.00"),
            )

    def test_allows_zero_rate(self) -> None:
        # A free tier (e.g. local models) is a legitimate zero-tribute plan.
        plan = PricingPlan(
            model="local-llama",
            input_per_million=Decimal(0),
            output_per_million=Decimal(0),
        )
        assert plan.tribute(1000, 1000) == Decimal(0)

    def test_plan_is_immutable(self) -> None:
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
        )
        with pytest.raises(AttributeError):
            plan.currency = "EUR"  # type: ignore[misc]

    def test_valid_plan_with_cached_rate(self) -> None:
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
            cached_input_per_million=Decimal("0.10"),
        )
        assert plan.cached_input_per_million == Decimal("0.10")

    def test_cached_rate_defaults_to_none(self) -> None:
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
        )
        assert plan.cached_input_per_million is None

    def test_rejects_negative_cached_rate(self) -> None:
        with pytest.raises(InvalidPricingError):
            PricingPlan(
                model="gpt-5-mini",
                input_per_million=Decimal("0.25"),
                output_per_million=Decimal("2.00"),
                cached_input_per_million=Decimal("-0.10"),
            )

    def test_allows_zero_cached_rate(self) -> None:
        # A fully-free cache read is legitimate (e.g. some providers today).
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
            cached_input_per_million=Decimal(0),
        )
        assert plan.tribute(0, 0, cached_tokens=1000) == Decimal(0)

    def test_valid_plan_with_cache_write_rates(self) -> None:
        plan = PricingPlan(
            model="claude-sonnet-5",
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
            cache_write_5m_per_million=Decimal("3.75"),
            cache_write_1h_per_million=Decimal("6.00"),
        )
        assert plan.cache_write_5m_per_million == Decimal("3.75")
        assert plan.cache_write_1h_per_million == Decimal("6.00")

    def test_cache_write_rates_default_to_none(self) -> None:
        plan = PricingPlan(
            model="claude-sonnet-5",
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
        )
        assert plan.cache_write_5m_per_million is None
        assert plan.cache_write_1h_per_million is None

    def test_rejects_negative_cache_write_5m_rate(self) -> None:
        with pytest.raises(InvalidPricingError):
            PricingPlan(
                model="claude-sonnet-5",
                input_per_million=Decimal("3.00"),
                output_per_million=Decimal("15.00"),
                cache_write_5m_per_million=Decimal("-3.75"),
            )

    def test_rejects_negative_cache_write_1h_rate(self) -> None:
        with pytest.raises(InvalidPricingError):
            PricingPlan(
                model="claude-sonnet-5",
                input_per_million=Decimal("3.00"),
                output_per_million=Decimal("15.00"),
                cache_write_1h_per_million=Decimal("-6.00"),
            )


class TestPricingPlanCost:
    def test_tribute_basic_calculation(self) -> None:
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
        )
        # 1,000,000 input tokens -> exactly input_per_million
        # 1,000,000 output tokens -> exactly output_per_million
        tribute = plan.tribute(1_000_000, 1_000_000)
        assert tribute == Decimal("2.25")

    def test_tribute_scales_proportionally(self) -> None:
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
        )
        tribute = plan.tribute(500_000, 0)
        assert tribute == Decimal("0.125")

    def test_tribute_with_zero_tokens(self) -> None:
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
        )
        assert plan.tribute(0, 0) == Decimal(0)

    def test_tribute_uses_decimal_precision(self) -> None:
        # Regression guard: this must never drift to float rounding errors
        # (e.g. 0.1 + 0.2 != 0.3 in binary float).
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
        )
        tribute = plan.tribute(1, 1)
        assert isinstance(tribute, Decimal)
        assert tribute == (Decimal("0.25") + Decimal("2.00")) / Decimal(1_000_000)

    def test_tribute_rejects_negative_input_tokens(self) -> None:
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
        )
        with pytest.raises(NegativeTokenCountError):
            plan.tribute(-1, 100)

    def test_tribute_rejects_negative_output_tokens(self) -> None:
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
        )
        with pytest.raises(NegativeTokenCountError):
            plan.tribute(100, -1)

    def test_tribute_with_cached_tokens_at_discounted_rate(self) -> None:
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
            cached_input_per_million=Decimal("0.10"),
        )
        # 1,000,000 regular input + 1,000,000 cached input + 1,000,000 output
        tribute = plan.tribute(1_000_000, 1_000_000, cached_tokens=1_000_000)
        assert tribute == Decimal("0.25") + Decimal("0.10") + Decimal("2.00")

    def test_tribute_cached_tokens_fallback_to_input_rate_when_not_configured(
        self,
    ) -> None:
        # No cached_input_per_million set: cached tokens must be billed at
        # the regular input rate, not silently discounted or dropped.
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
        )
        with_cached = plan.tribute(0, 0, cached_tokens=1_000_000)
        without_cached = plan.tribute(1_000_000, 0)
        assert with_cached == without_cached == Decimal("0.25")

    def test_tribute_cached_tokens_default_to_zero(self) -> None:
        # Backward compatibility: callers not using caching must see
        # identical behavior to before this feature existed.
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
            cached_input_per_million=Decimal("0.10"),
        )
        assert plan.tribute(1_000_000, 1_000_000) == Decimal("2.25")

    def test_tribute_rejects_negative_cached_tokens(self) -> None:
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
        )
        with pytest.raises(NegativeTokenCountError):
            plan.tribute(100, 100, cached_tokens=-1)

    def test_tribute_with_cache_write_5m_tokens(self) -> None:
        plan = PricingPlan(
            model="claude-sonnet-5",
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
            cache_write_5m_per_million=Decimal("3.75"),
        )
        tribute = plan.tribute(1_000_000, 1_000_000, cache_write_5m_tokens=1_000_000)
        assert tribute == Decimal("3.00") + Decimal("15.00") + Decimal("3.75")

    def test_tribute_with_cache_write_1h_tokens(self) -> None:
        plan = PricingPlan(
            model="claude-sonnet-5",
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
            cache_write_1h_per_million=Decimal("6.00"),
        )
        tribute = plan.tribute(1_000_000, 1_000_000, cache_write_1h_tokens=1_000_000)
        assert tribute == Decimal("3.00") + Decimal("15.00") + Decimal("6.00")

    def test_tribute_with_both_cache_write_tiers(self) -> None:
        # Anthropic reports both as distinct counts on the same call; a
        # single request could in principle hit both tiers.
        plan = PricingPlan(
            model="claude-sonnet-5",
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
            cache_write_5m_per_million=Decimal("3.75"),
            cache_write_1h_per_million=Decimal("6.00"),
        )
        tribute = plan.tribute(
            0, 0, cache_write_5m_tokens=1_000_000, cache_write_1h_tokens=1_000_000
        )
        assert tribute == Decimal("3.75") + Decimal("6.00")

    def test_tribute_raises_when_cache_write_5m_tokens_present_but_unconfigured(
        self,
    ) -> None:
        plan = PricingPlan(
            model="claude-sonnet-5",
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
        )
        with pytest.raises(CachePricingNotConfiguredError, match="claude-sonnet-5"):
            plan.tribute(100, 100, cache_write_5m_tokens=1000)

    def test_tribute_raises_when_cache_write_1h_tokens_present_but_unconfigured(
        self,
    ) -> None:
        plan = PricingPlan(
            model="claude-sonnet-5",
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
        )
        with pytest.raises(CachePricingNotConfiguredError, match="claude-sonnet-5"):
            plan.tribute(100, 100, cache_write_1h_tokens=1000)

    def test_tribute_no_error_when_cache_write_tokens_are_zero_and_unconfigured(
        self,
    ) -> None:
        # Zero write tokens is the default: must never raise, regardless of
        # whether the rates are configured.
        plan = PricingPlan(
            model="claude-sonnet-5",
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
        )
        assert plan.tribute(100, 100) == plan.tribute(
            100, 100, cache_write_5m_tokens=0, cache_write_1h_tokens=0
        )

    def test_tribute_rejects_negative_cache_write_5m_tokens(self) -> None:
        plan = PricingPlan(
            model="claude-sonnet-5",
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
        )
        with pytest.raises(NegativeTokenCountError):
            plan.tribute(100, 100, cache_write_5m_tokens=-1)

    def test_tribute_rejects_negative_cache_write_1h_tokens(self) -> None:
        plan = PricingPlan(
            model="claude-sonnet-5",
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
        )
        with pytest.raises(NegativeTokenCountError):
            plan.tribute(100, 100, cache_write_1h_tokens=-1)
