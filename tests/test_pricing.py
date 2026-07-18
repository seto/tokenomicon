from decimal import Decimal

import pytest

from tokenomicon.exceptions import (
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
            input_per_million=Decimal("0"),
            output_per_million=Decimal("0"),
        )
        assert plan.tribute(1000, 1000) == Decimal("0")

    def test_plan_is_immutable(self) -> None:
        plan = PricingPlan(
            model="gpt-5-mini",
            input_per_million=Decimal("0.25"),
            output_per_million=Decimal("2.00"),
        )
        with pytest.raises(AttributeError):
            plan.currency = "EUR"  # type: ignore[misc]


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
        assert plan.tribute(0, 0) == Decimal("0")

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
