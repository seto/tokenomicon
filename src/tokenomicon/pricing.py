# Copyright 2026 Roberto Matarazzo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pricing plan definition and tribute calculation.

PricingPlan holds the per-model rate (input/output cost per million
tokens, in a given ISO 4217 currency) and computes the tribute of a call
given its token counts.

An optional cached_input_per_million rate covers
prompt-cache reads; when unset, cached tokens are billed at the regular
input rate rather than being discounted or ignored.

All arithmetic uses
Decimal to avoid float rounding drift across repeated calculations.
"""

from dataclasses import dataclass
from decimal import Decimal

from .currencies import ISO_4217_CODES
from .exceptions import (
    CachePricingNotConfiguredError,
    InvalidCurrencyError,
    InvalidPricingError,
    NegativeTokenCountError,
)


@dataclass(frozen=True, slots=True)
class PricingPlan:
    """Pricing plan for a model, in `currency` per 1 million tokens."""

    model: str
    input_per_million: Decimal
    output_per_million: Decimal
    currency: str = "USD"
    cached_input_per_million: Decimal | None = None
    cache_write_5m_per_million: Decimal | None = None
    cache_write_1h_per_million: Decimal | None = None

    def __post_init__(self) -> None:
        if self.currency not in ISO_4217_CODES:
            raise InvalidCurrencyError(
                f"'{self.currency}' is not a valid ISO 4217 currency code"
            )
        if self.input_per_million < 0 or self.output_per_million < 0:
            raise InvalidPricingError(
                f"Rates cannot be negative: "
                f"input={self.input_per_million}, output={self.output_per_million}"
            )
        if self.cached_input_per_million is not None and self.cached_input_per_million < 0:
            raise InvalidPricingError(
                f"Rates cannot be negative: "
                f"cached_input={self.cached_input_per_million}"
            )
        if self.cache_write_5m_per_million is not None and self.cache_write_5m_per_million < 0:
            raise InvalidPricingError(
                f"Rates cannot be negative: "
                f"cache_write_5m={self.cache_write_5m_per_million}"
            )
        if self.cache_write_1h_per_million is not None and self.cache_write_1h_per_million < 0:
            raise InvalidPricingError(
                f"Rates cannot be negative: "
                f"cache_write_1h={self.cache_write_1h_per_million}"
            )

    def tribute(self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        cache_write_5m_tokens: int = 0,
        cache_write_1h_tokens: int = 0,
    ) -> Decimal:
        """Calculate the tribute for a call.

        cached_tokens (cache reads) are billed at cached_input_per_million if
        configured, otherwise fall back to input_per_million. They are billed
        separately from input_tokens, not as a subset of it.

        cache_write_5m_tokens and cache_write_1h_tokens (cache creation) are
        billed at their respective configured rates. Unlike cached_tokens,
        there is no fallback: if either count is nonzero and its rate isn't
        configured, tribute() raises CachePricingNotConfiguredError rather
        than silently underestimating the cost of a write premium it can't
        quantify.
        """

        if (
            input_tokens < 0
            or output_tokens < 0
            or cached_tokens < 0
            or cache_write_5m_tokens < 0
            or cache_write_1h_tokens < 0
        ):
            raise NegativeTokenCountError(
                f"Negative token count not allowed: "
                f"input={input_tokens}, output={output_tokens}, "
                f"cached={cached_tokens}, cache_write_5m={cache_write_5m_tokens}, "
                f"cache_write_1h={cache_write_1h_tokens}"
            )

        if cache_write_5m_tokens > 0 and self.cache_write_5m_per_million is None:
            raise CachePricingNotConfiguredError(
                f"Model '{self.model}' has cache_write_5m_tokens={cache_write_5m_tokens} "
                f"but no cache_write_5m_per_million rate is configured."
            )
        if cache_write_1h_tokens > 0 and self.cache_write_1h_per_million is None:
            raise CachePricingNotConfiguredError(
                f"Model '{self.model}' has cache_write_1h_tokens={cache_write_1h_tokens} "
                f"but no cache_write_1h_per_million rate is configured."
            )

        cached_rate = (
            self.cached_input_per_million
            if self.cached_input_per_million is not None
            else self.input_per_million
        )

        input_cost = (Decimal(input_tokens) * self.input_per_million) / Decimal(
            1_000_000
        )
        cached_cost = (Decimal(cached_tokens) * cached_rate) / Decimal(1_000_000)
        output_cost = (Decimal(output_tokens) * self.output_per_million) / Decimal(
            1_000_000
        )

        write_5m_cost = Decimal(0)
        if cache_write_5m_tokens > 0:
            write_5m_cost = (
                Decimal(cache_write_5m_tokens) * self.cache_write_5m_per_million
            ) / Decimal(1_000_000)

        write_1h_cost = Decimal(0)
        if cache_write_1h_tokens > 0:
            write_1h_cost = (
                Decimal(cache_write_1h_tokens) * self.cache_write_1h_per_million
            ) / Decimal(1_000_000)

        return input_cost + cached_cost + output_cost + write_5m_cost + write_1h_cost
