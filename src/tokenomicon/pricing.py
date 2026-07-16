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

"""Pricing plan definition and cost calculation.

PricingPlan holds the per-model rate (input/output cost per million
tokens, in a given ISO 4217 currency) and computes the cost of a call
given its token counts. All arithmetic uses Decimal to avoid float
rounding drift across repeated calculations.
"""

from dataclasses import dataclass
from decimal import Decimal

from .currencies import ISO_4217_CODES
from .exceptions import (
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

    def cost(self, input_tokens: int, output_tokens: int) -> Decimal:
        if input_tokens < 0 or output_tokens < 0:
            raise NegativeTokenCountError(
                f"Negative token count not allowed: "
                f"input={input_tokens}, output={output_tokens}"
            )

        input_cost = (Decimal(input_tokens) * self.input_per_million) / Decimal(
            1_000_000
        )
        output_cost = (Decimal(output_tokens) * self.output_per_million) / Decimal(
            1_000_000
        )

        return input_cost + output_cost
