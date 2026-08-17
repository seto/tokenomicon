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

"""Tokenomicon: post-hoc financial divination for LLM API calls.

Public API surface. Configure pricing via `config.load_toml(...)`,
then decorate any LLM call with `augur` to get its tribute back
alongside the original response.
"""

from .api import CallResult, augur
from .config import Config, config
from .exceptions import (
    CachePricingNotConfiguredError,
    ConfigError,
    InvalidCurrencyError,
    InvalidPricingError,
    ModelNotConfiguredError,
    NegativeTokenCountError,
    TokenExtractionWarning,
    TokenomiconError,
)
from .pricing import PricingPlan

__all__ = [
    "CachePricingNotConfiguredError",
    "CallResult",
    "Config",
    "ConfigError",
    "InvalidCurrencyError",
    "InvalidPricingError",
    "ModelNotConfiguredError",
    "NegativeTokenCountError",
    "PricingPlan",
    "TokenExtractionWarning",
    "TokenomiconError",
    "augur",
    "config",
]
