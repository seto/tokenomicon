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

""" """


class TokenomiconError(Exception):
    """Base exception for all errors in the Tokenomicon library."""


class ConfigError(TokenomiconError):
    """Raised for configuration loading/parsing errors."""


class ModelNotConfiguredError(TokenomiconError, KeyError):
    """Raised when a model not registered in Config is requested."""


class InvalidCurrencyError(TokenomiconError, ValueError):
    """Raised when a currency code is not a valid ISO 4217 code."""


class InvalidPricingError(TokenomiconError, ValueError):
    """Raised when a pricing rate is negative or not numeric."""


class NegativeTokenCountError(TokenomiconError, ValueError):
    """Raised when a token count is negative or inconsistent."""
