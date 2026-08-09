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

"""Model pricing registry, loadable from external TOML files.

Config is the single source of truth mapping a model name to its
PricingPlan. Plans are never bundled or assumed by the library itself:
callers register them explicitly, either in code via register() or by
loading a TOML file via load_toml(), so pricing can be updated without
redeploying the application. A module-level `config` singleton is
provided for the common case of one registry per process.
"""

import os
import re
import tomllib
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .exceptions import ConfigError, ModelNotConfiguredError
from .pricing import PricingPlan

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


class Config:
    """Configuration registry for models and their pricing plans."""

    def __init__(self) -> None:
        self._plans: dict[str, PricingPlan] = {}

    def register(self, plan: PricingPlan) -> None:
        self._plans[plan.model] = plan

    def load_toml(self, path: str | Path, *, expand_env: bool = False) -> None:
        """Load (or reload) pricing plans from an external TOML file.

        If expand_env=True, it replaces patterns like ${VAR_NAME} in the file
        with the corresponding environment variable value before parsing.
        It is disabled by default: no implicit expansion.
        """

        raw = Path(path).read_text()
        if expand_env:
            raw = _expand_env(raw)

        data = tomllib.loads(raw)
        for model, rates in data.items():
            try:
                input_per_million = Decimal(str(rates["input_per_million"]))
                output_per_million = Decimal(str(rates["output_per_million"]))
                cached_input_per_million = (
                    Decimal(str(rates["cached_input_per_million"]))
                    if "cached_input_per_million" in rates
                    else None
                )

            except InvalidOperation:
                raise ConfigError(
                    f"Invalid numeric value for model '{model}': "
                    f"input={rates.get('input_per_million')!r}, "
                    f"output={rates.get('output_per_million')!r}, "
                    f"cached_input={rates.get('cached_input_per_million')!r}"
                ) from None

            except KeyError as exc:
                raise ConfigError(
                    f"Model '{model}' is missing required field {exc}"
                ) from None

            self.register(
                PricingPlan(
                    model=model,
                    input_per_million=input_per_million,
                    output_per_million=output_per_million,
                    currency=rates.get("currency", "USD"),
                    cached_input_per_million=cached_input_per_million,
                )
            )

    def get(self, model: str) -> PricingPlan:
        try:
            return self._plans[model]

        except KeyError:
            raise ModelNotConfiguredError(
                f"Model '{model}' not configured. "
                f"Available models: {list(self._plans)}"
            ) from None


def _expand_env(raw: str) -> str:
    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        try:
            return os.environ[var_name]

        except KeyError:
            raise ConfigError(
                f"Environment variable '{var_name}' referenced in TOML but not set"
            ) from None

    return _ENV_PATTERN.sub(replace, raw)


config = Config()
