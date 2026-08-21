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

"""Decorator that turns any LLM call into a cost-aware one.

Wraps a function returning a provider response and returns a CallResult
with the original response plus the calculated tribute, leaving the
caller's access to that response untouched.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from functools import wraps
from typing import Any, TypeVar

from .config import config
from .exceptions import TokenExtractionWarning
from .extractors import extract_usage

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True, slots=True)
class CallResult:
    """Result of a decorated call: original response + calculated tribute."""

    result: Any
    tribute: Decimal | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    cache_write_5m_tokens: int | None
    cache_write_1h_tokens: int | None
    currency: str | None


def augur(
    model: str,
    *,
    manual_tokens: Callable[[Any], tuple[int, int]] | None = None,
) -> Callable[[F], Callable[..., CallResult]]:
    """Decorator that calculates the tribute of an LLM call.

    Args:
        model: name of the model, must match one registered in Config.
        manual_tokens: optional function (response) -> (input_tokens, output_tokens),
            used as a fallback only if automatic extraction does not recognize
            the response format.
    """

    def decorator(func: F) -> Callable[..., CallResult]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> CallResult:
            result = func(*args, **kwargs)

            usage = extract_usage(result)
            cached_tokens = 0
            cache_write_5m_tokens = 0
            cache_write_1h_tokens = 0
            if usage is None and manual_tokens is not None:
                input_tokens, output_tokens = manual_tokens(result)
                usage = (input_tokens, output_tokens)
            elif usage is not None:
                cached_tokens = usage.cached_tokens
                cache_write_5m_tokens = usage.cache_write_5m_tokens
                cache_write_1h_tokens = usage.cache_write_1h_tokens

            if usage is None:
                warnings.warn(
                    f"Not possible to determine token usage for model "
                    f"'{model}': response format not recognized and no "
                    f"'manual_tokens' provided. Tribute not calculated.",
                    category=TokenExtractionWarning,
                    stacklevel=2,
                )
                return CallResult(
                    result=result,
                    tribute=None,
                    input_tokens=None,
                    output_tokens=None,
                    cached_tokens=None,
                    cache_write_5m_tokens=None,
                    cache_write_1h_tokens=None,
                    currency=None,
                )

            input_tokens, output_tokens = usage[0], usage[1]
            plan = config.get(model)
            tribute = plan.tribute(
                input_tokens,
                output_tokens,
                cached_tokens,
                cache_write_5m_tokens,
                cache_write_1h_tokens,
            )

            return CallResult(
                result=result,
                tribute=tribute,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                cache_write_5m_tokens=cache_write_5m_tokens,
                cache_write_1h_tokens=cache_write_1h_tokens,
                currency=plan.currency,
            )

        return wrapper

    return decorator
