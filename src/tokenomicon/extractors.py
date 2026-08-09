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

"""Automatic token-usage extraction from known LLM provider response shapes.

Provides best-effort parsers for common providers (OpenAI, Anthropic, Google).
If no extractor recognizes a response, extract_usage() returns None and
the caller is expected to supply token counts manually.
"""

from collections.abc import Callable
from typing import Any, NamedTuple


class TokenUsage(NamedTuple):
    input_tokens: int
    output_tokens: int


def extract_usage(response: Any) -> TokenUsage | None:
    """Tries each known extractor. Returns None if none recognize the response."""

    for extractor in _EXTRACTORS:
        usage = extractor(response)
        if usage is not None:
            return usage

    return None


def _extract_openai(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    if input_tokens is None or output_tokens is None:
        return None

    return TokenUsage(input_tokens, output_tokens)


def _extract_anthropic(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is None or output_tokens is None:
        return None

    return TokenUsage(input_tokens, output_tokens)


def _extract_google(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None

    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    if input_tokens is None or output_tokens is None:
        return None

    return TokenUsage(input_tokens, output_tokens)


# Tried in order, most common provider first
_EXTRACTORS: tuple[Callable[[Any], TokenUsage | None], ...] = (
    _extract_openai,
    _extract_anthropic,
    _extract_google,
)
