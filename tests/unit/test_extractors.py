from types import SimpleNamespace

from tokenomicon.extractors import TokenUsage, extract_usage


class TestOpenAIExtraction:
    def test_extracts_usage_from_openai_shaped_response(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=150, completion_tokens=40)
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(input_tokens=150, output_tokens=40)

    def test_returns_none_if_openai_usage_missing_prompt_tokens(self) -> None:
        # Malformed/partial usage object: must not crash, must fall through.
        response = SimpleNamespace(usage=SimpleNamespace(completion_tokens=40))

        usage = extract_usage(response)

        assert usage is None

    def test_extracts_cached_tokens_and_subtracts_from_input(self) -> None:
        # OpenAI reports cached_tokens as a subset of prompt_tokens: the
        # extractor must normalize it to a separate count, not double-count.
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=150,
                completion_tokens=40,
                prompt_tokens_details=SimpleNamespace(cached_tokens=100),
            )
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(input_tokens=50, output_tokens=40, cached_tokens=100)

    def test_defaults_cached_tokens_to_zero_when_details_missing(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=150, completion_tokens=40)
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(input_tokens=150, output_tokens=40, cached_tokens=0)

    def test_defaults_cached_tokens_to_zero_when_cached_tokens_field_missing(
        self,
    ) -> None:
        # prompt_tokens_details present but without cached_tokens itself
        # (e.g. a partial/older response shape).
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=150,
                completion_tokens=40,
                prompt_tokens_details=SimpleNamespace(),
            )
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(input_tokens=150, output_tokens=40, cached_tokens=0)

    def test_cache_write_tokens_always_zero_for_openai(self) -> None:
        # OpenAI has no billable cache-write concept; the fields must stay
        # at their zero default regardless of response shape.
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=150, completion_tokens=40)
        )

        usage = extract_usage(response)

        assert usage is not None
        assert usage.cache_write_5m_tokens == 0
        assert usage.cache_write_1h_tokens == 0

    def test_openai_shape_without_deepseek_cache_field_not_claimed_by_deepseek(
        self,
    ) -> None:
        # A genuine OpenAI response (no prompt_cache_hit_tokens) must fall
        # through _extract_deepseek and be resolved by _extract_openai,
        # not be misidentified or dropped.
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=150, completion_tokens=40)
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(input_tokens=150, output_tokens=40)

    def test_mistral_shaped_response_recognized_via_shared_extractor(self) -> None:
        # Mistral's usage object is currently identical in shape to OpenAI's;
        # this documents that fact so a future divergence (which would need
        # its own _extract_mistral) shows up here as a failing assumption,
        # not a silent gap.
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=137, completion_tokens=914)
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(input_tokens=137, output_tokens=914)


class TestAnthropicExtraction:
    def test_extracts_usage_from_anthropic_shaped_response(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=150, output_tokens=40)
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(input_tokens=150, output_tokens=40)

    def test_openai_extractor_does_not_false_positive_on_anthropic_shape(self) -> None:
        # Anthropic's usage has input_tokens/output_tokens, not
        # prompt_tokens/completion_tokens: the OpenAI extractor must not
        # match by accident and must yield to the Anthropic extractor.
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=150, output_tokens=40)
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(input_tokens=150, output_tokens=40)

    def test_extracts_cached_tokens_as_separate_count(self) -> None:
        # Anthropic already reports input_tokens excluding cache reads:
        # no subtraction needed, unlike OpenAI/Google.
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=50, output_tokens=40, cache_read_input_tokens=100
            )
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(input_tokens=50, output_tokens=40, cached_tokens=100)

    def test_defaults_cached_tokens_to_zero_when_cache_read_field_missing(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=150, output_tokens=40)
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(input_tokens=150, output_tokens=40, cached_tokens=0)

    def test_extracts_cache_write_5m_tokens(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=21,
                output_tokens=393,
                cache_creation=SimpleNamespace(
                    ephemeral_5m_input_tokens=148, ephemeral_1h_input_tokens=0
                ),
            )
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(
            input_tokens=21,
            output_tokens=393,
            cache_write_5m_tokens=148,
            cache_write_1h_tokens=0,
        )

    def test_extracts_cache_write_1h_tokens(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=21,
                output_tokens=393,
                cache_creation=SimpleNamespace(
                    ephemeral_5m_input_tokens=0, ephemeral_1h_input_tokens=100
                ),
            )
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(
            input_tokens=21,
            output_tokens=393,
            cache_write_1h_tokens=100,
        )

    def test_extracts_both_cache_write_tiers_simultaneously(self) -> None:
        # Anthropic's docs show both tiers as independent counts on the
        # same response; a single call could in principle hit both.
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=21,
                output_tokens=393,
                cache_creation=SimpleNamespace(
                    ephemeral_5m_input_tokens=148, ephemeral_1h_input_tokens=100
                ),
            )
        )

        usage = extract_usage(response)

        assert usage is not None
        assert usage.cache_write_5m_tokens == 148
        assert usage.cache_write_1h_tokens == 100

    def test_defaults_cache_write_tokens_to_zero_when_cache_creation_missing(
        self,
    ) -> None:
        # Older SDK/API version without the cache_creation breakdown object.
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=150, output_tokens=40)
        )

        usage = extract_usage(response)

        assert usage is not None
        assert usage.cache_write_5m_tokens == 0
        assert usage.cache_write_1h_tokens == 0

    def test_defaults_cache_write_tokens_to_zero_when_fields_missing_within_cache_creation(
        self,
    ) -> None:
        # cache_creation present but empty/partial (e.g. a stripped-down mock).
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=150, output_tokens=40, cache_creation=SimpleNamespace()
            )
        )

        usage = extract_usage(response)

        assert usage is not None
        assert usage.cache_write_5m_tokens == 0
        assert usage.cache_write_1h_tokens == 0


class TestGoogleExtraction:
    def test_extracts_usage_from_google_shaped_response(self) -> None:
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=150, candidates_token_count=40
            )
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(input_tokens=150, output_tokens=40)

    def test_returns_none_if_google_usage_missing_prompt_count(self) -> None:
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(candidates_token_count=40)
        )

        assert extract_usage(response) is None

    def test_google_shape_does_not_false_positive_on_openai_or_anthropic(
        self,
    ) -> None:
        # usage_metadata is a distinct attribute name from usage, so an
        # OpenAI/Anthropic-shaped response (which has .usage, not
        # .usage_metadata) must never be misidentified as Google.
        response = SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=150, completion_tokens=40)
        )

        usage = extract_usage(response)

        # Must still resolve via the OpenAI extractor, not fall through.
        assert usage == TokenUsage(input_tokens=150, output_tokens=40)

    def test_extracts_cached_tokens_and_subtracts_from_input(self) -> None:
        # Google reports cached_content_token_count as a subset of
        # prompt_token_count, same accounting model as OpenAI.
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=150,
                candidates_token_count=40,
                cached_content_token_count=100,
            )
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(input_tokens=50, output_tokens=40, cached_tokens=100)

    def test_defaults_cached_tokens_to_zero_when_field_missing(self) -> None:
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=150, candidates_token_count=40
            )
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(input_tokens=150, output_tokens=40, cached_tokens=0)

    def test_cache_write_tokens_always_zero_for_google(self) -> None:
        response = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=150, candidates_token_count=40
            )
        )

        usage = extract_usage(response)

        assert usage is not None
        assert usage.cache_write_5m_tokens == 0
        assert usage.cache_write_1h_tokens == 0


class TestDeepSeekExtraction:
    def test_extracts_usage_from_deepseek_shaped_response(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=150,
                completion_tokens=40,
                prompt_cache_hit_tokens=0,
                prompt_cache_miss_tokens=150,
            )
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(input_tokens=150, output_tokens=40, cached_tokens=0)

    def test_extracts_cached_tokens_and_subtracts_from_input(self) -> None:
        # prompt_tokens (150) = prompt_cache_hit_tokens (100) + prompt_cache_miss_tokens (50),
        # matching DeepSeek's own documented invariant.
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=150,
                completion_tokens=40,
                prompt_cache_hit_tokens=100,
                prompt_cache_miss_tokens=50,
            )
        )

        usage = extract_usage(response)

        assert usage == TokenUsage(input_tokens=50, output_tokens=40, cached_tokens=100)

    def test_tried_before_openai_extractor_for_deepseek_shaped_response(self) -> None:
        # Both extractors could technically match (prompt_tokens/completion_tokens
        # are present in both), but only the DeepSeek-specific field must decide
        # which one actually claims the response — and it must claim it with
        # the cache-read data intact, not just fall through to OpenAI's shape.
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=150,
                completion_tokens=40,
                prompt_cache_hit_tokens=100,
                prompt_cache_miss_tokens=50,
            )
        )

        usage = extract_usage(response)

        assert usage is not None
        assert usage.cached_tokens == 100


class TestUnrecognizedResponses:
    def test_returns_none_for_response_without_usage_attribute(self) -> None:
        response = SimpleNamespace(choices=["something"])

        assert extract_usage(response) is None

    def test_returns_none_for_plain_dict_response(self) -> None:
        # Some providers (or local/custom clients) return plain dicts,
        # which have no attribute access -> must fall through cleanly.
        response = {"usage": {"prompt_tokens": 150, "completion_tokens": 40}}

        assert extract_usage(response) is None

    def test_returns_none_for_none_response(self) -> None:
        assert extract_usage(None) is None

    def test_returns_none_for_string_response(self) -> None:
        assert extract_usage("just a string") is None

    def test_returns_none_when_usage_is_none(self) -> None:
        response = SimpleNamespace(usage=None)

        assert extract_usage(response) is None
