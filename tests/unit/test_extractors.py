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
