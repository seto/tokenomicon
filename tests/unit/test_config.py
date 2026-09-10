import asyncio
from decimal import Decimal

import pytest

from tokenomicon.config import Config
from tokenomicon.exceptions import ConfigError, ModelNotConfiguredError
from tokenomicon.pricing import PricingPlan


class TestConfigRegistration:
    def test_register_and_get(self) -> None:
        cfg = Config()
        plan = PricingPlan(
            model="claude-sonnet-5",
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
        )
        cfg.register(plan)

        assert cfg.get("claude-sonnet-5") is plan

    def test_get_unregistered_model_raises(self) -> None:
        cfg = Config()
        with pytest.raises(ModelNotConfiguredError):
            cfg.get("nonexistent-model")

    def test_get_error_lists_available_models(self) -> None:
        cfg = Config()
        cfg.register(
            PricingPlan(
                model="claude-sonnet-5",
                input_per_million=Decimal("3.00"),
                output_per_million=Decimal("15.00"),
            )
        )
        with pytest.raises(ModelNotConfiguredError, match="claude-sonnet-5"):
            cfg.get("nonexistent-model")

    def test_register_overwrites_existing_model(self) -> None:
        cfg = Config()
        cfg.register(
            PricingPlan(
                model="claude-sonnet-5",
                input_per_million=Decimal("3.00"),
                output_per_million=Decimal("15.00"),
            )
        )
        cfg.register(
            PricingPlan(
                model="claude-sonnet-5",
                input_per_million=Decimal("4.00"),
                output_per_million=Decimal("16.00"),
            )
        )
        assert cfg.get("claude-sonnet-5").input_per_million == Decimal("4.00")


class TestConfigLoadToml:
    def test_load_valid_toml(self, tmp_path) -> None:
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [gpt-5-mini]
            input_per_million = "0.25"
            output_per_million = "2.00"

            [claude-sonnet-5]
            input_per_million = "3.00"
            output_per_million = "15.00"
            currency = "EUR"
            """
        )  # fmt: skip

        cfg = Config()
        cfg.load_toml(toml_file)

        mini = cfg.get("gpt-5-mini")
        assert mini.input_per_million == Decimal("0.25")
        assert mini.output_per_million == Decimal("2.00")
        assert mini.currency == "USD"  # default when omitted

        sonnet = cfg.get("claude-sonnet-5")
        assert sonnet.currency == "EUR"

    def test_load_toml_accepts_path_as_string(self, tmp_path) -> None:
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "3.00"
            output_per_million = "15.00"
            """
        )  # fmt: skip

        cfg = Config()
        cfg.load_toml(str(toml_file))  # str, not Path

        assert cfg.get("claude-sonnet-5") is not None

    def test_load_toml_reload_replaces_existing_plan(self, tmp_path) -> None:
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "3.00"
            output_per_million = "15.00"
            """
        )  # fmt: skip

        cfg = Config()
        cfg.load_toml(toml_file)
        assert cfg.get("claude-sonnet-5").input_per_million == Decimal("3.00")

        # Simulate a price update: same file, new rate, reloaded without a redeploy.
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "4.00"
            output_per_million = "16.00"
            """
        )  # fmt: skip

        cfg.load_toml(toml_file)
        assert cfg.get("claude-sonnet-5").input_per_million == Decimal("4.00")

    def test_load_toml_invalid_numeric_value_raises_config_error(
        self, tmp_path
    ) -> None:
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "not-a-number"
            output_per_million = "15.00"
            """
        )  # fmt: skip

        cfg = Config()
        with pytest.raises(ConfigError, match="claude-sonnet-5"):
            cfg.load_toml(toml_file)

    def test_load_toml_invalid_file_does_not_partially_update_registry(
        self, tmp_path
    ) -> None:
        existing = PricingPlan(
            model="claude-sonnet-5",
            input_per_million=Decimal("3.00"),
            output_per_million=Decimal("15.00"),
        )
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "4.00"
            output_per_million = "20.00"

            [gpt-5-mini]
            input_per_million = "not-a-number"
            output_per_million = "2.00"
            """
        )  # fmt: skip

        cfg = Config()
        cfg.register(existing)

        with pytest.raises(ConfigError, match="gpt-5-mini"):
            cfg.load_toml(toml_file)

        assert cfg.get("claude-sonnet-5").input_per_million == Decimal("3.00")

    def test_load_toml_missing_field_raises_config_error(self, tmp_path) -> None:
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "3.00"
            """
        )  # fmt: skip

        cfg = Config()
        with pytest.raises(ConfigError, match="claude-sonnet-5"):
            cfg.load_toml(toml_file)

    def test_load_toml_with_cached_input_per_million(self, tmp_path) -> None:
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "3.00"
            output_per_million = "15.00"
            cached_input_per_million = "0.30"
            """
        )  # fmt: skip

        cfg = Config()
        cfg.load_toml(toml_file)

        plan = cfg.get("claude-sonnet-5")
        assert plan.cached_input_per_million == Decimal("0.30")

    def test_load_toml_without_cached_input_per_million_defaults_to_none(
        self, tmp_path
    ) -> None:
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "3.00"
            output_per_million = "15.00"
            """
        )  # fmt: skip

        cfg = Config()
        cfg.load_toml(toml_file)

        assert cfg.get("claude-sonnet-5").cached_input_per_million is None

    def test_load_toml_invalid_cached_numeric_value_raises_config_error(
        self, tmp_path
    ) -> None:
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "3.00"
            output_per_million = "15.00"
            cached_input_per_million = "not-a-number"
            """
        )  # fmt: skip

        cfg = Config()
        with pytest.raises(ConfigError, match="claude-sonnet-5"):
            cfg.load_toml(toml_file)

    def test_load_toml_with_cache_write_rates(self, tmp_path) -> None:
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "3.00"
            output_per_million = "15.00"
            cache_write_5m_per_million = "3.75"
            cache_write_1h_per_million = "6.00"
            """
        )  # fmt: skip

        cfg = Config()
        cfg.load_toml(toml_file)

        plan = cfg.get("claude-sonnet-5")
        assert plan.cache_write_5m_per_million == Decimal("3.75")
        assert plan.cache_write_1h_per_million == Decimal("6.00")

    def test_load_toml_without_cache_write_rates_defaults_to_none(
        self, tmp_path
    ) -> None:
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "3.00"
            output_per_million = "15.00"
            """
        )  # fmt: skip

        cfg = Config()
        cfg.load_toml(toml_file)

        plan = cfg.get("claude-sonnet-5")
        assert plan.cache_write_5m_per_million is None
        assert plan.cache_write_1h_per_million is None

    def test_load_toml_invalid_cache_write_numeric_value_raises_config_error(
        self, tmp_path
    ) -> None:
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "3.00"
            output_per_million = "15.00"
            cache_write_5m_per_million = "not-a-number"
            """
        )  # fmt: skip

        cfg = Config()
        with pytest.raises(ConfigError, match="claude-sonnet-5"):
            cfg.load_toml(toml_file)


class TestConfigALoadToml:
    def test_aload_toml_loads_same_as_sync(self, tmp_path) -> None:
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "3.00"
            output_per_million = "15.00"
            """
        )  # fmt: skip

        cfg = Config()
        asyncio.run(cfg.aload_toml(toml_file))

        plan = cfg.get("claude-sonnet-5")
        assert plan.input_per_million == Decimal("3.00")
        assert plan.output_per_million == Decimal("15.00")

    def test_aload_toml_accepts_expand_env_kwarg(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("CLAUDE_SONNET_5_INPUT", "3.00")
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "${CLAUDE_SONNET_5_INPUT}"
            output_per_million = "15.00"
            """
        )  # fmt: skip

        cfg = Config()
        asyncio.run(cfg.aload_toml(toml_file, expand_env=True))

        assert cfg.get("claude-sonnet-5").input_per_million == Decimal("3.00")

    def test_aload_toml_propagates_config_error(self, tmp_path) -> None:
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "not-a-number"
            output_per_million = "15.00"
            """
        )  # fmt: skip

        cfg = Config()
        with pytest.raises(ConfigError, match="claude-sonnet-5"):
            asyncio.run(cfg.aload_toml(toml_file))

    def test_aload_toml_reload_replaces_existing_plan(self, tmp_path) -> None:
        # Mirrors test_load_toml_reload_replaces_existing_plan, async path.
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "3.00"
            output_per_million = "15.00"
            """
        )  # fmt: skip

        cfg = Config()
        asyncio.run(cfg.aload_toml(toml_file))
        assert cfg.get("claude-sonnet-5").input_per_million == Decimal("3.00")

        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "4.00"
            output_per_million = "16.00"
            """
        )  # fmt: skip

        asyncio.run(cfg.aload_toml(toml_file))
        assert cfg.get("claude-sonnet-5").input_per_million == Decimal("4.00")

    def test_aload_toml_does_not_block_event_loop(self, tmp_path) -> None:
        # Regression guard for the actual point of aload_toml: while the file
        # load runs on a thread, the event loop must remain free to run other
        # coroutines concurrently, not stall until the load finishes.
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "3.00"
            output_per_million = "15.00"
            """
        )  # fmt: skip

        cfg = Config()
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            for _ in range(50):
                await asyncio.sleep(0)
                ticks += 1

        async def scenario() -> None:
            await asyncio.gather(cfg.aload_toml(toml_file), ticker())

        asyncio.run(scenario())

        # If aload_toml blocked the loop, the ticker couldn't have advanced
        # concurrently with it; this is a coarse signal, not a precise timing
        # assertion, since both tasks share the same thread until the file
        # I/O actually hands off control.
        assert ticks == 50


class TestConfigExpandEnv:
    def test_expand_env_substitutes_variable(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("CLAUDE_SONNET_5_INPUT", "3.00")
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "${CLAUDE_SONNET_5_INPUT}"
            output_per_million = "15.00"
            """
        )  # fmt: skip

        cfg = Config()
        cfg.load_toml(toml_file, expand_env=True)

        assert cfg.get("claude-sonnet-5").input_per_million == Decimal("3.00")

    def test_expand_env_missing_variable_raises_config_error(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.delenv("CLAUDE_SONNET_5_INPUT", raising=False)
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "${CLAUDE_SONNET_5_INPUT}"
            output_per_million = "15.00"
            """
        )  # fmt: skip

        cfg = Config()
        with pytest.raises(ConfigError, match="CLAUDE_SONNET_5_INPUT"):
            cfg.load_toml(toml_file, expand_env=True)

    def test_without_expand_env_pattern_is_kept_literal(self, tmp_path) -> None:
        # Explicit-by-default: no implicit expansion unless expand_env=True.
        toml_file = tmp_path / "pricing.toml"
        toml_file.write_text(
            """
            [claude-sonnet-5]
            input_per_million = "${CLAUDE_SONNET_5_INPUT}"
            output_per_million = "15.00"
            """
        )  # fmt: skip

        cfg = Config()
        with pytest.raises(ConfigError):
            # The literal string "${CLAUDE_SONNET_5_INPUT}" is not valid Decimal input,
            # so it must fail as a bad numeric value, not silently resolve.
            cfg.load_toml(toml_file)  # expand_env defaults to False
