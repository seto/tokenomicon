# Changelog

## Version 0.1.0

- First published version, because the models are hungry and the budget is finite!
- `PricingPlan` with Decimal-based cost calculation and ISO 4217 currency validation.
- `Config` registry with TOML loading, reloadable at runtime, and explicit `expand_env`
  support (no implicit environment variable expansion).
- `augur` decorator: automatic token-usage extraction (OpenAI, Anthropic, Google) with
  `manual_tokens` fallback for unrecognized response shapes.
- `CallResult` dataclass exposing `tribute`, token counts, and currency alongside the
  original provider response, untouched.
- Full exception hierarchy under `TokenomiconError`, plus `TokenExtractionWarning` for
  non-fatal extraction failures.
- Zero runtime dependencies, pure standard library, Python 3.11+.
