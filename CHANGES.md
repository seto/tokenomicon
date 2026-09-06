# Changelog

## Version 0.5.0

### Async Support

- Add async/await support to `augur`: functions decorated with `async def` are now
  properly awaited and resolve to a `CallResult`, instead of silently returning an
  unresolved coroutine (previously, decorating an async function produced a `CallResult`
  wrapping the coroutine itself, never executing it). Sync functions are unaffected.
  Overloaded type hints let static type checkers correctly infer `await decorated()` on
  async-decorated functions.
- Add `Config.aload_toml()`, a non-blocking async twin of `load_toml()`: same signature
  and behavior, offloaded to a thread via `asyncio.to_thread` so it doesn't block the
  event loop when called from async code.

## Version 0.4.0

### Supported Providers

- Add DeepSeek support: token usage (including cache-read accounting via
  `prompt_cache_hit_tokens`) is extracted automatically from DeepSeek's
  OpenAI-compatible response shape.
- Add Mistral support: recognized automatically, since Mistral's response shape is
  currently identical to OpenAI's.

## Version 0.3.0

### Cache Write Rates

- Added prompt cache write pricing (Anthropic-only):
  `PricingPlan.cache_write_5m_per_million` and `cache_write_1h_per_million` price the
  two cache-creation TTL tiers, extracted automatically from Anthropic's
  `cache_creation` response field. Unlike cache reads, there's no rate fallback for
  writes: `tribute()` raises the new `CachePricingNotConfiguredError` if write tokens
  are present but the matching rate isn't configured, rather than silently understating
  the cost. `CallResult` now exposes `cache_write_5m_tokens` and `cache_write_1h_tokens`
  alongside the existing token counts.

## Version 0.2.0

### Prompt Caching

- Implemented prompt caching support: `PricingPlan.cached_input_per_million` prices
  cache reads at a discounted rate (falling back to the regular input rate when unset).
  Extractors recognize cache reads automatically across OpenAI, Anthropic, and Google
  response shapes, and cache token counts are billed separately from regular input
  tokens (not as a subset), matching how each provider ultimately reports them.
  `CallResult` now exposes `cached_tokens` alongside the existing token counts. Cache
  _write_ costs (e.g. Anthropic's cache-creation premium) aren't tracked yet.

### Code Style and Linting

- Enabled Ruff as linter with the following rules:
  - E: pycodestyle errors
  - W: pycodestyle warnings
  - F: Pyflakes errors
  - B: flake8-bugbear warnings
  - UP: pyupgrade warnings and modernizations
  - SIM: flake8-simplify rules
  - FURB: Refurb rules
- Ignore line length errors (E501) since Black handles formatting.

### Dependency Updates

- Fixed typo in Dependabot workflow that prevented it from running on schedule.

## Version 0.1.1

- Nothing really changed, but the README links to the license and changelog files on
  GitHub instead of the relative repo paths, which were broken when rendered on PyPI.
- The `Changelog` URL in `pyproject.toml` now points to `CHANGES.md` instead of the
  (mostly empty) GitHub Releases page.

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
