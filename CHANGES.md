# Changelog

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
