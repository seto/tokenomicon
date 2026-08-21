<h1 align="center">Tokenomicon</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://pypi.org/project/tokenomicon/"><img src="https://img.shields.io/pypi/v/tokenomicon.svg?maxAge=86400&color=blue" alt="Version"></a>
  <a href="https://pypi.org/project/tokenomicon"><img src="https://img.shields.io/pypi/pyversions/tokenomicon.svg" alt="Supported Versions"></a>
  <a href="https://github.com/seto/tokenomicon/actions"><img src="https://img.shields.io/github/actions/workflow/status/seto/tokenomicon/utests.yml?label=utests&logo=github" alt="Unit Tests"></a>
  <a href="https://github.com/seto/tokenomicon/actions"><img src="https://img.shields.io/github/actions/workflow/status/seto/tokenomicon/itests.yml?label=itests&logo=github" alt="Integration Tests"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code_style-black-000000.svg" alt="Code Style: Black"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Linting: Ruff"></a>
</p>

**Tokenomicon** is a post-hoc financial divination tool for LLM hunger and budget dread.
It calculates what a call to an LLM provider owes you back, in the form of a `tribute`.
Nothing more, nothing less.

_No token counting before the call. No pricing catalog to keep in sync with whatever
model shipped this week. Just the augur, reading the entrails of the response you
already got._

---

## Why

Most cost-tracking tools for LLM calls either bundle a tokenizer (useful only for one
provider, drifting out of date as models change) or ship a hardcoded pricing table
(equally stale the moment a provider changes a rate). Tokenomicon does neither.

- **Zero runtime dependencies.** Pure standard library, Python 3.11+.
- **No opinion on what exists.** Tokenomicon doesn't know or care which models are on
  the market. You register what you use, at the price you were quoted.
- **No redeploy for a price change.** Pricing lives in an external TOML file, reloadable
  at runtime.
- **Reads the bill, doesn't guess it.** Token usage is read from the provider's own
  response, the same numbers you'd already be billed on, not estimated with a local
  tokenizer.

## Installation

```bash
pip install tokenomicon
```

## Quick start

```python
from tokenomicon import augur, config

config.load_toml("pricing.toml")

@augur(model="gpt-5.4-mini")
def call_llm(prompt: str):
    return your_llm_client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
    )

outcome = call_llm("hi")

print(outcome.tribute)  # Decimal("3.14159"), what this call owes
print(outcome.result)  # the original provider response, untouched
```

`pricing.toml`:

```toml
[gpt-5.4-mini]
input_per_million = "0.75"
output_per_million = "4.50"
```

## Configuration

### TOML

Register pricing plans by loading a TOML file. There's no fixed or implied location:
pass whatever path fits your project (root, `config/`, a path from an environment
variable, etc.). Tokenomicon never looks for a file on its own.

```python
config.load_toml("pricing.toml")
```

Each table is a model name; `currency` defaults to `"USD"` if omitted:

```toml
[claude-sonnet-5]
input_per_million = "3.00"
output_per_million = "15.00"
currency = "EUR"
```

An unregistered model raises `ModelNotConfiguredError`: Tokenomicon never falls back to
a bundled "market price."

### Environment variable expansion

`${VAR_NAME}` patterns in the TOML file are left untouched by default. Opt in explicitly
to expand them from the environment:

```python
config.load_toml("pricing.toml", expand_env=True)
```

```toml
[gpt-5.4-mini]
input_per_million = "${GPT_5_4_MINI_INPUT}"
output_per_million = "${GPT_5_4_MINI_OUTPUT}"
```

A referenced variable that isn't set raises `ConfigError`.

### Prompt caching

Set `cached_input_per_million` on a plan to price cache reads at a discounted rate:

```toml
[claude-sonnet-5]
input_per_million = "3.00"
output_per_million = "15.00"
cached_input_per_million = "0.30"
```

Cached tokens are billed separately from regular input tokens, not as a subset of them,
matching how the token counts are reported back to `tribute()` and `CallResult`. If
`cached_input_per_million` isn't set, cached tokens fall back to the regular input rate
— no discount, but nothing lost or silently dropped either.

Extraction is automatic wherever the provider reports it: OpenAI's `cached_tokens`,
Anthropic's `cache_read_input_tokens`, and Google's `cached_content_token_count` are all
recognized.

**Cache writes** (Anthropic-only, the premium paid to populate the cache) are also
supported, split by TTL tier:

```toml
[claude-sonnet-5]
input_per_million = "3.00"
output_per_million = "15.00"
cache_write_5m_per_million = "3.75"
cache_write_1h_per_million = "6.00"
```

Unlike cache reads, there's no rate fallback for cache writes: billing a write premium
at the base input rate would silently understate the real cost. If Anthropic reports
cache-write tokens for either tier and the corresponding rate isn't configured,
`tribute()` raises `CachePricingNotConfiguredError` instead of guessing.

## The `augur` decorator

`augur` wraps a function that returns a provider response. It reads token usage off that
response, calculates the tribute owed, and returns a `CallResult` without touching your
original return value:

```python
@dataclass(frozen=True, slots=True)
class CallResult:
    result: Any  # the original, untouched response
    tribute: Decimal | None  # what this call owes, or None if undetermined
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None  # tokens served from a prompt cache, 0 if none
    cache_write_5m_tokens: int | None  # Anthropic-only, 5-minute cache TTL tier
    cache_write_1h_tokens: int | None  # Anthropic-only, 1-hour cache TTL tier
    currency: str | None
```

### Supported providers

Token usage is extracted automatically from known response shapes:

```python
import openai
client = openai.OpenAI()

@augur(model="gpt-5.4-mini")
def call():
    return client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": "hi"}],
    )
```

The same works out of the box for `anthropic` and `google-genai` clients; Tokenomicon
recognizes their respective `usage` / `usage_metadata` shapes without any extra
configuration.

### Fallback for unrecognized responses

If a response doesn't match a known shape (a local model, a custom client, a provider
not yet supported), supply a `manual_tokens` function:

```python
@augur(
    model="local-llama",
    manual_tokens=lambda response: (response["prompt_len"], response["gen_len"]),
)
def call_local(prompt: str):
    return my_local_client.generate(prompt)
```

`manual_tokens` is only called if automatic extraction fails, it never overrides a
successful automatic read. If neither succeeds, Tokenomicon emits a
`TokenExtractionWarning` and returns a `CallResult` with `tribute=None`, rather than
guessing.

## Error handling

All exceptions inherit from `TokenomiconError`:

| Exception                        | Raised when                                                                                  |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| `ModelNotConfiguredError`        | The requested model isn't registered in `Config`.                                            |
| `InvalidCurrencyError`           | `currency` isn't a valid ISO 4217 code.                                                      |
| `InvalidPricingError`            | A rate is negative or otherwise invalid.                                                     |
| `NegativeTokenCountError`        | A token count passed to `.tribute()` is negative.                                            |
| `CachePricingNotConfiguredError` | Cache-write tokens are present but the corresponding rate isn't configured.                  |
| `ConfigError`                    | TOML parsing fails, a field is missing, or an env var referenced via `expand_env` isn't set. |

`TokenExtractionWarning` is a `UserWarning`, not an exception: it doesn't interrupt the
call, it only signals that the tribute couldn't be determined.

> [!NOTE]  
> `ModelNotConfiguredError` is raised _after_ your wrapped function has already run. A
> misconfigured model name doesn't prevent the underlying LLM call from firing (and
> being billed by the provider); it only prevents Tokenomicon from calculating its
> tribute. Register your models before the calls that use them.

## Not yet supported

- **Cost accumulation / ledger** across multiple calls. Tokenomicon deliberately stays
  per-call; aggregate however fits your own storage.

## Development

```bash
git clone https://github.com/seto/tokenomicon.git
cd tokenomicon
pip install -r requirements-dev.txt
invoke utest
```

Integration tests make real, minimal calls against actual provider SDKs, to confirm
extractors recognize genuine response shapes rather than hand-built fixtures. They
require API keys and are opt-in only:

```bash
pip install -r tests/integration/requirements.txt
invoke itest
```

See `tests/integration/conftest.py` for the expected environment variables.

## License

This program is licensed under the
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).  
See the [LICENSE](https://github.com/seto/tokenomicon/blob/master/LICENSE) file for
details.

## Changelog

See [CHANGES.md](https://github.com/seto/tokenomicon/blob/master/CHANGES.md) for release
notes.
