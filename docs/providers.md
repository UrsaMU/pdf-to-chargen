# Provider Configuration

`chargen-extract` supports any provider with an OpenAI-compatible API, plus Anthropic's
native API. Provider is **auto-detected** from which API key is set in your `.env`.

## Auto-detection logic

| Condition | Provider used |
|-----------|--------------|
| `OPENAI_API_KEY` or `OPENAI_BASE_URL` is set | OpenAI-compatible |
| Only `ANTHROPIC_API_KEY` is set | Anthropic (native) |
| Both are set | OpenAI-compatible (override with `CHARGEN_PROVIDER=anthropic`) |
| `CHARGEN_PROVIDER=anthropic` | Anthropic (explicit) |
| `CHARGEN_PROVIDER=openai` | OpenAI-compatible (explicit) |

---

## Anthropic

Default provider. Uses the native Anthropic API — not the OpenAI-compatible shim —
so structured tool use, extended thinking, and prompt caching work correctly.

```env
ANTHROPIC_API_KEY=sk-ant-...
CHARGEN_MODEL=claude-sonnet-4-6   # default
```

**Recommended models:**

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| `claude-sonnet-4-6` | Fast | Excellent | Default — best balance |
| `claude-opus-4-6` | Slow | Best | Use for complex systems with dense rules |
| `claude-haiku-4-5` | Very fast | Good | Use for large text batches or budget CI |

---

## OpenAI

```env
OPENAI_API_KEY=sk-...
CHARGEN_MODEL=gpt-4o   # default when OPENAI_API_KEY is set
```

**Recommended models:**

| Model | Notes |
|-------|-------|
| `gpt-4o` | Default — strong reasoning, good structured output |
| `gpt-4o-mini` | Faster and cheaper; acceptable for straightforward systems |
| `o3-mini` | Strong reasoning; slower |

---

## Groq

Groq runs open-source models at very high inference speed — good for fast iteration.

```env
OPENAI_API_KEY=gsk_...
OPENAI_BASE_URL=https://api.groq.com/openai/v1
CHARGEN_MODEL=llama3-70b-8192
```

**Available models:**

| Model | Notes |
|-------|-------|
| `llama3-70b-8192` | Best quality on Groq |
| `llama3-8b-8192` | Fast; use for quick tests |
| `mixtral-8x7b-32768` | Larger context window |
| `gemma2-9b-it` | Lightweight |

Get a key at [console.groq.com](https://console.groq.com).

---

## Together AI

Large catalogue of open-source models with good availability.

```env
OPENAI_API_KEY=<together-key>
OPENAI_BASE_URL=https://api.together.xyz/v1
CHARGEN_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
```

**Recommended models:**

| Model | Notes |
|-------|-------|
| `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` | Strong general purpose |
| `meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo` | Highest quality |
| `mistralai/Mistral-7B-Instruct-v0.3` | Fast and cheap |
| `Qwen/Qwen2.5-72B-Instruct-Turbo` | Strong for structured output |

Get a key at [api.together.ai](https://api.together.ai).

---

## Mistral AI

```env
OPENAI_API_KEY=<mistral-key>
OPENAI_BASE_URL=https://api.mistral.ai/v1
CHARGEN_MODEL=mistral-large-latest
```

**Models:**

| Model | Notes |
|-------|-------|
| `mistral-large-latest` | Best quality |
| `mistral-small-latest` | Faster, cheaper |
| `codestral-latest` | Code-focused; good for plugin generation stage |

Get a key at [console.mistral.ai](https://console.mistral.ai).

---

## Ollama (local)

Run models locally with no API key required. Install Ollama and pull a model first.

```bash
brew install ollama
ollama pull llama3:70b   # or any other model
ollama serve             # starts on localhost:11434 by default
```

```env
OPENAI_BASE_URL=http://localhost:11434/v1
CHARGEN_MODEL=llama3:70b
# No OPENAI_API_KEY needed — the script uses "local" as a placeholder
```

**Recommended models for chargen:**

| Model | Notes |
|-------|-------|
| `llama3:70b` | Best quality; needs ~40 GB RAM |
| `llama3:8b` | Runs on 8 GB RAM; lower quality |
| `mistral:7b` | Good balance; runs on 8 GB RAM |
| `qwen2.5:72b` | Strong structured output |
| `deepseek-coder-v2` | Code-heavy stages |

**Performance note:** Local models will be significantly slower than hosted APIs for the
generation stage, which produces ~7 TypeScript files in one shot. `llama3:70b` typically
takes 3–8 minutes per run depending on hardware.

---

## LM Studio

LM Studio provides a local OpenAI-compatible server. Download a GGUF model and start
the local server from the LM Studio UI (default port 1234).

```env
OPENAI_BASE_URL=http://localhost:1234/v1
CHARGEN_MODEL=your-loaded-model-name
# No OPENAI_API_KEY needed
```

The model name must match exactly what appears in LM Studio's model dropdown.

---

## Fireworks AI

```env
OPENAI_API_KEY=<fireworks-key>
OPENAI_BASE_URL=https://api.fireworks.ai/inference/v1
CHARGEN_MODEL=accounts/fireworks/models/llama-v3p1-70b-instruct
```

---

## Perplexity

```env
OPENAI_API_KEY=<perplexity-key>
OPENAI_BASE_URL=https://api.perplexity.ai
CHARGEN_MODEL=llama-3.1-70b-instruct
```

---

## Other OpenAI-compatible endpoints

Any provider that implements the OpenAI chat completions API works:

```env
OPENAI_API_KEY=<your-key>
OPENAI_BASE_URL=https://your-provider.example.com/v1
CHARGEN_MODEL=your-model-name
```

The script posts to `POST $OPENAI_BASE_URL/chat/completions` with a standard messages
array — no provider-specific extensions are used.

---

## Choosing a model

The pipeline makes **3–5 LLM calls** per run, with the generation call being the longest
(produces 7 files). Rough guidance:

| Goal | Recommendation |
|------|---------------|
| Best output quality | `claude-sonnet-4-6` or `gpt-4o` |
| Fastest CI run | `claude-haiku-4-5` or `llama3-70b-8192` on Groq |
| No API cost | Ollama with `llama3:70b` |
| Dense rulebooks (SR5e, WoD) | `claude-opus-4-6` or `gpt-4o` — need large context |
| Simple systems (PbtA, FitD) | Any model works |

## Context window

The pipeline defaults to reading 150,000 characters of source text. Most models can
handle this comfortably. If you hit context errors, lower it:

```env
CHARGEN_MAX_CHARS=80000
```

or pass `--max-chars 80000` at the CLI.
