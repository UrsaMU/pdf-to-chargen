# @ursamu/chargen-tools

CLI tools for the UrsaMU chargen pipeline.

**Step 1** — Convert RPG rulebook PDFs to plain text (digital or scanned).
**Step 2** — Run a LangGraph agentic pipeline that extracts game mechanics, designs a plugin, generates all TypeScript, audits it, and writes it to your UrsaMU project.

Works with Anthropic (Claude) or any OpenAI-compatible provider (OpenAI, Groq, Together AI, Ollama, LM Studio, Mistral, and more).

---

## Requirements

```bash
# PDF conversion (Step 1)
brew install poppler     # pdftotext, pdfinfo, pdftoppm
brew install tesseract   # OCR for scanned books

# Deno runtime
brew install deno
```

---

## Quick start

### 1. Copy and fill in `.env`

```bash
cp .env.example .env
```

Minimum for Anthropic (default):

```env
ANTHROPIC_API_KEY=sk-ant-...
CHARGEN_TEXT_DIR=./books/text
CHARGEN_OUTPUT_DIR=/path/to/ursamu
CHARGEN_SYSTEM=Shadowrun 5e
```

Minimum for any OpenAI-compatible provider:

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.groq.com/openai/v1   # omit for api.openai.com
CHARGEN_MODEL=llama3-70b-8192
CHARGEN_TEXT_DIR=./books/text
CHARGEN_OUTPUT_DIR=/path/to/ursamu
CHARGEN_SYSTEM=Shadowrun 5e
```

See [docs/providers.md](docs/providers.md) for all supported providers and model names.

### 2. Convert PDFs to text

```bash
deno run --allow-read --allow-write --allow-run \
  jsr:@ursamu/chargen-tools/pdf-to-text \
  <pdf-dir> <text-dir>
```

| Flag | Description |
|------|-------------|
| `--flat` | Write all `.txt` files flat into `<text-dir>` — no subdirectory mirroring |
| `--overwrite` | Re-convert files that already exist |
| `--lang <code>` | Tesseract language pack (default: `eng`) |

Each PDF is auto-detected:
- **Digital** (vector text) — `pdftotext -layout`, preserves column spacing for stat tables
- **Scanned** (image-based) — `pdftoppm` at 300 DPI → `tesseract --psm 3` per page

Detection threshold: if `pdftotext` output is fewer than 150 bytes/page, the file is treated as scanned.

### 3. Extract mechanics and generate plugin

**Interactive** — pauses at each stage for your review:

```bash
deno run --allow-all jsr:@ursamu/chargen-tools/chargen-extract
```

**CI / headless** — auto-approves all checkpoints:

```bash
deno run --allow-all jsr:@ursamu/chargen-tools/chargen-extract --ci
```

All settings load from `.env`. Any flag overrides its env counterpart:

```bash
deno run --allow-all jsr:@ursamu/chargen-tools/chargen-extract \
  --system "Urban Shadows 2e" \
  --ci
```

See [docs/ci.md](docs/ci.md) for GitHub Actions examples.

---

## Flags

### `pdf-to-text`

```
<input-dir> <output-dir> [--flat] [--overwrite] [--lang <code>]
```

### `chargen-extract`

| Flag | Env var | Default | Description |
|------|---------|---------|-------------|
| `--text <dir>` | `CHARGEN_TEXT_DIR` | — | Directory of `.txt` files from Step 1 |
| `--output <dir>` | `CHARGEN_OUTPUT_DIR` | — | UrsaMU project root |
| `--system <name>` | `CHARGEN_SYSTEM` | — | Game system name |
| `--model <id>` | `CHARGEN_MODEL` | `claude-sonnet-4-6` / `gpt-4o` | Model override |
| `--max-chars <n>` | `CHARGEN_MAX_CHARS` | `150000` | Max characters to read from text files |
| `--ci` / `--yes` | `CHARGEN_CI=true` | `false` | Skip all confirmation checkpoints |
| `--provider <p>` | `CHARGEN_PROVIDER` | auto-detect | `anthropic` or `openai` |

**Priority order:** CLI flag → `.env` → built-in default.

---

## Pipeline

```
read_files → extract → [CONFIRM SCHEMA] → design → [CONFIRM DESIGN]
  → generate → audit → (auto-retry ×2) → write_files
```

1. **read_files** — reads all `.txt` files up to `--max-chars`
2. **extract** — LLM extracts stat schema + GM narrative context
3. **[CONFIRM SCHEMA]** — shows extracted schema; approve or type corrections *(skipped with `--ci`)*
4. **design** — LLM generates ursamu-dev plugin design plan
5. **[CONFIRM DESIGN]** — shows design plan; approve or type changes *(skipped with `--ci`)*
6. **generate** — LLM generates all plugin TypeScript files
7. **audit** — validates against ursamu-dev checklist; auto-retries on failure
8. **write_files** — writes `src/plugins/chargen/` to your UrsaMU project

---

## Output

```
<output>/src/plugins/chargen/
├── index.ts           IPlugin + IGameSystem + registerStatSystem + gm:system:register
├── commands.ts        +chargen, +stat, +skill, +ability, +chargen/submit, +chargen/approve
├── schema.ts          DBO types + collection instances (chargen.*)
├── validation.ts      Stat cap, pool, approval-lock checks
├── display.ts         Character sheet (rhost-vision) + formatCharacterContext for AI GM
├── game-system.ts     IGameSystem literal — extracted rules, thresholds, moves
├── tests/
│   └── chargen.test.ts
└── README.md
```

---

## AI GM bridge

Generated plugins implement `IGameSystem` (extends `IStatSystem`) and emit:

```typescript
gameHooks.emit("gm:system:register" as never, { system: gameSystem });
```

in `init()`. No direct `@ursamu/ai-gm` import — the AI GM picks up the system at runtime via `gameHooks`. This means the chargen plugin can be deployed independently of the AI GM package.

---

## Interactive mode

When running without `--ci`, the pipeline pauses twice for your input:

**Schema checkpoint** — After extraction, shows the full stat schema and GM narrative context. Press Enter to approve, or type corrections:

```
Strength column should go up to 8, not 6.
Also add Resonance as an attribute for Technomancers.
```

**Design checkpoint** — After the plugin design plan is generated. Approve or describe changes:

```
Add a +chargen/reset command that clears stats for characters in "incomplete" status.
```

Both corrections are fed back to the LLM before the next stage runs.

---

## Publishing

```bash
deno publish   # requires @ursamu scope access on jsr.io
```

---

## Docs

- [docs/providers.md](docs/providers.md) — Provider setup: Anthropic, OpenAI, Groq, Together AI, Ollama, LM Studio, Mistral
- [docs/ci.md](docs/ci.md) — CI/CD: GitHub Actions, env var reference, headless patterns
