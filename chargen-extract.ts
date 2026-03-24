#!/usr/bin/env -S deno run --allow-read --allow-write --allow-env --allow-net
/**
 * chargen-extract.ts — LangGraph agentic pipeline for UrsaMU chargen plugin generation.
 *
 * Supports any OpenAI-compatible provider (OpenAI, Groq, Together, Ollama, LM Studio,
 * Mistral, …) as well as Anthropic's native API. Provider is auto-detected from env vars.
 *
 * Settings are loaded from a .env file in the working directory, with CLI flags
 * taking precedence over env vars.
 *
 * Usage:
 *   deno run --allow-all chargen-extract.ts [flags]
 *   deno run --allow-all jsr:@ursamu/chargen-tools/chargen-extract [flags]
 *
 * Flags (all optional if set in .env):
 *   --text <dir>        Directory of .txt files
 *   --output <dir>      UrsaMU project root
 *   --system <name>     Game system name
 *   --model <id>        Model override
 *   --max-chars <n>     Max chars to read (default 150000)
 *   --ci / --yes        Non-interactive: auto-approve all checkpoints
 *
 * See .env.example for all supported environment variables.
 */

import "@std/dotenv/load";
import { Annotation, END, interrupt, MemorySaver, START, StateGraph } from "@langchain/langgraph";
import { Command } from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatOpenAI } from "@langchain/openai";
import type { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { walk } from "@std/fs/walk";
import { join, relative } from "@std/path";
import { ensureDir } from "@std/fs/ensure-dir";

// ─── Config — CLI flags > .env vars > defaults ────────────────────────────────

/** Returns value from CLI flag, then env var, then fallback. Exits if required and missing. */
function cfg(flag: string, envVar?: string, fallback?: string): string {
  const idx = Deno.args.indexOf(flag);
  const cli = idx !== -1 ? Deno.args[idx + 1] : undefined;
  if (cli) return cli;
  if (envVar) {
    const env = Deno.env.get(envVar);
    if (env) return env;
  }
  if (fallback !== undefined) return fallback;
  const src = [flag, envVar].filter(Boolean).join(" / ");
  console.error(`Error: required setting missing — provide ${src}`);
  console.error("Usage: chargen-extract --text <dir> --output <dir> --system \"Name\" [--ci]");
  Deno.exit(1);
}

const CI = Deno.args.includes("--ci") || Deno.args.includes("--yes") ||
           Deno.env.get("CHARGEN_CI") === "true";

// Provider detection: explicit flag/env > infer from which API key is present
const rawProvider = cfg("--provider", "CHARGEN_PROVIDER", "");
const useOpenAI   = rawProvider === "openai" ||
                    (!rawProvider && (!!Deno.env.get("OPENAI_API_KEY") || !!Deno.env.get("OPENAI_BASE_URL")));

const DEFAULT_MODEL = useOpenAI ? "gpt-4o" : "claude-sonnet-4-6";

const TEXT_DIR  = cfg("--text",       "CHARGEN_TEXT_DIR");
const OUTPUT    = cfg("--output",     "CHARGEN_OUTPUT_DIR");
const SYS_NAME  = cfg("--system",     "CHARGEN_SYSTEM");
const MODEL_ID  = cfg("--model",      "CHARGEN_MODEL", DEFAULT_MODEL);
const MAX_CHARS = parseInt(cfg("--max-chars", "CHARGEN_MAX_CHARS", "150000"), 10);

// ─── State ───────────────────────────────────────────────────────────────────

const ChargenState = Annotation.Root({
  textDir:        Annotation<string>({ reducer: (_p, n) => n, default: () => TEXT_DIR }),
  outputDir:      Annotation<string>({ reducer: (_p, n) => n, default: () => OUTPUT }),
  systemName:     Annotation<string>({ reducer: (_p, n) => n, default: () => SYS_NAME }),
  textContent:    Annotation<string>({ reducer: (_p, n) => n, default: () => "" }),
  statSchema:     Annotation<string>({ reducer: (_p, n) => n, default: () => "" }),
  gmContext:      Annotation<string>({ reducer: (_p, n) => n, default: () => "" }),
  designPlan:     Annotation<string>({ reducer: (_p, n) => n, default: () => "" }),
  pluginFiles:    Annotation<Record<string, string>>({ reducer: (_p, n) => n, default: () => ({}) }),
  auditIssues:    Annotation<string[]>({ reducer: (_p, n) => n, default: () => [] }),
  retryCount:     Annotation<number>({ reducer: (_p, n) => n, default: () => 0 }),
});

type State = typeof ChargenState.State;

// ─── Model factory ────────────────────────────────────────────────────────────

function createModel(): BaseChatModel {
  if (useOpenAI) {
    const apiKey  = Deno.env.get("OPENAI_API_KEY") ?? "local"; // local providers often skip auth
    const baseURL = Deno.env.get("OPENAI_BASE_URL");
    console.log(`[chargen-extract] Provider: openai-compatible  model: ${MODEL_ID}${baseURL ? `  base: ${baseURL}` : ""}`);
    return new ChatOpenAI({
      model:        MODEL_ID,
      temperature:  0.3,
      openAIApiKey: apiKey,
      configuration: baseURL ? { baseURL } : undefined,
    }) as unknown as BaseChatModel;
  }

  const apiKey = Deno.env.get("ANTHROPIC_API_KEY");
  if (!apiKey) {
    console.error("Error: ANTHROPIC_API_KEY not set (or set OPENAI_API_KEY to use an OpenAI-compatible provider)");
    Deno.exit(1);
  }
  console.log(`[chargen-extract] Provider: anthropic  model: ${MODEL_ID}`);
  return new ChatAnthropic({ model: MODEL_ID, temperature: 0.3, anthropicApiKey: apiKey }) as unknown as BaseChatModel;
}

const model = createModel();

async function llm(system: string, human: string): Promise<string> {
  const resp = await model.invoke([new SystemMessage(system), new HumanMessage(human)]);
  return typeof resp.content === "string"
    ? resp.content
    : (resp.content as Array<{ type?: string; text?: string }>)
        .filter((p) => p.type === "text").map((p) => p.text ?? "").join("");
}

// ─── Prompts ─────────────────────────────────────────────────────────────────

const EXTRACT_SYSTEM = `You are extracting RPG game mechanics from rulebook text.
Output two sections separated by "---GM_CONTEXT---":

Section 1 — STAT SCHEMA (markdown tables):
- Attributes table (Name | Min | Max | Notes)
- Skills table (Name | Linked Attribute | Max | Notes)
- Special Abilities section (category, examples, selection method)
- Derived Stats table (Name | Formula)
- Resource Pools table (Name | Total | Purpose)
- Metatypes/Archetypes table (Name | Stat Modifiers | Notes)

Section 2 — GM CONTEXT:
- Dice system description
- Full success threshold (number)
- Partial success threshold (number)
- Core rules prompt (≤150 words for LLM injection)
- Adjudication hint (fiction-first guidance)
- Hard moves list (3-5 examples)
- Soft moves list (3-5 examples)
- Miss consequence hint

Be comprehensive. Extract every stat, skill, and ability you find.`;

const DESIGN_SYSTEM = `You are designing a UrsaMU chargen plugin following the ursamu-dev standard.

Output a complete Design Plan block in this exact format:
\`\`\`
## Design Plan: <system> Chargen

Context:      plugin: chargen
Commands:     [list all commands with locks and descriptions]
Invariants:   [list all rules that must always hold]
DB:           chargen.chars: { schema }
              chargen.reviews: { schema }
Hooks:        player:login → incomplete chargen reminder
REST:         GET /api/v1/chargen/:playerId (auth: yes)
AI GM bridge: gm:system:register emitted in init()
Side-effects: On approve: set player flags, lock character
\`\`\`

The plugin MUST implement IGameSystem (extends IStatSystem) and emit gm:system:register.`;

const GENERATE_SYSTEM = `You are generating TypeScript code for a UrsaMU chargen plugin.

Rules:
- Use import from "jsr:@ursamu/ursamu" for all engine imports
- Stat names: lowercase with underscores
- All DBO collections prefixed with "chargen."
- Every addCmd has complete help text with Examples section
- Color codes always end with %cn
- Early return over nested conditions
- No function longer than 50 lines

Output a JSON object where keys are relative file paths and values are the complete file contents.
Example: { "index.ts": "...", "commands.ts": "...", ... }

Generate these files:
- index.ts    (IPlugin + IGameSystem + registerStatSystem + gm:system:register)
- commands.ts (all addCmd registrations)
- schema.ts   (DBO type definitions + collection instances)
- validation.ts (validateStat, validatePool, checkApproved, checkRequired)
- display.ts  (formatSheet for MUSH + formatCharacterContext for GM)
- game-system.ts (IGameSystem object literal with all extracted fields)
- README.md   (commands table, storage, REST routes, install)`;

// ─── Nodes ───────────────────────────────────────────────────────────────────

async function readFiles(state: State): Promise<Partial<State>> {
  console.log(`\n[chargen-extract] Reading text files from ${state.textDir}...`);
  const chunks: string[] = [];
  let total = 0;

  for await (const entry of walk(state.textDir, { exts: [".txt"], includeDirs: false })) {
    if (total >= MAX_CHARS) break;
    const rel = relative(state.textDir, entry.path);
    const content = await Deno.readTextFile(entry.path);
    const slice = content.slice(0, MAX_CHARS - total);
    chunks.push(`\n\n=== FILE: ${rel} ===\n${slice}`);
    total += slice.length;
    console.log(`  ✓ ${rel} (${content.length} chars${slice.length < content.length ? ", truncated" : ""})`);
  }

  if (chunks.length === 0) {
    throw new Error(`No .txt files found in ${state.textDir}`);
  }

  console.log(`  Total: ${total.toLocaleString()} chars from ${chunks.length} file(s)`);
  return { textContent: chunks.join("") };
}

async function extractMechanics(state: State): Promise<Partial<State>> {
  console.log("\n[chargen-extract] Extracting mechanics via Claude...");
  const raw = await llm(
    EXTRACT_SYSTEM,
    `Game system: ${state.systemName}\n\nRulebook text:\n${state.textContent}`,
  );

  const sep = raw.indexOf("---GM_CONTEXT---");
  const statSchema = sep === -1 ? raw : raw.slice(0, sep).trim();
  const gmContext  = sep === -1 ? "" : raw.slice(sep + "---GM_CONTEXT---".length).trim();

  console.log("  ✓ Stat schema extracted");
  console.log("  ✓ GM narrative context extracted");
  return { statSchema, gmContext };
}

async function confirmSchema(state: State): Promise<Partial<State>> {
  const display = [
    `\n${"=".repeat(70)}`,
    `MECHANICS SCHEMA: ${state.systemName}`,
    "=".repeat(70),
    state.statSchema,
    "\n--- GM CONTEXT ---",
    state.gmContext,
    "=".repeat(70),
    "\nReview the schema above.",
    "Press Enter to confirm, or type corrections to apply:",
  ].join("\n");

  const correction = interrupt(display) as string;

  if (correction && correction.trim()) {
    console.log("\n[chargen-extract] Applying corrections...");
    const corrected = await llm(
      "Update the mechanics schema based on the user's corrections. Return the full corrected schema.",
      `Current schema:\n${state.statSchema}\n\nGM context:\n${state.gmContext}\n\nCorrections:\n${correction}`,
    );
    const sep = corrected.indexOf("---GM_CONTEXT---");
    return {
      statSchema: sep === -1 ? corrected : corrected.slice(0, sep).trim(),
      gmContext: sep === -1 ? state.gmContext : corrected.slice(sep + "---GM_CONTEXT---".length).trim(),
    };
  }

  return {};
}

async function buildDesignPlan(state: State): Promise<Partial<State>> {
  console.log("\n[chargen-extract] Building design plan...");
  const designPlan = await llm(
    DESIGN_SYSTEM,
    `Game system: ${state.systemName}\n\nMechanics schema:\n${state.statSchema}\n\nGM context:\n${state.gmContext}`,
  );
  console.log("  ✓ Design plan generated");
  return { designPlan };
}

async function confirmDesign(state: State): Promise<Partial<State>> {
  const display = [
    `\n${"=".repeat(70)}`,
    `DESIGN PLAN: ${state.systemName} Chargen`,
    "=".repeat(70),
    state.designPlan,
    "=".repeat(70),
    "\nReview the design plan above.",
    "Press Enter to confirm, or type changes:",
  ].join("\n");

  const correction = interrupt(display) as string;

  if (correction && correction.trim()) {
    console.log("\n[chargen-extract] Revising design plan...");
    const revised = await llm(
      "Revise the design plan based on the user's changes. Return the full revised plan.",
      `Current plan:\n${state.designPlan}\n\nChanges:\n${correction}`,
    );
    return { designPlan: revised };
  }

  return {};
}

async function generatePlugin(state: State): Promise<Partial<State>> {
  const attempt = state.retryCount > 0 ? ` (retry ${state.retryCount})` : "";
  console.log(`\n[chargen-extract] Generating plugin code${attempt}...`);

  const auditFeedback = state.auditIssues.length > 0
    ? `\n\nPrevious audit issues to fix:\n${state.auditIssues.map((i) => `- ${i}`).join("\n")}`
    : "";

  const raw = await llm(
    GENERATE_SYSTEM,
    `Game system: ${state.systemName}

Mechanics schema:
${state.statSchema}

GM context:
${state.gmContext}

Design plan:
${state.designPlan}
${auditFeedback}

Output the JSON object with all 7 files.`,
  );

  // Extract JSON from potential markdown code fences
  const jsonMatch = raw.match(/```(?:json)?\s*([\s\S]+?)\s*```/) ?? [null, raw];
  const jsonStr = jsonMatch[1].trim();

  let pluginFiles: Record<string, string>;
  try {
    pluginFiles = JSON.parse(jsonStr);
  } catch {
    // If JSON parse fails, ask Claude to fix it
    const fixed = await llm(
      "Return only valid JSON, no other text. Fix the JSON syntax errors.",
      jsonStr.slice(0, 8000),
    );
    pluginFiles = JSON.parse(fixed);
  }

  const fileCount = Object.keys(pluginFiles).length;
  console.log(`  ✓ Generated ${fileCount} file(s): ${Object.keys(pluginFiles).join(", ")}`);
  return { pluginFiles, retryCount: state.retryCount + 1 };
}

function auditPlugin(state: State): Partial<State> {
  const issues: string[] = [];
  const files = state.pluginFiles;

  const check = (file: string, pattern: RegExp, msg: string) => {
    if (files[file] && !pattern.test(files[file])) issues.push(`${file}: ${msg}`);
  };

  // Critical checks
  check("index.ts", /gm:system:register/, "missing gm:system:register emit");
  check("index.ts", /registerStatSystem/, "missing registerStatSystem call");
  check("index.ts", /gameHooks\.off/, "missing gameHooks.off in remove()");
  check("commands.ts", /addCmd/, "no commands registered");
  check("validation.ts", /checkApproved/, "missing checkApproved function");
  check("game-system.ts", /coreRulesPrompt/, "missing coreRulesPrompt field");
  check("game-system.ts", /moveThresholds/, "missing moveThresholds field");
  check("schema.ts", /chargen\./, "DBO collection not namespaced with 'chargen.'");

  // Color code check
  if (files["display.ts"]) {
    const unresolved = files["display.ts"].match(/%c[a-z](?!.*%cn)/g);
    if (unresolved) issues.push("display.ts: color codes not closed with %cn");
  }

  if (issues.length > 0) {
    console.log(`\n[chargen-extract] Audit FAILED — ${issues.length} issue(s):`);
    issues.forEach((i) => console.log(`  ✗ ${i}`));
  } else {
    console.log("\n[chargen-extract] Audit PASSED ✓");
  }

  return { auditIssues: issues };
}

function shouldRetry(state: State): "generate" | "confirm_audit" | "write_files" {
  if (state.auditIssues.length === 0) return "write_files";
  if (state.retryCount < 2)           return "generate";
  return "confirm_audit";
}

async function confirmAudit(state: State): Promise<Partial<State>> {
  const display = [
    `\n${"=".repeat(70)}`,
    "AUDIT FAILED — Issues remain after retries:",
    ...state.auditIssues.map((i) => `  - ${i}`),
    "=".repeat(70),
    "Type 'proceed' to write files anyway, or press Enter to abort:",
  ].join("\n");

  const answer = interrupt(display) as string;
  // In CI mode the interrupt auto-resolves to "" — treat as proceed
  if (!CI && answer?.trim().toLowerCase() !== "proceed") {
    console.log("\n[chargen-extract] Aborted.");
    Deno.exit(1);
  }
  return {};
}

async function writeFiles(state: State): Promise<Partial<State>> {
  const pluginDir = join(state.outputDir, "src", "plugins", "chargen");
  await ensureDir(join(pluginDir, "tests"));

  console.log(`\n[chargen-extract] Writing plugin to ${pluginDir}...`);

  for (const [relPath, content] of Object.entries(state.pluginFiles)) {
    const absPath = relPath.startsWith("tests/")
      ? join(pluginDir, relPath)
      : join(pluginDir, relPath);
    await Deno.writeTextFile(absPath, content);
    console.log(`  ✓ ${relPath}`);
  }

  console.log(`\n✓ Chargen plugin written to ${pluginDir}`);
  console.log("  Next steps:");
  console.log("  1. Review generated files for accuracy");
  console.log("  2. Run: deno test src/plugins/chargen/tests/");
  console.log("  3. Add plugin to your UrsaMU instance");

  return {};
}

// ─── Graph ───────────────────────────────────────────────────────────────────

const graph = new StateGraph(ChargenState)
  .addNode("read_files",     readFiles)
  .addNode("extract",        extractMechanics)
  .addNode("confirm_schema", confirmSchema)
  .addNode("design",         buildDesignPlan)
  .addNode("confirm_design", confirmDesign)
  .addNode("generate",       generatePlugin)
  .addNode("audit",          auditPlugin)
  .addNode("confirm_audit",  confirmAudit)
  .addNode("write_files",    writeFiles)
  .addEdge(START,             "read_files")
  .addEdge("read_files",      "extract")
  .addEdge("extract",         "confirm_schema")
  .addEdge("confirm_schema",  "design")
  .addEdge("design",          "confirm_design")
  .addEdge("confirm_design",  "generate")
  .addEdge("generate",        "audit")
  .addConditionalEdges("audit", shouldRetry, {
    generate:      "generate",
    confirm_audit: "confirm_audit",
    write_files:   "write_files",
  })
  .addEdge("confirm_audit",   "write_files")
  .addEdge("write_files",     END);

const checkpointer = new MemorySaver();
const compiled     = graph.compile({ checkpointer });

// ─── CLI main ────────────────────────────────────────────────────────────────

const threadId = `chargen-${Date.now()}`;
const config   = { configurable: { thread_id: threadId } };

console.log(`\nUrsaMU Chargen Extractor — ${SYS_NAME}`);
console.log(`Text dir: ${TEXT_DIR}  |  Output: ${OUTPUT}  |  Model: ${MODEL_ID}  |  CI: ${CI}\n`);

// Run graph, handling interrupt checkpoints interactively
let input: unknown = { textDir: TEXT_DIR, outputDir: OUTPUT, systemName: SYS_NAME };

while (true) {
  let interrupted = false;
  let interruptValue = "";

  for await (const event of await compiled.stream(input as never, { ...config, streamMode: "values" })) {
    const pending = (event as Record<string, unknown>).__interrupt__;
    if (Array.isArray(pending) && pending.length > 0) {
      interruptValue = String((pending[0] as { value?: unknown }).value ?? "");
      interrupted = true;
      break;
    }
  }

  if (!interrupted) break;

  // In CI mode auto-approve; otherwise read from stdin
  console.log(interruptValue);
  let userInput = "";
  if (!CI) {
    const buf = new Uint8Array(4096);
    const n   = await Deno.stdin.read(buf);
    userInput = n ? new TextDecoder().decode(buf.subarray(0, n)).trim() : "";
  } else {
    console.log("[ci] auto-approved");
  }

  input = new Command({ resume: userInput });
}
