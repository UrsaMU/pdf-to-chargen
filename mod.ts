/**
 * @module @ursamu/chargen-tools
 *
 * CLI tools for UrsaMU chargen pipeline.
 *
 * ## Usage
 *
 * ### Convert PDFs to text
 * ```bash
 * deno run --allow-read --allow-write --allow-run \
 *   jsr:@ursamu/chargen-tools/pdf-to-text \
 *   <input-dir> <output-dir> [--flat] [--overwrite] [--lang eng]
 * ```
 *
 * ### Extract game mechanics and generate chargen plugin
 * ```bash
 * export ANTHROPIC_API_KEY=sk-...
 * deno run --allow-all \
 *   jsr:@ursamu/chargen-tools/chargen-extract \
 *   --text <text-dir> --output <ursamu-root> --system "Game Name"
 * ```
 *
 * ## Requirements
 * - `brew install poppler tesseract`  (for pdf-to-text)
 * - `ANTHROPIC_API_KEY` env var        (for chargen-extract)
 */

export const VERSION = "0.1.0";
export const PACKAGE = "@ursamu/chargen-tools";
