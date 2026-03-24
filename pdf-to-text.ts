#!/usr/bin/env -S deno run --allow-read --allow-write --allow-run
/**
 * pdf-to-text.ts — Walk a directory for PDFs and convert each to a .txt file.
 *
 * Auto-detects digital PDFs (vector text) vs scanned PDFs (image-based) and
 * routes to the appropriate extraction pipeline.
 *
 * Requirements:
 *   Digital:  pdftotext + pdfinfo  (poppler)   — brew install poppler
 *   Scanned:  + pdftoppm           (poppler)
 *             + tesseract          (OCR)        — brew install tesseract
 *
 * Usage:
 *   deno run --allow-read --allow-write --allow-run tools/pdf-to-text.ts \
 *     <input-dir> <output-dir> [--flat] [--overwrite] [--lang <lang>]
 *
 * Flags:
 *   --flat        All .txt files go directly into <output-dir> (no subdirs)
 *   --overwrite   Re-convert files that already exist in <output-dir>
 *   --lang <l>    Tesseract language pack (default: eng)
 *
 * Detection threshold:
 *   If pdftotext output < 150 bytes/page → classified as scanned → OCR pipeline
 */

import { walk }      from "@std/fs/walk";
import { ensureDir } from "@std/fs/ensure-dir";
import { exists }    from "@std/fs/exists";
import { join, relative, dirname, basename, extname } from "@std/path";

// ─── CLI ────────────────────────────────────────────────────────────────────

const positional  = Deno.args.filter((a) => !a.startsWith("--"));
const inputDir    = positional[0];
const outputDir   = positional[1];
const flat        = Deno.args.includes("--flat");
const overwrite   = Deno.args.includes("--overwrite");
const langIdx     = Deno.args.indexOf("--lang");
const lang        = langIdx !== -1 ? (Deno.args[langIdx + 1] ?? "eng") : "eng";

if (!inputDir || !outputDir) {
  console.error("Usage: pdf-to-text <input-dir> <output-dir> [--flat] [--overwrite] [--lang <lang>]");
  Deno.exit(1);
}

// ─── Dependency checks ──────────────────────────────────────────────────────

async function toolExists(name: string): Promise<boolean> {
  try {
    const cmd = new Deno.Command(name, { args: ["-v"], stderr: "piped", stdout: "piped" });
    await cmd.output();
    return true;
  } catch {
    return false;
  }
}

async function checkDependencies(): Promise<{ hasOcr: boolean }> {
  const [hasPdftotext, hasPdfinfo, hasPdftoppm, hasTesseract] = await Promise.all([
    toolExists("pdftotext"),
    toolExists("pdfinfo"),
    toolExists("pdftoppm"),
    toolExists("tesseract"),
  ]);

  const missing: string[] = [];
  if (!hasPdftotext) missing.push("pdftotext");
  if (!hasPdfinfo)   missing.push("pdfinfo");
  if (!hasPdftoppm)  missing.push("pdftoppm");

  if (missing.length > 0) {
    console.error(`Error: missing required tools: ${missing.join(", ")}`);
    console.error("Install with:  brew install poppler       (macOS)");
    console.error("           or: apt-get install poppler-utils  (Debian/Ubuntu)");
    Deno.exit(1);
  }

  if (!hasTesseract) {
    console.warn("Warning: tesseract not found — scanned PDFs will fail.");
    console.warn("Install with:  brew install tesseract");
  }

  return { hasOcr: hasTesseract };
}

// ─── Page count ─────────────────────────────────────────────────────────────

async function getPageCount(pdfPath: string): Promise<number> {
  const cmd = new Deno.Command("pdfinfo", {
    args: [pdfPath],
    stdout: "piped",
    stderr: "piped",
  });
  const { stdout } = await cmd.output();
  const info  = new TextDecoder().decode(stdout);
  const match = info.match(/^Pages:\s+(\d+)/m);
  return match ? parseInt(match[1], 10) : 1;
}

// ─── Detection ──────────────────────────────────────────────────────────────

// Returns "digital" and a temp .txt path, or "scanned" with tmpTxt deleted.
async function detectAndExtractDigital(
  pdfPath: string,
  tmpDir: string,
): Promise<{ type: "digital"; tmpTxt: string } | { type: "scanned" }> {
  const tmpTxt = join(tmpDir, "detect.txt");

  const cmd = new Deno.Command("pdftotext", {
    args: ["-layout", "-enc", "UTF-8", pdfPath, tmpTxt],
    stderr: "piped",
    stdout: "piped",
  });
  const { success } = await cmd.output();
  if (!success) return { type: "scanned" };

  const stat  = await Deno.stat(tmpTxt).catch(() => null);
  const bytes = stat?.size ?? 0;
  const pages = await getPageCount(pdfPath);

  if (bytes / pages < 150) {
    await Deno.remove(tmpTxt).catch(() => {});
    return { type: "scanned" };
  }

  return { type: "digital", tmpTxt };
}

// ─── OCR pipeline ───────────────────────────────────────────────────────────

async function convertOcr(
  pdfPath: string,
  outPath: string,
  tmpDir: string,
  ocrLang: string,
): Promise<void> {
  const pageBase = join(tmpDir, "page");

  // 1. Rasterise pages to 300 DPI PNG
  const raster = new Deno.Command("pdftoppm", {
    args: ["-r", "300", "-png", pdfPath, pageBase],
    stderr: "piped",
    stdout: "piped",
  });
  const { success: rasterOk, stderr: rasterErr } = await raster.output();
  if (!rasterOk) {
    throw new Error(new TextDecoder().decode(rasterErr).trim());
  }

  // 2. Collect page images in order
  const pageImages: string[] = [];
  for await (const entry of walk(tmpDir, { exts: [".png"], includeDirs: false })) {
    pageImages.push(entry.path);
  }
  pageImages.sort();

  if (pageImages.length === 0) throw new Error("pdftoppm produced no images");

  // 3. OCR each page
  const pageTexts: string[] = [];
  for (const img of pageImages) {
    const txtBase = img.replace(/\.png$/, "");
    const ocr = new Deno.Command("tesseract", {
      args: [img, txtBase, "-l", ocrLang, "--psm", "3"],
      stderr: "piped",
      stdout: "piped",
    });
    const { success: ocrOk, stderr: ocrErr } = await ocr.output();
    if (!ocrOk) {
      throw new Error(new TextDecoder().decode(ocrErr).trim());
    }
    pageTexts.push(txtBase + ".txt");
  }

  // 4. Concatenate pages, stripping form-feed chars tesseract adds
  const chunks: string[] = [];
  for (const txt of pageTexts) {
    const raw = await Deno.readTextFile(txt);
    chunks.push(raw.replaceAll("\f", ""));
  }

  await Deno.writeTextFile(outPath, chunks.join("\n"));
}

// ─── Per-file conversion ────────────────────────────────────────────────────

type Result = "digital" | "ocr" | "skipped" | "failed";

async function convertOne(opts: {
  relPath: string;
  inPath: string;
  outPath: string;
  ocrLang: string;
  hasOcr: boolean;
}): Promise<Result> {
  const { relPath, inPath, outPath, ocrLang, hasOcr } = opts;

  await ensureDir(dirname(outPath));

  const tmpDir = await Deno.makeTempDir({ prefix: "pdf2txt-" });
  try {
    const detection = await detectAndExtractDigital(inPath, tmpDir);

    if (detection.type === "digital") {
      await Deno.rename(detection.tmpTxt, outPath);
      console.log(`✓ digital  ${relPath}`);
      return "digital";
    }

    // Scanned path
    if (!hasOcr) {
      console.error(`✗ failed   ${relPath}: scanned PDF but tesseract is not installed`);
      return "failed";
    }

    const t0 = Date.now();
    await convertOcr(inPath, outPath, tmpDir, ocrLang);
    const pages = await getPageCount(inPath);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`✓ ocr      ${relPath}  (${pages} pages, ${elapsed}s)`);
    return "ocr";
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error(`✗ failed   ${relPath}: ${msg}`);
    return "failed";
  } finally {
    await Deno.remove(tmpDir, { recursive: true }).catch(() => {});
  }
}

// ─── Main ───────────────────────────────────────────────────────────────────

const { hasOcr } = await checkDependencies();
await ensureDir(outputDir);

let digital = 0, ocr = 0, skipped = 0, failed = 0;

for await (const entry of walk(inputDir, {
  match: [/\.pdf$/i],
  includeDirs: false,
})) {
  const relPath  = relative(inputDir, entry.path);
  const baseName = basename(entry.path, extname(entry.path)) + ".txt";
  const outPath  = flat
    ? join(outputDir, baseName)
    : join(outputDir, dirname(relPath), baseName);

  if (!overwrite && (await exists(outPath))) {
    console.log(`⏭  skip     ${relPath}`);
    skipped++;
    continue;
  }

  const result = await convertOne({
    relPath, inPath: entry.path, outPath, ocrLang: lang, hasOcr,
  });

  if (result === "digital") digital++;
  else if (result === "ocr") ocr++;
  else failed++;
}

console.log(
  `\nDone — ${digital + ocr} converted (${digital} digital, ${ocr} ocr), ${skipped} skipped, ${failed} failed`,
);
console.log(`Output: ${outputDir}`);

if (failed > 0) Deno.exit(1);
