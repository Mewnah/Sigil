import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
// @ts-expect-error plain .mjs module shared with vite.config.ts and the release scripts
import { loadEnvClassification } from "../../../scripts/release/env-classification.mjs";

/**
 * Guard: no secret may be named in frontend code. Client secrets and API keys are compiled into the Rust
 * binary (src-tauri/build.rs, `oauth-token` and `google-fonts` plugins); the bundle, which the embedded web
 * server hands to the LAN, only gets the public names of scripts/release/env-classification.json.
 */

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const classification: { public: string[]; secret: string[] } = loadEnvClassification(ROOT);

const TEXT_EXT = new Set([".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".css", ".html", ".json", ".md"]);
/** What Vite always defines on import.meta.env. */
const BUILT_IN = new Set(["DEV", "PROD", "MODE", "BASE_URL", "SSR"]);

function walk(dir: string, out: string[] = []): string[] {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === "node_modules" || entry.name === "dist") continue;
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(p, out);
    else if (TEXT_EXT.has(path.extname(entry.name))) out.push(p);
  }
  return out;
}

/** This file names the pattern it searches for, so it is left out of the scan for that pattern. */
const THIS_FILE = "src/utils/__tests__/noFrontendSecrets.test.ts";

const rel = (p: string) => path.relative(ROOT, p).split(path.sep).join("/");

/** Frontend code: src/, the Vite config, the HTML shell and the static pages the web server serves. */
function frontendFiles(): string[] {
  const files = walk(path.join(ROOT, "src"));
  files.push(path.join(ROOT, "vite.config.ts"), path.join(ROOT, "index.html"));
  for (const entry of fs.readdirSync(path.join(ROOT, "public"), { withFileTypes: true })) {
    if (entry.isFile() && [".html", ".js"].includes(path.extname(entry.name))) files.push(path.join(ROOT, "public", entry.name));
  }
  return files.filter((f) => fs.existsSync(f));
}

describe("no secret is named in frontend code", () => {
  const files = frontendFiles();

  it("scans a real set of files", () => {
    expect(files.length).toBeGreaterThan(100);
    expect(files.map(rel)).toEqual(expect.arrayContaining(["vite.config.ts", "src/utils/integrationEnv.ts", "src/core/services/kick/index.ts"]));
    expect(classification.secret.length).toBeGreaterThanOrEqual(6);
  });

  it("src/, vite.config.ts, index.html and the static pages never reference a secret variable name", () => {
    const hits: string[] = [];
    for (const file of files) {
      const text = fs.readFileSync(file, "utf8");
      for (const name of classification.secret) {
        if (text.includes(name)) hits.push(`${rel(file)}: ${name}`);
      }
    }
    expect(hits).toEqual([]);
  });

  it("every import.meta.env.NAME in src/ is a public name or a Vite built-in", () => {
    const bad: string[] = [];
    for (const file of files.filter((f) => rel(f).startsWith("src/") && rel(f) !== THIS_FILE)) {
      const text = fs.readFileSync(file, "utf8");
      for (const m of text.matchAll(/import\.meta\.env\.([A-Za-z0-9_]+)/g)) {
        const name = m[1]!;
        if (!BUILT_IN.has(name) && !classification.public.includes(name)) bad.push(`${rel(file)}: ${name}`);
      }
    }
    expect(bad).toEqual([]);
  });

  it("every name passed to firstEnv / firstEnvRaw is public", () => {
    const text = fs.readFileSync(path.join(ROOT, "src/utils/integrationEnv.ts"), "utf8");
    const names = new Set<string>();
    for (const call of text.matchAll(/firstEnv(?:Raw)?\(([^)]*)\)/g)) {
      for (const lit of call[1]!.matchAll(/"([A-Z0-9_]+)"/g)) names.add(lit[1]!);
    }
    expect(names.size).toBeGreaterThan(0);
    expect([...names].filter((n) => !classification.public.includes(n))).toEqual([]);
    // The public names are all really read somewhere (the list is not padded).
    const unused = classification.public.filter((n) => !text.includes(`"${n}"`) && n !== "TAURI_DEBUG");
    expect(unused).toEqual([]);
  });

  it("src/ reads no process.env (it is not defined in the bundle)", () => {
    const hits = files
      .filter((f) => rel(f).startsWith("src/") && !rel(f).includes("__tests__") && /\.(ts|tsx)$/.test(f))
      .filter((f) => /\bprocess\.env\b/.test(fs.readFileSync(f, "utf8")))
      .map(rel);
    expect(hits).toEqual([]);
  });

  it("the HTML shell and static pages use no %ENV% placeholder for a non-public name", () => {
    const bad: string[] = [];
    for (const file of files.filter((f) => f.endsWith(".html"))) {
      const text = fs.readFileSync(file, "utf8");
      for (const m of text.matchAll(/%((?:VITE|SIGIL|CURSES|TAURI)_[A-Z0-9_]+)%/g)) {
        if (!classification.public.includes(m[1]!)) bad.push(`${rel(file)}: ${m[1]}`);
      }
    }
    expect(bad).toEqual([]);
  });
});

describe("vite.config.ts hands Vite the allow-list, not prefixes", () => {
  const text = fs.readFileSync(path.join(ROOT, "vite.config.ts"), "utf8");

  it("sets envPrefix from viteEnvPrefix(loadEnvClassification(...))", () => {
    expect(text).toMatch(/envPrefix:\s*viteEnvPrefix\(\s*loadEnvClassification\(/);
  });

  it("has no literal envPrefix array and no define block that could inject env values", () => {
    expect(text).not.toMatch(/envPrefix:\s*\[/);
    expect(text).not.toMatch(/^\s*define\s*:/m);
    expect(text).not.toMatch(/loadEnv\(/);
  });
});
