#!/usr/bin/env node
/**
 * Release guard: no secret VALUE may appear in the frontend build or the binaries.
 *
 *   node scripts/release/check-no-secrets.mjs [path ...] [options]
 *
 * It collects the values of every variable whose name matches /SECRET|API_KEY|TOKEN|PASSWORD/i (and every
 * name classified `secret` in scripts/release/env-classification.json) from `.env`, `.env.local` and the
 * process environment, then looks for those values, as plain text, in `dist/` and in every `path` given
 * (Sigil.exe, an extracted installer folder, a copied bundle). A secret is allowed in exactly one place:
 * compiled into Sigil.exe by src-tauri/build.rs, XOR-obfuscated, so it is not a plain-text hit there.
 *
 * Each value is searched as UTF-8, UTF-16LE (Windows string tables, NSIS), JSON-escaped and URL-encoded.
 * Files are read in chunks, so a large executable is fine. An installer that is not extracted (a compressed
 * NSIS / MSI payload) can only be searched as the bytes it is: extract it first (verify-bundle.ps1 does).
 *
 * It prints VARIABLE NAMES and file paths only, never a value or any part of one.
 *
 * Options:
 *   --root <dir>        repository root (default: this script's repository)
 *   --env-file <file>   another dotenv file to read (repeatable); `.env` and `.env.local` are always read
 *   --no-process-env    ignore the process environment (only the dotenv files)
 *   --no-dist           do not scan `<root>/dist` (only the given paths)
 *   --require-values    exit 1 when there is no secret value to look for (a CI run that was given no
 *                       secrets would otherwise pass without checking anything)
 *
 * Exit codes: 0 no leak; 1 a leak, or --require-values without values; 2 usage error or a path that is
 * missing or unreadable.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SECRET_NAME_PATTERN, loadEnvClassification } from "./env-classification.mjs";

/** Values shorter than this are not searched: they would match by chance in a bundle. Real secrets are 20+. */
export const MIN_VALUE_LENGTH = 8;
const DEFAULT_CHUNK = 8 * 1024 * 1024;

/** `KEY=VALUE` pairs of a dotenv file (same shapes as src-tauri/embed_codec.rs parse_dotenv). */
export function parseDotenv(text) {
  const pairs = [];
  for (const raw of text.replace(/^﻿/, "").split(/\r?\n/)) {
    let line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    if (line.startsWith("export ")) line = line.slice(7).trimStart();
    const eq = line.indexOf("=");
    if (eq < 0) continue;
    const key = line.slice(0, eq).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) continue;
    pairs.push([key, parseValue(line.slice(eq + 1).trim())]);
  }
  return pairs;
}

function parseValue(value) {
  for (const quote of ['"', "'", "`"]) {
    if (value.startsWith(quote)) {
      const end = value.indexOf(quote, 1);
      if (end > 0) return value.slice(1, end);
    }
  }
  const comment = value.indexOf(" #");
  return comment >= 0 ? value.slice(0, comment).trimEnd() : value;
}

/** Whether a variable name is one whose value must never be in a build. */
export function isSecretName(name, classification) {
  return SECRET_NAME_PATTERN.test(name) || classification.secret.includes(name);
}

/**
 * The secret values to look for: `Map<name, Set<value>>`, plus the names skipped for being too short.
 * Sources: dotenv files (that exist) and, unless disabled, the process environment.
 */
export function collectSecrets({ envFiles, env, classification }) {
  const values = new Map();
  const tooShort = new Set();
  const add = (name, value) => {
    if (!isSecretName(name, classification)) return;
    const v = String(value ?? "").trim();
    if (!v) return;
    if (v.length < MIN_VALUE_LENGTH) {
      tooShort.add(name);
      return;
    }
    if (!values.has(name)) values.set(name, new Set());
    values.get(name).add(v);
  };
  for (const file of envFiles) {
    let text;
    try {
      text = fs.readFileSync(file, "utf8");
    } catch {
      continue;
    }
    for (const [name, value] of parseDotenv(text)) add(name, value);
  }
  if (env) for (const [name, value] of Object.entries(env)) add(name, value);
  return { values, tooShort: [...tooShort].filter((n) => !values.has(n)).sort() };
}

/** The byte patterns to search for one value: `{ form, buf }`, de-duplicated. */
export function needlesFor(value) {
  const candidates = [
    ["utf8", Buffer.from(value, "utf8")],
    ["utf16le", Buffer.from(value, "utf16le")],
    ["json", Buffer.from(JSON.stringify(value).slice(1, -1), "utf8")],
    ["url", Buffer.from(encodeURIComponent(value), "utf8")],
  ];
  const seen = new Set();
  const out = [];
  for (const [form, buf] of candidates) {
    const key = buf.toString("hex");
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ form, buf });
  }
  return out;
}

/** Every file under `p` (a file gives itself); symbolic links are not followed. */
export function listFiles(p) {
  const st = fs.lstatSync(p);
  if (st.isSymbolicLink()) return [];
  if (st.isFile()) return [p];
  if (!st.isDirectory()) return [];
  const out = [];
  for (const entry of fs.readdirSync(p, { withFileTypes: true })) {
    if (entry.isSymbolicLink()) continue;
    out.push(...listFiles(path.join(p, entry.name)));
  }
  return out;
}

/**
 * Search one file for the needles, in chunks that overlap by the longest needle so a value on a chunk
 * boundary is still found. Returns the `needle` entries that occur.
 */
export function scanFile(file, needles, chunkSize = DEFAULT_CHUNK) {
  if (needles.length === 0) return [];
  const overlap = Math.max(...needles.map((n) => n.buf.length)) - 1;
  const fd = fs.openSync(file, "r");
  try {
    const buf = Buffer.alloc(chunkSize + overlap);
    const hits = new Set();
    let carry = 0;
    for (;;) {
      const n = fs.readSync(fd, buf, carry, chunkSize, null);
      if (n === 0) break;
      const view = buf.subarray(0, carry + n);
      for (const needle of needles) {
        if (!hits.has(needle) && view.indexOf(needle.buf) !== -1) hits.add(needle);
      }
      carry = Math.min(overlap, view.length);
      if (carry > 0) view.copy(buf, 0, view.length - carry, view.length);
    }
    return [...hits];
  } finally {
    fs.closeSync(fd);
  }
}

/**
 * Scan `roots` for the secret values. Returns `{ files, leaks }` where each leak is
 * `{ name, form, file }` (names and paths only).
 */
export function scanPaths(roots, secretValues, { chunkSize = DEFAULT_CHUNK } = {}) {
  const needles = [];
  for (const [name, set] of secretValues) {
    for (const value of set) for (const n of needlesFor(value)) needles.push({ ...n, name });
  }
  const leaks = [];
  let files = 0;
  for (const root of roots) {
    for (const file of listFiles(root)) {
      files++;
      for (const hit of scanFile(file, needles, chunkSize)) {
        leaks.push({ name: hit.name, form: hit.form, file });
      }
    }
  }
  return { files, leaks };
}

export function parseArgs(argv) {
  const opts = { paths: [], envFiles: [], processEnv: true, dist: true, requireValues: false, root: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--root") opts.root = argv[++i];
    else if (a === "--env-file") opts.envFiles.push(argv[++i]);
    else if (a === "--no-process-env") opts.processEnv = false;
    else if (a === "--no-dist") opts.dist = false;
    else if (a === "--require-values") opts.requireValues = true;
    else if (a.startsWith("--")) throw new Error(`unknown option ${a}`);
    else opts.paths.push(a);
  }
  if (opts.root === undefined || opts.envFiles.includes(undefined)) throw new Error("an option is missing its value");
  return opts;
}

/**
 * The CLI. `io` carries `{ env, stdout, stderr, cwd }` so tests can run it without a process. Returns the
 * exit code.
 */
export function main(argv, io = {}) {
  const env = io.env ?? process.env;
  const out = io.stdout ?? ((s) => process.stdout.write(s));
  const err = io.stderr ?? ((s) => process.stderr.write(s));
  const cwd = io.cwd ?? process.cwd();

  let opts;
  try {
    opts = parseArgs(argv);
  } catch (e) {
    err(`check-no-secrets: ${e.message}\n`);
    return 2;
  }
  const root = path.resolve(cwd, opts.root ?? path.join(path.dirname(fileURLToPath(import.meta.url)), "..", ".."));
  let classification;
  try {
    classification = loadEnvClassification(root);
  } catch (e) {
    err(`check-no-secrets: cannot read the env classification under ${root}: ${e.message}\n`);
    return 2;
  }

  const roots = [];
  if (opts.dist) roots.push(path.join(root, "dist"));
  for (const p of opts.paths) roots.push(path.resolve(cwd, p));
  if (roots.length === 0) {
    err("check-no-secrets: nothing to scan (give a path, or drop --no-dist)\n");
    return 2;
  }
  for (const r of roots) {
    if (!fs.existsSync(r)) {
      err(`check-no-secrets: ${r} does not exist${r === path.join(root, "dist") ? " (run pnpm build first)" : ""}\n`);
      return 2;
    }
  }

  const envFiles = [path.join(root, ".env"), path.join(root, ".env.local"), ...opts.envFiles.map((f) => path.resolve(cwd, f))];
  const { values, tooShort } = collectSecrets({ envFiles, env: opts.processEnv ? env : null, classification });
  const names = [...values.keys()].sort();

  if (tooShort.length > 0) {
    out(`note: ${tooShort.length} secret-named variable(s) have values under ${MIN_VALUE_LENGTH} characters and are not checked: ${tooShort.join(", ")}\n`);
  }
  if (names.length === 0) {
    out("no secret values found in .env, .env.local or the environment: nothing to compare\n");
    if (opts.requireValues) {
      err("check-no-secrets: FAILED, --require-values was given but there is nothing to check (set the build secrets in the environment)\n");
      return 1;
    }
    return 0;
  }

  let result;
  try {
    result = scanPaths(roots, values);
  } catch (e) {
    err(`check-no-secrets: could not read a file: ${e.code ?? e.message}\n`);
    return 2;
  }
  out(`checked ${names.length} secret variable(s) (${names.join(", ")}) against ${result.files} file(s) in ${roots.length} location(s)\n`);
  if (result.leaks.length > 0) {
    const leakedNames = [...new Set(result.leaks.map((l) => l.name))].sort();
    for (const leak of result.leaks) {
      out(`LEAK: ${leak.name} (${leak.form}) in ${path.relative(root, leak.file) || leak.file}\n`);
    }
    err(`check-no-secrets: FAILED, the value of ${leakedNames.join(", ")} is in the build. Secrets are compiled into the Rust binary only; rotate them if this build was shared.\n`);
    return 1;
  }
  out("OK: no secret value found\n");
  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = main(process.argv.slice(2));
}
