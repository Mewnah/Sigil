/**
 * Merge per-workstream English UI strings from scripts/i18n-staging/<WS-ID>.json into
 * public/i18n/en/translation.json. Non-destructive: it only ADDS keys to English and never touches
 * other locales (run `pnpm i18n:sync` afterwards to copy the new English keys into every locale).
 *
 * Staging file format: the same nested shape as public/i18n/en/translation.json, English strings only.
 * Optional top-level "$remove": ["section.key", ...] deletes keys from EVERY locale (the only
 * destructive operation, explicit per key). Keys starting with "$" are reserved.
 *
 * Conflicts fail the run (exit 1) and nothing is written:
 * - a staged key already exists in English with a different value (pass --allow-en-changes to
 *   overwrite English wording on purpose; other locales keep their translation),
 * - two staging files stage the same key with different values,
 * - a staged key would replace a section with a string or vice versa.
 *
 * Usage (repo root):
 *   node scripts/merge-i18n-staging.mjs [--dry-run] [--allow-en-changes] [--only P0-B,P0-D]
 *   pnpm i18n:sync && pnpm i18n:check
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, "..");
const I18N = path.join(ROOT, "public", "i18n");
const STAGING = path.join(__dirname, "i18n-staging");

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);

/** Throw on anything that is not a nested object of strings (plus the reserved "$remove"). */
export function validateStaging(name, doc) {
  if (!isObject(doc)) throw new Error(`${name}: top level must be an object`);
  const walk = (obj, prefix) => {
    for (const [k, v] of Object.entries(obj)) {
      const key = prefix ? `${prefix}.${k}` : k;
      if (!prefix && k === "$remove") {
        if (!Array.isArray(v) || !v.every((x) => typeof x === "string" && x.length > 0)) {
          throw new Error(`${name}: "$remove" must be an array of dotted key strings`);
        }
        continue;
      }
      if (k.length === 0 || k.includes(".") || k.startsWith("$")) throw new Error(`${name}: invalid key "${key}"`);
      if (isObject(v)) walk(v, key);
      else if (typeof v !== "string") throw new Error(`${name}: "${key}" must be a string or an object`);
    }
  };
  walk(doc, "");
}

/** Flatten a nested strings object to dotted keys. */
export function flatten(obj, prefix = "", out = {}) {
  for (const [k, v] of Object.entries(obj)) {
    if (!prefix && k === "$remove") continue;
    const key = prefix ? `${prefix}.${k}` : k;
    if (isObject(v)) flatten(v, key, out);
    else out[key] = v;
  }
  return out;
}

/** Delete a dotted key; prunes sections left empty. Returns whether something was removed. */
export function removeKey(obj, dotted) {
  const parts = dotted.split(".");
  const stack = [];
  let cur = obj;
  for (const p of parts.slice(0, -1)) {
    if (!isObject(cur[p])) return false;
    stack.push([cur, p]);
    cur = cur[p];
  }
  const last = parts[parts.length - 1];
  if (!(last in cur)) return false;
  delete cur[last];
  for (let i = stack.length - 1; i >= 0; i--) {
    const [parent, key] = stack[i];
    if (Object.keys(parent[key]).length === 0) delete parent[key];
    else break;
  }
  return true;
}

/**
 * Pure merge. `staging` is `[{ name, doc }]` in merge order. Returns the new English object (a deep
 * copy) plus a report; `conflicts` non-empty means the caller must not write anything.
 */
export function mergeStaging(en, staging, { allowEnChanges = false } = {}) {
  const result = structuredClone(en);
  const added = [];
  const changed = [];
  const conflicts = [];
  const removals = [];
  /** dotted key -> { value, from } across staging files */
  const seen = new Map();

  for (const { name, doc } of staging) {
    validateStaging(name, doc);
    for (const key of doc.$remove ?? []) removals.push({ key, from: name });
    for (const [key, value] of Object.entries(flatten(doc))) {
      const prior = seen.get(key);
      if (prior && prior.value !== value) {
        conflicts.push(`${key}: staged as ${JSON.stringify(prior.value)} by ${prior.from} and ${JSON.stringify(value)} by ${name}`);
        continue;
      }
      seen.set(key, { value, from: name });

      const parts = key.split(".");
      let cur = result;
      let blocked = false;
      for (let i = 0; i < parts.length - 1; i++) {
        const p = parts[i];
        if (cur[p] === undefined) cur[p] = {};
        else if (!isObject(cur[p])) {
          conflicts.push(`${key}: ${parts.slice(0, i + 1).join(".")} is a string in English, not a section (${name})`);
          blocked = true;
          break;
        }
        cur = cur[p];
      }
      if (blocked) continue;
      const last = parts[parts.length - 1];
      const existing = cur[last];
      if (existing === undefined) {
        cur[last] = value;
        added.push(key);
      } else if (isObject(existing)) {
        conflicts.push(`${key}: is a section in English, staged as a string (${name})`);
      } else if (existing !== value) {
        if (allowEnChanges) {
          cur[last] = value;
          changed.push(key);
        } else {
          conflicts.push(`${key}: English has ${JSON.stringify(existing)}, ${name} stages ${JSON.stringify(value)} (use --allow-en-changes)`);
        }
      }
    }
  }
  for (const { key } of removals) {
    if (seen.has(key)) conflicts.push(`${key}: both staged and listed in "$remove"`);
  }
  return { en: result, added, changed, conflicts, removals };
}

function readJson(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function writeJson(p, obj) {
  fs.writeFileSync(p, JSON.stringify(obj, null, 2) + "\n", "utf8");
}

function main(argv) {
  const dryRun = argv.includes("--dry-run");
  const allowEnChanges = argv.includes("--allow-en-changes");
  const onlyIdx = argv.indexOf("--only");
  const only = onlyIdx >= 0 ? new Set((argv[onlyIdx + 1] ?? "").split(",").filter(Boolean)) : null;

  if (!fs.existsSync(STAGING)) {
    console.log("merge-i18n-staging: no scripts/i18n-staging directory; nothing to do");
    return 0;
  }
  const files = fs
    .readdirSync(STAGING)
    .filter((f) => f.endsWith(".json"))
    .filter((f) => !only || only.has(path.basename(f, ".json")))
    .sort();
  if (files.length === 0) {
    console.log("merge-i18n-staging: no staging files; nothing to do");
    return 0;
  }
  const staging = files.map((f) => ({ name: f, doc: readJson(path.join(STAGING, f)) }));
  const enPath = path.join(I18N, "en", "translation.json");
  const { en, added, changed, conflicts, removals } = mergeStaging(readJson(enPath), staging, { allowEnChanges });

  if (conflicts.length) {
    console.error("merge-i18n-staging: FAILED, nothing written:");
    for (const c of conflicts) console.error("  " + c);
    return 1;
  }
  console.log(`merge-i18n-staging: ${files.length} file(s), ${added.length} key(s) added, ${changed.length} changed, ${removals.length} removal(s)`);
  for (const k of added) console.log("  + " + k);
  for (const k of changed) console.log("  ~ " + k);
  if (dryRun) {
    for (const { key, from } of removals) console.log(`  - ${key} (${from}, all locales)`);
    console.log("merge-i18n-staging: dry run, nothing written");
    return 0;
  }

  for (const { key } of removals) removeKey(en, key);
  writeJson(enPath, en);
  for (const name of fs.readdirSync(I18N)) {
    if (name === "en" || removals.length === 0) continue;
    const p = path.join(I18N, name, "translation.json");
    if (!fs.existsSync(p)) continue;
    const loc = readJson(p);
    let touched = false;
    for (const { key } of removals) touched = removeKey(loc, key) || touched;
    if (touched) writeJson(p, loc);
  }
  for (const { key, from } of removals) console.log(`  - ${key} (${from}, all locales)`);
  console.log("merge-i18n-staging: done. Next: pnpm i18n:sync && pnpm i18n:check");
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  process.exit(main(process.argv.slice(2)));
}
