/**
 * Generate src-tauri/model-catalog/mt-bergamot.json (the Bergamot translation models, one catalog entry per
 * direction) from Mozilla's model registry: the Remote Settings collection `main/translations-models` that Firefox
 * Translations downloads its models from. Owned by P4-E (docs/adr/0008-bergamot.md).
 *
 * What it picks (the desktop release channel, like Firefox on Windows):
 * - records whose `filter_expression` applies to Windows on the release channel (known expressions only; an unknown
 *   expression is skipped with a warning, so a new rule never slips an untested model in);
 * - no pre-release versions (`2.0a1`);
 * - per direction, the highest version; the entry's files are that version's `model` (intgemm int8 student),
 *   `lex` (shortlist) and `vocab`, or `srcvocab` + `trgvocab` for the CJK models with separate vocabularies.
 * Sizes and sha256 are the registry's attachment digests (Firefox verifies downloads against the same values); the
 * files are served, unchanged and immutable, from the attachments CDN.
 *
 * Ids are `mt-bergamot/<source>_<target>` with Sigil's canonical language codes lower-cased (`zh` = Simplified,
 * `zh-hant`), family `mt-bergamot` (its own group in Models & Runtimes), and every entry requires the helper runtime
 * `runtime/bergamot`. Every direction goes to or from English: other pairs pivot through English
 * (`mt_local/bergamot.rs`).
 *
 * Network: one request, only with `--fetch` (the records JSON, ~0.4 MB, cached in
 * %LOCALAPPDATA%\sigil-dev-cache\bergamot\translations-models-records.json). Nothing else is downloaded.
 *
 * Usage (repo root):
 *   node scripts/runtime-packs/bergamot/gen-catalog.mjs              # summary of what would be written (dry run)
 *   node scripts/runtime-packs/bergamot/gen-catalog.mjs --write      # write the fragment
 *   node scripts/runtime-packs/bergamot/gen-catalog.mjs --check      # exit 1 if the fragment is out of date
 *   node scripts/runtime-packs/bergamot/gen-catalog.mjs --fetch --write   # refresh the records first (1 request)
 * Options:
 *   --records <file>   records JSON (default: the dev-cache file above)
 *   --channel <c>      release (default) or nightly (also takes nightly-only models)
 *   --out <file>       fragment to write (default: src-tauri/model-catalog/mt-bergamot.json)
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.join(__dirname, "..", "..", "..");
export const DEFAULT_OUT = path.join(ROOT, "src-tauri", "model-catalog", "mt-bergamot.json");
export const RECORDS_URL = "https://firefox.settings.services.mozilla.com/v1/buckets/main/collections/translations-models/records";
/** `capabilities.attachments.base_url` of https://firefox.settings.services.mozilla.com/v1/ (checked 2026-09-30). */
export const ATTACHMENTS_BASE = "https://firefox-settings-attachments.cdn.mozilla.net/";
export const FAMILY = "mt-bergamot";
export const RUNTIME_ID = "runtime/bergamot";
const USER_AGENT = "Sigil-gen-catalog (https://github.com/Mewnah/Sigil)";

/** English names of the registry's languages (a new registry language must be added here). */
export const LANGUAGE_NAMES = {
  af: "Afrikaans",
  ar: "Arabic",
  az: "Azerbaijani",
  be: "Belarusian",
  bg: "Bulgarian",
  bn: "Bengali",
  bs: "Bosnian",
  ca: "Catalan",
  cs: "Czech",
  da: "Danish",
  de: "German",
  el: "Greek",
  en: "English",
  es: "Spanish",
  et: "Estonian",
  eu: "Basque",
  fa: "Persian",
  fi: "Finnish",
  fr: "French",
  gl: "Galician",
  gu: "Gujarati",
  he: "Hebrew",
  hi: "Hindi",
  hr: "Croatian",
  hu: "Hungarian",
  id: "Indonesian",
  is: "Icelandic",
  it: "Italian",
  ja: "Japanese",
  kn: "Kannada",
  ko: "Korean",
  lt: "Lithuanian",
  lv: "Latvian",
  ml: "Malayalam",
  mr: "Marathi",
  ms: "Malay",
  mt: "Maltese",
  nb: "Norwegian Bokmål",
  nl: "Dutch",
  nn: "Norwegian Nynorsk",
  pl: "Polish",
  pt: "Portuguese",
  ro: "Romanian",
  ru: "Russian",
  sk: "Slovak",
  sl: "Slovenian",
  sq: "Albanian",
  sr: "Serbian",
  sv: "Swedish",
  ta: "Tamil",
  te: "Telugu",
  th: "Thai",
  tr: "Turkish",
  uk: "Ukrainian",
  ur: "Urdu",
  vi: "Vietnamese",
  "zh-Hans": "Chinese (Simplified)",
  "zh-Hant": "Chinese (Traditional)",
};

/** Registry language code -> Sigil's canonical code (`translation/languages.ts`). */
export function canonicalCode(registryCode) {
  if (registryCode === "zh-Hans") return "zh";
  return registryCode;
}

/** Catalog id of one direction (canonical codes, lower-cased; `_` separates source and target). */
export function directionId(fromRegistry, toRegistry) {
  return `${FAMILY}/${canonicalCode(fromRegistry).toLowerCase()}_${canonicalCode(toRegistry).toLowerCase()}`;
}

/**
 * Whether a record's `filter_expression` applies to Firefox on Windows for `channel`. Only expressions seen in the
 * registry are understood; anything else returns null (the caller skips the record and warns).
 */
export function filterApplies(expression, channel = "release") {
  const expr = (expression ?? "").trim();
  if (expr === "") return true;
  const known = {
    "env.channel == 'default' || env.channel == 'nightly'": channel === "nightly",
    "env.channel == 'default' || env.channel == 'nightly' || env.channel == 'beta'": channel === "nightly",
    "env.appinfo.OS != 'Android' || env.channel != 'release'": true,
    "env.appinfo.OS == 'Android'": false,
  };
  return Object.prototype.hasOwnProperty.call(known, expr) ? known[expr] : null;
}

/** Parse `2.1`, `2.0a1` into a sortable key; null for anything else. */
export function parseVersion(v) {
  const m = /^(\d+)\.(\d+)(?:a(\d*))?$/.exec(String(v ?? ""));
  if (!m) return null;
  return { major: Number(m[1]), minor: Number(m[2]), prerelease: m[3] !== undefined, pre: Number(m[3] || 0) };
}

export function compareVersions(a, b) {
  const x = parseVersion(a);
  const y = parseVersion(b);
  if (!x || !y) throw new Error(`cannot compare versions ${a} and ${b}`);
  return (
    x.major - y.major ||
    x.minor - y.minor ||
    // A release sorts after its pre-releases.
    Number(!x.prerelease) - Number(!y.prerelease) ||
    x.pre - y.pre
  );
}

const FILE_ORDER = ["model", "lex", "vocab", "srcvocab", "trgvocab"];
const SAFE_NAME_RE = /^[A-Za-z0-9._+-]+$/;
const SHA256_RE = /^[0-9a-f]{64}$/;

/**
 * Pick one version per direction. Returns `{ directions: [{ from, to, version, files: [record] }], warnings }`,
 * sorted by source then target (canonical codes).
 */
export function selectDirections(records, { channel = "release" } = {}) {
  const warnings = [];
  const byDirection = new Map();
  for (const r of records) {
    const applies = filterApplies(r.filter_expression, channel);
    if (applies === null) {
      warnings.push(`skipped ${r.name} ${r.fromLang}-${r.toLang} ${r.version}: unknown filter expression ${JSON.stringify(r.filter_expression)}`);
      continue;
    }
    if (!applies) continue;
    const v = parseVersion(r.version);
    if (!v) {
      warnings.push(`skipped ${r.name}: unparseable version ${JSON.stringify(r.version)}`);
      continue;
    }
    if (v.prerelease && channel !== "nightly") continue;
    const key = `${r.fromLang}\u0000${r.toLang}`;
    if (!byDirection.has(key)) byDirection.set(key, new Map());
    const versions = byDirection.get(key);
    if (!versions.has(r.version)) versions.set(r.version, []);
    versions.get(r.version).push(r);
  }

  const directions = [];
  for (const [key, versions] of byDirection) {
    const [from, to] = key.split("\u0000");
    const ordered = [...versions.keys()].sort(compareVersions).reverse();
    let picked = null;
    for (const version of ordered) {
      const files = versions.get(version);
      const problem = checkFileSet(files);
      if (problem) {
        warnings.push(`skipped ${from}-${to} ${version}: ${problem}`);
        continue;
      }
      picked = { from, to, version, files: [...files].sort((a, b) => FILE_ORDER.indexOf(a.fileType) - FILE_ORDER.indexOf(b.fileType)) };
      break;
    }
    if (picked) directions.push(picked);
  }
  directions.sort((a, b) => {
    const fa = canonicalCode(a.from).toLowerCase();
    const fb = canonicalCode(b.from).toLowerCase();
    if (fa !== fb) return fa < fb ? -1 : 1;
    const ta = canonicalCode(a.to).toLowerCase();
    const tb = canonicalCode(b.to).toLowerCase();
    return ta < tb ? -1 : ta > tb ? 1 : 0;
  });
  return { directions, warnings };
}

/** Why a version's file set is unusable, or "" when it is complete. */
function checkFileSet(files) {
  const types = new Map();
  for (const f of files) {
    if (types.has(f.fileType)) return `two ${f.fileType} files`;
    types.set(f.fileType, f);
    if (!FILE_ORDER.includes(f.fileType)) return `unknown file type ${f.fileType}`;
    const a = f.attachment ?? {};
    if (!SAFE_NAME_RE.test(f.name ?? "")) return `unsafe file name ${JSON.stringify(f.name)}`;
    if (!SHA256_RE.test(a.hash ?? "")) return `${f.name}: no sha256`;
    if (!Number.isInteger(a.size) || a.size <= 0) return `${f.name}: no size`;
    if (typeof a.location !== "string" || !a.location || a.location.startsWith("/") || a.location.includes("..")) {
      return `${f.name}: bad attachment location`;
    }
  }
  if (!types.has("model")) return "no model file";
  if (!types.has("lex")) return "no shortlist (lex) file";
  const shared = types.has("vocab");
  const split = types.has("srcvocab") && types.has("trgvocab");
  if (shared === split) return "needs either vocab, or srcvocab and trgvocab";
  if (!/\.intgemm(\.alphas|8)\.bin$/.test(types.get("model").name)) return `model ${types.get("model").name} is not an intgemm int8 model`;
  return "";
}

function languageName(registryCode) {
  const name = LANGUAGE_NAMES[registryCode];
  if (!name) throw new Error(`no English name for the registry language ${JSON.stringify(registryCode)}: add it to LANGUAGE_NAMES`);
  return name;
}

/** The catalog entry of one selected direction. */
export function directionEntry(d) {
  const src = languageName(d.from);
  const tgt = languageName(d.to);
  const bytes = d.files.reduce((n, f) => n + f.attachment.size, 0);
  const mb = Math.max(1, Math.round(bytes / 1_000_000));
  return {
    id: directionId(d.from, d.to),
    family: FAMILY,
    kind: "model",
    version: d.version,
    title: `${src} → ${tgt} (Firefox Translations)`,
    description: `Offline ${src} to ${tgt} translation on the processor, no graphics memory. Mozilla's Firefox Translations model, version ${d.version}; about ${mb} MB.`,
    license: "MPL-2.0",
    status: "published",
    requires: [RUNTIME_ID],
    files: d.files.map((f) => ({
      path: f.name,
      size: f.attachment.size,
      sha256: f.attachment.hash,
      sources: [
        {
          kind: "direct",
          url: new URL(f.attachment.location, ATTACHMENTS_BASE).href,
          revision: `translations-models ${d.from}-${d.to} ${d.version}`,
        },
      ],
    })),
  };
}

/** The whole fragment (deterministic for the same records). */
export function buildFragment(records, { channel = "release" } = {}) {
  const { directions, warnings } = selectDirections(records, { channel });
  const entries = directions.map(directionEntry);
  const fragment = {
    $schema: "./_schema.json",
    $comment: [
      "Mozilla Bergamot (Firefox Translations) models for the sigil-bergamot helper (runtime/bergamot, mt_local/bergamot.rs). Owned by P4-E; decision record docs/adr/0008-bergamot.md.",
      `GENERATED by scripts/runtime-packs/bergamot/gen-catalog.mjs from Mozilla's Remote Settings collection main/translations-models (${channel} channel on Windows: the highest non-prerelease version whose filter expression applies). Do not edit by hand; re-run the script to update.`,
      "One entry per direction, all to or from English (other pairs pivot through English). Sizes and sha256 are the registry's attachment digests, the same values Firefox verifies; the files are served unchanged from Mozilla's attachments CDN (immutable, one request per file).",
      "Ids are mt-bergamot/<source>_<target> with Sigil's canonical language codes lower-cased (zh = Simplified Chinese, zh-hant = Traditional). Files: model.*.intgemm.alphas.bin (int8 student), lex.50.50.*.s2t.bin (shortlist), vocab.*.spm or srcvocab/trgvocab.*.spm.",
    ],
    entries,
  };
  return { fragment, directions, warnings };
}

export function fragmentText(fragment) {
  return `${JSON.stringify(fragment, null, 2)}\n`;
}

// ---------------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------------

function defaultRecordsPath() {
  const base = process.env.SIGIL_DEV_CACHE || path.join(process.env.LOCALAPPDATA || os.tmpdir(), "sigil-dev-cache");
  return path.join(base, "bergamot", "translations-models-records.json");
}

function parseArgs(argv) {
  const args = { write: false, check: false, fetch: false, channel: "release", records: defaultRecordsPath(), out: DEFAULT_OUT };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const value = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`${a} needs a value`);
      return v;
    };
    if (a === "--write") args.write = true;
    else if (a === "--check") args.check = true;
    else if (a === "--fetch") args.fetch = true;
    else if (a === "--records") args.records = value();
    else if (a === "--out") args.out = value();
    else if (a === "--channel") {
      args.channel = value();
      if (!["release", "nightly"].includes(args.channel)) throw new Error("--channel must be release or nightly");
    } else throw new Error(`unknown argument ${a}`);
  }
  return args;
}

async function fetchRecords(file) {
  console.error(`fetching ${RECORDS_URL} (one request)`);
  const res = await fetch(RECORDS_URL, { headers: { "User-Agent": USER_AGENT, Accept: "application/json" } });
  if (!res.ok) throw new Error(`HTTP ${res.status} from ${RECORDS_URL}`);
  const text = await res.text();
  JSON.parse(text); // refuse garbage before caching it
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(`${file}.partial`, text);
  fs.renameSync(`${file}.partial`, file);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.fetch) await fetchRecords(args.records);
  if (!fs.existsSync(args.records)) {
    throw new Error(`records not found: ${args.records} (run with --fetch once)`);
  }
  const doc = JSON.parse(fs.readFileSync(args.records, "utf8"));
  const records = Array.isArray(doc) ? doc : doc.data;
  if (!Array.isArray(records)) throw new Error("the records file has no data array");
  const { fragment, directions, warnings } = buildFragment(records, { channel: args.channel });
  for (const w of warnings) console.error(`warning: ${w}`);
  const bytes = directions.reduce((n, d) => n + d.files.reduce((m, f) => m + f.attachment.size, 0), 0);
  console.log(`${directions.length} directions, ${(bytes / 1e9).toFixed(2)} GB in total if every one were installed`);
  const text = fragmentText(fragment);
  if (args.check) {
    const current = fs.existsSync(args.out) ? fs.readFileSync(args.out, "utf8") : "";
    if (current.replace(/\r\n/g, "\n") !== text) {
      console.error(`${path.relative(ROOT, args.out)} is out of date: run with --write`);
      process.exit(1);
    }
    console.log(`${path.relative(ROOT, args.out)} is up to date`);
    return;
  }
  if (args.write) {
    fs.writeFileSync(args.out, text);
    console.log(`wrote ${path.relative(ROOT, args.out)}`);
  } else {
    for (const d of directions) console.log(`  ${directionId(d.from, d.to)}  ${d.version}  ${d.files.map((f) => f.name).join(", ")}`);
    console.log("dry run: pass --write to write the fragment");
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((e) => {
    console.error(e instanceof Error ? e.message : e);
    process.exit(1);
  });
}
