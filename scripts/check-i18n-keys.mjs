/**
 * Fail if any locale is missing or has extra keys vs public/i18n/en/translation.json,
 * or if a code listed in src/i18n.ts (i18nLanguages) has no translation.json.
 * Usage: node scripts/check-i18n-keys.mjs
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, "..");
const I18N = path.join(ROOT, "public", "i18n");
const I18N_TS = path.join(ROOT, "src", "i18n.ts");

function flatten(obj, prefix = "") {
  const out = {};
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v !== null && typeof v === "object" && !Array.isArray(v)) {
      Object.assign(out, flatten(v, key));
    } else {
      out[key] = v;
    }
  }
  return out;
}

function registeredLocaleCodes() {
  const src = fs.readFileSync(I18N_TS, "utf8");
  const codes = [...src.matchAll(/code:\s*'([^']+)'/g)].map((m) => m[1]);
  return [...new Set(codes)];
}

const enPath = path.join(I18N, "en", "translation.json");
const enFlat = flatten(JSON.parse(fs.readFileSync(enPath, "utf8")));
const enKeys = new Set(Object.keys(enFlat));

let failed = false;

for (const code of registeredLocaleCodes()) {
  const p = path.join(I18N, code, "translation.json");
  if (!fs.existsSync(p)) {
    console.error(`\nMissing locale file for registered code '${code}': ${path.relative(ROOT, p)}`);
    failed = true;
  }
}

for (const name of fs.readdirSync(I18N)) {
  if (name === "en") continue;
  const p = path.join(I18N, name, "translation.json");
  if (!fs.existsSync(p)) continue;
  const locFlat = flatten(JSON.parse(fs.readFileSync(p, "utf8")));
  const locKeys = new Set(Object.keys(locFlat));
  const missing = [...enKeys].filter((k) => !locKeys.has(k));
  const extra = [...locKeys].filter((k) => !enKeys.has(k));
  if (missing.length || extra.length) {
    failed = true;
    console.error(`\n[${name}]`);
    if (missing.length) console.error("  missing:", missing.join(", "));
    if (extra.length) console.error("  extra:", extra.join(", "));
  }
}

if (failed) {
  console.error("\ncheck-i18n-keys: FAILED");
  process.exit(1);
}
console.log("check-i18n-keys: OK");
