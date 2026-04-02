/**
 * List translation keys where a locale value still matches English (likely untranslated sync fill).
 * Usage: node scripts/i18n-report-untranslated.mjs [--locale de] [--max 80]
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, "..");
const I18N = path.join(ROOT, "public", "i18n");

function flatten(obj, prefix = "") {
  const out = {};
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v !== null && typeof v === "object" && !Array.isArray(v)) Object.assign(out, flatten(v, key));
    else out[key] = v;
  }
  return out;
}

/** Strip {{x}} and <code>…</code> for loose compare (optional). */
function normalizeForCompare(s) {
  if (typeof s !== "string") return s;
  return s.replace(/\{\{[^}]+\}\}/g, "{{}}").replace(/<code>[^<]*<\/code>/gi, "<code></code>");
}

const raw = process.argv.slice(2).filter((a) => a !== "--");
let onlyLocale = null;
let maxPerLocale = 200;
for (let i = 0; i < raw.length; i++) {
  if (raw[i] === "--locale" && raw[i + 1]) {
    onlyLocale = raw[i + 1];
    i++;
  } else if (raw[i] === "--max" && raw[i + 1]) {
    maxPerLocale = Math.max(1, parseInt(raw[i + 1], 10) || 200);
    i++;
  }
}

const enPath = path.join(I18N, "en", "translation.json");
const enFlat = flatten(JSON.parse(fs.readFileSync(enPath, "utf8")));

for (const name of fs.readdirSync(I18N).sort()) {
  if (name === "en") continue;
  if (onlyLocale && name !== onlyLocale) continue;
  const p = path.join(I18N, name, "translation.json");
  if (!fs.existsSync(p)) continue;
  const locFlat = flatten(JSON.parse(fs.readFileSync(p, "utf8")));
  const same = [];
  for (const k of Object.keys(enFlat)) {
    const ev = enFlat[k];
    const lv = locFlat[k];
    if (typeof ev !== "string" || typeof lv !== "string") continue;
    if (ev === lv || normalizeForCompare(ev) === normalizeForCompare(lv)) same.push(k);
  }
  const shown = same.slice(0, maxPerLocale);
  console.log(`\n[${name}] ${same.length} key(s) identical to en (showing ${shown.length})`);
  for (const k of shown) console.log(`  ${k}`);
  if (same.length > shown.length) console.log(`  … +${same.length - shown.length} more`);
}

console.log("\nTip: refresh MT locales with `pnpm i18n:translate` (slow) or `python scripts/build-translations.py --only de,fr`.");