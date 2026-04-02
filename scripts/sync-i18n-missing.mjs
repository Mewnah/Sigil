/**
 * Deep-merge missing keys from public/i18n/en/translation.json into every other locale file.
 * New keys copy the English string until translators update them.
 */
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, "..");
const I18N = path.join(ROOT, "public", "i18n");

function deepMergeMissing(target, source) {
  if (source === null || typeof source !== "object" || Array.isArray(source)) return target;
  for (const k of Object.keys(source)) {
    const sv = source[k];
    if (sv !== null && typeof sv === "object" && !Array.isArray(sv)) {
      if (!target[k] || typeof target[k] !== "object" || Array.isArray(target[k])) target[k] = {};
      deepMergeMissing(target[k], sv);
    } else if (target[k] === undefined) {
      target[k] = sv;
    }
  }
  return target;
}

const enPath = path.join(I18N, "en", "translation.json");
const en = JSON.parse(fs.readFileSync(enPath, "utf8"));

for (const name of fs.readdirSync(I18N)) {
  if (name === "en") continue;
  const p = path.join(I18N, name, "translation.json");
  if (!fs.existsSync(p)) continue;
  const cur = JSON.parse(fs.readFileSync(p, "utf8"));
  deepMergeMissing(cur, en);
  fs.writeFileSync(p, JSON.stringify(cur, null, 2) + "\n", "utf8");
  console.log("synced", name);
}
