import fs from "node:fs";
import path from "node:path";
import i18n from "i18next";

/**
 * Initializes i18next for STT unit tests with the real English strings plus every staged workstream file
 * (`scripts/i18n-staging/*.json`, merged at INTEGRATE), so tests assert the texts users see and a missing key shows
 * up as its raw key. Not a test file (no `.test.` in the name).
 */
export async function initSttTestI18n(): Promise<void> {
  const root = path.resolve(__dirname, "../../../../..");
  const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8")) as Record<string, unknown>;
  const merge = (into: Record<string, unknown>, from: Record<string, unknown>) => {
    for (const [k, v] of Object.entries(from)) {
      if (k.startsWith("$")) continue;
      if (v && typeof v === "object" && !Array.isArray(v)) {
        const cur = into[k];
        const target = cur && typeof cur === "object" ? (cur as Record<string, unknown>) : {};
        into[k] = target;
        merge(target, v as Record<string, unknown>);
      } else if (!(k in into)) into[k] = v;
    }
  };
  const en = read(path.join(root, "public/i18n/en/translation.json"));
  const staging = path.join(root, "scripts/i18n-staging");
  for (const file of fs.readdirSync(staging).filter((f) => f.endsWith(".json")).sort())
    merge(en, read(path.join(staging, file)));
  await i18n.init({ lng: "en", resources: { en: { translation: en } }, interpolation: { escapeValue: false } });
}

/** True when `text` is a translated string rather than a raw `section.key`. */
export const translated = (text: unknown): boolean => typeof text === "string" && !/^[a-z_]+\.[a-z0-9_]+$/.test(text);
