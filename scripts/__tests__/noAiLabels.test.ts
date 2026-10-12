/**
 * Guard: the UI does not call features "AI". Rewrite is "Rewrite", the model behind it is a "language model", and the
 * engine that runs on the PC is "Built-in (on your PC)". "AI" stays only where it is a proper name or a notice to the
 * audience that a voice is synthetic. English strings are checked together with the staged ones
 * (`scripts/i18n-staging/*.json`), so a string waiting for the merge is held to the same rule.
 */
import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const root = path.resolve(__dirname, "../..");
type Json = { [k: string]: string | Json };

const readJson = (file: string): Json => JSON.parse(fs.readFileSync(file, "utf8")) as Json;

/** Every string leaf of a translation file, by dotted key. */
function leaves(doc: Json, prefix = "", out: Map<string, string> = new Map()): Map<string, string> {
  for (const [k, v] of Object.entries(doc)) {
    if (k.startsWith("$")) continue;
    const key = prefix ? `${prefix}.${k}` : k;
    if (typeof v === "string") out.set(key, v);
    else if (v && typeof v === "object") leaves(v, key, out);
  }
  return out;
}

/** English strings as the app will have them once the staged files are merged. */
function englishWithStaging(): Map<string, string> {
  const en = leaves(readJson(path.join(root, "public/i18n/en/translation.json")));
  const staging = path.join(root, "scripts/i18n-staging");
  for (const file of fs.readdirSync(staging).filter((f) => f.endsWith(".json")).sort())
    for (const [k, v] of leaves(readJson(path.join(staging, file)))) en.set(k, v);
  return en;
}

/** Strings that may say "AI": a proper name, or the disclosure to the audience that a cloned voice is synthetic. */
const KEEP_AI_KEYS = new Set([
  "stt.gemini_get_key", // Google AI Studio
  "tts.gemini_get_key", // Google AI Studio
  "tts.gemini_key_hint", // Google AI Studio
  "transform.gemini_hint", // Google AI Studio
  "transl.llm_gemini_hint", // Google AI Studio
  "tts.chatterbox_local_desc", // Resemble AI
  "tts.ggml_omnivoice_disclose",
  "tts.ggml_omnivoice_disclose_on",
  "tts.ggml_omnivoice_disclose_off",
]);
/** Proper names that contain the word. */
const PROPER_NAMES = /Google AI Studio|Resemble AI/g;

describe("no 'AI' or 'LLM' in English UI labels", () => {
  const english = englishWithStaging();

  it("says Rewrite and language model, not AI, in every string", () => {
    const offenders: string[] = [];
    for (const [key, value] of english) {
      if (KEEP_AI_KEYS.has(key)) continue;
      const text = value.replace(PROPER_NAMES, "");
      if (/\bAI\b|\bLLM\b/.test(text)) offenders.push(`${key} = ${value}`);
    }
    expect(offenders).toEqual([]);
  });

  it("keeps AI only in the strings it belongs in", () => {
    for (const key of KEEP_AI_KEYS) {
      const value = english.get(key);
      expect(value, `${key} exists`).toBeTypeOf("string");
      expect(/\bAI\b/.test(value ?? ""), `${key} still names AI`).toBe(true);
    }
  });

  it("uses the new names for the renamed features", () => {
    expect(english.get("transform.title")).toBe("Rewrite");
    expect(english.get("transform.source_label")).toBe("Rewritten text");
    expect(english.get("transform.engine_llm")).toBe("Language model");
    expect(english.get("transform.field_provider")).toBe("Provider");
    expect(english.get("transform.field_system_prompt")).toBe("Instructions");
    expect(english.get("transl.backend_llm")).toBe("Language model");
    expect(english.get("transl.llm_title")).toBe("Language model");
    expect(english.get("transl.llm_server_holder_transform")).toBe("Rewrite");
    expect(english.get("manual.nav_audio")).toBe("Speech and voice");
    expect(english.get("stt.provider_local")).toBe("Built-in (on your PC)");
  });

  it("does not use the old engine name in a label that points at it", () => {
    for (const key of ["tts.commit_turn_end_needs_smart_turn", "vrc.textbox_commit_turn_end_needs_smart_turn"])
      expect(english.get(key), key).not.toContain("(On this PC)");
  });
});

describe("no 'AI Transform' in the app's source", () => {
  it("is gone from every source file outside the tests", () => {
    const hits: string[] = [];
    const walk = (dir: string) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          if (entry.name !== "__tests__") walk(full);
        } else if (/\.(ts|tsx)$/.test(entry.name) && !/\.test\./.test(entry.name)) {
          if (fs.readFileSync(full, "utf8").includes("AI Transform")) hits.push(path.relative(root, full));
        }
      }
    };
    walk(path.join(root, "src"));
    expect(hits).toEqual([]);
  });
});

describe("every locale follows the renames", () => {
  const dirs = fs.readdirSync(path.join(root, "public/i18n")).filter((d) => fs.existsSync(path.join(root, "public/i18n", d, "translation.json")));
  const all = dirs.map((d) => [d, leaves(readJson(path.join(root, "public/i18n", d, "translation.json")))] as const);

  it("has found the locales", () => {
    expect(dirs.length).toBeGreaterThanOrEqual(20);
  });

  it("no longer uses the old names (a translation of the new ones is fine)", () => {
    for (const [dir, strings] of all) {
      const old: string[] = [];
      for (const [key, value] of strings) if (/AI Transform|AI transform/.test(value) || value.includes("On this PC (automatic)")) old.push(key);
      expect(old, dir).toEqual([]);
    }
  });

  it("no longer carries the retired Moonshine server panel strings, but keeps the three the service still uses", () => {
    const retired = [
      "stt.moonshine_title",
      "stt.moonshine_desc",
      "stt.moonshine_endpoint",
      "stt.moonshine_language",
      "stt.moonshine_language_desc",
      "stt.moonshine_legacy_title",
      "stt.moonshine_legacy_notice",
      "stt.provider_moonshine_legacy",
      "stt.error_moonshine_not_found",
    ];
    const kept = ["stt.provider_moonshine", "stt.status_moonshine_connecting", "stt.error_moonshine_start"];
    for (const [dir, strings] of all) {
      for (const key of retired) expect(strings.has(key), `${dir} ${key}`).toBe(false);
      for (const key of kept) expect(strings.has(key), `${dir} ${key}`).toBe(true);
    }
  });
});
