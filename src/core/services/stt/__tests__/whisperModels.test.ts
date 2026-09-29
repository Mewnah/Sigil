import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { formatModelSize, WHISPER_LANGUAGES, WHISPER_MODEL_OPTIONS, whisperLanguageFromBcp47 } from "../whisperModels";

/** Model ids the Rust plugin resolves (`CATALOG`, `ALIASES`, `ANIME_WHISPER_ID` in whisper.rs). */
function rustModelIds(): Set<string> {
  const src = readFileSync(resolve(__dirname, "../../../../../src-tauri/src/services/whisper.rs"), "utf8");
  const ids = new Set<string>();
  const catalog = src.slice(src.indexOf("const CATALOG"), src.indexOf("];", src.indexOf("const CATALOG")));
  for (const m of catalog.matchAll(/\(\s*"([^"]+)"\s*,\s*"ggml-[^"]+\.bin"\s*\)/g)) ids.add(m[1]);
  const anime = src.match(/ANIME_WHISPER_ID:\s*&str\s*=\s*"([^"]+)"/);
  if (anime) ids.add(anime[1]);
  return ids;
}

describe("Whisper model options", () => {
  it("every offered id is one the Rust plugin knows", () => {
    const rust = rustModelIds();
    expect(rust.size).toBeGreaterThan(20);
    for (const option of WHISPER_MODEL_OPTIONS) expect(rust, option.id).toContain(option.id);
  });

  it("offers each id once, including the schema default and the migration picks", () => {
    const ids = WHISPER_MODEL_OPTIONS.map((o) => o.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const id of ["base.en", "small.en-q5_1", "small-q5_1", "base.en-q5_1", "base-q5_1", "large-v3-turbo"]) expect(ids).toContain(id);
  });

  it("formats sizes", () => {
    expect(formatModelSize(57)).toBe("~57 MB");
    expect(formatModelSize(1600)).toBe("~1.6 GB");
  });

  it("maps BCP-47 tags to offered Whisper languages", () => {
    expect(whisperLanguageFromBcp47("en-US")).toBe("en");
    expect(whisperLanguageFromBcp47("zh-TW")).toBe("zh");
    expect(whisperLanguageFromBcp47("fil-PH")).toBe("tl");
    expect(whisperLanguageFromBcp47("nb-NO")).toBe("no");
    expect(whisperLanguageFromBcp47("xx")).toBe("auto");
    expect(whisperLanguageFromBcp47(undefined)).toBe("auto");
    expect(new Set(WHISPER_LANGUAGES).size).toBe(WHISPER_LANGUAGES.length);
  });
});
