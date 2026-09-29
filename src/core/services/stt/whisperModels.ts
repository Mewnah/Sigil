/**
 * Whisper models offered in the STT inspector. Ids must exist in the Rust catalog (`get_model_info` in
 * src-tauri/src/services/whisper.rs); `__tests__/whisperModels.test.ts` checks that against the Rust source.
 */

export type WhisperModelFamily = "tiny" | "base" | "small" | "medium" | "large-v3-turbo" | "large-v3" | "anime-whisper";
export type WhisperQuant = "f16" | "q8_0" | "q5_1" | "q5_0";

export interface WhisperModelOption {
  id: string;
  family: WhisperModelFamily;
  /** English-only (`.en`) model: always transcribes English, a little more accurate for it. */
  englishOnly: boolean;
  quant: WhisperQuant;
  /** Approximate download size in MB. */
  sizeMb: number;
  /** Must be supplied as a local .bin (never downloaded). */
  bringYourOwn?: boolean;
  /** Language the model is fixed to (single-language fine-tunes). */
  language?: string;
}

const m = (
  id: string,
  family: WhisperModelFamily,
  englishOnly: boolean,
  quant: WhisperQuant,
  sizeMb: number,
  extra: Partial<WhisperModelOption> = {},
): WhisperModelOption => ({ id, family, englishOnly, quant, sizeMb, ...extra });

/** Ordered small → large; quantized builds first (faster on CPU, near-identical accuracy). */
export const WHISPER_MODEL_OPTIONS: readonly WhisperModelOption[] = [
  m("tiny.en-q5_1", "tiny", true, "q5_1", 31),
  m("tiny-q5_1", "tiny", false, "q5_1", 31),
  m("tiny.en", "tiny", true, "f16", 75),
  m("tiny", "tiny", false, "f16", 75),
  m("base.en-q5_1", "base", true, "q5_1", 57),
  m("base-q5_1", "base", false, "q5_1", 57),
  m("base.en-q8_0", "base", true, "q8_0", 78),
  m("base-q8_0", "base", false, "q8_0", 78),
  m("base.en", "base", true, "f16", 142),
  m("base", "base", false, "f16", 142),
  m("small.en-q5_1", "small", true, "q5_1", 181),
  m("small-q5_1", "small", false, "q5_1", 181),
  m("small.en-q8_0", "small", true, "q8_0", 252),
  m("small-q8_0", "small", false, "q8_0", 252),
  m("small.en", "small", true, "f16", 466),
  m("small", "small", false, "f16", 466),
  m("medium.en-q5_0", "medium", true, "q5_0", 514),
  m("medium-q5_0", "medium", false, "q5_0", 514),
  m("medium.en-q8_0", "medium", true, "q8_0", 785),
  m("medium-q8_0", "medium", false, "q8_0", 785),
  m("medium.en", "medium", true, "f16", 1500),
  m("medium", "medium", false, "f16", 1500),
  m("large-v3-turbo-q5_0", "large-v3-turbo", false, "q5_0", 547),
  m("large-v3-turbo-q8_0", "large-v3-turbo", false, "q8_0", 834),
  m("large-v3-turbo", "large-v3-turbo", false, "f16", 1600),
  m("large-v3-q5_0", "large-v3", false, "q5_0", 1080),
  m("large-v3", "large-v3", false, "f16", 3100),
  m("anime-whisper", "anime-whisper", false, "f16", 1600, { bringYourOwn: true, language: "ja" }),
];

export function findWhisperModel(id: string): WhisperModelOption | undefined {
  const key = id.trim();
  return WHISPER_MODEL_OPTIONS.find((o) => o.id === key);
}

/** "~57 MB" / "~1.6 GB". */
export function formatModelSize(sizeMb: number): string {
  return sizeMb >= 1000 ? `~${(sizeMb / 1000).toFixed(1)} GB` : `~${sizeMb} MB`;
}

/**
 * Whisper language codes offered in the inspector (plus "auto"). Names are rendered with `Intl.DisplayNames` in the
 * UI language. Covers Sigil's UI locales and the languages Whisper handles well.
 */
export const WHISPER_LANGUAGES: readonly string[] = [
  "en", "ja", "ko", "zh", "es", "fr", "de", "it", "pt", "ru", "pl", "tr", "nl", "uk", "cs", "sv", "da", "no", "fi",
  "el", "hu", "ro", "bg", "ar", "he", "fa", "ur", "hi", "bn", "ta", "th", "vi", "id", "ms", "tl",
];

/** Whisper language for a BCP-47 tag (`ja-JP` → `ja`, `zh-TW` / `yue-*` → `zh`); "auto" when not offered. */
export function whisperLanguageFromBcp47(tag: string | undefined): string {
  const primary = String(tag ?? "").trim().split(/[-_]/)[0].toLowerCase();
  if (primary === "yue" || primary === "cmn") return "zh";
  if (primary === "fil") return "tl";
  if (primary === "nb" || primary === "nn") return "no";
  return WHISPER_LANGUAGES.includes(primary) ? primary : "auto";
}
