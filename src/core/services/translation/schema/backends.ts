/**
 * Persisted translation backend ids: add values, never rename or remove them (zSafe falls back to the default).
 * `auto` is the router (P4-C): it runs a per-language-pair chain of the other engines with fallback.
 */
export enum Translation_Backends {
  azure = "azure",
  libretranslate = "libretranslate",
  onDevice = "onDevice",
  /** OpenAI-compatible LLM: the bundled llama-server (Hy-MT2, MiLMMT, ...), Gemini, OpenRouter, OpenAI, custom. */
  llm = "llm",
  /** DeepL API Free / Pro (called from Rust). */
  deepl = "deepl",
  /** Mozilla Bergamot CPU helper (zero VRAM). */
  bergamot = "bergamot",
  /** NLLB-200 distilled on Sigil's ONNX Runtime (long tail, ~200 languages). */
  nllb = "nllb",
  /** Router: per-pair engine chain with fallback, circuit breaker and a separate interim engine. */
  auto = "auto",
}

/** Backends the router can chain (every backend but `auto` itself), in the default preference order. */
export const TRANSLATION_ENGINE_IDS = [
  Translation_Backends.llm,
  Translation_Backends.deepl,
  Translation_Backends.bergamot,
  Translation_Backends.nllb,
  Translation_Backends.azure,
  Translation_Backends.libretranslate,
  Translation_Backends.onDevice,
] as const;
export type TranslationEngineId = (typeof TRANSLATION_ENGINE_IDS)[number];

/** Backends that use the shared `sourceLanguage` / `targetLanguage`; the legacy ones keep per-slice pairs. */
export const SHARED_PAIR_BACKENDS: readonly Translation_Backends[] = [
  Translation_Backends.llm,
  Translation_Backends.deepl,
  Translation_Backends.bergamot,
  Translation_Backends.nllb,
  Translation_Backends.auto,
];

/** Keeps array entries that are plain objects (one malformed entry must not reset a whole list). */
export function keepObjects(value: unknown): unknown {
  return Array.isArray(value) ? value.filter((v) => v !== null && typeof v === "object" && !Array.isArray(v)) : value;
}
