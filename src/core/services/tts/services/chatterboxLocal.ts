import i18n from "i18next";
import type { RustCloningTtsRequest } from "@/types";
import { clampedOr } from "../num";
import { TTS_State } from "../schema";
import { ITTSReceiver } from "../types";
import { PREVIEW_TEXT } from "./localNeural";
import { CloningTtsBackend, cloningThreads } from "./pocket";

/** The model (`model-catalog/tts-cloning.json`; Rust `chatterbox_ort::MODEL_ID`). */
export const CHATTERBOX_LOCAL_MODEL = "tts/chatterbox-mtl-v3";
/** The optional fp16 decoder step for graphics cards (Rust `chatterbox_ort::GPU_FLOW_ID`). */
export const CHATTERBOX_LOCAL_GPU_FLOW = "tts/chatterbox-mtl-v3-gpu-flow";

/** The languages Chatterbox Multilingual V3 speaks (Rust `chatterbox_ort::text::LANGUAGES`; a test keeps them equal). */
export const CHATTERBOX_LOCAL_LANGUAGES = [
  "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
] as const;

/** Ranges of the panel's sliders (Rust clamps to the same or wider). */
export const CHATTERBOX_LOCAL_RANGES = {
  exaggeration: { min: 0.25, max: 2, default: 0.5 },
  cfgWeight: { min: 0, max: 1, default: 0.5 },
  temperature: { min: 0.05, max: 2, default: 0.8 },
} as const;

/** Decoder steps per quality setting (`null` = the engine's choice: 10 on a graphics card, 6 on the CPU). */
export const CHATTERBOX_LOCAL_STEPS: Readonly<Record<string, number | null>> = { auto: null, fast: 6, best: 10 };

/** The model's tag for a BCP-47 language (`pt-BR` → `pt`, `nb` → `no`), or `null` when it does not speak it. */
export function chatterboxLanguage(tag: string | null | undefined): string | null {
  const base = (tag ?? "").trim().toLowerCase().replace("_", "-").split("-")[0];
  const mapped = base === "nb" || base === "nn" ? "no" : base === "iw" ? "he" : base === "yue" || base === "cmn" ? "zh" : base;
  return (CHATTERBOX_LOCAL_LANGUAGES as readonly string[]).includes(mapped) ? mapped : null;
}

/**
 * The `language` of a sentence's request: the panel's language when one is chosen, else the sentence's caption
 * language when the model speaks it, else `null` (Rust then goes by the script, and English for Latin text).
 */
export function chatterboxRequestLanguage(setting: string, sentenceLang: string | null): string | null {
  const fixed = setting && setting !== "auto" ? chatterboxLanguage(setting) : null;
  return fixed ?? chatterboxLanguage(sentenceLang);
}

/** Preview lines for the Chatterbox languages the shared preview table (`localNeural.PREVIEW_TEXT`) lacks. */
export const CHATTERBOX_PREVIEW_TEXT: Readonly<Record<string, string>> = {
  he: "שלום! כך נשמע הקול שלי.",
  ko: "안녕하세요! 제 목소리는 이렇게 들려요.",
  ms: "Helo! Beginilah bunyi suara saya.",
  sw: "Habari! Hivi ndivyo sauti yangu inavyosikika.",
};

/** The preview line in Chatterbox language `lang`, or `null` when there is none. */
function previewLine(lang: string): string | null {
  return CHATTERBOX_PREVIEW_TEXT[lang] ?? PREVIEW_TEXT[lang] ?? null;
}

/** The language the panel previews in: the panel's choice, else the UI language when Chatterbox speaks it, else English. */
export function chatterboxPreviewLanguage(setting: string, uiLanguage: string | null): string {
  return chatterboxRequestLanguage(setting, uiLanguage) ?? "en";
}

/** The panel's sample line (the placeholder of the preview field): in the preview language, English when there is none. */
export function chatterboxPreviewText(setting: string, uiLanguage: string | null): string {
  return previewLine(chatterboxPreviewLanguage(setting, uiLanguage)) ?? PREVIEW_TEXT.en;
}

/**
 * What the panel's preview speaks and the language tag it sends. Chatterbox applies the tag literally (the `[xx]`
 * token and that language's text handling), so it must match the text:
 * - a typed line goes with the panel's language when one is chosen, else with none (Rust goes by the script);
 * - the sample line goes with its own language (English when the preview language has no line).
 */
export function chatterboxPreview(setting: string, uiLanguage: string | null, typed: string): { text: string; language: string | null } {
  const text = typed.trim();
  if (text) return { text, language: chatterboxRequestLanguage(setting, null) };
  const lang = chatterboxPreviewLanguage(setting, uiLanguage);
  const line = previewLine(lang);
  return line ? { text: line, language: lang } : { text: PREVIEW_TEXT.en, language: "en" };
}

/** A localized problem that stops Start, or `null`. */
export function chatterboxSettingsProblem(c: TTS_State["chatterboxLocal"]): string | null {
  if (!(c.referenceVoiceId ?? "").trim()) return i18n.t("tts.chatterbox_local_voice_required");
  return null;
}

/** The Chatterbox request for one sentence (`text` "" = load and prepare the voice). */
export function chatterboxRequest(c: TTS_State["chatterboxLocal"], text: string, sentenceLang: string | null): RustCloningTtsRequest {
  const r = CHATTERBOX_LOCAL_RANGES;
  const steps = CHATTERBOX_LOCAL_STEPS[c.quality] ?? null;
  const options: Record<string, unknown> = {
    ep: c.ep,
    exaggeration: clampedOr(c.exaggeration, r.exaggeration.default, r.exaggeration.min, r.exaggeration.max),
    cfgWeight: clampedOr(c.cfgWeight, r.cfgWeight.default, r.cfgWeight.min, r.cfgWeight.max),
    temperature: clampedOr(c.temperature, r.temperature.default, r.temperature.min, r.temperature.max),
  };
  if (steps) options.flowSteps = steps;
  return {
    model: CHATTERBOX_LOCAL_MODEL,
    referenceVoiceId: (c.referenceVoiceId ?? "").trim(),
    text,
    language: chatterboxRequestLanguage(c.language, sentenceLang),
    // The model has no rate control.
    speed: 1,
    threads: cloningThreads(c.threads, 8),
    options,
  };
}

/** Optional downloads worth offering for the settings: the graphics-card decoder unless the CPU is chosen. */
export function chatterboxOptionalIds(c: TTS_State["chatterboxLocal"]): string[] {
  return c.ep === "cpu" ? [] : [CHATTERBOX_LOCAL_GPU_FLOW];
}

/**
 * `chatterboxLocal` backend: Chatterbox Multilingual V3 on Sigil's ONNX Runtime (`plugin:local-tts|chatterbox_local_speak`),
 * the in-process voice-cloning fallback in 23 languages, on the graphics card when there is one. Clones a voice from
 * the shared voice library (required). Not the `chatterbox` backend (a user-run HTTP server).
 */
export class TTS_ChatterboxLocalService extends CloningTtsBackend {
  constructor(receiver: ITTSReceiver) {
    super(receiver, "chatterbox", "Chatterbox");
  }

  protected request(state: TTS_State, text: string, lang: string | null): RustCloningTtsRequest | string {
    const c = state.chatterboxLocal;
    const problem = chatterboxSettingsProblem(c);
    if (problem) return problem;
    return chatterboxRequest(c, text, lang);
  }

  protected requirementIds(_state: TTS_State): string[] {
    // The graphics-card decoder is optional: only the model itself must be installed to start.
    return [CHATTERBOX_LOCAL_MODEL];
  }
}
