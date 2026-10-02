import { invoke } from "@tauri-apps/api/core";
import i18n from "i18next";
import type { RustGeminiTtsRequest, RustTtsSinkTarget, RustTtsSpeakReport } from "@/types";
import { pushSystemLog } from "../../systemLog";
import type { FallbackBreaker } from "../pipeline";
import { TTS_State } from "../schema";
import { ITTSReceiver, TTS_OPTIONS_MISSING, TtsSinkContext, errorMessage } from "../types";
import { CloudTtsBackend, previewCloud, type CloudFallbackSettings } from "./edge";

/**
 * Model ids offered in the panel (Google's speech-generation docs, 2026-10; editable: preview names change). The first
 * is the default. `label` is an i18n key; models without one show their id.
 */
export const GEMINI_TTS_MODELS: readonly { id: string; label?: string }[] = [
  { id: "gemini-3.8-flash-tts", label: "tts.gemini_model_flash" },
  { id: "gemini-3.8-flash-lite-tts", label: "tts.gemini_model_flash_lite" },
  { id: "gemini-3.1-flash-tts-preview" },
  { id: "gemini-2.5-pro-preview-tts" },
];

/** Default model and voice (Rust `gemini_tts::DEFAULT_MODEL` / `DEFAULT_VOICE`). */
export const GEMINI_DEFAULT_MODEL = GEMINI_TTS_MODELS[0].id;
export const GEMINI_DEFAULT_VOICE = "Kore";

/**
 * Style instructions offered as one-click presets (`style`; sent as `speech_metadata`, never spoken). English works for
 * every language the model speaks. `key` names the i18n label `tts.gemini_style_preset_<key>`.
 */
export const GEMINI_STYLE_PRESETS: readonly { key: string; style: string }[] = [
  { key: "cheerful", style: "cheerful and friendly" },
  { key: "calm", style: "calm and warm" },
  { key: "excited", style: "excited and energetic" },
  { key: "whisper", style: "whispering softly" },
  { key: "serious", style: "serious, like a news anchor" },
];

/** Gemini's prebuilt voices and their character (Google's descriptions; the names are not translated). */
export const GEMINI_VOICES: readonly { name: string; style: string }[] = [
  { name: "Zephyr", style: "bright" },
  { name: "Puck", style: "upbeat" },
  { name: "Charon", style: "informative" },
  { name: "Kore", style: "firm" },
  { name: "Fenrir", style: "excitable" },
  { name: "Leda", style: "youthful" },
  { name: "Orus", style: "firm" },
  { name: "Aoede", style: "breezy" },
  { name: "Callirrhoe", style: "easy_going" },
  { name: "Autonoe", style: "bright" },
  { name: "Enceladus", style: "breathy" },
  { name: "Iapetus", style: "clear" },
  { name: "Umbriel", style: "easy_going" },
  { name: "Algieba", style: "smooth" },
  { name: "Despina", style: "smooth" },
  { name: "Erinome", style: "clear" },
  { name: "Algenib", style: "gravelly" },
  { name: "Rasalgethi", style: "informative" },
  { name: "Laomedeia", style: "upbeat" },
  { name: "Achernar", style: "soft" },
  { name: "Alnilam", style: "firm" },
  { name: "Schedar", style: "even" },
  { name: "Gacrux", style: "mature" },
  { name: "Pulcherrima", style: "forward" },
  { name: "Achird", style: "friendly" },
  { name: "Zubenelgenubi", style: "casual" },
  { name: "Vindemiatrix", style: "gentle" },
  { name: "Sadachbia", style: "lively" },
  { name: "Sadaltager", style: "knowledgeable" },
  { name: "Sulafat", style: "warm" },
];

/** Where to get a free key. */
export const GEMINI_KEY_URL = "https://aistudio.google.com/apikey";

/** The Gemini request of a sentence (the key travels to Rust only, which puts it in a header). */
export function geminiRequest(c: TTS_State["gemini"], text: string, timeoutMs: number, language?: string | null): RustGeminiTtsRequest {
  return {
    apiKey: c.apiKey.trim(),
    baseUrl: c.baseUrl.trim(),
    model: c.model.trim() || GEMINI_DEFAULT_MODEL,
    text,
    voice: c.voice.trim() || GEMINI_DEFAULT_VOICE,
    style: c.style.trim(),
    language: language?.trim() || null,
    timeoutMs: Math.max(0, Math.round(timeoutMs)),
  };
}

/**
 * Whether the key and model work (`gemini_tts_check`): `true`, `false` when Google cannot be reached (offline: Start
 * still works, sentences fall back), or a rejection naming the problem (refused key, unknown model).
 */
export function checkGemini(c: TTS_State["gemini"]): Promise<boolean> {
  return invoke<boolean>("plugin:gemini-tts|gemini_tts_check", {
    apiKey: c.apiKey.trim(),
    model: c.model.trim() || GEMINI_DEFAULT_MODEL,
    baseUrl: c.baseUrl.trim() || null,
  });
}

/** Speak `text` with the slice's voice into a new sink utterance on `target` (the panel's "Test voice"). */
export function previewGeminiVoice(c: TTS_State["gemini"], text: string, target: RustTtsSinkTarget): Promise<void> {
  return previewCloud(target, (id) =>
    invoke<RustTtsSpeakReport>("plugin:gemini-tts|gemini_tts_speak", { request: geminiRequest(c, text, 0), utteranceId: id }),
  );
}

/** Fire-and-forget IPC: never throws, synchronously or later. */
function send(command: string, args?: Record<string, unknown>): void {
  try {
    void Promise.resolve(invoke(command, args)).catch(() => {});
  } catch {
    // Without Tauri (web build, tests) there is nothing to tell.
  }
}

/**
 * Google Gemini TTS on the user's own (free-tier) key (backend `gemini`): Gemini 3.8 Flash / Flash-Lite TTS, 30
 * voices that follow a style instruction, streaming PCM through the Rust `gemini-tts` plugin. On 429 (free-tier
 * limits), 5xx, network failures and stalls the local Kokoro voice takes over (fallback chain, `pipeline.ts`). Start
 * needs a key; a refused key or an unknown model refuses Start, an unreachable Google does not (sentences fall back).
 */
export class TTS_GeminiService extends CloudTtsBackend {
  constructor(receiver: ITTSReceiver, breaker?: FallbackBreaker) {
    super(receiver, i18n.t("tts.gemini_service_name"), breaker);
  }

  protected fallbackSettings(state: TTS_State): CloudFallbackSettings {
    return state.gemini;
  }

  protected setupKey(state: TTS_State): string {
    const c = state.gemini;
    return [c.apiKey.trim(), c.model.trim(), c.baseUrl.trim(), c.voice.trim()].join("\n");
  }

  protected async connect(state: TTS_State): Promise<string | null> {
    const c = state.gemini;
    if (!c.apiKey.trim()) return TTS_OPTIONS_MISSING;
    try {
      const reachable = await checkGemini(c);
      if (!reachable) pushSystemLog("TTS", i18n.t("tts.gemini_unreachable"), "warning");
      return null;
    } catch (error) {
      return errorMessage(error);
    }
  }

  protected speakCloud(text: string, sink: TtsSinkContext, timeoutMs: number): Promise<unknown> {
    const request = geminiRequest(this.state.gemini, text, timeoutMs, sink.lang);
    return invoke<RustTtsSpeakReport>("plugin:gemini-tts|gemini_tts_speak", { request, utteranceId: sink.utteranceId });
  }

  protected cancelCloud(utteranceId: string | null): void {
    send("plugin:gemini-tts|gemini_tts_cancel", { utteranceId });
  }
}
