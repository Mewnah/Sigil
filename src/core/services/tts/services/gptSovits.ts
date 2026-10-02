import { invoke } from "@tauri-apps/api/core";
import i18n from "i18next";
import type { RustGptSovitsRequest, RustTtsSpeakReport } from "@/types";
import { isSinkCancelled } from "../../sound/sink";
import { clampedOr } from "../num";
import type { TTS_State } from "../schema";
import type { GptSovitsStreamMode, TTS_GptSovits_State } from "../schema/gptSovits";
import { ITTSReceiver, TtsSinkContext, errorMessage } from "../types";
import { HttpSinkBackend, cancelHttpSpeech, parseExtra } from "./openaiSpeech";

/**
 * `gptSovits` backend (P6-G): a GPT-SoVITS server the user runs (`api_v2.py`, v2 / v2Pro / v3 / v4), through the Rust
 * `gpt-sovits` plugin, which streams each sentence's WAV straight into the TTS sink (decoded as it arrives, so the
 * first audio plays while the rest is generated; no audio crosses IPC).
 *
 * - The reference is a voice-library voice (its clip and transcript; the server must run on this PC to read the clip)
 *   or a clip path on the server's machine with its transcript.
 * - Start checks that the server is `api_v2.py`, then switches the GPT / SoVITS weights when paths are set (the server
 *   keeps them; loading takes a few seconds).
 * - The text language `auto` uses the caption's language when the publisher tagged it (GPT-SoVITS reads Japanese kanji
 *   as Chinese when it has to guess), else lets the server detect.
 * - Reference, transcript, languages, streaming, speed and extra fields apply to the next sentence; the server URL and
 *   the weights restart the backend.
 */

/** Extra fields of `RustGptSovitsRequest` added by P6-G (Rust `GptSovitsRequest::streaming_mode`). */
export type GptSovitsRequest = RustGptSovitsRequest & { streamingMode?: number };

/** `api_v2.py`'s default address (Rust `gpt_sovits::DEFAULT_BASE_URL`). */
export const GPT_SOVITS_DEFAULT_URL = "http://127.0.0.1:9880";

/** Audio files a reference clip on this PC may be (what GPT-SoVITS loads with librosa / ffmpeg). */
export const GPT_SOVITS_REFERENCE_EXTENSIONS = ["wav", "flac", "mp3", "ogg", "m4a"] as const;

/** `streaming_mode` sent for a setting (0 = `true`, every api_v2 version). */
export const STREAM_MODE_VALUE: Readonly<Record<GptSovitsStreamMode, number>> = { fragments: 0, medium: 2, fast: 3 };

/**
 * GPT-SoVITS's language code for a caption's language tag, or `null` when GPT-SoVITS has none for it. Cantonese
 * (`yue`) only for an explicit marker (`yue`, `zh-HK`, `zh-MO`, `zh-yue`): Traditional Chinese (`zh-Hant`, which the
 * translation layer makes of zh-TW, zh-HK and zh-MO alike) is read as Mandarin. Same rule as Rust `gsv_language`.
 */
export function gsvLanguageOf(lang: string | null | undefined): string | null {
  const l = (lang ?? "").trim().toLowerCase().replace(/_/g, "-");
  if (!l) return null;
  const [base, ...subtags] = l.split("-");
  if (base === "yue" || (base === "zh" && subtags.some((t) => t === "hk" || t === "mo" || t === "yue"))) return "yue";
  if (base === "zh" || base === "cmn") return "zh";
  if (base === "ja" || base === "en" || base === "ko") return base;
  return null;
}

/** The `text_lang` of a sentence: the setting, or (for `auto`) the caption's language when GPT-SoVITS has it. */
export function sentenceLanguage(setting: string, captionLang: string | null | undefined): string {
  const s = (setting ?? "").trim() || "auto";
  if (s !== "auto") return s;
  return gsvLanguageOf(captionLang) ?? "auto";
}

/**
 * Whether a server URL points at this PC (loopback or `localhost`), so it can read the voice library's clips. Same rule
 * as Rust `gpt_sovits::is_this_pc`.
 */
export function isLocalServer(url: string): boolean {
  try {
    const u = new URL(url.includes("://") ? url : `http://${url}`);
    const host = u.hostname.replace(/^\[|\]$/g, "").replace(/\.$/, "").toLowerCase();
    return host === "localhost" || host.endsWith(".localhost") || host === "::1" || /^127\.\d+\.\d+\.\d+$/.test(host);
  } catch {
    return false;
  }
}

/**
 * Everything but the text of a sentence, or a localized reason to refuse. A voice-library reference is used only with
 * a server on this PC; otherwise the clip path on the server is.
 */
export function gptSovitsSettings(c: TTS_GptSovits_State): Omit<GptSovitsRequest, "text"> | string {
  const baseUrl = (c.baseUrl ?? "").trim() || GPT_SOVITS_DEFAULT_URL;
  const library = isLocalServer(baseUrl) ? (c.referenceVoiceId ?? "").trim() : "";
  const refAudioPath = library ? "" : (c.refAudioPath ?? "").trim();
  if (!library && !refAudioPath) return i18n.t("tts.gpt_sovits_no_reference");
  const extra = parseExtra(c.extra);
  if (extra === undefined) return i18n.t("tts.openai_speech_extra_invalid");
  return {
    baseUrl,
    textLang: (c.textLang ?? "").trim() || "auto",
    referenceVoiceId: library,
    refAudioPath,
    promptText: (c.promptText ?? "").trim(),
    promptLang: (c.promptLang ?? "").trim() || "auto",
    streaming: !!c.streaming,
    streamingMode: STREAM_MODE_VALUE[c.streamingMode] ?? 0,
    speed: clampedOr(c.speed, 1, 0.5, 2),
    extra,
    timeoutMs: Math.round(clampedOr(c.timeout, 0, 0, 600) * 1000),
  };
}

/** Speak `text` with `settings` into the begun sink utterance `utteranceId`. */
export function speakGptSovits(
  settings: Omit<GptSovitsRequest, "text">,
  text: string,
  utteranceId: string,
  captionLang?: string | null,
): Promise<RustTtsSpeakReport> {
  const request: GptSovitsRequest = { ...settings, text, textLang: sentenceLanguage(settings.textLang ?? "auto", captionLang) };
  return invoke<RustTtsSpeakReport>("plugin:gpt-sovits|gpt_sovits_speak", { request, utteranceId });
}

/** Whether `api_v2.py` answers at `baseUrl`; rejects with the reason when another server answers there. */
export function checkGptSovits(baseUrl: string): Promise<boolean> {
  return invoke<boolean>("plugin:gpt-sovits|gpt_sovits_check", { baseUrl: baseUrl.trim() || GPT_SOVITS_DEFAULT_URL });
}

/** Switch the server's weights ("" keeps that half); rejects with the server's message. */
export function setGptSovitsWeights(baseUrl: string, gptWeights: string, sovitsWeights: string): Promise<void> {
  return invoke<void>("plugin:gpt-sovits|gpt_sovits_set_weights", {
    baseUrl: baseUrl.trim() || GPT_SOVITS_DEFAULT_URL,
    gptWeights: gptWeights.trim(),
    sovitsWeights: sovitsWeights.trim(),
  });
}

export class TTS_GptSovitsService extends HttpSinkBackend {
  constructor(bindings: ITTSReceiver) {
    super(bindings, "GPT-SoVITS");
  }

  protected async connect(state: TTS_State): Promise<string | null> {
    const c = state.gptSovits;
    const settings = gptSovitsSettings(c);
    if (typeof settings === "string") return settings;
    const url = settings.baseUrl ?? GPT_SOVITS_DEFAULT_URL;
    const ok = await checkGptSovits(url);
    if (!ok) return i18n.t("tts.gpt_sovits_unreachable", { url });
    if (c.gptWeights.trim() || c.sovitsWeights.trim()) {
      try {
        await setGptSovitsWeights(url, c.gptWeights, c.sovitsWeights);
      } catch (e) {
        return i18n.t("tts.gpt_sovits_weights_failed", { error: errorMessage(e) });
      }
    }
    return null;
  }

  /** Live settings (reference, languages, speed and streaming apply to the next sentence). */
  #settings(): Omit<GptSovitsRequest, "text"> {
    const live = gptSovitsSettings(window.ApiServer.state.services.tts.data.gptSovits);
    if (typeof live === "string") throw new Error(live);
    return live;
  }

  protected speak(text: string, utteranceId: string): Promise<unknown> {
    return speakGptSovits(this.#settings(), text, utteranceId, null);
  }

  /** v2, with the caption's language for `auto` (see the class docs). */
  async synthesizeToSink(text: string, sink: TtsSinkContext, signal: AbortSignal): Promise<void> {
    if (!this.running || signal.aborted || !text.trim()) return;
    const onAbort = () => cancelHttpSpeech(sink.utteranceId);
    signal.addEventListener("abort", onAbort, { once: true });
    try {
      await speakGptSovits(this.#settings(), text, sink.utteranceId, sink.lang);
    } catch (error) {
      if (signal.aborted || isSinkCancelled(error) || !this.running) return;
      throw new Error(`${this.label}: ${errorMessage(error)}`);
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
  }
}
