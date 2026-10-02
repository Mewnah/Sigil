import { invoke, isTauri } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import i18n from "i18next";
import { proxy, subscribe } from "valtio";
import {
  ServiceNetworkState,
  type RustQwenCloneVoice,
  type RustTtsHelperEndpoint,
  type RustTtsHelperStatus,
  type RustTtsSinkTarget,
  type RustTtsSpeakReport,
} from "@/types";
import { isSinkCancelled } from "../../sound/sink";
import { primaryGpu, type SysSystemInfo } from "../../sysinfo/format";
import type { TTS_State } from "../schema";
import type { Qwen3TtsMode, Qwen3TtsModel, Qwen3TtsVariant, TTS_Qwen3_State } from "../schema/qwen3";
import { ITTSReceiver, ITTSService, TTS_OPTIONS_MISSING, TtsSinkContext, errorMessage } from "../types";
import { cancelHttpSpeech, previewToSink, speakOpenAi, type OpenAiSpeechSettings } from "./openaiSpeech";

/**
 * `qwen3` backend (P6-E): Qwen3-TTS 0.6B / 1.7B through Sigil's qwentts.cpp server, a helper process the Rust
 * `tts-helpers` plugin starts (`tts_helper_start { engine: "qwen3", config }`: 127.0.0.1, a random key, a kill-on-close
 * Job Object, a VRAM claim). Every sentence streams raw 24 kHz PCM through the OpenAI-compatible client
 * (`openai_speech_speak`) straight into the TTS sink, so the first audio plays while the rest is generated.
 *
 * - Modes: 9 built-in voices with a style / emotion instruction (1.7B), a voice-library voice cloned from its clip and
 *   transcript (`qwen_tts_prepare_clone` registers it; Rust caches its speaker data in the library), or a voice
 *   described in words (1.7B VoiceDesign). Each mode loads its own model.
 * - Size `auto` follows the resource governor: 1.7B when it fits, 0.6B while the voice changer runs on the GPU of a
 *   card under 16 GB. When the recommendation changes while speaking stops, the helper is reloaded with the other size
 *   (only when that size is installed; otherwise the loaded one keeps speaking).
 * - Settings that decide the model (mode, size, runtime) reload the helper before the next sentence; voice, language,
 *   instruction, description and seed apply to the next sentence (each request names them).
 * - A helper that crashed restarts by itself (process host); the next sentence picks up its new port, and cloned
 *   voices register again.
 */

// -------------------------------------------------------------------------------------------------
// Catalog ids and the plan (mirror of Rust `tts_helpers::qwen::plan`)
// -------------------------------------------------------------------------------------------------

export const QWEN3_TOKENIZER_ID = "tts/qwen3-tts-tokenizer";
export const QWEN3_RUNTIME_IDS = {
  vulkan: "runtime/qwentts-server-vulkan",
  cuda13: "runtime/qwentts-server-cuda13",
  cpu: "runtime/qwentts-server-cpu",
} as const;
export const NVIDIA_RUNTIME_ID = "runtime/nvidia-cuda13";

export type Qwen3Size = "1.7b" | "0.6b";
export type Qwen3Variant = Exclude<Qwen3TtsVariant, "auto">;

const TALKERS: Record<Qwen3Size, Partial<Record<Qwen3TtsMode, string>>> = {
  "1.7b": { preset: "tts/qwen3-tts-1.7b", clone: "tts/qwen3-tts-1.7b-base", design: "tts/qwen3-tts-1.7b-voicedesign" },
  "0.6b": { preset: "tts/qwen3-tts-0.6b", clone: "tts/qwen3-tts-0.6b-base" },
};

/** The governor's size ids (`resources::policy::ids`). */
const POLICY_SIZE: Record<string, Qwen3Size> = { "tts/qwen3-tts-1.7b": "1.7b", "tts/qwen3-tts-0.6b": "0.6b" };

/** Catalog id of the talker of `size` in `mode` (`null`: there is no 0.6B voice design). */
export function qwen3TalkerId(size: Qwen3Size, mode: Qwen3TtsMode): string | null {
  return TALKERS[size][mode] ?? null;
}

/** Size of a talker id (the helper status' `model`), `null` when it is not one. */
export function qwen3SizeOf(talkerId: string | null | undefined): Qwen3Size | null {
  if (!talkerId) return null;
  for (const size of ["1.7b", "0.6b"] as const) {
    if (Object.values(TALKERS[size]).includes(talkerId)) return size;
  }
  return null;
}

/** What the resource governor says about Qwen3-TTS (`resources` service, topic `qwen3_tts`). */
export interface Qwen3Recommendation {
  choice: string;
  alternatives: string[];
  /** `gpu` or `cpu`. */
  device?: string;
  reason_code?: string;
}

export interface Qwen3PlanEnv {
  recommendation: Qwen3Recommendation | null | undefined;
  installed: (id: string) => boolean;
  /** The primary GPU can run the CUDA 13 build (NVIDIA Turing or newer, driver 580+). */
  cudaUsable: boolean;
}

export interface Qwen3Plan {
  size: Qwen3Size;
  mode: Qwen3TtsMode;
  variant: Qwen3Variant;
  talker: string;
  runtime: string;
  /**
   * Catalog ids the plan needs, in Rust's `missing` order: talker, tokenizer, runtime pack, and the NVIDIA runtime for
   * the CUDA pack (`<ModelRequirement>` adds any other `requires`).
   */
  ids: string[];
  /** Style instructions work (CustomVoice 1.7B; the 0.6B has no instruction control). */
  instructions: boolean;
  /** Voice design was asked with 0.6B: it runs on 1.7B. */
  designForcedLarge: boolean;
}

/**
 * The model, runtime pack and downloads for these settings, as Rust plans them (Rust decides at start; this drives the
 * panel's requirement card and the size swap). Size `auto`: the governor's choice when installed, else its first
 * installed alternative, else its choice. Variant `auto`: the processor when the governor found no usable GPU, CUDA 13
 * when usable and installed with the NVIDIA runtime, else an installed Vulkan pack, else CUDA 13 when half of it is
 * installed (its pack or the NVIDIA runtime), else Vulkan.
 */
export function planQwen3(s: Pick<TTS_Qwen3_State, "model" | "variant" | "mode">, env: Qwen3PlanEnv): Qwen3Plan {
  const mode: Qwen3TtsMode = s.mode === "clone" || s.mode === "design" ? s.mode : "preset";
  const rec = env.recommendation;
  const recSize: Qwen3Size = (rec && POLICY_SIZE[rec.choice]) || "0.6b";
  const ready = (size: Qwen3Size) => {
    const t = qwen3TalkerId(size, mode);
    return !!t && env.installed(t) && env.installed(QWEN3_TOKENIZER_ID);
  };
  let size: Qwen3Size;
  const model: Qwen3TtsModel = s.model;
  if (mode === "design") size = "1.7b";
  else if (model === "1.7b" || model === "0.6b") size = model;
  else if (!rec) size = "1.7b";
  else {
    const candidates = [rec.choice, ...(rec.alternatives ?? [])].map((id) => POLICY_SIZE[id]).filter((x): x is Qwen3Size => !!x);
    size = candidates.find(ready) ?? recSize;
  }
  const talker = qwen3TalkerId(size, mode) ?? "tts/qwen3-tts-1.7b-voicedesign";
  let variant: Qwen3Variant;
  if (s.variant === "vulkan" || s.variant === "cuda13" || s.variant === "cpu") variant = s.variant;
  else if (rec?.device === "cpu") variant = "cpu";
  else if (env.cudaUsable && env.installed(QWEN3_RUNTIME_IDS.cuda13) && env.installed(NVIDIA_RUNTIME_ID)) variant = "cuda13";
  else if (env.installed(QWEN3_RUNTIME_IDS.vulkan)) variant = "vulkan";
  else if (env.cudaUsable && (env.installed(QWEN3_RUNTIME_IDS.cuda13) || env.installed(NVIDIA_RUNTIME_ID))) variant = "cuda13";
  else variant = "vulkan";
  const runtime = QWEN3_RUNTIME_IDS[variant];
  return {
    size,
    mode,
    variant,
    talker,
    runtime,
    ids: variant === "cuda13" ? [talker, QWEN3_TOKENIZER_ID, runtime, NVIDIA_RUNTIME_ID] : [talker, QWEN3_TOKENIZER_ID, runtime],
    instructions: mode === "preset" && size === "1.7b",
    designForcedLarge: mode === "design" && model === "0.6b",
  };
}

// -------------------------------------------------------------------------------------------------
// Voices, instructions, languages
// -------------------------------------------------------------------------------------------------

/** The 9 built-in voices (CustomVoice), with their native language (they speak every supported language). */
export const QWEN3_SPEAKERS = [
  { id: "ryan", name: "Ryan", lang: "en", gender: "male" },
  { id: "aiden", name: "Aiden", lang: "en", gender: "male" },
  { id: "vivian", name: "Vivian", lang: "zh", gender: "female" },
  { id: "serena", name: "Serena", lang: "zh", gender: "female" },
  { id: "uncle_fu", name: "Uncle Fu", lang: "zh", gender: "male" },
  { id: "dylan", name: "Dylan", lang: "zh", gender: "male" },
  { id: "eric", name: "Eric", lang: "zh", gender: "male" },
  { id: "ono_anna", name: "Ono Anna", lang: "ja", gender: "female" },
  { id: "sohee", name: "Sohee", lang: "ko", gender: "female" },
] as const;

export const QWEN3_DEFAULT_SPEAKER = "ryan";

/** Quick style / emotion instructions (the model reads them as written; labels are translated in the panel). */
export const QWEN3_STYLE_PRESETS = [
  { key: "happy", instruction: "Speak in a happy, cheerful tone." },
  { key: "excited", instruction: "Speak with excitement and lots of energy." },
  { key: "calm", instruction: "Speak calmly and gently, at a relaxed pace." },
  { key: "sad", instruction: "Speak in a sad, subdued voice." },
  { key: "angry", instruction: "Speak in an angry, intense tone." },
  { key: "whisper", instruction: "Whisper softly, close to the microphone." },
  { key: "serious", instruction: "Speak in a serious, steady tone." },
  { key: "playful", instruction: "Speak playfully, with a teasing smile." },
] as const;

/** Voice design examples (English descriptions; Chinese works as well). */
export const QWEN3_DESIGN_EXAMPLES = [
  "Young woman in her twenties, bright and friendly, slightly high pitch, lively and expressive.",
  "Middle-aged man, deep and warm voice, calm and confident, like a late-night radio host.",
  "Energetic teenage boy, clear tenor, fast and enthusiastic, always about to laugh.",
] as const;

/** Languages Qwen3-TTS speaks: [BCP-47 base, qwentts.cpp name]. */
export const QWEN3_LANGUAGES = [
  ["en", "english"],
  ["zh", "chinese"],
  ["ja", "japanese"],
  ["ko", "korean"],
  ["de", "german"],
  ["fr", "french"],
  ["ru", "russian"],
  ["pt", "portuguese"],
  ["es", "spanish"],
  ["it", "italian"],
] as const;

/**
 * The qwentts.cpp language name of a BCP-47 tag (`ja`, `pt-BR`, `zh-Hant`) or of a name (`Japanese`); `auto` for an
 * empty tag, `auto` or a language the model lacks (the model then guesses from the text).
 */
export function qwen3Language(tag: string | null | undefined): string {
  const t = (tag ?? "").trim().toLowerCase().replace(/_/g, "-");
  if (!t || t === "auto") return "auto";
  let base = t.split("-")[0];
  if (base === "cmn" || base === "yue" || base === "zho") base = "zh";
  if (base === "jpn") base = "ja";
  if (base === "eng") base = "en";
  const hit = QWEN3_LANGUAGES.find(([code, name]) => name === t || code === base);
  return hit ? hit[1] : "auto";
}

/** Longest instruction / description sent (the server takes any length; long ones only slow the first audio). */
export const QWEN3_MAX_INSTRUCTION = 600;
/** Seed of design mode when none is set: one designed voice for every sentence. */
export const QWEN3_DESIGN_SEED = 1234;

function seedOf(s: Pick<TTS_Qwen3_State, "seed" | "mode">): number | null {
  const raw = String(s.seed ?? "").trim();
  if (/^-?\d{1,15}$/.test(raw)) {
    const n = Number(raw);
    if (Number.isSafeInteger(n) && n >= 0) return n;
  }
  return s.mode === "design" ? QWEN3_DESIGN_SEED : null;
}

/** Why these settings cannot speak (a localized reason, or `null`). */
export function qwen3SettingsProblem(s: TTS_Qwen3_State): string | null {
  if (s.mode === "clone" && !s.referenceVoiceId.trim()) return i18n.t("tts.qwen3_clone_voice_missing");
  if (s.mode === "design" && !s.voiceDescription.trim()) return i18n.t("tts.qwen3_design_description_missing");
  return null;
}

/** The launch part of the settings, sent to `tts_helper_start` (Rust `qwen::plan::QwenConfig`). */
export interface Qwen3HelperConfig {
  model: Qwen3TtsModel;
  variant: Qwen3TtsVariant;
  mode: Qwen3TtsMode;
  language: string;
}

export function qwen3HelperConfig(s: TTS_Qwen3_State): Qwen3HelperConfig {
  return { model: s.model, variant: s.variant, mode: s.mode, language: qwen3Language(s.language) };
}

/**
 * Two configs that load the same helper (Rust may still reuse a running one for different ones). The language is not
 * part of it: every sentence names its language in the request (`qwen3SpeechSettings`), so changing it never reloads
 * the model (Rust's `Launch::same_server` leaves it out too, and the server always runs `--lang auto`).
 */
export function qwen3ConfigKey(c: Qwen3HelperConfig): string {
  return `${c.model}|${c.variant}|${c.mode}`;
}

/**
 * The `openai_speech_speak` settings of one sentence: raw 24 kHz PCM streamed, the voice, and the language / style /
 * seed as extra request members of the server. `lang` is the caption's language (used when the setting is `auto`);
 * `running` is the helper's talker id (style instructions only on the 1.7B CustomVoice).
 */
export function qwen3SpeechSettings(
  endpoint: Pick<RustTtsHelperEndpoint, "baseUrl" | "apiKey" | "model" | "pcmSampleRate">,
  s: TTS_Qwen3_State,
  lang: string | null | undefined,
  voice: string,
): OpenAiSpeechSettings {
  const language = s.language.trim() && s.language.trim().toLowerCase() !== "auto" ? qwen3Language(s.language) : qwen3Language(lang);
  const extra: Record<string, unknown> = { language };
  const size = qwen3SizeOf(endpoint.model);
  if (s.mode === "preset" && size === "1.7b") {
    const instruction = s.instruction.trim().slice(0, QWEN3_MAX_INSTRUCTION);
    if (instruction) extra.instructions = instruction;
  } else if (s.mode === "design") {
    extra.instructions = s.voiceDescription.trim().slice(0, QWEN3_MAX_INSTRUCTION);
  }
  const seed = seedOf(s);
  if (seed !== null) extra.seed = seed;
  return {
    baseUrl: endpoint.baseUrl,
    apiKey: endpoint.apiKey,
    model: "",
    voice,
    responseFormat: "pcm",
    speed: 1,
    stream: true,
    pcmSampleRate: endpoint.pcmSampleRate > 0 ? endpoint.pcmSampleRate : 24_000,
    extra,
    timeoutMs: 0,
  };
}

// -------------------------------------------------------------------------------------------------
// IPC
// -------------------------------------------------------------------------------------------------

/** Start (or reuse) the helper; rejects with Rust's reason (missing downloads, no GPU build, a crash while loading). */
export function startQwen3Helper(config: Qwen3HelperConfig): Promise<RustTtsHelperEndpoint> {
  return invoke<RustTtsHelperEndpoint>("plugin:tts-helpers|tts_helper_start", { engine: "qwen3", config });
}

/** Stop the helper (frees its video memory). Never throws. */
export function stopQwen3Helper(): void {
  try {
    void Promise.resolve(invoke("plugin:tts-helpers|tts_helper_stop", { engine: "qwen3" })).catch(() => {});
  } catch {
    // Without Tauri there is nothing to stop.
  }
}

export function qwen3HelperStatus(): Promise<RustTtsHelperStatus> {
  return invoke<RustTtsHelperStatus>("plugin:tts-helpers|tts_helper_status", { engine: "qwen3" });
}

/** Register library voice `voiceId` with the running helper; resolves with the server voice name. */
export function prepareQwen3Clone(voiceId: string): Promise<RustQwenCloneVoice> {
  return invoke<RustQwenCloneVoice>("plugin:tts-helpers|qwen_tts_prepare_clone", { voiceId });
}

/** Tauri event of the helpers (Rust `tts_helpers::STATE_EVENT`). */
export const QWEN3_STATE_EVENT = "tts_helpers:state";

// -------------------------------------------------------------------------------------------------
// Shared UI state (panel)
// -------------------------------------------------------------------------------------------------

export interface Qwen3SentenceStats {
  /** Request to the first pushed audio (ms). */
  firstAudioMs: number;
  /** Seconds of audio. */
  audioSec: number;
  /** Request to the last pushed audio (ms; includes waiting for playback). */
  totalMs: number;
  at: number;
}

export const qwen3Ui = proxy({
  /** The last sentence spoken (for the panel). */
  last: null as Qwen3SentenceStats | null,
  /** The governor asked for another size that is not installed (talker id), while another one keeps speaking. */
  swapBlocked: null as string | null,
});

export function noteQwen3Report(report: RustTtsSpeakReport | null | undefined): Qwen3SentenceStats | null {
  if (!report || !(report.sampleRate > 0)) return null;
  const stats: Qwen3SentenceStats = {
    firstAudioMs: Math.round(report.firstAudioMs),
    audioSec: Math.round((report.frames / report.sampleRate) * 100) / 100,
    totalMs: Math.round(report.totalMs),
    at: Date.now(),
  };
  qwen3Ui.last = stats;
  return stats;
}

/** Whether the primary GPU can run the CUDA 13 build, from the sysinfo service (false when unknown). */
export function qwen3CudaUsable(): boolean {
  const info = window.ApiServer?.sysinfo?.state?.info as SysSystemInfo | null | undefined;
  if (!info?.eps?.cuda?.eligible) return false;
  const gpu = primaryGpu(info);
  return !!gpu && gpu.vendor === "nvidia" && !gpu.integrated;
}

/** The plan for `s` with the live recommendation, installed models and GPU facts. */
export function currentQwen3Plan(s: TTS_Qwen3_State): Qwen3Plan {
  const states = (window.ApiServer?.models?.state?.states ?? {}) as Record<string, { state?: string } | undefined>;
  return planQwen3(s, {
    recommendation: window.ApiServer?.resources?.state?.recommendations?.qwen3_tts as Qwen3Recommendation | undefined,
    installed: (id) => states[id]?.state === "installed",
    cudaUsable: qwen3CudaUsable(),
  });
}

// -------------------------------------------------------------------------------------------------
// Preview (the panel's test button)
// -------------------------------------------------------------------------------------------------

/** A preview started the helper while text to speech was off: stop it after this long without another preview. */
export const QWEN3_PREVIEW_IDLE_MS = 120_000;
let previewIdleTimer: ReturnType<typeof setTimeout> | null = null;

function clearPreviewIdle() {
  if (previewIdleTimer) clearTimeout(previewIdleTimer);
  previewIdleTimer = null;
}

/** Whether text to speech runs on this backend now (then the helper belongs to it). */
function serviceUsesQwen3(): boolean {
  const tts = window.ApiServer?.tts;
  const data = window.ApiServer?.state?.services?.tts?.data;
  return !!tts && data?.backend === "qwen3" && tts.serviceState?.status !== ServiceNetworkState.disconnected;
}

/** Speak `text` once with the panel's settings into a new sink utterance on `target`. */
export async function previewQwen3(s: TTS_Qwen3_State, text: string, target: RustTtsSinkTarget): Promise<void> {
  const problem = qwen3SettingsProblem(s);
  if (problem) throw new Error(problem);
  clearPreviewIdle();
  try {
    const endpoint = await startQwen3Helper(qwen3HelperConfig(s));
    const voice = s.mode === "clone" ? (await prepareQwen3Clone(s.referenceVoiceId.trim())).voice : s.mode === "preset" ? s.voice.trim() || QWEN3_DEFAULT_SPEAKER : "";
    const settings = qwen3SpeechSettings(endpoint, s, null, voice);
    await previewToSink(target, async (id) => noteQwen3Report(await speakOpenAi(settings, text, id)));
  } finally {
    if (!serviceUsesQwen3()) {
      previewIdleTimer = setTimeout(() => {
        previewIdleTimer = null;
        if (!serviceUsesQwen3()) stopQwen3Helper();
      }, QWEN3_PREVIEW_IDLE_MS);
    }
  }
}

// -------------------------------------------------------------------------------------------------
// The backend
// -------------------------------------------------------------------------------------------------

/** How long a reload waits for earlier sentences to finish streaming from the old helper. */
export const QWEN3_RELOAD_WAIT_MS = 30_000;
/** Quiet time after a recommendation change before the size swap (bursts of resource events coalesce). */
const SWAP_DEBOUNCE_MS = 2_000;
/** Reloads one sentence may wait for when the settings keep changing; then it speaks with what runs. */
const MAX_RELOADS_PER_SENTENCE = 3;

function looksLikeConnectionError(error: unknown): boolean {
  const m = errorMessage(error).toLowerCase();
  return /connect|connection|refused|unreachable|reset|closed|not running|restarting|no endpoint/.test(m);
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export class TTS_Qwen3Service implements ITTSService {
  #running = false;
  #stopped = false;
  #endpoint: RustTtsHelperEndpoint | null = null;
  /** Config the helper was (re)started with. */
  #launchKey = "";
  /** The helper may have a new port (it restarted) or was replaced elsewhere: start again before the next sentence. */
  #dirty = false;
  #reload: Promise<void> | null = null;
  /**
   * Sentences holding the current endpoint (from the moment they pass the reload check until their stream ends). A
   * reload waits for these only, never for sentences queued behind it, so the pipeline's look-ahead cannot stall it.
   */
  #leases = 0;
  #disposers: (() => void)[] = [];
  #swapTimer: ReturnType<typeof setTimeout> | null = null;
  #recChoice: string | null = null;

  constructor(private readonly receiver: ITTSReceiver) {}

  async start(state: TTS_State): Promise<void> {
    if (!isTauri()) {
      this.receiver.onStop(i18n.t("tts.qwen3_desktop_only"));
      return;
    }
    const s = state.qwen3;
    const problem = qwen3SettingsProblem(s);
    if (problem) {
      this.receiver.onStop(`${TTS_OPTIONS_MISSING}: ${problem}`);
      return;
    }
    clearPreviewIdle();
    try {
      const config = qwen3HelperConfig(s);
      const endpoint = await startQwen3Helper(config);
      if (this.#stopped) return;
      this.#endpoint = endpoint;
      this.#launchKey = qwen3ConfigKey(config);
      if (s.mode === "clone") await prepareQwen3Clone(s.referenceVoiceId.trim());
      if (this.#stopped) return;
      await this.#watch();
      if (this.#stopped) {
        this.#unwatch();
        return;
      }
      this.#running = true;
      this.receiver.onStart();
    } catch (error) {
      if (this.#stopped) return;
      this.receiver.onStop(i18n.t("tts.qwen3_start_failed", { error: errorMessage(error) }));
    }
  }

  /** v2: stream one sentence into the begun sink utterance. */
  async synthesizeToSink(text: string, sink: TtsSinkContext, signal: AbortSignal): Promise<void> {
    if (!this.#running || signal.aborted || !text.trim()) return;
    const onAbort = () => cancelHttpSpeech(sink.utteranceId);
    signal.addEventListener("abort", onAbort, { once: true });
    let leased = false;
    try {
      const s = window.ApiServer.state.services.tts.data.qwen3 as TTS_Qwen3_State;
      const problem = qwen3SettingsProblem(s);
      if (problem) throw new Error(problem);
      let endpoint: RustTtsHelperEndpoint | null = null;
      let voice = "";
      for (let attempt = 0; ; attempt++) {
        leased = await this.#current(s, true);
        if (!leased || signal.aborted || !this.#running || !this.#endpoint) return;
        endpoint = this.#endpoint;
        try {
          voice = await this.#voiceFor(s);
          break;
        } catch (error) {
          if (attempt > 0 || !looksLikeConnectionError(error)) throw error;
          // The helper restarted (or moved to another port): let go of it, find it again, register again.
          this.#dirty = true;
          this.#leases -= 1;
          leased = false;
        }
      }
      if (signal.aborted || !this.#running) return;
      const settings = qwen3SpeechSettings(endpoint, s, sink.lang, voice);
      const report = await speakOpenAi(settings, text, sink.utteranceId);
      noteQwen3Report(report);
    } catch (error) {
      if (signal.aborted || isSinkCancelled(error) || !this.#running) return;
      if (looksLikeConnectionError(error)) this.#dirty = true;
      throw new Error(`Qwen3-TTS: ${errorMessage(error)}`);
    } finally {
      if (leased) this.#leases -= 1;
      signal.removeEventListener("abort", onAbort);
    }
  }

  /** Legacy whole-line path: unused (the sentence pipeline drives `synthesizeToSink`). */
  play(_value: string): void {}

  stop(): void {
    this.#halt();
    this.receiver.onStop();
  }

  dispose(): void {
    this.#halt();
  }

  #halt() {
    const wasActive = !this.#stopped;
    this.#stopped = true;
    this.#running = false;
    this.#unwatch();
    if (wasActive) {
      cancelHttpSpeech(null);
      stopQwen3Helper();
    }
  }

  /**
   * The server voice of this sentence: a built-in voice, the library voice registered with the running helper (cheap
   * once registered; a restarted helper registers it again), or none (design).
   */
  async #voiceFor(s: TTS_Qwen3_State): Promise<string> {
    if (s.mode === "clone") return (await prepareQwen3Clone(s.referenceVoiceId.trim())).voice;
    if (s.mode === "preset") return s.voice.trim() || QWEN3_DEFAULT_SPEAKER;
    return "";
  }

  /**
   * Make the helper run `s`'s launch settings: reload it when they changed, or find it again after a restart (a new
   * port and key). With `lease`, the caller then holds the endpoint for one sentence (release with `#leases -= 1`);
   * the check and the lease happen in one step, so a reload can never start in between. Resolves `false` once the
   * backend stopped; rejects with the reason a reload failed.
   */
  async #current(s: TTS_Qwen3_State, lease: boolean): Promise<boolean> {
    for (let reloads = 0; ; ) {
      if (this.#reload) {
        await this.#reload;
        continue;
      }
      if (!this.#running) return false;
      const config = qwen3HelperConfig(s);
      const key = qwen3ConfigKey(config);
      const current = key === this.#launchKey && !this.#dirty && !!this.#endpoint;
      if (current || (reloads >= MAX_RELOADS_PER_SENTENCE && this.#endpoint)) {
        if (lease) this.#leases += 1;
        return true;
      }
      reloads += 1;
      this.#reload = this.#reloadTo(config, key).finally(() => {
        this.#reload = null;
      });
      await this.#reload;
    }
  }

  /** Start the helper with `config` once the sentences streaming from the running one are done (bounded wait). */
  async #reloadTo(config: Qwen3HelperConfig, key: string): Promise<void> {
    const deadline = Date.now() + QWEN3_RELOAD_WAIT_MS;
    while (this.#leases > 0 && this.#running && Date.now() < deadline) await sleep(50);
    if (!this.#running) return;
    const endpoint = await startQwen3Helper(config);
    if (!this.#running) return;
    this.#endpoint = endpoint;
    this.#launchKey = key;
    this.#dirty = false;
  }

  async #watch(): Promise<void> {
    try {
      const unlisten: UnlistenFn = await listen<RustTtsHelperStatus>(QWEN3_STATE_EVENT, (e) => this.#onHelperState(e.payload));
      this.#disposers.push(unlisten);
    } catch {
      // No event bridge: connection errors still mark the helper for a fresh start.
    }
    const recs = window.ApiServer?.resources?.state?.recommendations;
    if (recs) {
      this.#recChoice = recs.qwen3_tts?.choice ?? null;
      this.#disposers.push(subscribe(recs, () => this.#onRecommendation()));
    }
  }

  #unwatch() {
    if (this.#swapTimer) clearTimeout(this.#swapTimer);
    this.#swapTimer = null;
    const disposers = this.#disposers;
    this.#disposers = [];
    for (const d of disposers) {
      try {
        d();
      } catch {
        // already gone
      }
    }
  }

  #onHelperState(status: RustTtsHelperStatus | null | undefined) {
    if (!this.#running || status?.engine !== "qwen3" || this.#reload) return;
    if (status.state === "starting" || status.state === "stopped") {
      // Restarting by itself (a crash), or stopped from elsewhere: start / find it again before the next sentence.
      this.#dirty = true;
    } else if (status.state === "failed") {
      this.#dirty = true;
      this.receiver.onError(i18n.t("tts.qwen3_helper_failed", { error: status.error ?? "" }));
    }
  }

  /** The governor changed its mind (the voice changer started or stopped on the GPU): swap the size when idle. */
  #onRecommendation() {
    const choice = window.ApiServer?.resources?.state?.recommendations?.qwen3_tts?.choice ?? null;
    if (choice === this.#recChoice) return;
    this.#recChoice = choice;
    const s = window.ApiServer?.state?.services?.tts?.data?.qwen3 as TTS_Qwen3_State | undefined;
    if (!this.#running || !s || s.model !== "auto" || s.mode === "design") return;
    if (this.#swapTimer) clearTimeout(this.#swapTimer);
    this.#swapTimer = setTimeout(() => {
      this.#swapTimer = null;
      void this.#swapSize();
    }, SWAP_DEBOUNCE_MS);
  }

  async #swapSize(): Promise<void> {
    const s = window.ApiServer?.state?.services?.tts?.data?.qwen3 as TTS_Qwen3_State | undefined;
    if (!this.#running || !s || s.model !== "auto") return;
    const plan = currentQwen3Plan(s);
    const runningSize = qwen3SizeOf(this.#endpoint?.model);
    if (!runningSize || plan.size === runningSize) {
      qwen3Ui.swapBlocked = null;
      return;
    }
    const states = (window.ApiServer?.models?.state?.states ?? {}) as Record<string, { state?: string } | undefined>;
    const installed = (id: string) => states[id]?.state === "installed";
    if (!plan.ids.every(installed)) {
      // The recommended size is not downloaded: the loaded one keeps speaking.
      qwen3Ui.swapBlocked = plan.talker;
      return;
    }
    qwen3Ui.swapBlocked = null;
    try {
      // Rust plans the size again; sentences speaking now finish on the loaded model first, later ones wait for the
      // other size.
      this.#dirty = true;
      await this.#current(s, false);
    } catch (error) {
      if (this.#running) this.receiver.onError(i18n.t("tts.qwen3_swap_failed", { error: errorMessage(error) }));
    }
  }
}
