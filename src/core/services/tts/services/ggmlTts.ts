import { invoke, isTauri } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import i18n from "i18next";
import { proxy } from "valtio";
import {
  ServiceNetworkState,
  type RustTtsHelperEndpoint,
  type RustTtsHelperStatus,
  type RustTtsSinkTarget,
  type RustTtsSpeakReport,
} from "@/types";
import { isSinkCancelled } from "../../sound/sink";
import { primaryGpu, type SysSystemInfo } from "../../sysinfo/format";
import { clampedOr } from "../num";
import type { TTS_State } from "../schema";
import type { GgmlTtsEngine, GgmlTtsVariant, TTS_GgmlTts_State } from "../schema/ggmlTts";
import { ITTSReceiver, ITTSService, TTS_OPTIONS_MISSING, TtsSinkContext, errorMessage } from "../types";
import { cancelHttpSpeech, previewToSink, speakOpenAi, type OpenAiSpeechSettings } from "./openaiSpeech";

/**
 * `ggmlTts` backend (P6-F): the experimental ggml TTS helpers, each a server the Rust `tts-helpers` plugin starts
 * (`tts_helper_start { engine, config }`: 127.0.0.1, a random key, a kill-on-close Job Object, a VRAM claim):
 * - `magpie`: NVIDIA Magpie TTS Multilingual through NeMo-Speech.cpp (5 voices; en es de fr it vi hi, plus ko ar pt-BR
 *   with the v2607 checkpoint), one 22.05 kHz clip per sentence.
 * - `breeze`: Breeze TTS 2 through Breeze-TTS-2.cpp (English and Mandarin): a voice designed or directed in words, or a
 *   voice-library voice cloned; streamed 24 kHz.
 * - `omnivoice`: OmniVoice through CrispASR (600+ languages): an automatic voice or a cloned library voice.
 * - `s2`: Fish S2 Pro through s2.cpp: an automatic voice or a cloned library voice, inline tags such as `[laugh]`.
 *
 * Every sentence is one `openai_speech_speak` request (Breeze and S2 go through Rust's in-process adapter, which turns
 * it into their own request shape), streamed into the TTS sink. Requests are sent one at a time, in sentence order:
 * Breeze and s2.cpp refuse a second concurrent request, and the others serialize internally anyway. Settings that decide
 * what the helper loads reload it before the next sentence; a helper that crashed restarts by itself (process host) and
 * the next sentence picks up its new port.
 */

// -------------------------------------------------------------------------------------------------
// Engine facts (mirror of Rust tts_helpers::{magpie,breeze,omnivoice,s2})
// -------------------------------------------------------------------------------------------------

export type GgmlVariant = Exclude<GgmlTtsVariant, "auto">;

export const NVIDIA_RUNTIME_ID = "runtime/nvidia-cuda13";

export interface GgmlPack {
  variant: GgmlVariant;
  runtime: string;
}

export interface GgmlModel {
  /** Settings value. */
  key: string;
  /** Catalog id. */
  id: string;
  /** Catalog ids it runs with (tokenizer, codec). */
  companions: string[];
}

export interface GgmlEngineFacts {
  id: GgmlTtsEngine;
  /** English product name (not translated). */
  name: string;
  /** Runtime packs in preference order. */
  packs: GgmlPack[];
  /** The first is the default. */
  models: GgmlModel[];
  /** Clones voices from the voice library. */
  cloning: boolean;
  /** Takes a voice instruction (Breeze). */
  instructions: boolean;
  /** Packs Sigil builds (Import, never downloaded). */
  sigilBuiltPacks: boolean;
}

const MAGPIE_TOKENIZER_V2602 = "tts/magpie-tts-357m-v2602-tokenizer";
const MAGPIE_TOKENIZER_V2607 = "tts/magpie-tts-357m-v2607-tokenizer";
const MAGPIE_CODEC = "tts/nemo-nano-codec-22khz";
const OMNIVOICE_CODEC = "tts/omnivoice-tokenizer-f16";
const S2_TOKENIZER = "tts/s2-pro-tokenizer";

export const GGML_ENGINES: Readonly<Record<GgmlTtsEngine, GgmlEngineFacts>> = {
  magpie: {
    id: "magpie",
    name: "NVIDIA Magpie TTS",
    packs: [
      { variant: "cuda", runtime: "runtime/nemo-speech-cuda" },
      { variant: "vulkan", runtime: "runtime/nemo-speech-vulkan" },
      { variant: "cpu", runtime: "runtime/nemo-speech-cpu" },
    ],
    models: [
      { key: "v2602", id: "tts/magpie-tts-357m-v2602", companions: [MAGPIE_TOKENIZER_V2602, MAGPIE_CODEC] },
      { key: "v2607", id: "tts/magpie-tts-357m-v2607", companions: [MAGPIE_TOKENIZER_V2607, MAGPIE_CODEC] },
    ],
    cloning: false,
    instructions: false,
    sigilBuiltPacks: false,
  },
  breeze: {
    id: "breeze",
    name: "Breeze TTS 2",
    packs: [
      { variant: "vulkan", runtime: "runtime/breeze-server-vulkan" },
      // The same pack, started with --cpu.
      { variant: "cpu", runtime: "runtime/breeze-server-vulkan" },
    ],
    models: [
      { key: "q8_0", id: "tts/breeze-tts-2-q8_0", companions: [] },
      { key: "q4_k", id: "tts/breeze-tts-2-q4_k", companions: [] },
    ],
    cloning: true,
    instructions: true,
    sigilBuiltPacks: false,
  },
  omnivoice: {
    id: "omnivoice",
    name: "OmniVoice",
    packs: [
      { variant: "cuda13", runtime: "runtime/crispasr-cuda13" },
      { variant: "vulkan", runtime: "runtime/crispasr-vulkan" },
      { variant: "cpu", runtime: "runtime/crispasr-cpu" },
    ],
    models: [
      { key: "q8_0", id: "tts/omnivoice-q8_0", companions: [OMNIVOICE_CODEC] },
      { key: "f16", id: "tts/omnivoice-f16", companions: [OMNIVOICE_CODEC] },
    ],
    cloning: true,
    instructions: false,
    sigilBuiltPacks: false,
  },
  s2: {
    id: "s2",
    name: "Fish S2 Pro",
    packs: [
      { variant: "cuda13", runtime: "runtime/s2-server-cuda13" },
      { variant: "vulkan", runtime: "runtime/s2-server-vulkan" },
      { variant: "cpu", runtime: "runtime/s2-server-cpu" },
    ],
    models: [
      { key: "q4_k_m", id: "tts/s2-pro-q4_k_m", companions: [S2_TOKENIZER] },
      { key: "q6_k", id: "tts/s2-pro-q6_k", companions: [S2_TOKENIZER] },
      { key: "q8_0", id: "tts/s2-pro-q8_0", companions: [S2_TOKENIZER] },
    ],
    cloning: true,
    instructions: false,
    sigilBuiltPacks: true,
  },
};

/**
 * Catalog entries the app cannot download (Import until they are published): the Sigil-hosted Magpie tokenizers and the
 * v2607 checkpoint (NVIDIA's own GGUF, pinned, published together with its tokenizer once `real_ggml_engine_speaks` has
 * passed on the v0.2.0 packs), and the Sigil-built s2 packs. A test keeps every id here `unpublished` in the catalog, so
 * publishing one means removing it from this set.
 */
export const GGML_LOCAL_ONLY_IDS: ReadonlySet<string> = new Set([
  MAGPIE_TOKENIZER_V2602,
  MAGPIE_TOKENIZER_V2607,
  "tts/magpie-tts-357m-v2607",
  ...GGML_ENGINES.s2.packs.map((p) => p.runtime),
]);

export function ggmlEngine(engine: string | null | undefined): GgmlEngineFacts {
  return GGML_ENGINES[(engine ?? "") as GgmlTtsEngine] ?? GGML_ENGINES.magpie;
}

/** The engine's model for a settings value (unknown or another engine's value = its default). */
export function ggmlModel(engine: GgmlEngineFacts, key: string | null | undefined): GgmlModel {
  const k = (key ?? "").trim().toLowerCase();
  return engine.models.find((m) => m.key === k || m.id === k) ?? engine.models[0];
}

export interface GgmlGpu {
  /** A hardware GPU is present. */
  anyGpu: boolean;
  /** The primary GPU runs CUDA 13 (NVIDIA Turing or newer, driver 580+). */
  cuda: boolean;
}

export interface GgmlPlanEnv {
  installed: (id: string) => boolean;
  gpu: GgmlGpu;
}

export interface GgmlPlan {
  engine: GgmlEngineFacts;
  model: GgmlModel;
  pack: GgmlPack;
  /** Every catalog id the plan needs (model, companions, pack, NVIDIA runtime for CUDA 13), for `<ModelRequirement>`. */
  ids: string[];
  /** Why this variant cannot run on this PC (Rust refuses it with the same reason). */
  refused: string | null;
}

const isNvidia = (v: GgmlVariant) => v === "cuda13" || v === "cuda";

/** The runtime pack, as Rust's `launch::pick_pack` chooses it. */
export function pickGgmlPack(
  engine: GgmlEngineFacts,
  choice: GgmlTtsVariant,
  env: GgmlPlanEnv,
): { pack: GgmlPack; refused: string | null } {
  const find = (v: GgmlVariant) => engine.packs.find((p) => p.variant === v);
  const ready = (p: GgmlPack) => env.installed(p.runtime) && (p.variant !== "cuda13" || env.installed(NVIDIA_RUNTIME_ID));
  if (choice !== "auto") {
    const p = find(choice);
    if (!p) return { pack: engine.packs[0], refused: i18n.t("tts.ggml_variant_missing", { engine: engine.name }) };
    if (isNvidia(choice) && !env.gpu.cuda) return { pack: p, refused: i18n.t("tts.ggml_variant_needs_cuda") };
    if (choice !== "cpu" && !env.gpu.anyGpu) return { pack: p, refused: i18n.t("tts.ggml_variant_needs_gpu") };
    return { pack: p, refused: null };
  }
  const cpu = find("cpu");
  if (!env.gpu.anyGpu) {
    return cpu ? { pack: cpu, refused: null } : { pack: engine.packs[0], refused: i18n.t("tts.ggml_variant_needs_gpu") };
  }
  const nvidia = engine.packs.filter((p) => isNvidia(p.variant) && env.gpu.cuda);
  const pick =
    nvidia.find(ready) ??
    [find("vulkan")].find((p): p is GgmlPack => !!p && ready(p)) ??
    nvidia.find((p) => env.installed(p.runtime) || (p.variant === "cuda13" && env.installed(NVIDIA_RUNTIME_ID))) ??
    find("vulkan") ??
    nvidia[0] ??
    cpu ??
    engine.packs[0];
  return { pack: pick, refused: null };
}

/** The model, pack and downloads for these settings, as Rust plans them (Rust decides at start). */
export function planGgml(s: Pick<TTS_GgmlTts_State, "engine" | "model" | "variant">, env: GgmlPlanEnv): GgmlPlan {
  const engine = ggmlEngine(s.engine);
  const model = ggmlModel(engine, s.model);
  const { pack, refused } = pickGgmlPack(engine, s.variant, env);
  const ids = [model.id, ...model.companions, pack.runtime];
  if (pack.variant === "cuda13") ids.push(NVIDIA_RUNTIME_ID);
  return { engine, model, pack, ids: [...new Set(ids)], refused };
}

// -------------------------------------------------------------------------------------------------
// Voices and languages
// -------------------------------------------------------------------------------------------------

/** Magpie's five built-in voices. */
export const MAGPIE_SPEAKERS = ["John", "Sofia", "Aria", "Jason", "Leo"] as const;

/**
 * Languages the official NeMo-Speech.cpp packs speak per Magpie checkpoint: [BCP-47 base, server code]. Mandarin and
 * Japanese need a source build with their frontends, so they are not offered.
 */
const MAGPIE_V2602_LANGUAGES: ReadonlyArray<readonly [string, string]> = [
  ["en", "en-US"],
  ["es", "es-ES"],
  ["de", "de-DE"],
  ["fr", "fr-FR"],
  ["it", "it-IT"],
  ["vi", "vi-VN"],
  ["hi", "hi-IN"],
];
const MAGPIE_V2607_EXTRA: ReadonlyArray<readonly [string, string]> = [
  ["ko", "ko-KR"],
  ["ar", "ar-MSA"],
  ["pt", "pt-BR"],
];

export function magpieLanguages(modelKey: string): ReadonlyArray<readonly [string, string]> {
  return modelKey === "v2607" ? [...MAGPIE_V2602_LANGUAGES, ...MAGPIE_V2607_EXTRA] : MAGPIE_V2602_LANGUAGES;
}

/** Lower-case BCP-47 base of a tag (`pt-BR` -> `pt`), `null` for empty or `auto`. */
export function languageBase(tag: string | null | undefined): string | null {
  const t = (tag ?? "").trim().toLowerCase().replace(/_/g, "-");
  if (!t || t === "auto") return null;
  const base = t.split("-")[0];
  if (base === "cmn" || base === "yue" || base === "zho") return "zh";
  if (base === "jpn") return "ja";
  if (base === "eng") return "en";
  return /^[a-z]{2,3}$/.test(base) ? base : null;
}

/** Languages offered in the panel's language select for an engine (`null`: the engine takes no language). */
export function ggmlLanguageChoices(s: Pick<TTS_GgmlTts_State, "engine" | "model">): string[] | null {
  const engine = ggmlEngine(s.engine);
  if (engine.id === "magpie") return magpieLanguages(ggmlModel(engine, s.model).key).map(([base]) => base);
  if (engine.id === "omnivoice") return OMNIVOICE_COMMON_LANGUAGES;
  return null;
}

/** A short list of common languages for OmniVoice's select (it speaks any BCP-47 language; others stay as set). */
export const OMNIVOICE_COMMON_LANGUAGES = ["en", "ja", "zh", "ko", "es", "pt", "fr", "de", "it", "ru", "ar", "hi", "id", "th", "vi", "tr", "pl", "nl", "uk", "sv"];

/**
 * Whether the language setting can be used with these settings' engine and model: `auto`, an engine that takes no
 * language (Breeze, S2: the setting is ignored), any language for OmniVoice, a language of the checkpoint for Magpie.
 */
export function ggmlLanguageUsable(s: Pick<TTS_GgmlTts_State, "engine" | "model" | "language">): boolean {
  const wanted = languageBase(s.language);
  if (!wanted) return true;
  const engine = ggmlEngine(s.engine);
  if (engine.id !== "magpie") return true;
  return magpieLanguages(ggmlModel(engine, s.model).key).some(([base]) => base === wanted);
}

/** The language setting to keep after the engine or the model changed: `auto` when the new one cannot speak it. */
export function ggmlLanguageAfterChange(s: Pick<TTS_GgmlTts_State, "engine" | "model" | "language">): string {
  return ggmlLanguageUsable(s) ? s.language : "auto";
}

/**
 * The language of one sentence for the engine: the setting when it names one, else the caption's. Magpie: its server
 * code, or an error for a language its checkpoint does not speak. OmniVoice: the BCP-47 base (`undefined`: let it read
 * the script). Breeze and S2 take none.
 */
export function ggmlSentenceLanguage(s: Pick<TTS_GgmlTts_State, "engine" | "model" | "language">, captionLang: string | null | undefined): string | undefined {
  const engine = ggmlEngine(s.engine);
  const wanted = languageBase(s.language) ?? languageBase(captionLang);
  if (engine.id === "magpie") {
    const langs = magpieLanguages(ggmlModel(engine, s.model).key);
    if (!wanted) return "en-US";
    const hit = langs.find(([base]) => base === wanted);
    if (!hit) throw new Error(i18n.t("tts.ggml_magpie_language_unsupported", { language: wanted }));
    return hit[1];
  }
  if (engine.id === "omnivoice") return wanted ?? undefined;
  return undefined;
}

/** Default seeds that keep a designed or automatic voice steady (Breeze's own default is 42). */
const DEFAULT_SEED: Partial<Record<GgmlTtsEngine, number>> = { breeze: 42, omnivoice: 7 };

export function ggmlSeed(s: Pick<TTS_GgmlTts_State, "engine" | "seed">): number | null {
  const raw = String(s.seed ?? "").trim();
  if (/^\d{1,9}$/.test(raw)) return Number(raw);
  return DEFAULT_SEED[s.engine] ?? null;
}

/** Longest instruction sent (Rust's adapter cuts at 1000 characters too). */
export const GGML_MAX_INSTRUCTION = 1000;

/** The text CrispASR logs as the clone consent (the user ticked the switch). */
export const OMNIVOICE_CONSENT_TEXT = "The Sigil user confirmed they have the consent of the person whose voice is cloned.";
/** The text CrispASR logs when the spoken AI notice is skipped (the user ticked the switch). */
export const OMNIVOICE_DISCLOSURE_TEXT = "The Sigil user discloses to their audience that this speech is AI-generated.";

/** Why these settings cannot speak (a localized reason, or `null`). */
export function ggmlSettingsProblem(s: TTS_GgmlTts_State): string | null {
  const engine = ggmlEngine(s.engine);
  const cloning = engine.cloning && !!s.referenceVoiceId.trim();
  if (engine.id === "omnivoice" && cloning && !s.cloneConsent) return i18n.t("tts.ggml_omnivoice_consent_missing");
  if (engine.id === "magpie") {
    const voice = s.voice.trim();
    if (voice && !MAGPIE_SPEAKERS.some((v) => v.toLowerCase() === voice.toLowerCase())) {
      return i18n.t("tts.ggml_magpie_voice_unknown", { voice });
    }
    // A language chosen for another engine or checkpoint (the panel resets it on a change; older settings may not).
    if (!ggmlLanguageUsable(s)) return i18n.t("tts.ggml_magpie_language_unsupported", { language: languageBase(s.language) });
  }
  return null;
}

/** The launch part of the settings, sent to `tts_helper_start` (Rust `ggml::GgmlConfig`). */
export interface GgmlHelperConfig {
  model: string;
  variant: GgmlTtsVariant;
  language: string;
  voice: string;
  referenceVoiceId: string;
  quality: string;
  lowVram: boolean;
}

export function ggmlHelperConfig(s: TTS_GgmlTts_State): GgmlHelperConfig {
  const engine = ggmlEngine(s.engine);
  const magpie = engine.id === "magpie";
  return {
    model: ggmlModel(engine, s.model).key,
    variant: s.variant,
    language: languageBase(s.language) ?? "",
    voice: magpie ? s.voice.trim() : "",
    // Only OmniVoice copies the library voice at start; Breeze and S2 read it per request.
    referenceVoiceId: engine.id === "omnivoice" ? s.referenceVoiceId.trim() : "",
    quality: engine.id === "omnivoice" ? s.quality : "",
    lowVram: engine.id === "s2" && !!s.lowVram,
  };
}

/**
 * Two settings that run the same helper (Rust `Launch::same_server` agrees). Magpie's voice and language go with every
 * request and are never launch arguments (Rust uses them only for the warm-up), so they are not part of the key.
 */
export function ggmlLaunchKey(engine: GgmlTtsEngine, c: GgmlHelperConfig): string {
  return [engine, c.model, c.variant, c.referenceVoiceId, c.quality, c.lowVram ? "low" : ""].join("|");
}

/**
 * The `openai_speech_speak` settings of one sentence. `lang` is the caption's language; throws a readable error when
 * the engine cannot speak it.
 */
export function ggmlSpeechSettings(
  endpoint: Pick<RustTtsHelperEndpoint, "baseUrl" | "apiKey" | "pcmSampleRate">,
  s: TTS_GgmlTts_State,
  lang: string | null | undefined,
): OpenAiSpeechSettings {
  const engine = ggmlEngine(s.engine);
  const reference = engine.cloning ? s.referenceVoiceId.trim() : "";
  const extra: Record<string, unknown> = {};
  let voice = "";
  let stream = true;
  const seed = ggmlSeed(s);
  switch (engine.id) {
    case "magpie": {
      const wanted = s.voice.trim();
      voice = MAGPIE_SPEAKERS.find((v) => v.toLowerCase() === wanted.toLowerCase()) ?? MAGPIE_SPEAKERS[0];
      extra.language = ggmlSentenceLanguage(s, lang);
      // The server answers a whole clip per request.
      stream = false;
      break;
    }
    case "breeze": {
      const instruction = s.instruction.trim().slice(0, GGML_MAX_INSTRUCTION);
      if (instruction) extra.instructions = instruction;
      extra.cfg_scale = Math.round(clampedOr(s.cfgScale, 1, 1, 3) * 100) / 100;
      if (seed !== null) extra.seed = seed;
      if (reference) extra.sigil_reference_voice = reference;
      break;
    }
    case "omnivoice": {
      const language = ggmlSentenceLanguage(s, lang);
      if (language) extra.language = language;
      if (seed !== null) extra.seed = seed;
      if (reference) {
        voice = reference;
        if (s.cloneConsent) extra.consent_attestation = OMNIVOICE_CONSENT_TEXT;
        if (s.discloseAi) {
          extra.spoken_disclaimer = false;
          extra.marking_attestation = OMNIVOICE_DISCLOSURE_TEXT;
        }
      }
      break;
    }
    case "s2": {
      if (reference) extra.sigil_reference_voice = reference;
      // Whole clips: s2.cpp's live chunks re-decode the confirmed prefix (slower overall).
      stream = false;
      break;
    }
  }
  return {
    baseUrl: endpoint.baseUrl,
    apiKey: endpoint.apiKey,
    model: "",
    voice,
    responseFormat: "pcm",
    speed: 1,
    stream,
    pcmSampleRate: endpoint.pcmSampleRate > 0 ? endpoint.pcmSampleRate : 24_000,
    extra,
    timeoutMs: 0,
  };
}

// -------------------------------------------------------------------------------------------------
// IPC
// -------------------------------------------------------------------------------------------------

/** Longest wait for an engine's pending stop before a start of it (or a load of another engine) goes ahead. */
export const GGML_STOP_WAIT_MS = 15_000;

/** Stops sent and not finished yet, per engine. */
const pendingStops = new Map<GgmlTtsEngine, Promise<void>>();

/** `p`, or `ms` elapsed first (never rejects when `p` does not). */
function waitAtMost(p: Promise<unknown>, ms: number): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const elapsed = new Promise<void>((r) => (timer = setTimeout(r, ms)));
  return Promise.race([p.then(() => undefined), elapsed]).finally(() => clearTimeout(timer));
}

/**
 * Start (or reuse) an engine's helper; rejects with Rust's reason (missing downloads, no GPU build, a crash while
 * loading). A stop of the same engine sent before is waited for first: Rust cancels a start that a stop overtakes, and
 * two commands sent back to back may run in either order.
 */
export async function startGgmlHelper(engine: GgmlTtsEngine, config: GgmlHelperConfig): Promise<RustTtsHelperEndpoint> {
  const stopping = pendingStops.get(engine);
  if (stopping) await waitAtMost(stopping, GGML_STOP_WAIT_MS);
  return invoke<RustTtsHelperEndpoint>("plugin:tts-helpers|tts_helper_start", { engine, config });
}

/**
 * Stop an engine's helper (frees its video memory). Resolves once Rust stopped it; never rejects (a failure goes to
 * `onError`). A later start of the same engine waits for it.
 */
export function stopGgmlHelper(engine: GgmlTtsEngine, onError?: (error: unknown) => void): Promise<void> {
  let sent: Promise<void>;
  try {
    sent = Promise.resolve(invoke("plugin:tts-helpers|tts_helper_stop", { engine })).then(
      () => undefined,
      (error) => {
        try {
          onError?.(error);
        } catch {
          // A reporting failure is not the stop's.
        }
      },
    );
  } catch {
    // Without Tauri there is nothing to stop.
    sent = Promise.resolve();
  }
  const tracked: Promise<void> = sent.finally(() => {
    if (pendingStops.get(engine) === tracked) pendingStops.delete(engine);
  });
  pendingStops.set(engine, tracked);
  return tracked;
}

export function ggmlHelperStatus(engine: GgmlTtsEngine): Promise<RustTtsHelperStatus> {
  return invoke<RustTtsHelperStatus>("plugin:tts-helpers|tts_helper_status", { engine });
}

/** Tauri event of the helpers (Rust `tts_helpers::STATE_EVENT`). */
export const GGML_STATE_EVENT = "tts_helpers:state";

// -------------------------------------------------------------------------------------------------
// Shared UI state (panel)
// -------------------------------------------------------------------------------------------------

export interface GgmlSentenceStats {
  engine: GgmlTtsEngine;
  /** Request to the first pushed audio (ms). */
  firstAudioMs: number;
  /** Seconds of audio. */
  audioSec: number;
  /** Request to the last pushed audio (ms; includes waiting for playback). */
  totalMs: number;
  at: number;
}

export const ggmlUi = proxy({
  /** The last sentence spoken (for the panel). */
  last: null as GgmlSentenceStats | null,
});

export function noteGgmlReport(engine: GgmlTtsEngine, report: RustTtsSpeakReport | null | undefined): GgmlSentenceStats | null {
  if (!report || !(report.sampleRate > 0)) return null;
  const stats: GgmlSentenceStats = {
    engine,
    firstAudioMs: Math.round(report.firstAudioMs),
    audioSec: Math.round((report.frames / report.sampleRate) * 100) / 100,
    totalMs: Math.round(report.totalMs),
    at: Date.now(),
  };
  ggmlUi.last = stats;
  return stats;
}

/** GPU facts from the sysinfo service (no GPU / no CUDA when unknown). */
export function ggmlGpu(): GgmlGpu {
  const info = window.ApiServer?.sysinfo?.state?.info as SysSystemInfo | null | undefined;
  const gpu = info ? primaryGpu(info) : null;
  return {
    // Without a probe result, assume a GPU: Rust decides with the real facts.
    anyGpu: info ? !!gpu : true,
    cuda: !!info?.eps?.cuda?.eligible && !!gpu && gpu.vendor === "nvidia" && !gpu.integrated,
  };
}

/** The plan for `s` with the installed models and GPU facts. */
export function currentGgmlPlan(s: TTS_GgmlTts_State): GgmlPlan {
  const states = (window.ApiServer?.models?.state?.states ?? {}) as Record<string, { state?: string } | undefined>;
  return planGgml(s, { installed: (id) => states[id]?.state === "installed", gpu: ggmlGpu() });
}

// -------------------------------------------------------------------------------------------------
// Preview (the panel's test button)
// -------------------------------------------------------------------------------------------------

/** A preview started a helper while text to speech did not use it: stop it after this long without another preview. */
export const GGML_PREVIEW_IDLE_MS = 120_000;
/** The idle stop of each helper a preview left loaded (one per engine: previews of several engines each get theirs). */
const previewIdle = new Map<GgmlTtsEngine, ReturnType<typeof setTimeout>>();

function clearPreviewIdle(engine: GgmlTtsEngine): void {
  const timer = previewIdle.get(engine);
  if (timer !== undefined) clearTimeout(timer);
  previewIdle.delete(engine);
}

function armPreviewIdle(engine: GgmlTtsEngine): void {
  clearPreviewIdle(engine);
  previewIdle.set(
    engine,
    setTimeout(() => {
      previewIdle.delete(engine);
      if (!serviceUses(engine)) void stopGgmlHelper(engine);
    }, GGML_PREVIEW_IDLE_MS),
  );
}

/** Engines with a pending preview idle stop (tests). */
export function ggmlPreviewIdleEngines(): GgmlTtsEngine[] {
  return [...previewIdle.keys()];
}

/**
 * `engine` is about to load (a preview, or text to speech starting or switching to it): its own idle stop is dropped
 * (it is in use again), and every other helper a preview left loaded is stopped now unless text to speech uses it, so
 * its video memory is free before `engine` loads. Resolves once those stops finished (bounded).
 */
async function releasePreviewHelpers(engine: GgmlTtsEngine): Promise<void> {
  clearPreviewIdle(engine);
  const stops: Promise<void>[] = [];
  for (const other of [...previewIdle.keys()]) {
    clearPreviewIdle(other);
    if (!serviceUses(other)) stops.push(stopGgmlHelper(other));
  }
  if (stops.length) await waitAtMost(Promise.all(stops), GGML_STOP_WAIT_MS);
}

/** Whether text to speech runs on this backend and engine now (then the helper belongs to it). */
function serviceUses(engine: GgmlTtsEngine): boolean {
  const tts = window.ApiServer?.tts;
  const data = window.ApiServer?.state?.services?.tts?.data;
  return !!tts && data?.backend === "ggmlTts" && data?.ggmlTts?.engine === engine && tts.serviceState?.status !== ServiceNetworkState.disconnected;
}

/** Speak `text` once with the panel's settings into a new sink utterance on `target`. */
export async function previewGgml(s: TTS_GgmlTts_State, text: string, target: RustTtsSinkTarget): Promise<void> {
  const problem = ggmlSettingsProblem(s);
  if (problem) throw new Error(problem);
  const engine = ggmlEngine(s.engine).id;
  try {
    await releasePreviewHelpers(engine);
    const endpoint = await startGgmlHelper(engine, ggmlHelperConfig(s));
    const settings = ggmlSpeechSettings(endpoint, s, null);
    await previewToSink(target, async (id) => noteGgmlReport(engine, await speakOpenAi(settings, text, id)));
  } finally {
    if (!serviceUses(engine)) armPreviewIdle(engine);
  }
}

// -------------------------------------------------------------------------------------------------
// The backend
// -------------------------------------------------------------------------------------------------

/** How long a reload waits for the sentence streaming from the old helper. */
export const GGML_RELOAD_WAIT_MS = 30_000;
/** Reloads one sentence may wait for when the settings keep changing; then it speaks with what runs. */
const MAX_RELOADS_PER_SENTENCE = 3;

/** The helper may have restarted or moved: find it again before the next sentence. */
function looksLikeConnectionError(error: unknown): boolean {
  const m = errorMessage(error).toLowerCase();
  return /connect|connection|refused|unreachable|reset|closed|not running|restarting|no endpoint/.test(m);
}

/** The request never reached a running engine (openai_speech's and the adapter's "cannot connect"). */
const NEVER_REACHED = /cannot connect|connection refused|actively refused|not running|still restarting|no endpoint/;
/** The connection broke while the engine worked on the sentence. */
const CONNECTION_BROKE = /connection (?:was )?(?:closed|reset|aborted)|forcibly closed|closed before message completed|broken pipe/;

/**
 * Whether a failed sentence can be sent once more without being heard twice: the request never reached the engine, or
 * the connection broke while a whole-clip engine (no audio before the sentence is done: Magpie, S2) worked on it. A
 * streaming engine that broke off mid-sentence may already have played its first part, so that sentence fails instead.
 */
export function ggmlRetryable(error: unknown, wholeClip: boolean): boolean {
  const m = errorMessage(error).toLowerCase();
  return NEVER_REACHED.test(m) || (wholeClip && CONNECTION_BROKE.test(m));
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/**
 * One request at a time, in arrival order (the sentence pipeline calls in sentence order). A waiter whose signal aborts
 * leaves the queue at once.
 */
export class SerialGate {
  #tail: Promise<void> = Promise.resolve();

  /** Resolves with a release function once every earlier holder released, or `null` when `signal` aborted first. */
  async acquire(signal: AbortSignal): Promise<(() => void) | null> {
    let release!: () => void;
    const mine = new Promise<void>((r) => (release = r));
    const before = this.#tail;
    this.#tail = before.then(() => mine);
    if (signal.aborted) {
      release();
      return null;
    }
    let resolveAbort!: (v: "aborted") => void;
    const aborted = new Promise<"aborted">((r) => (resolveAbort = r));
    const onAbort = () => resolveAbort("aborted");
    signal.addEventListener("abort", onAbort, { once: true });
    const won = await Promise.race([before.then(() => "turn" as const), aborted]);
    signal.removeEventListener("abort", onAbort);
    if (won === "aborted" || signal.aborted) {
      // Keep the order for the ones behind: this slot frees itself once its turn comes.
      void before.then(release);
      return null;
    }
    return release;
  }
}

export class TTS_GgmlTtsService implements ITTSService {
  #running = false;
  #stopped = false;
  #engine: GgmlTtsEngine = "magpie";
  #endpoint: RustTtsHelperEndpoint | null = null;
  /** Launch key the helper was (re)started with. */
  #launchKey = "";
  /** The helper may have a new port (it restarted) or was replaced elsewhere: start again before the next sentence. */
  #dirty = false;
  #reload: Promise<void> | null = null;
  /** Sentences holding the current endpoint (between the reload check and the end of their stream). */
  #leases = 0;
  #gate = new SerialGate();
  #disposers: (() => void)[] = [];

  constructor(private readonly receiver: ITTSReceiver) {}

  async start(state: TTS_State): Promise<void> {
    if (!isTauri()) {
      this.receiver.onStop(i18n.t("tts.ggml_desktop_only"));
      return;
    }
    const s = state.ggmlTts;
    const problem = ggmlSettingsProblem(s);
    if (problem) {
      this.receiver.onStop(`${TTS_OPTIONS_MISSING}: ${problem}`);
      return;
    }
    const engine = ggmlEngine(s.engine).id;
    // Set first: a stop while the helper loads stops this engine's helper.
    this.#engine = engine;
    try {
      // A helper a preview left loaded for this engine is taken over; one of another engine is stopped first.
      await releasePreviewHelpers(engine);
      if (this.#stopped) return;
      const config = ggmlHelperConfig(s);
      const endpoint = await startGgmlHelper(engine, config);
      if (this.#stopped) {
        void stopGgmlHelper(engine);
        return;
      }
      this.#endpoint = endpoint;
      this.#launchKey = ggmlLaunchKey(engine, config);
      await this.#watch();
      if (this.#stopped) {
        this.#unwatch();
        return;
      }
      this.#running = true;
      this.receiver.onStart();
    } catch (error) {
      if (this.#stopped) return;
      this.receiver.onStop(i18n.t("tts.ggml_start_failed", { engine: ggmlEngine(engine).name, error: errorMessage(error) }));
    }
  }

  /** v2: stream one sentence into the begun sink utterance (one request at a time, in order). */
  async synthesizeToSink(text: string, sink: TtsSinkContext, signal: AbortSignal): Promise<void> {
    if (!this.#running || signal.aborted || !text.trim()) return;
    const onAbort = () => cancelHttpSpeech(sink.utteranceId);
    signal.addEventListener("abort", onAbort, { once: true });
    const release = await this.#gate.acquire(signal);
    let leased = false;
    try {
      if (!release || signal.aborted || !this.#running) return;
      const s = window.ApiServer.state.services.tts.data.ggmlTts as TTS_GgmlTts_State;
      const problem = ggmlSettingsProblem(s);
      if (problem) throw new Error(problem);
      for (let attempt = 0; ; attempt++) {
        leased = await this.#current(s, true);
        if (!leased || signal.aborted || !this.#running || !this.#endpoint) return;
        const settings = ggmlSpeechSettings(this.#endpoint, s, sink.lang);
        try {
          const report = await speakOpenAi(settings, text, sink.utteranceId);
          noteGgmlReport(this.#engine, report);
          return;
        } catch (error) {
          // Only a sentence nobody heard yet is sent again (into the same sink utterance).
          if (attempt > 0 || signal.aborted || !ggmlRetryable(error, !settings.stream)) throw error;
          // The helper restarted (or moved to another port): let go of it, find it again, try once more.
          this.#dirty = true;
          this.#leases -= 1;
          leased = false;
        }
      }
    } catch (error) {
      if (signal.aborted || isSinkCancelled(error) || !this.#running) return;
      if (looksLikeConnectionError(error)) this.#dirty = true;
      throw new Error(`${ggmlEngine(this.#engine).name}: ${errorMessage(error)}`);
    } finally {
      if (leased) this.#leases -= 1;
      release?.();
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
      void stopGgmlHelper(this.#engine);
    }
  }

  /**
   * Make the helper run `s`'s launch settings: reload it when they changed (stopping the old engine's helper when the
   * engine changed), or find it again after a restart. With `lease`, the caller then holds the endpoint for one
   * sentence (release with `#leases -= 1`). Resolves `false` once the backend stopped; rejects with the reason a
   * reload failed.
   */
  async #current(s: TTS_GgmlTts_State, lease: boolean): Promise<boolean> {
    for (let reloads = 0; ; ) {
      if (this.#reload) {
        await this.#reload;
        continue;
      }
      if (!this.#running) return false;
      const engine = ggmlEngine(s.engine).id;
      const config = ggmlHelperConfig(s);
      const key = ggmlLaunchKey(engine, config);
      const current = key === this.#launchKey && !this.#dirty && !!this.#endpoint;
      if (current || (reloads >= MAX_RELOADS_PER_SENTENCE && this.#endpoint)) {
        if (lease) this.#leases += 1;
        return true;
      }
      reloads += 1;
      this.#reload = this.#reloadTo(engine, config, key).finally(() => {
        this.#reload = null;
      });
      await this.#reload;
    }
  }

  async #reloadTo(engine: GgmlTtsEngine, config: GgmlHelperConfig, key: string): Promise<void> {
    const deadline = Date.now() + GGML_RELOAD_WAIT_MS;
    while (this.#leases > 0 && this.#running && Date.now() < deadline) await sleep(50);
    if (!this.#running) return;
    const previous = this.#engine;
    if (previous !== engine) {
      // Another engine: free the old one's video memory (and any a preview left loaded) before the new one loads.
      this.#endpoint = null;
      this.#engine = engine;
      await waitAtMost(Promise.all([stopGgmlHelper(previous), releasePreviewHelpers(engine)]), GGML_STOP_WAIT_MS);
      if (!this.#running) return;
    }
    const endpoint = await startGgmlHelper(engine, config);
    if (!this.#running) {
      void stopGgmlHelper(engine);
      return;
    }
    this.#endpoint = endpoint;
    this.#launchKey = key;
    this.#dirty = false;
  }

  async #watch(): Promise<void> {
    try {
      const unlisten: UnlistenFn = await listen<RustTtsHelperStatus>(GGML_STATE_EVENT, (e) => this.#onHelperState(e.payload));
      this.#disposers.push(unlisten);
    } catch {
      // No event bridge: connection errors still mark the helper for a fresh start.
    }
  }

  #unwatch() {
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
    if (!this.#running || !status || status.engine !== this.#engine || this.#reload) return;
    if (status.state === "starting" || status.state === "stopped") {
      // Restarting by itself (a crash), or stopped from elsewhere: start / find it again before the next sentence.
      this.#dirty = true;
    } else if (status.state === "failed") {
      this.#dirty = true;
      this.receiver.onError(i18n.t("tts.ggml_helper_failed", { engine: ggmlEngine(this.#engine).name, error: status.error ?? "" }));
    }
  }
}
