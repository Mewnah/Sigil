import { invoke } from "@tauri-apps/api/core";
import i18n from "i18next";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import type {
  RustCatalogEntry,
  RustEdgeTtsRequest,
  RustEdgeTtsVoice,
  RustLocalTtsRequest,
  RustModelEntryStatus,
  RustTtsSinkTarget,
  RustTtsSpeakReport,
} from "@/types";
import { isSinkCancelled } from "../../sound/sink";
import { pushSystemLog } from "../../systemLog";
import { clampedOr } from "../num";
import {
  FallbackBreaker,
  TtsFallbackChain,
  parseTtsFailure,
  type TtsFailure,
  type TtsFailureKind,
  type TtsFallbackEvent,
  type TtsSinkSpeaker,
} from "../pipeline";
import { TTS_State } from "../schema";
import { ITTSReceiver, ITTSService, TtsSinkContext, errorMessage } from "../types";
import { LOCAL_TTS_DEFAULT_MODEL, missingDownloads, requirementIdsOf, scriptLanguage, threadsOf } from "./localNeural";
import { previewToSink } from "./openaiSpeech";

// =================================================================================================
// The local fallback voice (shared by the cloud backends: `edge` here, `gemini` in gemini.ts)
// =================================================================================================

/** The local voice cloud backends fall back to (Kokoro v1.0, the zero-setup voice). */
export const FALLBACK_MODEL = LOCAL_TTS_DEFAULT_MODEL;
/** Languages Kokoro v1.0 speaks (base codes); others get its default (English) voice. */
export const KOKORO_LANGUAGES: readonly string[] = ["en", "ja", "zh", "es", "fr", "hi", "it", "pt"];
/** The local voice is re-touched this often so the engine's 10-minute idle unload never frees it. */
const KEEP_WARM_MS = 4 * 60_000;

/** Base language code of a tag (`pt-BR` → `pt`), or `null`. */
export function baseLanguage(tag: string | null | undefined): string | null {
  const base = (tag ?? "").trim().toLowerCase().split(/[-_]/)[0];
  return /^[a-z]{2,3}$/.test(base) ? base : null;
}

/**
 * The Kokoro language for a sentence the local voice speaks instead of a cloud voice: the text event's language tag
 * when the publisher set one; else what the script shows (Japanese, Chinese, Hindi); else, for Latin-script text, the
 * cloud voice's language (`hint`; an Edge voice says it); a sentence without letters keeps `previous`. A language
 * Kokoro does not speak gives `null` (its default voice).
 */
export function fallbackLanguage(text: string, tag: string | null | undefined, hint: string | null, previous: string | null): string | null {
  const supported = (l: string | null) => (l && KOKORO_LANGUAGES.includes(l) ? l : null);
  const tagged = baseLanguage(tag);
  if (tagged) return supported(tagged);
  const script = scriptLanguage(text);
  if (script === undefined) return previous;
  if (script) return script;
  return supported(baseLanguage(hint));
}

/** Fire-and-forget IPC: never throws, synchronously or later. */
function send(command: string, args?: Record<string, unknown>): void {
  try {
    void Promise.resolve(invoke(command, args)).catch(() => {});
  } catch {
    // Without Tauri (web build, tests) there is nothing to tell.
  }
}

/** Whether the fallback model still has to be downloaded, from the model store's current state (`null` = unknown). */
function fallbackMissingNow(): boolean | null {
  const state = window.ApiServer?.models?.state;
  if (!state) return null;
  const missing = missingDownloads(
    requirementIdsOf(FALLBACK_MODEL),
    state.catalog as RustCatalogEntry[],
    state.states as Record<string, RustModelEntryStatus>,
  );
  return missing === null ? null : missing.length > 0;
}

/** Whether the fallback model still has to be downloaded (`null` = unknown: try loading it). */
export async function fallbackMissing(): Promise<boolean | null> {
  const models = window.ApiServer?.models;
  if (!models) return null;
  try {
    await models.load();
  } catch {
    return null;
  }
  return fallbackMissingNow();
}

/** What the local fallback voice can do now. */
export type LocalFallbackState = "loading" | "ready" | "missing" | "failed" | "disposed";

/** A local voice that failed to load is tried again after this long (by the next sentence or failure). */
export const FALLBACK_RETRY_MS = 60_000;

export interface LocalFallbackOptions {
  /** Threads of the local engine (`null` = automatic). */
  threads: number | null;
  /** Speaking rate (0.5-2), read per sentence. */
  speed: () => number;
  /** The cloud voice's language (Edge: its locale), read per sentence. */
  languageHint: () => string | null;
}

/**
 * The local Kokoro voice behind a cloud backend: loaded (and kept loaded) while the backend runs, so a failed
 * sentence starts on it at once; speaks into the sink utterance the cloud voice was given. Only when Kokoro is
 * downloaded: the panel offers the download, nothing downloads by itself.
 *
 * Not downloaded (`missing`) or failed to load (`failed`) is not final while the backend runs: the next sentence
 * ({@link observe}) or failure ({@link speaker}) checks again, so a Kokoro downloaded from the panel's card is used
 * from the next sentence on without restarting TTS (a failed load is retried after {@link FALLBACK_RETRY_MS}).
 */
export class LocalFallbackVoice {
  #state: LocalFallbackState = "loading";
  #ready: Promise<boolean> | null = null;
  /** A load attempt is running. */
  #loading = false;
  /** When a failed load may be tried again (epoch ms). */
  #retryAt = 0;
  #language: string | null;
  #keepWarm: ReturnType<typeof setInterval> | null = null;
  #error = "";

  constructor(
    private readonly opts: LocalFallbackOptions,
    private readonly now: () => number = Date.now,
  ) {
    // Loaded for the cloud voice's language up front (Kokoro reloads when the language changes).
    const hint = baseLanguage(opts.languageHint());
    this.#language = hint && KOKORO_LANGUAGES.includes(hint) ? hint : null;
  }

  get state(): LocalFallbackState {
    return this.#state;
  }

  /** Why loading failed (state `failed`). */
  get error(): string {
    return this.#error;
  }

  /** Ready, or still loading (a failure will be spoken by it). */
  get usable(): boolean {
    return this.#state === "ready" || this.#state === "loading";
  }

  /**
   * Start loading. Resolves true when the voice is ready. Called again while it is `missing` (downloaded since?) or
   * `failed` (retry due?), it starts a new attempt; otherwise it returns the current one.
   */
  prepare(): Promise<boolean> {
    if (this.#ready && !this.#loading && this.#retryDue()) this.#ready = null;
    if (!this.#ready) {
      // A retry that will load (downloaded since, or a failed load due again) counts as loading at once, so the
      // sentence that started it already gets the short "switch to the local voice" timeout.
      if (this.#state === "failed" || (this.#state === "missing" && fallbackMissingNow() === false)) this.#state = "loading";
      this.#loading = true;
      const attempt = this.#load().finally(() => {
        this.#loading = false;
      });
      this.#ready = attempt;
    }
    return this.#ready;
  }

  /** Whether a settled `missing` / `failed` attempt should be retried now. */
  #retryDue(): boolean {
    if (this.#state === "missing") return fallbackMissingNow() !== true;
    if (this.#state === "failed") return this.now() >= this.#retryAt;
    return false;
  }

  /** The speaker once ready, or `null` when it cannot speak (not downloaded, failed, stopped). */
  async speaker(): Promise<TtsSinkSpeaker | null> {
    const ok = await this.prepare();
    return ok && this.#state === "ready" ? this.#speak : null;
  }

  #loadArgs() {
    return { model: FALLBACK_MODEL, threads: this.opts.threads, voice: null, language: this.#language };
  }

  /**
   * The cloud voice is about to speak `text` (tagged `tag`): keep Kokoro loaded for that language, so a failure of this
   * sentence needs no reload (about 1 s) before the local voice starts. Only a change of language loads anything.
   */
  observe(text: string, tag: string | null | undefined): void {
    if (this.#state === "disposed") return;
    // Not downloaded or failed to load: check again (downloaded from the panel since? a failed load due again?).
    if (this.#state === "missing" || this.#state === "failed") void this.prepare();
    const next = fallbackLanguage(text, tag, this.opts.languageHint(), this.#language);
    if (next === this.#language) return;
    this.#language = next;
    if (this.#state === "ready") send("plugin:local-tts|load_local_tts", this.#loadArgs());
  }

  /** Disposed (a method, so a check after an await is not narrowed away). */
  #gone(): boolean {
    return this.#state === "disposed";
  }

  async #load(): Promise<boolean> {
    const missing = await fallbackMissing();
    if (this.#gone()) return false;
    if (missing) {
      this.#state = "missing";
      return false;
    }
    this.#state = "loading";
    const args = this.#loadArgs();
    try {
      await invoke("plugin:local-tts|load_local_tts", args);
    } catch (error) {
      if (this.#gone()) return false;
      this.#state = "failed";
      this.#error = errorMessage(error);
      this.#retryAt = this.now() + FALLBACK_RETRY_MS;
      return false;
    }
    if (this.#gone()) return false;
    this.#state = "ready";
    this.#error = "";
    // A sentence in another language arrived while it loaded: get that one ready too.
    if (this.#language !== args.language) send("plugin:local-tts|load_local_tts", this.#loadArgs());
    this.#keepWarm ??= setInterval(() => {
      if (this.#state === "ready") send("plugin:local-tts|load_local_tts", this.#loadArgs());
    }, KEEP_WARM_MS);
    return true;
  }

  #speak: TtsSinkSpeaker = async (text, sink, signal) => {
    if (signal.aborted || this.#state !== "ready") return;
    this.#language = fallbackLanguage(text, sink.lang, this.opts.languageHint(), this.#language);
    const request: RustLocalTtsRequest = {
      model: FALLBACK_MODEL,
      voice: "",
      speed: clampedOr(this.opts.speed(), 1, 0.5, 2),
      language: this.#language,
      text,
      threads: this.opts.threads,
    };
    const onAbort = () => send("plugin:local-tts|cancel_local_tts", { utteranceId: sink.utteranceId });
    signal.addEventListener("abort", onAbort, { once: true });
    try {
      await invoke<RustTtsSpeakReport>("plugin:local-tts|speak_local_tts", { request, utteranceId: sink.utteranceId });
    } catch (error) {
      if (signal.aborted || this.#state !== "ready") return;
      // The sink dropped the utterance by itself: the pipeline reports that once, from the sink's event.
      if (isSinkCancelled(error)) throw error;
      throw new Error(i18n.t("tts.cloud_local_failed", { error: errorMessage(error) }));
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
  };

  /** Stop and free the model (Kokoro holds about 0.5-0.8 GB of RAM). */
  dispose(): void {
    const loaded = this.#state === "ready" || this.#state === "loading";
    this.#state = "disposed";
    if (this.#keepWarm) clearInterval(this.#keepWarm);
    this.#keepWarm = null;
    if (!loaded) return;
    send("plugin:local-tts|cancel_local_tts", { utteranceId: null });
    send("plugin:local-tts|unload_local_tts");
  }
}

// =================================================================================================
// Cloud backend base (Edge, Gemini)
// =================================================================================================

/** Fallback settings shared by the cloud slices. */
export interface CloudFallbackSettings {
  fallbackToLocal: boolean;
  /** Seconds without audio before the local voice takes over. */
  fallbackAfter: string;
}

/** The "switch to the local voice" timeout (ms) of a slice: 1-30 s, default 3 s. */
export function fallbackAfterMs(c: Pick<CloudFallbackSettings, "fallbackAfter">): number {
  return Math.round(clampedOr(c.fallbackAfter, 3, 1, 30) * 1000);
}

/** The timeout a cloud request gets: the short one while the local voice can take over, else automatic (0). */
export function cloudTimeoutMs(c: CloudFallbackSettings, local: Pick<LocalFallbackVoice, "usable"> | null): number {
  return c.fallbackToLocal && local?.usable ? fallbackAfterMs(c) : 0;
}

/** Localized name of a failure kind (for notices). */
export function failureReason(kind: TtsFailureKind): string {
  return i18n.t(`tts.cloud_reason_${kind}`);
}

/**
 * Base of the cloud backends (`edge`, `gemini`; ITTSService v2 `synthesizeToSink`): each sentence goes to the Rust
 * cloud plugin, which streams it into the sink utterance; a {@link TtsFallbackChain} speaks it with the local Kokoro
 * voice instead when the service fails or stalls, and keeps later sentences on Kokoro while the service is down.
 *
 * - `start` checks the settings (`connect`) and, with the fallback on, loads Kokoro in the background. Requests get
 *   the short "switch to the local voice" timeout only while Kokoro can take over; without it they wait as long as
 *   the plugin's automatic limit (there is nothing better to play).
 * - The first failure of an outage shows one notice; the service's return is logged.
 * - Settings apply to the next sentence; changing the service setup (voice, key, model) resets the breaker.
 * - Stop / skip cancel the cloud request and the local job; `stop` frees Kokoro.
 */
export abstract class CloudTtsBackend implements ITTSService {
  #running = false;
  #stopped = false;
  #local: LocalFallbackVoice | null = null;
  /** {@link setupKey} of the last sentence (`null` before the first). */
  #setup: string | null = null;
  readonly chain: TtsFallbackChain;

  constructor(
    protected readonly bindings: ITTSReceiver,
    /** Service name in messages ("Edge voices"). */
    protected readonly label: string,
    breaker?: FallbackBreaker,
  ) {
    this.chain = new TtsFallbackChain({
      primary: (text, sink, signal) => this.#speakCloud(text, sink, signal),
      fallback: async () => (this.#local ? this.#local.speaker() : null),
      breaker,
      onFallback: (event) => this.#onFallback(event),
      onRecovered: () => pushSystemLog("TTS", i18n.t("tts.cloud_recovered", { service: this.label }), "info"),
    });
  }

  /** The fallback settings of the backend's slice. */
  protected abstract fallbackSettings(state: TTS_State): CloudFallbackSettings;
  /**
   * What identifies the service setup (voice, key, model, URL). When it changes while the backend runs, the breaker
   * resets: a corrected key or voice is tried on the next sentence instead of after the breaker's pause.
   */
  protected abstract setupKey(state: TTS_State): string;
  /** Check the settings (and the service, if cheap): a reason to refuse Start, or `null`. */
  protected abstract connect(state: TTS_State): Promise<string | null>;
  /** Speak one sentence into the begun sink utterance with the cloud plugin. */
  protected abstract speakCloud(text: string, sink: TtsSinkContext, timeoutMs: number): Promise<unknown>;
  /** Cancel the cloud request of `utteranceId`, or every one. */
  protected abstract cancelCloud(utteranceId: string | null): void;
  /** The cloud voice's language (a hint for the local voice). */
  protected languageHint(_state: TTS_State): string | null {
    return null;
  }
  /** The local voice's speaking rate. */
  protected localSpeed(_state: TTS_State): number {
    return 1;
  }

  get running(): boolean {
    return this.#running;
  }

  /** The local fallback voice (`null` when switched off). */
  get local(): LocalFallbackVoice | null {
    return this.#local;
  }

  protected get state(): TTS_State {
    return window.ApiServer.state.services.tts.data;
  }

  async start(state: TTS_State): Promise<void> {
    const fb = this.fallbackSettings(state);
    if (fb.fallbackToLocal) {
      this.#local = new LocalFallbackVoice({
        threads: threadsOf(state.localNeural?.threads),
        speed: () => this.localSpeed(this.state),
        languageHint: () => this.languageHint(this.state),
      });
      // In the background: Start does not wait for Kokoro (a failure meanwhile waits for it instead).
      void this.#local.prepare().then((ok) => {
        const local = this.#local;
        if (ok || !local || this.#stopped) return;
        const message =
          local.state === "missing"
            ? i18n.t("tts.cloud_local_missing", { service: this.label })
            : i18n.t("tts.cloud_local_load_failed", { error: local.error });
        pushSystemLog("TTS", message, "warning");
      });
    }
    try {
      const refusal = await this.connect(state);
      if (this.#stopped) return;
      if (refusal) {
        this.#halt();
        this.bindings.onStop(refusal);
        return;
      }
      this.#running = true;
      this.bindings.onStart();
    } catch (error) {
      if (this.#stopped) return;
      this.#halt();
      this.bindings.onStop(`${this.label}: ${errorMessage(error)}`);
    }
  }

  /** v2: one sentence through the fallback chain into the begun sink utterance `sink.utteranceId`. */
  async synthesizeToSink(text: string, sink: TtsSinkContext, signal: AbortSignal): Promise<void> {
    if (!this.#running || signal.aborted || !text.trim()) return;
    const setup = this.setupKey(this.state);
    if (this.#setup !== null && setup !== this.#setup) this.chain.breaker.reset();
    this.#setup = setup;
    await this.chain.synthesize(text, sink, signal);
  }

  async #speakCloud(text: string, sink: TtsSinkContext, signal: AbortSignal): Promise<void> {
    this.#local?.observe(text, sink.lang);
    const timeoutMs = cloudTimeoutMs(this.fallbackSettings(this.state), this.#local);
    const onAbort = () => this.cancelCloud(sink.utteranceId);
    signal.addEventListener("abort", onAbort, { once: true });
    try {
      await this.speakCloud(text, sink, timeoutMs);
    } catch (error) {
      // Stop / skip (and stopping the backend) cancel the request; Rust then answers "cancelled". A sink that dropped
      // the utterance by itself also answers "cancelled": that is rethrown (never a service failure, never a
      // success), and the pipeline reports it from the sink's event.
      if (signal.aborted || !this.#running) return;
      throw error;
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
  }

  #onFallback(event: TtsFallbackEvent) {
    const f: TtsFailure | null = event.failure;
    if (!f) return;
    pushSystemLog("TTS", `${f.message} (${f.kind}${f.partial ? ", partial" : ""})`, "warning");
    if (!event.opened || !this.#local?.usable) return;
    toast(i18n.t("tts.cloud_fallback_active", { service: this.label, reason: failureReason(f.kind) }), {
      type: "warning",
      autoClose: 8_000,
      toastId: `tts-cloud-fallback-${this.label}`,
    });
  }

  /**
   * Legacy whole-line path. Unused: a backend with `synthesizeToSink` always speaks through the sentence pipeline.
   */
  play(_value: string): void {}

  stop(): void {
    this.#halt();
    this.bindings.onStop();
  }

  dispose(): void {
    this.#halt();
  }

  #halt() {
    const wasActive = !this.#stopped;
    this.#stopped = true;
    this.#running = false;
    if (!wasActive) return;
    this.cancelCloud(null);
    this.#local?.dispose();
  }
}

// =================================================================================================
// Edge
// =================================================================================================

/** Default voice (Rust `edge_tts::DEFAULT_VOICE`). */
export const EDGE_DEFAULT_VOICE = "en-US-AvaMultilingualNeural";

/**
 * A short Edge voice name: `<lang>[-<Script>]-<REGION>[-<subregion>…]-<Name>Neural` (`en-US-AvaMultilingualNeural`,
 * `iu-Cans-CA-SiqiniqNeural`, `zh-CN-liaoning-XiaobeiNeural`).
 */
const EDGE_SHORT_NAME = /^[a-z]{2,8}(?:-[A-Z][a-z]{3})?-(?:[A-Z]{2,}|\d{3})(?:-[a-z]+)*-[A-Za-z0-9]+Neural$/;

/**
 * Whether `voice` is an Edge voice name the plugin accepts (Rust `edge_tts::protocol::voice_name`): a short name
 * ({@link EDGE_SHORT_NAME}) or the long form.
 */
export function isEdgeVoiceName(voice: string): boolean {
  const v = voice.trim();
  if (!v || v.length > 160 || !/^[A-Za-z0-9 \-_(),.]+$/.test(v)) return false;
  if (v.startsWith("Microsoft Server Speech Text to Speech Voice (") && v.endsWith(")")) return true;
  return EDGE_SHORT_NAME.test(v);
}

/** The language of an Edge short voice name (`ja-JP-NanamiNeural` → `ja`). */
export function edgeVoiceLanguage(voice: string): string | null {
  const m = /^([a-z]{2,3})-/.exec(voice.trim());
  return m ? m[1] : null;
}

/**
 * The locale part of a short voice name (`iu-Cans-CA-SiqiniqNeural` → `iu-Cans-CA`): its list locale when the name
 * starts with it, else everything before the name.
 */
export function edgeVoiceLocale(v: Pick<RustEdgeTtsVoice, "shortName" | "locale">): string {
  if (v.locale && v.shortName.startsWith(`${v.locale}-`)) return v.locale;
  const parts = v.shortName.split("-");
  return parts.length > 1 ? parts.slice(0, -1).join("-") : v.locale || v.shortName;
}

/** The Edge request of a sentence (everything but the timeout and text comes from the slice). */
export function edgeRequest(c: TTS_State["edge"], text: string, timeoutMs: number): RustEdgeTtsRequest {
  return {
    text,
    voice: c.voice.trim() || EDGE_DEFAULT_VOICE,
    rate: clampedOr(c.rate, 1, 0.5, 2),
    pitchHz: Math.round(clampedOr(c.pitchHz, 0, -50, 50)),
    timeoutMs: Math.max(0, Math.round(timeoutMs)),
  };
}

/** The Edge voice list (cached by Rust and here; `[]` until loaded). */
export const edgeVoicesState = proxy({
  voices: [] as RustEdgeTtsVoice[],
  loading: false,
  error: "",
});

let voicesRequest: Promise<RustEdgeTtsVoice[]> | null = null;

/** Load the voice list once (a failed load can be retried). */
export function loadEdgeVoices(force = false): Promise<RustEdgeTtsVoice[]> {
  if (voicesRequest && !force) return voicesRequest;
  edgeVoicesState.loading = true;
  edgeVoicesState.error = "";
  const request = invoke<RustEdgeTtsVoice[]>("plugin:edge-tts|edge_tts_list_voices")
    .then((list) => {
      const voices = Array.isArray(list) ? list : [];
      edgeVoicesState.voices = voices;
      return voices;
    })
    .catch((error) => {
      edgeVoicesState.error = errorMessage(error);
      voicesRequest = null;
      return [] as RustEdgeTtsVoice[];
    })
    .finally(() => {
      edgeVoicesState.loading = false;
    });
  voicesRequest = request;
  return request;
}

/** Voices grouped by locale (`en-US`), locales sorted by their display name in `uiLanguage`. */
export function groupEdgeVoices(
  voices: readonly RustEdgeTtsVoice[],
  uiLanguage: string,
): { locale: string; label: string; voices: RustEdgeTtsVoice[] }[] {
  let names: Intl.DisplayNames | null = null;
  try {
    names = new Intl.DisplayNames([uiLanguage.replace("_", "-"), "en"], { type: "language" });
  } catch {
    names = null;
  }
  const groups = new Map<string, RustEdgeTtsVoice[]>();
  for (const v of voices) {
    const locale = v.locale || edgeVoiceLocale(v);
    const list = groups.get(locale) ?? [];
    list.push(v);
    groups.set(locale, list);
  }
  return [...groups.entries()]
    .map(([locale, list]) => {
      let label = locale;
      try {
        label = names?.of(locale) ?? locale;
      } catch {
        label = locale;
      }
      return { locale, label: label === locale ? locale : `${label} (${locale})`, voices: [...list].sort((a, b) => a.shortName.localeCompare(b.shortName)) };
    })
    .sort((a, b) => a.label.localeCompare(b.label));
}

/** A voice's display name: `Nanami (female)` from `ja-JP-NanamiNeural`, `Siqiniq` from `iu-Cans-CA-SiqiniqNeural`. */
export function edgeVoiceLabel(v: Pick<RustEdgeTtsVoice, "shortName" | "gender"> & Partial<Pick<RustEdgeTtsVoice, "locale">>): string {
  const locale = edgeVoiceLocale({ shortName: v.shortName, locale: v.locale ?? "" });
  const rest = v.shortName.startsWith(`${locale}-`) ? v.shortName.slice(locale.length + 1) : v.shortName;
  const name = (rest || v.shortName).replace(/Neural$/, "").replace(/Multilingual$/, " (multilingual)");
  const gender = v.gender?.toLowerCase();
  if (gender === "female") return i18n.t("tts.edge_voice_female", { name });
  if (gender === "male") return i18n.t("tts.edge_voice_male", { name });
  return name;
}

/**
 * A "Test voice" run of a cloud plugin: like {@link previewToSink}, but a cloud failure rejects with its readable
 * message (the wire tag the fallback chain reads is internal and never shown).
 */
export async function previewCloud(target: RustTtsSinkTarget, speak: (utteranceId: string) => Promise<unknown>): Promise<void> {
  try {
    await previewToSink(target, speak);
  } catch (error) {
    const failure = parseTtsFailure(error);
    if (!failure.tagged) throw error;
    throw new Error(failure.message);
  }
}

/** Speak `text` with the slice's voice into a new sink utterance on `target` (the panel's "Test voice"). */
export function previewEdgeVoice(c: TTS_State["edge"], text: string, target: RustTtsSinkTarget): Promise<void> {
  return previewCloud(target, (id) =>
    invoke<RustTtsSpeakReport>("plugin:edge-tts|edge_tts_speak", { request: edgeRequest(c, text, 0), utteranceId: id }),
  );
}

/**
 * Microsoft Edge "Read aloud" voices (backend `edge`): hundreds of natural voices in over 70 languages, no key and no
 * download, through the Rust `edge-tts` plugin. The service is unofficial and sometimes fails or stalls, so the local
 * Kokoro voice takes over (fallback chain, `pipeline.ts`). Voice, rate and pitch apply to the next sentence.
 */
export class TTS_EdgeService extends CloudTtsBackend {
  constructor(receiver: ITTSReceiver, breaker?: FallbackBreaker) {
    super(receiver, i18n.t("tts.edge_service_name"), breaker);
  }

  protected fallbackSettings(state: TTS_State): CloudFallbackSettings {
    return state.edge;
  }

  protected setupKey(state: TTS_State): string {
    return state.edge.voice.trim() || EDGE_DEFAULT_VOICE;
  }

  protected async connect(state: TTS_State): Promise<string | null> {
    const voice = state.edge.voice.trim() || EDGE_DEFAULT_VOICE;
    return isEdgeVoiceName(voice) ? null : i18n.t("tts.edge_voice_invalid", { voice });
  }

  protected speakCloud(text: string, sink: TtsSinkContext, timeoutMs: number): Promise<unknown> {
    const request = edgeRequest(this.state.edge, text, timeoutMs);
    return invoke<RustTtsSpeakReport>("plugin:edge-tts|edge_tts_speak", { request, utteranceId: sink.utteranceId });
  }

  protected cancelCloud(utteranceId: string | null): void {
    send("plugin:edge-tts|edge_tts_cancel", { utteranceId });
  }

  protected languageHint(state: TTS_State): string | null {
    return edgeVoiceLanguage(state.edge.voice);
  }

  protected localSpeed(state: TTS_State): number {
    return clampedOr(state.edge.rate, 1, 0.5, 2);
  }
}
