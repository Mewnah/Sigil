import { invoke } from "@tauri-apps/api/core";
import type { UnlistenFn } from "@tauri-apps/api/event";
import i18n from "i18next";
import {
  SPEECH_EVENTS,
  type RustSpeechEngineSpec,
  type RustSpeechFinal,
  type RustSpeechPipelineConfig,
  type RustSpeechPipelineStarted,
  type RustSpeechStatus,
  type RustSpeechStopPipelineArgs,
  type RustSpeechUtterance,
} from "@/types";
import { devLog } from "@/utils/devLog";
import { resolveRustInputDeviceForInvoke } from "@/utils/rustAudioInput";
import { isAbortError, listenWithAbort, raceWithAbort } from "./abort";
import { SttStartLifecycle } from "./localEngineLifecycle";
import type { STT_State } from "./schema";
import { TRANSIENT_WARNING_MS } from "./statusChannel";
import { buildVadConfig } from "./sttOptions";
import type { ISTTReceiver, ISpeechRecognitionService, SttStatusOptions, SttTextMeta } from "./types";

/**
 * STT backends whose engine runs inside Sigil's Rust speech pipeline (`speech` plugin) as its explicit utterance
 * engine: the HTTP engine for OpenAI-compatible servers, Groq and the legacy Moonshine server (P2-F).
 *
 * Rust owns everything between the microphone and the transcript: the capture hub's raw 16 kHz feed (withheld while
 * STT is hard-muted), the Silero segmenter, the ordered utterance worker (latency budget, backlog cap, merging when
 * requests fall behind, Groq's pacing and billing-minimum batches), the request itself (keys only in the
 * `Authorization` header; no webview CSP, so LAN servers work), retries and the hallucination post-filter. No PCM
 * crosses IPC and silence never produces a request. This driver maps one pipeline session's events to the receiver:
 *
 * - `speech:final` → `onFinal(text, { segmentId, revision, lang })` (one final per segment);
 * - `speech:status` → the status line (never published); state `error` ends the session with a message;
 * - `speech:utterance` `dropped` → a transient "speech was skipped" line when speech was lost.
 *
 * Late speech is merged, not dropped: HTTP sessions run with Rust's longest latency budget
 * ({@link HTTP_LATENCY_BUDGET_MS}), because the engine's waits (Groq pacing, a 429's `Retry-After`, network retries)
 * happen inside the request while the next utterances queue up behind it. Only speech that would wait longer than
 * that, or more than {@link HTTP_MAX_BACKLOG} waiting utterances, is skipped (and reported on the status line).
 *
 * Only events of this session's `(pipelineId, epoch)` are used. Events that arrive before `start_pipeline` returned
 * the epoch are held and replayed once it is known (Tauri does not order events against command results).
 *
 * The file keeps its historical name (the backends once polled captured audio every 500 ms).
 */

/**
 * Pipeline key of speech-to-text (Rust `speech::engine_api::PIPELINE_MIC`). Shared with the `local` backend: only one
 * STT backend runs at a time, and `Service_STT` awaits the previous engine's stop before the next start.
 */
export const STT_PIPELINE_ID = "mic";
/** Events held while `start_pipeline` is in flight. */
const MAX_EARLY_EVENTS = 64;
/** How long a stop waits for a `start_pipeline` still in flight (Moonshine's health probe may take 10 s). */
export const START_SETTLE_WAIT_MS = 15_000;
/** Segment ids remembered to drop a repeated final. */
const MAX_FINISHED_SEGMENTS = 256;
/** Closed utterances that may wait for a request (Rust clamps 1-10): waiting ones are merged, not dropped. */
export const HTTP_MAX_BACKLOG = 10;
/**
 * How long a closed utterance may wait for its request before Rust skips it (`latencyBudgetMs`; Rust's maximum,
 * `speech::pipeline::MAX_LATENCY_BUDGET_MS`). Rust's 20 s default suits local engines, not HTTP: a Groq 429 asks to
 * wait 30 s or more, a LAN server at RTF 0.8 takes ~22 s for a merged 28 s batch, and a Wi-Fi drop costs retries. With
 * the maximum, everything said meanwhile waits and is sent merged once the request goes through.
 */
export const HTTP_LATENCY_BUDGET_MS = 120_000;
/** Longest error detail shown in a status line or stop message. */
const MAX_DETAIL = 200;
/** Longest a rate-limit wait line stays up (Rust reports a wait once, not its end). */
const MAX_WAIT_LINE_MS = 120_000;

/** One line for the status channel. */
export type StatusLine = [text: string, options?: SttStatusOptions];

/** What a backend runs in the pipeline, and the decode settings of the session. */
export interface PipelineEngineSetup {
  /** The utterance engine (`final` of the pipeline config). May carry a key in `options`: never log it. */
  engine: RustSpeechEngineSpec;
  /** "auto" or a language tag (Rust normalizes it to a base subtag). */
  language?: string;
  /** Decoder prompt (the HTTP engine sends it as the OpenAI `prompt`). */
  prompt?: string;
  hotwords?: string[];
}

// ---------------------------------------------------------------------------------------------------------------
// Pure helpers (unit-tested)

/** One line of an error detail for the UI: whitespace collapsed, capped. */
export function shortDetail(text: unknown, max = MAX_DETAIL): string {
  const t = String(text ?? "").replace(/\s+/g, " ").trim();
  return t.length <= max ? t : `${t.slice(0, max - 1)}…`;
}

/** Text of a rejected `invoke` (Rust commands reject with a string). */
export function errorText(error: unknown): string {
  if (typeof error === "string") return error;
  if (error instanceof Error) return error.message;
  try {
    return JSON.stringify(error) ?? String(error);
  } catch {
    return String(error);
  }
}

/** Lower-case base language subtag (`en-US` → `en`), or `null` for blank, "auto" and malformed tags. */
export function baseLanguageTag(tag: unknown): string | null {
  const base = String(tag ?? "").trim().split(/[-_]/)[0].toLowerCase();
  return /^[a-z]{2,3}$/.test(base) ? base : null;
}

/**
 * A server base URL the way Rust's `http_client::normalize_base_url` accepts it: a missing scheme means `http://`,
 * only http(s) with a host, trailing slashes removed. `null` when Rust would reject it.
 */
export function normalizeEndpointUrl(raw: unknown): string | null {
  const trimmed = String(raw ?? "").trim();
  if (!trimmed) return null;
  const withScheme = trimmed.includes("://") ? trimmed : `http://${trimmed}`;
  let url: URL;
  try {
    url = new URL(withScheme);
  } catch {
    return null;
  }
  if ((url.protocol !== "http:" && url.protocol !== "https:") || !url.hostname) return null;
  return withScheme.replace(/\/+$/, "");
}

/** `speech:start_pipeline` config for a backend's explicit utterance engine. */
export function buildUtterancePipelineConfig(
  setup: PipelineEngineSetup,
  vad: Partial<STT_State["vad"]> | undefined,
  source: string | null,
): RustSpeechPipelineConfig {
  return {
    pipelineId: STT_PIPELINE_ID,
    source: source ?? "",
    vad: buildVadConfig(vad),
    language: String(setup.language ?? "").trim() || "auto",
    streaming: null,
    final: setup.engine,
    smartTurn: false,
    hotwords: setup.hotwords ?? [],
    prompt: String(setup.prompt ?? "").trim(),
    // Strips non-speech markers and repetition loops; without a no-speech probability (HTTP) it drops only the hard
    // subtitle-credit phrases, never a plain "thank you".
    hallucinationFilter: true,
    latencyBudgetMs: HTTP_LATENCY_BUDGET_MS,
    maxBacklog: HTTP_MAX_BACKLOG,
    retainPcm: false,
  };
}

/** Revision metadata of a final (`TextEvent.segmentId` / `revision` / `lang` / `turnEnd`). */
export function speechFinalMeta(ev: Pick<RustSpeechFinal, "pipelineId" | "epoch" | "segmentId" | "revision" | "lang" | "turnEnd">): SttTextMeta {
  const meta: SttTextMeta = { segmentId: `${ev.pipelineId}:${ev.epoch}:${ev.segmentId}` };
  if (Number.isInteger(ev.revision) && ev.revision >= 0) meta.revision = ev.revision;
  const lang = baseLanguageTag(ev.lang);
  if (lang) meta.lang = lang;
  if (typeof ev.turnEnd === "boolean") meta.turnEnd = ev.turnEnd;
  return meta;
}

/**
 * The wait of a `rate_limited` status that announces one (Rust `engines::http`: "waiting 12 s for the request limit",
 * "rate limited by the server; retrying in 20 s"), or `null` for any other `rate_limited` detail. Those others follow
 * a lost utterance (the worker's "an utterance was dropped: …" or a failed request's error, whose server text may
 * hold its own durations such as Groq's "try again in 7m12.5s"), so they must never read as a wait.
 */
export function rateLimitWaitSeconds(detail: string): number | null {
  const m = /\b(?:waiting|retrying in) (\d+) s\b/.exec(detail);
  return m ? Number(m[1]) : null;
}

/**
 * The wait in minutes (rounded up) of a fatal rate-limit detail: Rust's "the next request would wait 42 min", or a
 * server's own "try again in 7m12.5s" / "in 1h2m3s" (Groq). `null` when it names none (or only seconds).
 */
export function rateLimitWaitMinutes(detail: string): number | null {
  const own = /\b(\d+)\s*min\b/.exec(detail);
  if (own) return Number(own[1]);
  const server = /\bin (?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?(?![a-z0-9])/i.exec(detail);
  if (!server || (server[1] === undefined && server[2] === undefined)) return null;
  const [, hours = "0", minutes = "0", seconds = "0"] = server;
  return Number(hours) * 60 + Number(minutes) + (parseFloat(seconds) > 0 ? 1 : 0);
}

const transient = (): SttStatusOptions => ({ level: "warning", expireMs: TRANSIENT_WARNING_MS });

/**
 * The status line of a `speech:status` event while the session runs: `null` clears the line, `undefined` leaves it
 * alone. Lifecycle statuses of the start (`starting` / `loading`) and the terminal ones are handled by the driver.
 */
export function speechStatusLine(ev: Pick<RustSpeechStatus, "code" | "detail" | "level">, engine: string): StatusLine | null | undefined {
  const detail = shortDetail(ev.detail);
  switch (ev.code) {
    case "listening":
      return null;
    case "starting":
    case "stopped":
    case "loading_model":
    case "loading_runtime":
    case "downloading":
      return undefined;
    case "rate_limited": {
      const seconds = rateLimitWaitSeconds(detail);
      // Not a wait: the warning that follows an utterance given up (limit used up, or a wait past the budget). The
      // `dropped` event already said so; this line must not replace it with "sentences are sent together".
      if (seconds === null) return [i18n.t("stt.status_dropped_rate_limited"), transient()];
      const expireMs = Math.min(MAX_WAIT_LINE_MS, Math.max(TRANSIENT_WARNING_MS, (seconds + 5) * 1000));
      return [i18n.t("stt.status_request_limit_wait", { seconds: Math.max(1, seconds) }), { level: "warning", expireMs }];
    }
    case "backlog_dropped":
      return [i18n.t("stt.status_utterance_dropped"), transient()];
    case "capture_gap":
      return [i18n.t("stt.status_capture_gap"), transient()];
    case "capture_error":
      return [i18n.t("stt.error_capture", { detail }), transient()];
    default:
      if (!detail) return undefined;
      return ev.level === "info"
        ? [i18n.t("stt.status_engine_note", { engine, detail })]
        : [i18n.t("stt.status_engine_note", { engine, detail }), transient()];
  }
}

/** The stop message of a fatal `speech:status` (state `error`). */
export function speechFatalMessage(ev: Pick<RustSpeechStatus, "code" | "detail">, engine: string): string {
  const detail = shortDetail(ev.detail) || String(ev.code ?? "");
  switch (ev.code) {
    case "capture_error":
      return i18n.t("stt.error_capture", { detail });
    case "rate_limited": {
      const minutes = rateLimitWaitMinutes(detail);
      if (minutes !== null) return i18n.t("stt.error_rate_limit_wait", { engine, minutes: Math.max(1, minutes) });
      return i18n.t("stt.error_request_rejected", { engine, detail });
    }
    default:
      return i18n.t("stt.error_request_rejected", { engine, detail });
  }
}

/** Status line for a `speech:utterance` `dropped` event; `null` when nothing was lost (merged, no speech, muted). */
export function droppedStatusLine(reason: string): StatusLine | null {
  switch (reason) {
    case "backlog":
    case "latency_budget":
    case "queue_full":
    case "stream_skipped":
      return [i18n.t("stt.status_utterance_dropped"), transient()];
    case "rate_limited":
      return [i18n.t("stt.status_dropped_rate_limited"), transient()];
    case "engine_error":
      return [i18n.t("stt.status_dropped_request_failed"), transient()];
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Driver

type PipelineEvent =
  | { kind: "final"; payload: RustSpeechFinal }
  | { kind: "status"; payload: RustSpeechStatus }
  | { kind: "utterance"; payload: RustSpeechUtterance };

/** A `start_pipeline` call of one session. */
interface StartCall {
  /** Resolves with the started session, or `null` when the call failed (never rejects). */
  promise: Promise<RustSpeechPipelineStarted | null>;
  settled: boolean;
  result: RustSpeechPipelineStarted | null;
  /** {@link noteSttPipelineStart} generation of this call. */
  generation: number;
}

/** `start_pipeline` calls on {@link STT_PIPELINE_ID} issued from this webview (every STT driver that notes its own). */
let sttPipelineStarts = 0;

/**
 * Notes a `start_pipeline` on {@link STT_PIPELINE_ID}; call it right before the invoke. Returns the call's generation.
 * A start replaces whatever session the key runs, so a driver whose own start resolved late skips its cleanup stop
 * once a newer start was issued ({@link sttPipelineStartSuperseded}); that cleanup stop also names the late session's
 * epoch, so Rust ignores it once any newer session (of a driver that notes its starts or not) took the key.
 */
export function noteSttPipelineStart(): number {
  sttPipelineStarts += 1;
  return sttPipelineStarts;
}

/** True once a start newer than `generation` was issued on {@link STT_PIPELINE_ID}. */
export function sttPipelineStartSuperseded(generation: number): boolean {
  return sttPipelineStarts !== generation;
}

const TIMED_OUT = Symbol("timed out");

function within<T>(p: Promise<T>, ms: number): Promise<T | typeof TIMED_OUT> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<typeof TIMED_OUT>((resolve) => {
    timer = setTimeout(() => resolve(TIMED_OUT), ms);
  });
  return Promise.race([p, timeout]).finally(() => clearTimeout(timer));
}

/**
 * Base driver of a backend whose engine runs in the Rust speech pipeline. Subclasses validate their settings and
 * describe the engine ({@link prepare}); everything else is shared.
 */
export abstract class RustUtteranceService implements ISpeechRecognitionService {
  startAbortSignal: AbortSignal | null = null;
  protected readonly lifecycle: SttStartLifecycle;

  #unlisteners: UnlistenFn[] = [];
  /** Epoch of the running session; `null` until `start_pipeline` returned it (and after teardown). */
  #epoch: number | null = null;
  /** Events held until the epoch is known; `null` = not collecting. */
  #early: PipelineEvent[] | null = null;
  #start: StartCall | null = null;
  /** Rust ended the session itself (`error` or `stopped` status): nothing is left to stop. */
  #ended = false;
  /** `onStart` was sent. */
  #listening = false;
  /** Segments whose final was handled (a finals-only route has one final per segment). */
  #finished = new Set<number>();
  /**
   * The last Rust stop issued by `#teardown`. Every stop() awaits it, including the early return after a failed
   * start already tore down, so the orchestrator never starts the next engine while an older stop is in flight.
   */
  #pendingRustStop: Promise<void> = Promise.resolve();

  /** Console tag, e.g. `[Moonshine]`. */
  protected abstract readonly logTag: string;

  constructor(protected readonly receiver: ISTTReceiver) {
    this.lifecycle = new SttStartLifecycle((error) => this.receiver.onStop(error));
  }

  /** User-facing engine name for status lines and stop messages. */
  protected abstract engineLabel(): string;

  /** Per-service cpal device ("" = Settings native default). */
  protected abstract inputDevice(params: STT_State): string;

  /** Validate the settings and describe the engine; a string is the error message to stop with. */
  protected abstract prepare(params: STT_State): PipelineEngineSetup | string | Promise<PipelineEngineSetup | string>;

  /** Status line while the session starts. */
  protected connectingStatus(): string {
    return i18n.t("stt.status_connecting_engine", { engine: this.engineLabel() });
  }

  /** Stop message when `start_pipeline` failed for a reason other than the microphone. */
  protected startFailedMessage(detail: string): string {
    return i18n.t("stt.error_engine_start", { engine: this.engineLabel(), detail: shortDetail(detail) });
  }

  /** Cleans a transcript before it is published; "" drops it. */
  protected postProcess(text: string): string {
    return text.trim();
  }

  async start(params: STT_State): Promise<void> {
    this.lifecycle.beginStart();
    this.#finished.clear();
    this.#ended = false;
    try {
      const setup = await this.prepare(params);
      if (await this.#stopIfAborted()) return;
      if (typeof setup === "string") {
        this.receiver.onStatus("");
        this.lifecycle.notifyStopOnce(setup);
        return;
      }
      this.receiver.onStatus(this.connectingStatus());

      this.#early = [];
      await this.#listen();
      if (await this.#stopIfAborted()) return;

      const config = buildUtterancePipelineConfig(
        setup,
        params.vad,
        resolveRustInputDeviceForInvoke(this.inputDevice(params)),
      );
      const started = await raceWithAbort(this.#requestStart(config), this.startAbortSignal);
      if (await this.#stopIfAborted()) return;

      this.#accept(started.epoch);
      // A held event already ended the session (reported by #fail / #endedByRust).
      if (this.#ended || this.lifecycle.aborted) return;
      this.#listening = true;
      this.receiver.onStatus("");
      this.receiver.onStart();
      devLog(`${this.logTag} Listening (pipeline ${STT_PIPELINE_ID}, epoch ${started.epoch})`);
    } catch (error) {
      if (isAbortError(error) || this.lifecycle.aborted) {
        await this.#teardown();
        return;
      }
      const detail = errorText(error);
      const code = this.#heldFailureCode(detail);
      console.error(`${this.logTag} Error starting:`, detail);
      await this.#teardown();
      this.receiver.onStatus("");
      this.lifecycle.notifyStopOnce(
        code === "capture_error" ? i18n.t("stt.error_capture", { detail: shortDetail(detail) }) : this.startFailedMessage(detail),
      );
    } finally {
      this.lifecycle.finishStart();
    }
  }

  async stop(): Promise<void> {
    this.lifecycle.abort();
    const live = this.#start !== null || this.#unlisteners.length > 0 || this.#listening;
    if (live) devLog(`${this.logTag} Stopping...`);
    await this.#teardown();
    if (live) this.lifecycle.notifyStopOnce();
  }

  dispose(): void {
    void this.stop();
  }

  // --- events ------------------------------------------------------------------------------------------------------

  async #listen(): Promise<void> {
    const on = async <T>(event: string, kind: PipelineEvent["kind"]) => {
      const unlisten = await listenWithAbort<T>(
        event,
        (e) => this.#onEvent({ kind, payload: e.payload } as PipelineEvent),
        this.startAbortSignal,
      );
      this.#unlisteners.push(unlisten);
    };
    await on<RustSpeechStatus>(SPEECH_EVENTS.status, "status");
    await on<RustSpeechFinal>(SPEECH_EVENTS.final, "final");
    await on<RustSpeechUtterance>(SPEECH_EVENTS.utterance, "utterance");
  }

  #onEvent(event: PipelineEvent): void {
    const env = event.payload;
    if (!env || env.pipelineId !== STT_PIPELINE_ID || typeof env.epoch !== "number") return;
    if (this.#epoch === null) {
      if (this.#early && this.#early.length < MAX_EARLY_EVENTS) this.#early.push(event);
      return;
    }
    if (env.epoch !== this.#epoch || this.#ended) return;
    switch (event.kind) {
      case "final":
        this.#onFinal(event.payload);
        break;
      case "status":
        this.#onStatus(event.payload);
        break;
      case "utterance":
        this.#onUtterance(event.payload);
        break;
    }
  }

  /** The session's epoch is known: replay what was held for it, drop the rest. */
  #accept(epoch: number): void {
    const held = this.#early ?? [];
    this.#early = null;
    this.#epoch = epoch;
    for (const event of held) {
      if (this.#epoch === null || this.#ended) break;
      this.#onEvent(event);
    }
  }

  /** Code of the fatal status Rust emitted for a failed start (`detail` is the command's error), if it arrived. */
  #heldFailureCode(detail: string): string | null {
    const held = this.#early ?? [];
    for (let i = held.length - 1; i >= 0; i--) {
      const e = held[i];
      if (e.kind === "status" && e.payload.state === "error" && String(e.payload.detail ?? "").trim() === detail.trim())
        return e.payload.code;
    }
    return null;
  }

  #onFinal(ev: RustSpeechFinal): void {
    if (typeof ev.segmentId !== "number" || this.#finished.has(ev.segmentId)) return;
    this.#finished.add(ev.segmentId);
    if (this.#finished.size > MAX_FINISHED_SEGMENTS) {
      const oldest = this.#finished.values().next().value;
      if (oldest !== undefined) this.#finished.delete(oldest);
    }
    const text = this.postProcess(String(ev.text ?? ""));
    if (!text) return;
    devLog(`${this.logTag} Result:`, text);
    this.receiver.onFinal(text, speechFinalMeta(ev));
  }

  #onStatus(ev: RustSpeechStatus): void {
    if (ev.state === "error") {
      this.#ended = true;
      void this.#fail(speechFatalMessage(ev, this.engineLabel()));
      return;
    }
    if (ev.state === "stopped") {
      // Not our stop (our listeners are gone before it): another start of the same pipeline key replaced us.
      this.#ended = true;
      void this.#endedByRust();
      return;
    }
    // Start progress is shown by the driver itself while `start_pipeline` runs; afterwards it is stale.
    if (ev.state === "starting" || ev.state === "loading") return;
    const line = speechStatusLine(ev, this.engineLabel());
    if (line === undefined) return;
    if (line === null) this.receiver.onStatus("");
    else this.receiver.onStatus(...line);
  }

  #onUtterance(ev: RustSpeechUtterance): void {
    if (ev.kind !== "dropped") return;
    const line = droppedStatusLine(ev.reason);
    if (!line) return;
    console.warn(`${this.logTag} utterance ${ev.segmentId} dropped (${ev.reason})`);
    this.receiver.onStatus(...line);
  }

  // --- lifecycle ---------------------------------------------------------------------------------------------------

  #requestStart(config: RustSpeechPipelineConfig): Promise<RustSpeechPipelineStarted> {
    const generation = noteSttPipelineStart();
    const call = invoke<RustSpeechPipelineStarted>("plugin:speech|start_pipeline", { config });
    const start: StartCall = { promise: Promise.resolve(null), settled: false, result: null, generation };
    start.promise = call.then(
      (started) => {
        start.settled = true;
        start.result = started && typeof started.epoch === "number" ? started : null;
        return start.result;
      },
      () => {
        start.settled = true;
        return null;
      },
    );
    this.#start = start;
    return call.then((started) => {
      if (!started || typeof started.epoch !== "number") throw new Error("start_pipeline returned no session");
      return started;
    });
  }

  async #fail(message: string): Promise<void> {
    if (this.lifecycle.aborted) return;
    console.error(`${this.logTag} ${message}`);
    await this.#teardown();
    this.receiver.onStatus("");
    this.lifecycle.notifyStopOnce(message);
  }

  async #endedByRust(): Promise<void> {
    if (this.lifecycle.aborted) return;
    console.warn(`${this.logTag} the speech pipeline was stopped from elsewhere`);
    await this.#teardown();
    this.receiver.onStatus("");
    this.lifecycle.notifyStopOnce();
  }

  async #stopIfAborted(): Promise<boolean> {
    if (!this.lifecycle.aborted) return false;
    await this.#teardown();
    this.lifecycle.notifyStopOnce();
    return true;
  }

  async #teardown(): Promise<void> {
    for (const unlisten of this.#unlisteners.splice(0)) unlisten();
    const start = this.#start;
    const ended = this.#ended;
    this.#start = null;
    this.#epoch = null;
    this.#early = null;
    this.#listening = false;
    // Stop the Rust session once per start. A teardown after another one already issued the stop (e.g. stop() after
    // a fatal error) sends nothing new; it only waits for that stop to settle.
    if (start) this.#pendingRustStop = this.#stopRust(start, ended);
    await this.#pendingRustStop;
  }

  /** Stop what `start` started, without ever stopping a newer session of the pipeline key. */
  async #stopRust(start: StartCall, ended: boolean): Promise<void> {
    if (start.settled) {
      // Running unless the start failed, or Rust ended the session itself (then the key may belong to someone else).
      if (start.result && !ended) await this.#invokeStop();
      return;
    }
    // `start_pipeline` is still in flight. Stopping now cancels a start Rust is already building; a start that only
    // reached Rust after this stop still starts a session, which is stopped once it resolves.
    await this.#invokeStop();
    const result = await within(start.promise, START_SETTLE_WAIT_MS);
    if (result === TIMED_OUT) {
      // The orchestrator moves on; if the start completes later, stop it only while it is still the key's session.
      void start.promise.then((late) => (late ? this.#stopIfCurrent(late.epoch, start.generation) : undefined));
      return;
    }
    if (result) await this.#invokeStop();
  }

  /**
   * Stops a start that resolved after the driver gave up on it, unless a newer start was issued (it replaces this
   * session itself, and a stop now would cancel it). The stop names the late session's `epoch`: Rust stops it only
   * while it is still the key's current session (checked under the key's emit lock) and never cancels a start queued
   * behind it, so a newer session of any driver is safe.
   */
  async #stopIfCurrent(epoch: number, generation: number): Promise<void> {
    if (sttPipelineStartSuperseded(generation)) return;
    await this.#invokeStop(epoch);
  }

  /** `stop_pipeline` for the key, or with `epoch` only for that session ({@link #stopIfCurrent}). */
  #invokeStop(epoch?: number): Promise<void> {
    const args: RustSpeechStopPipelineArgs = { pipelineId: STT_PIPELINE_ID };
    if (epoch !== undefined) args.epoch = epoch;
    return invoke("plugin:speech|stop_pipeline", { ...args }).then(
      () => {},
      (e: unknown) => devLog(`${this.logTag} stop_pipeline failed:`, e),
    );
  }
}
