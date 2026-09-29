import { invoke } from "@tauri-apps/api/core";
import { UnlistenFn } from "@tauri-apps/api/event";
import i18n from "i18next";
import type { RustUtteranceEvent } from "@/types";
import { devLog } from "@/utils/devLog";
import { resolveRustInputDeviceForInvoke } from "@/utils/rustAudioInput";
import { isAbortError, listenWithAbort, raceWithAbort } from "./abort";
import { SttStartLifecycle } from "./localEngineLifecycle";
import type { RequestLimiter } from "./rateLimiter";
import { STT_State } from "./schema";
import { TRANSIENT_WARNING_MS } from "./statusChannel";
import { pipelineFatalMessage, pipelineStatus } from "./statusMessages";
import { buildVadConfig } from "./sttOptions";
import { ISTTReceiver, ISpeechRecognitionService } from "./types";
import { pcmFromIpc, UtterancePipeline } from "./utterancePipeline";

/** Rate of the mono f32 utterances handed to {@link RustUtteranceService.transcribe} (Rust `speech_core::SAMPLE_RATE`). */
export const RUST_CAPTURE_SAMPLE_RATE = 16000;
/** Events that may arrive before `start_utterance_capture` returned the capture id (replayed once it is known). */
const MAX_EARLY_EVENTS = 32;

export function truncateForUi(s: string, max: number): string {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length <= max ? t : `${t.slice(0, max)}…`;
}

/**
 * Shared engine for HTTP-style STT (Moonshine, OpenAI-compatible servers, Groq): Rust captures the microphone and
 * closes utterances with the Silero VAD (`plugin:audio|start_utterance_capture` → `audio:utterance`); each utterance
 * is fetched once as binary PCM (`take_utterance`) and transcribed with one request, in order, through an
 * {@link UtterancePipeline} (optional rate limiter, merge-on-backlog, retries). Silence never produces a request.
 *
 * Subclasses provide engine setup ({@link prepare}) and one transcription call ({@link transcribe}).
 */
export abstract class RustUtteranceService implements ISpeechRecognitionService {
  startAbortSignal: AbortSignal | null = null;
  protected readonly lifecycle: SttStartLifecycle;

  #isRunning = false;
  #unlisten?: UnlistenFn;
  #pipeline: UtterancePipeline | null = null;
  /** Id returned by `start_utterance_capture`; events of other captures are ignored. */
  #captureId: number | null = null;
  #earlyEvents: RustUtteranceEvent[] = [];
  /** Takes run one after another so utterances enter the pipeline in capture order. */
  #takeChain: Promise<void> = Promise.resolve();
  /** Set right before `start_utterance_capture` is sent; the teardown that issues the matching stop clears it. */
  #captureRequested = false;
  /**
   * The last `stop_utterance_capture` issued by `#teardown`. Every stop() awaits it, including the early return
   * after a failed/aborted start already tore down, so the orchestrator never starts the next engine while an
   * older stop is still in flight.
   */
  #pendingRustStop: Promise<void> = Promise.resolve();

  /** Console tag, e.g. `[Moonshine]`. */
  protected abstract readonly logTag: string;

  constructor(protected readonly receiver: ISTTReceiver) {
    this.lifecycle = new SttStartLifecycle((error) => this.receiver.onStop(error));
  }

  /** User-facing engine name for stop messages. */
  protected abstract engineLabel(): string;

  /** Engine setup before capture starts. Return an error message to stop with, or `null` to continue. */
  protected abstract prepare(params: STT_State): Promise<string | null>;

  /** Per-service cpal device ("" = Settings native default). */
  protected abstract inputDevice(params: STT_State): string;

  /**
   * Transcribe mono {@link RUST_CAPTURE_SAMPLE_RATE} PCM (one utterance, or several merged when requests fell
   * behind). Throw a `TranscriptionError` to classify a failure; `signal` aborts when the session is torn down.
   */
  protected abstract transcribe(samples: Float32Array, signal: AbortSignal): Promise<string>;

  /** Request pacing for this session (called after a successful `prepare`); `null` = none. */
  protected limiter(): RequestLimiter | null {
    return null;
  }

  /** Cleans a transcript before it is published (e.g. strip non-speech markers); "" drops it. */
  protected postProcess(text: string): string {
    return text.trim();
  }

  async start(params: STT_State): Promise<void> {
    this.lifecycle.beginStart();
    try {
      const setupError = await this.prepare(params);
      if (await this.#stopIfAborted()) return;
      if (setupError !== null) {
        this.receiver.onStatus("");
        this.lifecycle.notifyStopOnce(setupError);
        return;
      }

      const pipeline = new UtterancePipeline({
        sampleRate: RUST_CAPTURE_SAMPLE_RATE,
        limiter: this.limiter(),
        transcribe: (samples, signal) => this.transcribe(samples, signal),
        onText: (text) => {
          const clean = this.postProcess(text);
          if (clean) {
            devLog(`${this.logTag} Result:`, clean);
            this.receiver.onFinal(clean);
          }
        },
        onStatus: (status) => {
          if (status) this.receiver.onStatus(...pipelineStatus(status));
          else this.receiver.onStatus("");
        },
        onFatal: (fatal) => void this.#fail(pipelineFatalMessage(this.engineLabel(), fatal)),
      });
      this.#pipeline = pipeline;
      this.#captureId = null;
      this.#earlyEvents = [];
      this.#takeChain = Promise.resolve();

      this.#unlisten = await listenWithAbort<RustUtteranceEvent>(
        "audio:utterance",
        (event) => this.#onUtteranceEvent(pipeline, event.payload),
        this.startAbortSignal,
      );
      if (await this.#stopIfAborted()) return;

      this.#captureRequested = true;
      const captureId = await raceWithAbort(
        invoke<number>("plugin:audio|start_utterance_capture", {
          deviceName: resolveRustInputDeviceForInvoke(this.inputDevice(params)),
          vad: buildVadConfig(params.vad),
        }),
        this.startAbortSignal,
      );
      if (await this.#stopIfAborted()) return;

      this.#captureId = captureId;
      this.#isRunning = true;
      const early = this.#earlyEvents;
      this.#earlyEvents = [];
      for (const event of early) this.#onUtteranceEvent(pipeline, event);

      this.receiver.onStatus("");
      this.receiver.onStart();
      devLog(`${this.logTag} Listening (capture ${captureId})`);
    } catch (error) {
      await this.#teardown();
      if (isAbortError(error) || this.lifecycle.aborted) return;
      console.error(`${this.logTag} Error starting:`, error);
      this.receiver.onStatus("");
      this.lifecycle.notifyStopOnce(String(error));
    } finally {
      this.lifecycle.finishStart();
    }
  }

  async stop(): Promise<void> {
    this.lifecycle.abort();
    if (!this.#isRunning && !this.#unlisten && !this.#pipeline && !this.#captureRequested) {
      // Nothing live: a failed or aborted start (or a fatal pipeline error) already tore down. Still wait for the
      // Rust stop that teardown issued, so the caller's next start cannot be cancelled by it landing late.
      await this.#pendingRustStop;
      return;
    }
    devLog(`${this.logTag} Stopping...`);
    await this.#teardown();
    this.lifecycle.notifyStopOnce();
  }

  dispose(): void {
    void this.stop();
  }

  #onUtteranceEvent(pipeline: UtterancePipeline, event: RustUtteranceEvent): void {
    if (this.#pipeline !== pipeline || pipeline.disposed || !event) return;
    if (this.#captureId === null) {
      if (this.#earlyEvents.length < MAX_EARLY_EVENTS) this.#earlyEvents.push(event);
      return;
    }
    if (event.capture_id !== this.#captureId) return;
    switch (event.kind) {
      case "utterance": {
        const { capture_id: captureId, seq } = event;
        const continues = event.forced === true;
        this.#takeChain = this.#takeChain.then(() => this.#take(pipeline, captureId, seq, continues));
        break;
      }
      case "dropped":
        console.warn(`${this.logTag} utterance ${event.seq} dropped (${event.reason})`);
        pipeline.speechEnded(event.seq);
        // Transient: it clears itself once transcription has caught up instead of staying for the whole session.
        this.receiver.onStatus(i18n.t("stt.status_utterance_dropped"), { level: "warning", expireMs: TRANSIENT_WARNING_MS });
        break;
      case "error":
        void this.#fail(i18n.t("stt.error_capture", { detail: event.message }));
        break;
      case "speech_start":
        // More speech is coming: a short batch held for the billing minimum may wait for it.
        pipeline.speechStarted(event.seq);
        break;
    }
  }

  async #take(pipeline: UtterancePipeline, captureId: number, seq: number, continues: boolean): Promise<void> {
    let samples: Float32Array;
    try {
      samples = pcmFromIpc(await invoke<unknown>("plugin:audio|take_utterance", { captureId, seq }));
    } catch (e) {
      // Stopped meanwhile, or taken too late (the Rust backlog dropped it and reported `dropped`).
      devLog(`${this.logTag} utterance ${seq} unavailable:`, e);
      pipeline.speechEnded(seq);
      return;
    }
    if (this.#pipeline !== pipeline || pipeline.disposed) return;
    // Muted speech is never sent (privacy; cloud requests are billed and rate limited). Only a hard mute counts: a
    // pending unmute (waiting for an open interim sentence to end) is already an unmute for engines.
    if (this.receiver.isMuted()) {
      devLog(`${this.logTag} utterance ${seq} skipped (muted)`);
      pipeline.speechEnded(seq);
      return;
    }
    pipeline.push({ seq, samples, continues });
  }

  async #fail(message: string): Promise<void> {
    if (this.lifecycle.aborted) return;
    console.error(`${this.logTag} ${message}`);
    await this.#teardown();
    this.receiver.onStatus("");
    this.lifecycle.notifyStopOnce(message);
  }

  async #stopIfAborted(): Promise<boolean> {
    if (!this.lifecycle.aborted) return false;
    await this.#teardown();
    this.lifecycle.notifyStopOnce();
    return true;
  }

  async #teardown(): Promise<void> {
    this.#isRunning = false;
    this.#pipeline?.dispose();
    this.#pipeline = null;
    this.#captureId = null;
    this.#earlyEvents = [];
    if (this.#unlisten) {
      this.#unlisten();
      this.#unlisten = undefined;
    }
    // Stop the Rust capture once per requested start. A teardown that runs after another one already issued the
    // stop (e.g. the start's catch path after stop()) sends nothing new; it only waits for that stop to settle.
    if (this.#captureRequested) {
      this.#captureRequested = false;
      this.#pendingRustStop = invoke("plugin:audio|stop_utterance_capture").then(
        () => {},
        () => {},
      );
    }
    await this.#pendingRustStop;
  }
}
