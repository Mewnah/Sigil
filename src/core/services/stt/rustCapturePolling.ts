import { invoke } from "@tauri-apps/api/core";
import { UnlistenFn } from "@tauri-apps/api/event";
import { devLog } from "@/utils/devLog";
import { resolveRustInputDeviceForInvoke } from "@/utils/rustAudioInput";
import { isAbortError, listenWithAbort, raceWithAbort } from "./abort";
import { CaptureToMonoConverter } from "./audioConvert";
import { SttStartLifecycle } from "./localEngineLifecycle";
import { STT_State } from "./schema";
import { ISTTReceiver, ISpeechRecognitionService, RustAudioChunkPayload } from "./types";

/**
 * Rate of the mono f32 windows handed to {@link RustCapturePollingService.transcribe}. Rust captures at the
 * device's own rate/channels (the requested rate is only a hint), so chunks are downmixed and resampled to this.
 */
export const RUST_CAPTURE_SAMPLE_RATE = 16000;
const POLL_MS = 500;
/** Do not send windows shorter than 0.25 s. */
const MIN_SAMPLES = 4000;
/** After this many failed transcriptions in a row, stop so the user is not left "connected" with only interim errors. */
const MAX_CONSECUTIVE_ERRORS = 3;
const MAX_ERROR_DETAIL = 200;

export function truncateForUi(s: string, max: number): string {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length <= max ? t : `${t.slice(0, max)}…`;
}

/**
 * Shared pipeline for STT engines that capture audio in Rust (`plugin:audio|start_audio_capture`), buffer
 * `audio:chunk` events and transcribe the buffer every {@link POLL_MS} (Moonshine, local OpenAI-compatible HTTP).
 *
 * Owns start/stop bookkeeping, teardown, the consecutive-error budget and aborting in-flight transcriptions on
 * stop/teardown. Subclasses provide engine setup ({@link prepare}) and one transcription call ({@link transcribe}).
 */
export abstract class RustCapturePollingService implements ISpeechRecognitionService {
  startAbortSignal: AbortSignal | null = null;
  protected readonly lifecycle: SttStartLifecycle;

  #isRunning = false;
  #unlistenAudio?: UnlistenFn;
  /** Mono {@link RUST_CAPTURE_SAMPLE_RATE} windows, already converted from the device format. */
  #audioBuffer: Float32Array[] = [];
  /** Device-format `audio:chunk` → mono 16 kHz; stateful (resampler history), so one per start. */
  #converter = new CaptureToMonoConverter(RUST_CAPTURE_SAMPLE_RATE);
  #processingInterval?: ReturnType<typeof setInterval>;
  /** Set right before `start_audio_capture` is sent; the teardown that issues the matching stop clears it. */
  #rustCaptureRequested = false;
  /**
   * The last `stop_audio_capture` issued by `#teardown`. Every stop() awaits it, including the early return
   * after a failed/aborted start already tore down, so the orchestrator never starts the next engine while an
   * older stop is still in flight (a late stop would cancel the next `start_audio_capture`).
   */
  #pendingRustStop: Promise<void> = Promise.resolve();
  /** Prevents overlapping transcriptions when one outlasts the poll interval. */
  #transcribeInFlight = false;
  #consecutiveErrors = 0;
  /** Aborted by teardown so in-flight transcription requests do not outlive stop(). */
  #requestAbort: AbortController | null = null;

  /** Console tag, e.g. `[Moonshine]`. */
  protected abstract readonly logTag: string;
  /** User-facing prefix for interim errors and the stop message. */
  protected abstract readonly errorLabel: string;

  constructor(protected readonly receiver: ISTTReceiver) {
    this.lifecycle = new SttStartLifecycle((error) => this.receiver.onStop(error));
  }

  /** Engine setup before capture starts. Return an error message to stop with, or `null` to continue. */
  protected abstract prepare(params: STT_State): Promise<string | null>;

  /** Per-service cpal device ("" = Settings native default). */
  protected abstract inputDevice(params: STT_State): string;

  /**
   * Transcribe one mono {@link RUST_CAPTURE_SAMPLE_RATE} window. Throw an `Error` whose message is the user-facing
   * detail to count a failure. `signal` aborts when the pipeline is torn down.
   */
  protected abstract transcribe(samples: Float32Array, signal: AbortSignal): Promise<string>;

  async start(params: STT_State): Promise<void> {
    this.lifecycle.beginStart();
    try {
      const setupError = await this.prepare(params);
      if (await this.#stopIfAborted()) return;
      if (setupError !== null) {
        this.lifecycle.notifyStopOnce(setupError);
        return;
      }

      const converter = new CaptureToMonoConverter(RUST_CAPTURE_SAMPLE_RATE);
      this.#converter = converter;
      this.#unlistenAudio = await listenWithAbort<RustAudioChunkPayload>(
        "audio:chunk",
        (event) => {
          if (!this.#isRunning || this.#converter !== converter) return;
          // Payloads carry the DEVICE rate and interleaved channels (e.g. 48 kHz stereo), not 16 kHz mono.
          const { samples, sample_rate, channels } = event.payload;
          const mono = converter.convert(samples, sample_rate, channels);
          if (mono.length > 0) this.#audioBuffer.push(mono);
        },
        this.startAbortSignal,
      );
      if (await this.#stopIfAborted()) return;

      this.#isRunning = true;
      this.#consecutiveErrors = 0;
      this.#requestAbort = new AbortController();

      this.#rustCaptureRequested = true;
      await raceWithAbort(
        invoke("plugin:audio|start_audio_capture", {
          deviceName: resolveRustInputDeviceForInvoke(this.inputDevice(params)),
          sampleRate: RUST_CAPTURE_SAMPLE_RATE,
        }),
        this.startAbortSignal,
      );
      if (await this.#stopIfAborted()) return;

      this.#processingInterval = setInterval(() => {
        void this.#processAudioBuffer();
      }, POLL_MS);

      this.receiver.onStart();
      devLog(`${this.logTag} Recording started`);
    } catch (error) {
      await this.#teardown();
      if (isAbortError(error) || this.lifecycle.aborted) return;
      console.error(`${this.logTag} Error starting:`, error);
      this.lifecycle.notifyStopOnce(String(error));
    } finally {
      this.lifecycle.finishStart();
    }
  }

  async stop(): Promise<void> {
    this.lifecycle.abort();
    if (!this.#isRunning && !this.#unlistenAudio && !this.#processingInterval && !this.#rustCaptureRequested) {
      // Nothing live: a failed or aborted start (or the error budget) already tore down. Still wait for the Rust
      // stop that teardown issued, so the caller's next start cannot be cancelled by it landing late.
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

  async #stopIfAborted(): Promise<boolean> {
    if (!this.lifecycle.aborted) return false;
    await this.#teardown();
    this.lifecycle.notifyStopOnce();
    return true;
  }

  async #teardown(): Promise<void> {
    this.#isRunning = false;
    this.#requestAbort?.abort();
    this.#requestAbort = null;
    if (this.#processingInterval) {
      clearInterval(this.#processingInterval);
      this.#processingInterval = undefined;
    }
    if (this.#unlistenAudio) {
      this.#unlistenAudio();
      this.#unlistenAudio = undefined;
    }
    this.#audioBuffer = [];
    // Stop Rust capture once per requested start. A teardown that runs after another one already issued the stop
    // (e.g. the start's catch path after stop()) sends nothing new; it only waits for that stop to settle.
    if (this.#rustCaptureRequested) {
      this.#rustCaptureRequested = false;
      this.#pendingRustStop = invoke("plugin:audio|stop_audio_capture").then(
        () => {},
        () => {},
      );
    }
    await this.#pendingRustStop;
  }

  async #processAudioBuffer(): Promise<void> {
    const requestAbort = this.#requestAbort;
    if (!this.#isRunning || !requestAbort || this.#transcribeInFlight || this.#audioBuffer.length === 0) return;

    const totalLength = this.#audioBuffer.reduce((acc, chunk) => acc + chunk.length, 0);
    if (totalLength < MIN_SAMPLES) return;

    const combined = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.#audioBuffer) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }
    this.#audioBuffer = [];

    const { signal } = requestAbort;
    this.#transcribeInFlight = true;
    this.receiver.onInterim("...");
    try {
      const text = (await this.transcribe(combined, signal)).trim();
      if (signal.aborted) return;
      this.#consecutiveErrors = 0;
      if (text) {
        devLog(`${this.logTag} Result:`, text);
        this.receiver.onFinal(text);
      }
    } catch (error) {
      if (signal.aborted) return;
      console.error(`${this.logTag} Transcription error:`, error);
      const detail = truncateForUi(error instanceof Error ? error.message : String(error), MAX_ERROR_DETAIL);
      this.#consecutiveErrors++;
      if (this.#consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
        await this.#teardown();
        this.lifecycle.notifyStopOnce(
          `${this.errorLabel}: stopped after ${MAX_CONSECUTIVE_ERRORS} failed requests. Last error: ${detail}`,
        );
        return;
      }
      this.receiver.onInterim(`${this.errorLabel}: ${detail}`);
    } finally {
      this.#transcribeInFlight = false;
    }
  }
}
