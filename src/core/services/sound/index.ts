import i18n from "i18next";
import { proxy } from "valtio";
import { IServiceInterface, type RustTtsSinkRoute, type RustTtsSinkUtteranceEvent } from "@/types";
import { decodeAudioClip, type DecodedPcm } from "./pcm";
import { TtsSinkClient, isFinalSinkState, isSinkCancelled } from "./sink";

export type VoiceClipOptions = {
  /** Output device name (cpal); empty = the app's output device, else the system default. */
  device_name: string;
  /** 1 = unchanged (0..2). */
  volume: number;
  /** Playback speed; 1 = unchanged (tempo and pitch together, like the old player). */
  rate: number;
  /** Where the clip plays (default: the device). */
  route?: RustTtsSinkRoute;
};

/** Bounds of the sink: NaN would fail to deserialize, and a speed of 0 would never finish. */
const MAX_VOLUME = 2;
const MIN_RATE = 0.1;
const MAX_RATE = 10;

function finite(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/** Clamp clip options to values the sink accepts. */
export function sanitizeVoiceClipOptions(options: VoiceClipOptions, defaultDevice: string): VoiceClipOptions {
  const rate = finite(options.rate, 1);
  return {
    device_name: (options.device_name || defaultDevice || "").trim(),
    volume: Math.min(MAX_VOLUME, Math.max(0, finite(options.volume, 1))),
    rate: rate > 0 ? Math.min(MAX_RATE, Math.max(MIN_RATE, rate)) : 1,
    route: options.route ?? "device",
  };
}

function detail(error: unknown): string {
  return typeof error === "string" ? error : error instanceof Error ? error.message : String(error ?? "");
}

function describePlaybackError(error: unknown, device: string): string {
  return device
    ? i18n.t("toasts.sound_playback_failed_device", { device, error: detail(error) })
    : i18n.t("toasts.sound_playback_failed", { error: detail(error) });
}

type Waiter = { resolve: () => void; reject: (error: Error) => void; device: string };

/**
 * Voice clips through the unified Rust TTS sink (P2-G folded the old serial `play_async` player into it; no
 * clip crosses IPC as a JSON number array any more).
 *
 * - {@link enqueueVoiceClip} decodes a whole clip (WAV in TypeScript, other formats with the webview's decoder)
 *   and {@link enqueuePcm} takes PCM; both begin a sink utterance in call order, push the audio over binary IPC
 *   and resolve when it finished playing (or was dropped by {@link clearVoiceClips}). They reject with a readable,
 *   localized message when it could not be played (missing output device, undecodable audio, …).
 * - {@link sink} is the shared sink client; the TTS sentence pipeline uses it too.
 */
class Service_Sound implements IServiceInterface {
  public serviceState = proxy({
    muted: false,
  });

  readonly sink: TtsSinkClient;
  #waiters = new Map<string, Waiter>();
  #tail: Promise<void> = Promise.resolve();
  #seq = 0;
  #session = Math.random().toString(36).slice(2, 8);
  /** Bumped by clearVoiceClips(): clips queued before it are dropped. */
  #generation = 0;
  #unsubscribe: (() => void) | null = null;

  constructor(sink: TtsSinkClient = new TtsSinkClient()) {
    this.sink = sink;
  }

  async init() {
    this.#subscribe();
  }

  #subscribe() {
    if (!this.#unsubscribe) this.#unsubscribe = this.sink.onUtterance((event) => this.#onEvent(event));
  }

  #onEvent(event: RustTtsSinkUtteranceEvent) {
    const waiter = this.#waiters.get(event.id);
    if (!waiter || !isFinalSinkState(event.state)) return;
    this.#waiters.delete(event.id);
    if (event.state === "error") waiter.reject(new Error(describePlaybackError(event.error ?? "", waiter.device)));
    else waiter.resolve();
  }

  /**
   * Queue an encoded clip (WAV, MP3, …). Resolves when it finished playing (or was dropped by `clearVoiceClips`);
   * rejects with a readable, localized message when it could not be played.
   */
  enqueueVoiceClip(buffer: ArrayBuffer, options: VoiceClipOptions): Promise<void> {
    if (!buffer || buffer.byteLength === 0) return Promise.resolve();
    return this.#enqueue(() => decodeAudioClip(buffer), options);
  }

  /** Queue decoded PCM (same contract as {@link enqueueVoiceClip}). */
  enqueuePcm(pcm: DecodedPcm, options: VoiceClipOptions): Promise<void> {
    if (!pcm.samples.length) return Promise.resolve();
    return this.#enqueue(async () => pcm, options);
  }

  #enqueue(produce: () => Promise<DecodedPcm>, raw: VoiceClipOptions): Promise<void> {
    const defaultDevice = window.ApiServer?.state?.audioOutputDevice ?? "";
    const options = sanitizeVoiceClipOptions(raw, defaultDevice);
    const id = `clip-${this.#session}-${this.#seq++}`;
    const generation = this.#generation;
    this.#subscribe();
    return new Promise<void>((resolve, reject) => {
      const fail = (error: unknown) => {
        this.#waiters.delete(id);
        reject(new Error(describePlaybackError(error, options.device_name)));
      };
      // Begins are issued in call order; decoding the next clip overlaps playback of this one.
      this.#tail = this.#tail.then(async () => {
        if (generation !== this.#generation) return resolve();
        let pcm: DecodedPcm;
        try {
          pcm = await produce();
        } catch (error) {
          return fail(error);
        }
        if (generation !== this.#generation || !pcm.samples.length) return resolve();
        this.#waiters.set(id, { resolve, reject, device: options.device_name });
        try {
          await this.sink.ready();
          await this.sink.begin(id, { route: options.route ?? "device", device: options.device_name, volume: options.volume });
        } catch (error) {
          return fail(error);
        }
        if (generation !== this.#generation || !this.#waiters.has(id)) {
          // clearVoiceClips() ran while the begin was in flight: its cancel may have reached Rust before the begin
          // (a no-op there), so cancel again now that the utterance exists. The waiter was already resolved.
          void this.sink.cancel(id).catch(() => {});
          return;
        }
        // Push and end run on after the begin, so the next clip's begin is not held up by back-pressure.
        void (async () => {
          try {
            await this.sink.push(id, {
              samples: pcm.samples,
              sampleRate: Math.round(pcm.sampleRate * (options.rate || 1)),
              channels: pcm.channels,
            });
            await this.sink.end(id);
          } catch (error) {
            if (!this.#waiters.has(id)) return; // cleared meanwhile
            if (isSinkCancelled(error)) {
              this.#waiters.delete(id);
              resolve();
              return;
            }
            void this.sink.cancel(id).catch(() => {});
            fail(error);
          }
        })();
      });
    });
  }

  /** Stop voice clips: the one playing, every queued one and those still being decoded. */
  clearVoiceClips(): void {
    this.#generation++;
    for (const [id, waiter] of [...this.#waiters]) {
      this.#waiters.delete(id);
      waiter.resolve();
      void this.sink.cancel(id).catch(() => {});
    }
  }

  /** Clips queued in the sink (waiting or playing). */
  get pendingVoiceClips(): number {
    return this.#waiters.size;
  }

  dispose() {
    this.clearVoiceClips();
    this.#unsubscribe?.();
    this.#unsubscribe = null;
  }
}

export default Service_Sound;
