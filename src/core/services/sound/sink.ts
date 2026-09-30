/**
 * TypeScript client of the unified Rust TTS sink (`tts-sink` plugin, P2-G).
 *
 * - Control calls (`begin` / `end` / `cancel` / `flush` / `status`) are plain IPC commands.
 * - {@link TtsSinkClient.push} sends PCM as a raw little-endian f32 body with the `x-sigil-*` headers
 *   ({@link TTS_SINK_HEADERS}): binary IPC, never JSON number arrays. It resolves once the sink queued the audio
 *   (after waiting for room when its queue is full: back-pressure) and rejects with `"cancelled"` once the
 *   utterance was cancelled ({@link isSinkCancelled}).
 * - Utterance events (`tts_sink:utterance`) fan out to every {@link TtsSinkClient.onUtterance} subscriber through a
 *   single Tauri listener, attached on first use; {@link TtsSinkClient.ready} resolves once it is attached, so a
 *   caller that awaits it before its first `begin` never misses an event. The first `ready()` also flushes what an
 *   earlier webview session left in the sink.
 *
 * Utterances play strictly in begin order: callers that begin several must issue the begins one after another.
 */
import { invoke, type InvokeArgs, type InvokeOptions } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  TTS_SINK_EVENTS,
  TTS_SINK_HEADERS,
  type RustTtsSinkStatus,
  type RustTtsSinkTarget,
  type RustTtsSinkUtteranceEvent,
} from "@/types";

/** How the client reaches Rust (the real Tauri APIs, or a fake in tests). */
export interface TtsSinkTransport {
  invoke<T>(cmd: string, args?: InvokeArgs, options?: InvokeOptions): Promise<T>;
  listen<T>(event: string, handler: (payload: T) => void): Promise<() => void>;
}

export const tauriSinkTransport: TtsSinkTransport = {
  invoke: (cmd, args, options) => invoke(cmd, args, options),
  listen: async (event, handler) => {
    const unlisten = await listen(event, (e) => handler(e.payload as never));
    return unlisten;
  },
};

/** PCM for {@link TtsSinkClient.push}: interleaved f32 in [-1, 1]. */
export interface SinkPcm {
  samples: Float32Array;
  sampleRate: number;
  channels: number;
}

/** Error text Rust returns for a cancelled utterance (`SinkError::Cancelled`). */
export const SINK_CANCELLED = "cancelled";

/** Whether `error` means the utterance was cancelled (stop / skip): not a failure to report. */
export function isSinkCancelled(error: unknown): boolean {
  const message = typeof error === "string" ? error : error instanceof Error ? error.message : "";
  return message === SINK_CANCELLED;
}

/** Terminal utterance states. */
export function isFinalSinkState(state: RustTtsSinkUtteranceEvent["state"]): boolean {
  return state === "done" || state === "cancelled" || state === "error";
}

/** Upper bound of one push (samples); longer clips are sent in several pushes. */
const MAX_PUSH_SAMPLES = 48_000 * 2 * 30;

export class TtsSinkClient {
  #handlers = new Set<(event: RustTtsSinkUtteranceEvent) => void>();
  #listening: Promise<void> | null = null;
  /** The start-of-session flush was issued (see {@link ready}). */
  #flushed = false;

  constructor(private readonly transport: TtsSinkTransport = tauriSinkTransport) {}

  /** Subscribe to utterance events; returns the unsubscribe function. Attaches the Tauri listener on first use. */
  onUtterance(handler: (event: RustTtsSinkUtteranceEvent) => void): () => void {
    this.#handlers.add(handler);
    void this.ready();
    return () => {
      this.#handlers.delete(handler);
    };
  }

  /**
   * Resolves once the event listener is attached (never rejects; without Tauri it resolves at once). The first time,
   * it also flushes the sink: utterances an earlier webview session left behind (a reload in the middle of speech)
   * would keep playing, or, begun but never fed, hold up this session's speech until the sink timed them out. This
   * session has no utterance yet then: every producer awaits `ready()` before its first `begin`.
   */
  ready(): Promise<void> {
    if (!this.#listening) {
      const first = !this.#flushed;
      this.#flushed = true;
      this.#listening = this.transport
        .listen<RustTtsSinkUtteranceEvent>(TTS_SINK_EVENTS.utterance, (event) => {
          for (const handler of [...this.#handlers]) {
            try {
              handler(event);
            } catch (error) {
              console.error("[TTS sink] event handler failed", error);
            }
          }
        })
        .then(
          async () => {
            if (!first) return;
            try {
              await this.flush();
            } catch {
              // No sink to flush (web mode): nothing was left behind either.
            }
          },
          (error) => {
            console.warn("[TTS sink] cannot listen for utterance events:", error);
            this.#listening = null;
          },
        );
    }
    return this.#listening ?? Promise.resolve();
  }

  /** Begin utterance `id` (queued after every begun one). */
  begin(id: string, target: RustTtsSinkTarget): Promise<void> {
    return this.transport.invoke<void>("plugin:tts-sink|begin_utterance", { spec: { id, target } });
  }

  /** Queue PCM for `id` (binary IPC; resolves once queued, rejects with "cancelled" after a cancel). */
  async push(id: string, pcm: SinkPcm): Promise<void> {
    const channels = Math.max(1, Math.floor(pcm.channels));
    const samples = pcm.samples;
    const frames = Math.floor(samples.length / channels);
    if (frames === 0) return;
    const headers = {
      [TTS_SINK_HEADERS.utterance]: id,
      [TTS_SINK_HEADERS.sampleRate]: String(Math.round(pcm.sampleRate)),
      [TTS_SINK_HEADERS.channels]: String(channels),
    };
    const step = Math.max(channels, Math.floor(MAX_PUSH_SAMPLES / channels) * channels);
    for (let at = 0; at < frames * channels; at += step) {
      const part = samples.subarray(at, Math.min(frames * channels, at + step));
      // Float32Array is little-endian on every platform Sigil runs on (x86-64, ARM64).
      const body = new Uint8Array(part.buffer, part.byteOffset, part.byteLength);
      await this.transport.invoke<void>("plugin:tts-sink|push_pcm", body, { headers });
    }
  }

  /** Mark `id` complete: it finishes once its audio played. */
  end(id: string): Promise<void> {
    return this.transport.invoke<void>("plugin:tts-sink|end_utterance", { id });
  }

  /** Cancel `id`, or every utterance (`null`). */
  cancel(id: string | null): Promise<void> {
    return this.transport.invoke<void>("plugin:tts-sink|cancel_utterance", { id });
  }

  /** Cancel everything and drop queued audio (outputs stay open). */
  flush(): Promise<void> {
    return this.transport.invoke<void>("plugin:tts-sink|flush_sink");
  }

  status(): Promise<RustTtsSinkStatus> {
    return this.transport.invoke<RustTtsSinkStatus>("plugin:tts-sink|get_sink_status");
  }
}
