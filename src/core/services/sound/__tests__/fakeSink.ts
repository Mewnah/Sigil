/**
 * Test helpers for the TTS sink client (not a test file): an in-memory stand-in for the Rust `tts-sink` plugin and a
 * WAV builder. Used by the sound and TTS pipeline tests.
 */
import type { InvokeArgs, InvokeOptions } from "@tauri-apps/api/core";
import { TTS_SINK_HEADERS, type RustTtsSinkTarget, type RustTtsSinkUtteranceEvent } from "@/types";
import type { TtsSinkTransport } from "../sink";

export interface FakeUtterance {
  id: string;
  target: RustTtsSinkTarget;
  pushes: { samples: Float32Array; sampleRate: number; channels: number }[];
  ended: boolean;
  state: "queued" | "playing" | "done" | "cancelled" | "error";
}

/** Mirrors the sink's IPC semantics closely enough for TS tests; events are emitted by the test. */
export class FakeSinkTransport implements TtsSinkTransport {
  readonly calls: { cmd: string; args?: InvokeArgs; headers?: Record<string, string> }[] = [];
  readonly utterances = new Map<string, FakeUtterance>();
  /** Begin order. */
  readonly order: string[] = [];
  readonly listeners = new Set<(payload: RustTtsSinkUtteranceEvent) => void>();
  /** Command -> error thrown by its next call. */
  readonly failNext = new Map<string, string>();
  /** Awaited by every push (back-pressure in tests). */
  pushGate: ((id: string) => Promise<void>) | null = null;
  listenCalls = 0;

  async invoke<T>(cmd: string, args?: InvokeArgs, options?: InvokeOptions): Promise<T> {
    const headers = (options?.headers ?? undefined) as Record<string, string> | undefined;
    this.calls.push({ cmd, args, headers });
    const failure = this.failNext.get(cmd);
    if (failure !== undefined) {
      this.failNext.delete(cmd);
      throw failure;
    }
    const a = (args ?? {}) as Record<string, unknown>;
    switch (cmd) {
      case "plugin:tts-sink|begin_utterance": {
        const spec = a.spec as { id: string; target: RustTtsSinkTarget };
        if (this.utterances.has(spec.id)) throw `utterance "${spec.id}" was already begun`;
        this.utterances.set(spec.id, { id: spec.id, target: spec.target, pushes: [], ended: false, state: "queued" });
        this.order.push(spec.id);
        break;
      }
      case "plugin:tts-sink|push_pcm": {
        const id = headers?.[TTS_SINK_HEADERS.utterance] ?? "";
        const u = this.utterances.get(id);
        if (!u) throw `no TTS utterance "${id}"`;
        if (u.state === "cancelled") throw "cancelled";
        await this.pushGate?.(id);
        // Cancelled while the push waited.
        if ((u.state as FakeUtterance["state"]) === "cancelled") throw "cancelled";
        const bytes = args as Uint8Array;
        const copy = bytes.slice();
        u.pushes.push({
          samples: new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4),
          sampleRate: Number(headers?.[TTS_SINK_HEADERS.sampleRate]),
          channels: Number(headers?.[TTS_SINK_HEADERS.channels]),
        });
        break;
      }
      case "plugin:tts-sink|end_utterance": {
        const u = this.utterances.get(String(a.id));
        if (!u) throw `no TTS utterance "${String(a.id)}"`;
        if (u.state === "cancelled") throw "cancelled";
        u.ended = true;
        break;
      }
      case "plugin:tts-sink|flush_sink":
      case "plugin:tts-sink|cancel_utterance": {
        for (const u of this.utterances.values()) {
          if ((a.id === null || a.id === undefined || a.id === u.id) && (u.state === "queued" || u.state === "playing")) {
            u.state = "cancelled";
            this.emit({ id: u.id, state: "cancelled", error: null, playedMs: 0 });
          }
        }
        break;
      }
      case "plugin:tts-sink|get_sink_status": {
        const live = this.order.filter((id) => {
          const s = this.utterances.get(id)?.state;
          return s === "queued" || s === "playing";
        });
        const playing = live.find((id) => this.utterances.get(id)?.state === "playing") ?? null;
        return { playing, queued: live.filter((id) => id !== playing), queuedMs: 0, devices: [] } as T;
      }
      default:
        break;
    }
    return undefined as T;
  }

  async listen<T>(_event: string, handler: (payload: T) => void): Promise<() => void> {
    this.listenCalls++;
    const h = handler as unknown as (payload: RustTtsSinkUtteranceEvent) => void;
    this.listeners.add(h);
    return () => this.listeners.delete(h);
  }

  emit(event: RustTtsSinkUtteranceEvent) {
    for (const l of [...this.listeners]) l(event);
  }

  /** Report `id` audible. */
  play(id: string) {
    const u = this.utterances.get(id);
    if (u && u.state === "queued") u.state = "playing";
    this.emit({ id, state: "playing", error: null, playedMs: 0 });
  }

  /** Report `id` finished playing. */
  done(id: string) {
    const u = this.utterances.get(id);
    if (u) u.state = "done";
    this.emit({ id, state: "done", error: null, playedMs: 100 });
  }

  fail(id: string, error: string) {
    const u = this.utterances.get(id);
    if (u) u.state = "error";
    this.emit({ id, state: "error", error, playedMs: 0 });
  }

  /** All samples pushed for `id`, concatenated. */
  audio(id: string): number[] {
    return (this.utterances.get(id)?.pushes ?? []).flatMap((p) => Array.from(p.samples));
  }

  commands(): string[] {
    return this.calls.map((c) => c.cmd.replace("plugin:tts-sink|", ""));
  }
}

/** A WAV file with `samples` (interleaved; integers for PCM, floats for float formats). */
export function makeWav(
  samples: number[],
  { rate = 24_000, channels = 1, bits = 16, float = false, extensible = false, extraChunk = false } = {},
): ArrayBuffer {
  const bytesPer = bits / 8;
  const dataBytes = samples.length * bytesPer;
  const fmtSize = extensible ? 40 : 16;
  const extra = extraChunk ? 8 + 3 + 1 : 0; // an odd-sized LIST chunk plus its pad byte
  const buf = new ArrayBuffer(12 + 8 + fmtSize + extra + 8 + dataBytes);
  const dv = new DataView(buf);
  const str = (at: number, s: string) => [...s].forEach((c, i) => dv.setUint8(at + i, c.charCodeAt(0)));
  str(0, "RIFF");
  dv.setUint32(4, buf.byteLength - 8, true);
  str(8, "WAVE");
  str(12, "fmt ");
  dv.setUint32(16, fmtSize, true);
  const code = float ? 3 : 1;
  dv.setUint16(20, extensible ? 0xfffe : code, true);
  dv.setUint16(22, channels, true);
  dv.setUint32(24, rate, true);
  dv.setUint32(28, rate * channels * bytesPer, true);
  dv.setUint16(32, channels * bytesPer, true);
  dv.setUint16(34, bits, true);
  let at = 20 + fmtSize;
  if (extensible) {
    dv.setUint16(36, 22, true);
    dv.setUint16(38, bits, true);
    dv.setUint16(44, code, true); // sub-format GUID starts with the format code
  }
  if (extraChunk) {
    str(at, "LIST");
    dv.setUint32(at + 4, 3, true);
    at += 8 + 3 + 1;
  }
  str(at, "data");
  dv.setUint32(at + 4, dataBytes, true);
  at += 8;
  samples.forEach((s, i) => {
    const p = at + i * bytesPer;
    if (float && bits === 32) dv.setFloat32(p, s, true);
    else if (float && bits === 64) dv.setFloat64(p, s, true);
    else if (bits === 8) dv.setUint8(p, s);
    else if (bits === 16) dv.setInt16(p, s, true);
    else if (bits === 24) {
      dv.setUint8(p, s & 0xff);
      dv.setUint8(p + 1, (s >> 8) & 0xff);
      dv.setInt8(p + 2, s >> 16);
    } else if (bits === 32) dv.setInt32(p, s, true);
  });
  return buf;
}

export const flush = async (n = 20) => {
  for (let i = 0; i < n; i++) await Promise.resolve();
};
