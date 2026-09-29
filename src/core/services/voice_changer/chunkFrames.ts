/** Must match `ALLOWED_CHUNK_FRAMES` in `src-tauri/.../chunk_processor.rs`. */
export const VOICE_CHANGER_CHUNK_FRAMES = [
  160, 192, 224, 256, 320, 384, 448, 512, 576, 640, 704, 768, 896, 1024, 1280, 1536, 2048, 2304,
  2560, 2816, 3072, 3584, 4096, 5120, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 40960,
  48000,
] as const;

export type VoiceChangerChunkFrames = (typeof VOICE_CHANGER_CHUNK_FRAMES)[number];

const CHUNK_SET = new Set<number>(VOICE_CHANGER_CHUNK_FRAMES);

export function isVoiceChangerChunkFrames(n: number): n is VoiceChangerChunkFrames {
  return CHUNK_SET.has(n);
}

/** Snap persisted or IPC values to the nearest allowed chunk size. */
export function normalizeChunkFramesValue(n: number): VoiceChangerChunkFrames {
  const rounded = Math.round(n);
  if (isVoiceChangerChunkFrames(rounded)) return rounded;
  let best: VoiceChangerChunkFrames = 512;
  let bestDist = Infinity;
  for (const a of VOICE_CHANGER_CHUNK_FRAMES) {
    const d = Math.abs(a - rounded);
    if (d < bestDist) {
      bestDist = d;
      best = a;
    }
  }
  return best;
}

/** ~Milliseconds of audio per chunk at 48 kHz (for UI labels). */
export function chunkFramesToMsAt48k(frames: number): string {
  const ms = (frames / 48000) * 1000;
  if (ms >= 10) return ms.toFixed(0);
  if (ms >= 1) return ms.toFixed(1);
  return ms.toFixed(2);
}

/** Persisted crossfade range (schema / Rust `crossfade_ms.clamp(0.0, 200.0)`). */
export const CROSSFADE_MS_MAX = 200;

/**
 * Duration (ms) of one processing chunk. While the engine runs, pass its `chunk_ms` metric (chunk frames at the
 * real capture rate, e.g. 44.1 kHz); otherwise the capture rate is unknown and 48 kHz is assumed.
 */
export function chunkDurationMs(frames: number, runningChunkMs?: number | null): number {
  if (runningChunkMs != null && Number.isFinite(runningChunkMs) && runningChunkMs > 0) return runningChunkMs;
  return (frames / 48000) * 1000;
}

/**
 * Crossfade (ms) the RVC harness actually uses for `crossfadeMs`: it clamps the SOLA overlap to one
 * processing chunk (`stream.rs`: `overlap.min(block_expected)`), so anything longer behaves like one chunk.
 * `runningChunkMs` is the running engine's `chunk_ms` metric (see `chunkDurationMs`).
 */
export function effectiveCrossfadeMs(crossfadeMs: number, frames: number, runningChunkMs?: number | null): number {
  return Math.min(crossfadeMs, chunkDurationMs(frames, runningChunkMs));
}

/** Largest whole-ms crossfade worth offering for this chunk size (one chunk, at least 1 ms). */
export function crossfadeMaxMs(frames: number, runningChunkMs?: number | null): number {
  return Math.max(1, Math.min(CROSSFADE_MS_MAX, Math.floor(chunkDurationMs(frames, runningChunkMs))));
}
