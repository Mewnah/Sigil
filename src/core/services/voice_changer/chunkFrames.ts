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
