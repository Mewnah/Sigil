import { runsOnTemplateEngine, type VoiceKind } from "./families";

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

/**
 * The engine block (ms) a template voice runs for a `block_ms` setting: rounded up to the plan's 10 ms
 * frames and then to a whole number of 20 ms (`engine/plan.rs` `block_frames`), at most 200 ms. 0 (the
 * block follows the device chunk) stays 0.
 */
export function engineBlockMs(blockMs: number): number {
  if (!Number.isFinite(blockMs) || blockMs <= 0) return 0;
  const frames = Math.max(1, Math.ceil(blockMs / 10 - 1e-9));
  return Math.min(200, Math.ceil(frames / 2) * 20);
}

/**
 * The engine block (ms) that clamps the crossfade of the selected voice: `engineBlockMs(blockMs)` for a
 * template voice (an unlisted voice counts as one, the default setup) in RVC mode, 0 for everything else.
 * A bring-your-own (legacy) voice follows the device chunk, and a voice family's SOLA overlap is clamped to
 * the family's own fixed block (its preset may round the block), not to the `block_ms`-derived one.
 */
export function crossfadeEngineBlockMs(kind: VoiceKind, isRvc: boolean, blockMs: number): number {
  return isRvc && runsOnTemplateEngine(kind) ? engineBlockMs(blockMs) : 0;
}

/**
 * Largest crossfade worth offering: one device chunk for the legacy harness, but a template voice's
 * harness clamps the SOLA overlap to one emitted ENGINE block (`stream.rs`: `overlap.min(block_expected)`
 * with the fixed block), so a 10 ms device block (P3B-H `device_block`) does not cap a 40 ms engine block.
 * `engineBlock` is `engineBlockMs(blockMs)` for a template voice, 0 otherwise.
 */
export function crossfadeMaxMsForEngine(frames: number, runningChunkMs: number | null | undefined, engineBlock: number): number {
  const byChunk = crossfadeMaxMs(frames, runningChunkMs);
  if (!(engineBlock > 0)) return byChunk;
  return Math.max(byChunk, Math.max(1, Math.min(CROSSFADE_MS_MAX, Math.floor(engineBlock))));
}

/** The block (ms) `device_block = model_divisor` runs when it applies (`chunk_processor::model_divisor_chunk`: 10 ms at the capture rate). */
export const MODEL_DIVISOR_BLOCK_MS = 10;

/** What the device-block row says under `model_divisor`: nothing, that the voice cannot use it, the block running, or that it did not apply. */
export type DeviceBlockNote = "template_only" | "live" | "not_applied" | null;

/**
 * The note of the device-block row. `runningChunkMs` is the running engine's `chunk_ms` metric (null while stopped).
 * The mode applies to template voices: with an explicit engine block it picks a divisor of that block, with the
 * block on automatic a divisor that derives the same engine block. When neither fits (the engine block follows a
 * longer chunk, or the rate has no whole 10 ms frame) the run keeps the chunk size, which shows as a running
 * block other than 10 ms.
 */
export function deviceBlockNote(modelDivisor: boolean, template: boolean, runningChunkMs: number | null | undefined): DeviceBlockNote {
  if (!modelDivisor) return null;
  if (!template) return "template_only";
  if (runningChunkMs == null || !(runningChunkMs > 0)) return null;
  return Math.abs(runningChunkMs - MODEL_DIVISOR_BLOCK_MS) < 0.05 ? "live" : "not_applied";
}
