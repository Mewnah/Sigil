import { z } from "zod";

import { zSafe } from "@/utils";

import { normalizeChunkFramesValue } from "./chunkFrames";

export const VoiceChangerModeSchema = z.enum(["classic", "onnx", "sidecar", "rvc"]);

export const LatencyPresetSchema = z.enum(["low_latency", "stable", "custom"]);

/** RVC quality/latency profile (Streamer / Low latency / Quality). */
export const RvcProfileSchema = z.enum(["streamer", "low_latency", "quality"]);

export const PitchExtractorSchema = z.enum(["fcpe", "rmvpe", "crepe", "dio"]);

export const ExecutionProviderSchema = z.enum(["cpu", "cuda", "dml"]);

export const VoiceChangerStateSchema = z.object({
    enabled: zSafe(z.boolean(), false),
    pitch: zSafe(z.number(), 0), // semitones, ±24
    formant: zSafe(z.number(), 0), // -1.0 to +1.0
    /** FFT window length (ms). Higher = usually better quality, more latency and CPU. */
    vocoderWindowMs: zSafe(z.number().min(30).max(60), 45),
    /** Phase vocoder oversampling: 1 = lowest CPU (more artifacts), 32 = heaviest, often cleaner. */
    vocoderOversample: zSafe(
        z.union([z.literal(1), z.literal(2), z.literal(4), z.literal(8), z.literal(16), z.literal(32)]),
        8
    ),
    /** Frames per processing chunk (per channel). Changing while running restarts the voice changer. */
    chunkFrames: zSafe(
        z.coerce.number().transform((n) => normalizeChunkFramesValue(n)),
        512,
    ),
    inputGain: zSafe(z.number().min(0).max(4), 1),
    outputGain: zSafe(z.number().min(0).max(4), 1),
    outputPrefillMs: zSafe(z.number(), 35),
    outputMaxQueueMs: zSafe(z.number(), 70),
    bufferTruncateEnabled: zSafe(z.boolean(), false),
    bufferTruncateRms: zSafe(z.number(), 0.0005),
    bufferTruncateChunks: zSafe(z.number().min(1).max(500), 32),
    inputDevice: zSafe(z.string(), ""),
    outputDevice: zSafe(z.string(), "default"),
    /** Second output for headphones (streamers). Empty = off; `"default"` = system default output device. */
    monitorOutputDevice: zSafe(z.string(), ""),
    monitorGain: zSafe(z.number().min(0).max(4), 1),
    /** `classic` = built-in phase vocoder; `sidecar` / `onnx` / `rvc` = converter/model first, then shared post-DSP. */
    mode: zSafe(VoiceChangerModeSchema, "classic"),
    /** WebSocket URL for sidecar mode, e.g. ws://127.0.0.1:8765 */
    sidecarWsUrl: zSafe(z.string(), ""),
    /** Path to ONNX model: float32 `[1, chunk_samples]` in/out (see scripts/neural-sidecar). */
    onnxModelPath: zSafe(z.string(), ""),
    /** Execution provider for ONNX chunk mode only (RVC has its own `rvcExecutionProvider`). */
    onnxExecutionProvider: zSafe(ExecutionProviderSchema, "cpu"),
    /** Buffer templates; `custom` when prefill/max ms edited manually. */
    latencyPreset: zSafe(LatencyPresetSchema, "low_latency"),
    /** Directory containing BYO RVC slot files (`gen.onnx`, optional `added.index`). */
    rvcSlotDir: zSafe(z.string(), ""),
    rvcHubertPath: zSafe(z.string(), ""),
    rvcFcpePath: zSafe(z.string(), ""),
    rvcRmvpePath: zSafe(z.string(), ""),
    /** Streamer / Low latency / Quality — maps to Rust `rvc_preset`. */
    rvcProfile: zSafe(RvcProfileSchema, "streamer"),
    /** Execution provider for RVC (HuBERT, pitch extractor, generator) — maps to Rust `rvc_execution_provider`. */
    rvcExecutionProvider: zSafe(ExecutionProviderSchema, "cpu"),
    pitchExtractor: zSafe(PitchExtractorSchema, "fcpe"),
    /** Read when the RVC harness is built (restart to apply). Streamer 64 / Low latency 96 / Quality 192. */
    extraContextMs: zSafe(z.number().min(0).max(2000), 64),
    /** Read when the RVC harness is built (restart to apply). */
    crossfadeMs: zSafe(z.number().min(0).max(200), 20),
    indexRate: zSafe(z.number().min(0).max(1), 0),
    protect: zSafe(z.number().min(0).max(0.5), 0.33),
    speakerId: zSafe(z.number().int().min(0), 0),
    gpuDeviceIndex: zSafe(z.number().int().min(0), 0),
    silenceSkip: zSafe(z.boolean(), true),
    vramCapMb: zSafe(z.number().int().min(256).max(8192), 768),
});

export type VoiceChangerState = z.infer<typeof VoiceChangerStateSchema>;
export type RvcProfile = z.infer<typeof RvcProfileSchema>;
export type PitchExtractor = z.infer<typeof PitchExtractorSchema>;
export type ExecutionProvider = z.infer<typeof ExecutionProviderSchema>;
