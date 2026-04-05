import { z } from "zod";

import { zSafe } from "@/utils";

import { normalizeChunkFramesValue } from "./chunkFrames";



export const VoiceChangerModeSchema = z.enum(["classic", "onnx", "sidecar"]);



export const LatencyPresetSchema = z.enum(["low_latency", "stable", "custom"]);



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

    preset: zSafe(z.string(), "default"),

    inputDevice: zSafe(z.string(), ""),

    outputDevice: zSafe(z.string(), "default"),

    /** Second output for headphones (streamers). Empty = off; `"default"` = system default output device. */

    monitorOutputDevice: zSafe(z.string(), ""),

    monitorGain: zSafe(z.number().min(0).max(4), 1),

    /** `classic` = built-in phase vocoder path; `sidecar` / `onnx` = remote or model first, then shared post-DSP (pitch, formant, vocoder). */

    mode: zSafe(VoiceChangerModeSchema, "classic"),

    /** WebSocket URL for sidecar mode, e.g. ws://127.0.0.1:8765 */

    sidecarWsUrl: zSafe(z.string(), ""),

    /** Path to ONNX model: float32 `[1, chunk_samples]` in/out (see scripts/neural-sidecar). */

    onnxModelPath: zSafe(z.string(), ""),

    onnxExecutionProvider: zSafe(z.enum(["cpu", "cuda", "dml"]), "cpu"),

    /** Buffer templates; `custom` when prefill/max ms edited manually. */

    latencyPreset: zSafe(LatencyPresetSchema, "low_latency"),

});



export type VoiceChangerState = z.infer<typeof VoiceChangerStateSchema>;


