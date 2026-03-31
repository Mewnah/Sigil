import { z } from "zod";
import { zSafe } from "@/utils";

export const VoiceChangerStateSchema = z.object({
    enabled: zSafe(z.boolean(), false),
    pitch: zSafe(z.number(), 0), // -12 to +12 semitones
    formant: zSafe(z.number(), 0), // -1.0 to +1.0
    /** FFT window length (ms). Higher = usually better quality, more latency and CPU. */
    vocoderWindowMs: zSafe(z.number().min(30).max(60), 45),
    /** Phase vocoder oversampling: 4 = fastest, 32 = heaviest CPU, often cleaner. */
    vocoderOversample: zSafe(z.union([z.literal(4), z.literal(8), z.literal(16), z.literal(32)]), 8),
    preset: zSafe(z.string(), "default"),
    inputDevice: zSafe(z.string(), ""),
    outputDevice: zSafe(z.string(), "default"),
});

export type VoiceChangerState = z.infer<typeof VoiceChangerStateSchema>;
