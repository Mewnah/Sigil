import { z } from "zod";
import { zSafe } from "@/utils";

export const VoiceChangerStateSchema = z.object({
    enabled: zSafe(z.boolean(), false),
    pitch: zSafe(z.number(), 0), // -12 to +12 semitones
    formant: zSafe(z.number(), 0), // -1.0 to +1.0
    preset: zSafe(z.string(), "default"),
    inputDevice: zSafe(z.string(), ""),
    outputDevice: zSafe(z.string(), "default"),
});

export type VoiceChangerState = z.infer<typeof VoiceChangerStateSchema>;
