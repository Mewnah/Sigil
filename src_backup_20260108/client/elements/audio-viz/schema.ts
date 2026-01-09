import { zSafe } from "@/utils";
import z from "zod";

export const Element_AudioVizStateSchemaN = z.object({
    barColor: zSafe(z.string(), "#ef4444"), // Default red
    barCount: zSafe(z.number(), 64),
    gap: zSafe(z.number(), 2),
    sensitivity: zSafe(z.number(), 1.5),
    radius: zSafe(z.number(), 4),
    mirror: zSafe(z.boolean(), false),
}).default({});


export type Element_AudioVizState = z.infer<typeof Element_AudioVizStateSchemaN>;
