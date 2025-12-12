import { zSafe } from "@/utils";
import { z } from "zod";

export enum Transform_Backends {
    dummy = "dummy",
    openai = "openai",
}

export const Service_Transform_Schema = z.object({
    backend: zSafe(z.nativeEnum(Transform_Backends), Transform_Backends.openai),
    autoStart: zSafe(z.boolean(), true),
    stopWithStream: zSafe(z.boolean(), false),
    contextHistory: zSafe(z.boolean(), false),

    // AI Provider Selection
    provider: zSafe(z.enum(["openai", "openrouter", "custom"]), "openrouter"),

    // OpenAI Config
    openaiKey: zSafe(z.string(), ""),
    openaiUrl: zSafe(z.string(), "https://api.openai.com/v1"),
    openaiModel: zSafe(z.string(), "gpt-4o-mini"),

    // OpenRouter Config
    openRouterKey: zSafe(z.string(), ""),
    openRouterUrl: zSafe(z.string(), "https://openrouter.ai/api/v1"),
    openRouterModel: zSafe(z.string(), "google/gemini-1.5-flash"),

    // Custom Config
    customKey: zSafe(z.string(), ""),
    customUrl: zSafe(z.string(), ""),
    customModel: zSafe(z.string(), ""),

    // Common
    systemPrompt: zSafe(z.string(), "Rewrite the input text to be in an 'uwu' style. Replace words with cute alternatives (e.g. 'hello' -> 'hewwo') and add emoticons. Keep the original meaning."),
}).default({});
