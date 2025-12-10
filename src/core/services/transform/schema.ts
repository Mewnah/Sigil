import { zSafe } from "@/utils";
import { z } from "zod";

export enum Transform_Backends {
    dummy = "dummy",
    openai = "openai",
}

export const Service_Transform_Schema = z.object({
    backend: zSafe(z.nativeEnum(Transform_Backends), Transform_Backends.dummy),
    autoStart: zSafe(z.boolean(), true),

    // AI Provider Selection
    provider: zSafe(z.enum(["openai", "openrouter", "custom"]), "openai"),

    // OpenAI Config
    openaiKey: zSafe(z.string(), ""),
    openaiUrl: zSafe(z.string(), "https://api.openai.com/v1"),
    openaiModel: zSafe(z.string(), "gpt-3.5-turbo"),

    // OpenRouter Config
    openRouterKey: zSafe(z.string(), ""),
    openRouterUrl: zSafe(z.string(), "https://openrouter.ai/api/v1"),
    openRouterModel: zSafe(z.string(), "mistralai/mistral-7b-instruct:free"),

    // Custom Config
    customKey: zSafe(z.string(), ""),
    customUrl: zSafe(z.string(), ""),
    customModel: zSafe(z.string(), ""),

    // Common
    systemPrompt: zSafe(z.string(), "Rewrite the STT text for clarity and tone. Keep meaning the same. No new content. Reduce harshness if present. Fix grammar lightly. Preserve natural voice."),
}).default({});
