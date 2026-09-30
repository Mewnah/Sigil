import { LLM_TRANSPORTS } from "@/core/services/llm/types";
import { zSafe } from "@/utils";
import { z } from "zod";

export enum Transform_Backends {
    dummy = "dummy",
    openai = "openai",
}

/** Persisted: add values, never rename or remove them. Requests go through the shared LLM client (`@/core/services/llm`). */
export const TRANSFORM_PROVIDERS = ["openai", "openrouter", "custom", "gemini"] as const;
export type Transform_Provider = (typeof TRANSFORM_PROVIDERS)[number];

/**
 * "default" sends nothing (the model's own behaviour, and the only safe value for models without
 * reasoning); the rest map to `reasoning_effort` (OpenAI, Gemini's OpenAI endpoint, llama.cpp, vLLM)
 * or `reasoning.effort` (OpenRouter). "none" turns thinking off where the model allows it.
 */
export const TRANSFORM_REASONING_EFFORTS = ["default", "none", "minimal", "low", "medium", "high"] as const;
export type Transform_ReasoningEffort = (typeof TRANSFORM_REASONING_EFFORTS)[number];

export const TRANSFORM_MAX_TOKENS_MIN = 16;
export const TRANSFORM_MAX_TOKENS_MAX = 32_768;
/** Room for a rewritten caption plus some thinking: reasoning tokens count against the limit. */
export const TRANSFORM_MAX_TOKENS_DEFAULT = 1024;

export const TRANSFORM_OPENROUTER_MODEL_DEFAULT = "google/gemini-2.5-flash-lite";
/**
 * Gemini's newest stable Flash-Lite (free tier; checked 2026-09-29 on ai.google.dev: Google limits the 2.5 models to
 * projects that already used them, so new keys need a 3.x model). The id is editable in the panel.
 */
export const TRANSFORM_GEMINI_MODEL_DEFAULT = "gemini-3.5-flash-lite";

/** Per-line deadline (the request is aborted, and the original text sent, when it expires). */
export const TRANSFORM_TIMEOUT_MS_DEFAULT = 15_000;
export const TRANSFORM_TIMEOUT_MS_MIN = 3_000;
export const TRANSFORM_TIMEOUT_MS_MAX = 120_000;
/**
 * Retired OpenRouter model ids that installs may still have stored: the state is saved with its
 * defaults, so everyone who launched before the default changed kept "google/gemini-1.5-flash".
 * Gemini 1.5 is shut down, so these are mapped to the current default (see `openRouterModelFor`).
 */
export const RETIRED_OPENROUTER_MODELS: readonly string[] = [
    "google/gemini-1.5-flash",
    "google/gemini-1.5-flash-8b",
    "google/gemini-flash-1.5",
    "google/gemini-flash-1.5-8b",
];

/** The model to request for a stored OpenRouter model id (retired ids become the current default). */
export function openRouterModelFor(stored: string): string {
    const model = stored.trim();
    return RETIRED_OPENROUTER_MODELS.includes(model.toLowerCase()) ? TRANSFORM_OPENROUTER_MODEL_DEFAULT : model;
}

export const Service_Transform_Schema = z.object({
    backend: zSafe(z.nativeEnum(Transform_Backends), Transform_Backends.openai),
    autoStart: zSafe(z.boolean(), true),
    stopWithStream: zSafe(z.boolean(), false),
    contextHistory: zSafe(z.boolean(), false),

    // AI Provider Selection
    provider: zSafe(z.enum(TRANSFORM_PROVIDERS), "openrouter"),

    // OpenAI Config
    openaiKey: zSafe(z.string(), ""),
    openaiUrl: zSafe(z.string(), "https://api.openai.com/v1"),
    openaiModel: zSafe(z.string(), "gpt-4o-mini"),

    // OpenRouter Config
    openRouterKey: zSafe(z.string(), ""),
    openRouterUrl: zSafe(z.string(), "https://openrouter.ai/api/v1"),
    openRouterModel: zSafe(z.string(), TRANSFORM_OPENROUTER_MODEL_DEFAULT),

    // Custom Config
    customKey: zSafe(z.string(), ""),
    customUrl: zSafe(z.string(), ""),
    customModel: zSafe(z.string(), ""),

    // Gemini (OpenAI-compatible endpoint; the key travels in the x-goog-api-key header)
    geminiKey: zSafe(z.string(), ""),
    geminiModel: zSafe(z.string(), TRANSFORM_GEMINI_MODEL_DEFAULT),

    /**
     * How requests reach the server: `auto` sends cloud requests from the webview and routes LAN servers through the
     * Rust proxy (and local servers that refuse the browser); `webview` / `rust` force one route.
     */
    transport: zSafe(z.enum(LLM_TRANSPORTS), "auto"),
    /**
     * Stream replies: captions on text.transform_source show the newest line's reply as it arrives, until the next
     * sentence starts. Finals are unchanged. Servers that refuse streaming get plain requests (see the LLM client).
     */
    stream: zSafe(z.boolean(), true),
    timeoutMs: zSafe(
        z.coerce.number().int().min(TRANSFORM_TIMEOUT_MS_MIN).max(TRANSFORM_TIMEOUT_MS_MAX),
        TRANSFORM_TIMEOUT_MS_DEFAULT,
    ),

    // Common
    systemPrompt: zSafe(z.string(), "Rewrite the input text to be in an 'uwu' style. Replace words with cute alternatives (e.g. 'hello' -> 'hewwo') and add emoticons. Keep the original meaning."),
    /** Upper bound for one reply (was a hard-coded 100, which truncated output and starved thinking models). */
    maxTokens: zSafe(
        z.coerce.number().int().min(TRANSFORM_MAX_TOKENS_MIN).max(TRANSFORM_MAX_TOKENS_MAX),
        TRANSFORM_MAX_TOKENS_DEFAULT,
    ),
    reasoningEffort: zSafe(z.enum(TRANSFORM_REASONING_EFFORTS), "default"),
    /**
     * Sends each line fenced in `<text>` tags with a system note that it is text to rewrite, not a
     * message to answer or instructions to follow (spoken questions, or injection through the LAN STT
     * path). On by default; responder-style prompts that should reply to the speaker turn it off.
     */
    fenceInput: zSafe(z.boolean(), true),
}).default({});

export type Transform_State = z.infer<typeof Service_Transform_Schema>;
