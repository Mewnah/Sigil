import { LLM_PROVIDERS, LLM_REASONING_EFFORTS, LLM_TRANSPORTS } from "@/core/services/llm/types";
import { zSafe } from "@/utils";
import { z } from "zod";

/**
 * Prompt families of the prompt builder (P4-C `prompt.ts`): `auto` picks by model (catalog `promptFormat` for the
 * bundled server, a name heuristic for remote models, else `generic`); `custom` uses `customPrompt`. Persisted:
 * add values, never rename or remove them.
 */
export const TRANSLATION_PROMPT_FORMATS = [
  "auto",
  "generic",
  "hy-mt2",
  "translategemma",
  "milmmt",
  "seed-x",
  "lmt",
  "sakura",
  "vntl",
  "madlad",
  "custom",
] as const;
export type TranslationPromptFormat = (typeof TRANSLATION_PROMPT_FORMATS)[number];

/** Runtime pack variants of the bundled llama-server (`auto` = the VRAM governor chooses). */
export const TRANSLATION_LLM_SERVER_VARIANTS = ["auto", "vulkan", "cuda13", "cpu"] as const;

/** Zero-setup default GGUF (Hy-MT2 1.8B Q4; `resources::policy::ids::MT_HY_MT2_1_8B`). */
export const TRANSLATION_LLM_DEFAULT_MODEL = "mt/hy-mt2-1.8b-q4_k_m";

/**
 * LLM translation (`llm` backend). Connection fields mirror `LlmConnectionSettings` (one key / model per provider,
 * so switching providers keeps them); `server*` configure the bundled llama-server (`RustLlmServerConfig`).
 * The language pair is the shared `sourceLanguage` / `targetLanguage`.
 */
export const Translation_Llm_Schema = z.object({
  provider: zSafe(z.enum(LLM_PROVIDERS), "bundled"),

  /** Bundled server: catalog id, `auto` (governor) or an absolute `.gguf` path. */
  serverModel: zSafe(z.coerce.string(), TRANSLATION_LLM_DEFAULT_MODEL),
  serverVariant: zSafe(z.enum(TRANSLATION_LLM_SERVER_VARIANTS), "auto"),
  serverContext: zSafe(z.coerce.number().int().min(512).max(131_072), 4096),
  /** `-ngl`; -1 = from the VRAM plan. */
  serverGpuLayers: zSafe(z.coerce.number().int().min(-1).max(999), -1),
  serverFlashAttn: zSafe(z.enum(["auto", "on", "off"]), "auto"),

  geminiKey: zSafe(z.coerce.string(), ""),
  geminiModel: zSafe(z.coerce.string(), "gemini-3.5-flash-lite"),
  openRouterKey: zSafe(z.coerce.string(), ""),
  openRouterModel: zSafe(z.coerce.string(), "google/gemini-2.5-flash-lite"),
  openaiKey: zSafe(z.coerce.string(), ""),
  openaiModel: zSafe(z.coerce.string(), "gpt-4o-mini"),
  customUrl: zSafe(z.coerce.string(), ""),
  customKey: zSafe(z.coerce.string(), ""),
  customModel: zSafe(z.coerce.string(), ""),
  transport: zSafe(z.enum(LLM_TRANSPORTS), "auto"),

  promptFormat: zSafe(z.enum(TRANSLATION_PROMPT_FORMATS), "auto"),
  /**
   * Template for `promptFormat: "custom"`. Placeholders: `{source_lang}`, `{target_lang}`, `{text}` (fenced as
   * data by the builder), `{context}`, `{glossary}`.
   */
  customPrompt: zSafe(z.coerce.string(), ""),
  reasoningEffort: zSafe(z.enum(LLM_REASONING_EFFORTS), "default"),
  maxTokens: zSafe(z.coerce.number().int().min(16).max(8192), 512),
  temperature: zSafe(z.coerce.number().min(0).max(2), 0.2),
  /** Deadline of one final line; interims use the router's shorter budget. */
  timeoutMs: zSafe(z.coerce.number().int().min(1000).max(120_000), 15_000),
  /** Stream replies (earlier first words for captions). */
  stream: zSafe(z.coerce.boolean(), true),
  interim: zSafe(z.coerce.boolean(), true),
}).default({});
