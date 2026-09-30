/**
 * Pure helpers for the llama-server panel (`LlmServer.tsx`): model grouping and option values, reason and state
 * i18n keys, memory formatting, the custom-file sentinel. No Tauri, DOM or valtio here (unit-tested under Node).
 * Owned by P4-B.
 */
import type { RustLlmServerPlan, RustLlmServerState } from "@/types";
import type { LlmModelInfoExt } from "./index";

/** Select value standing for "a GGUF file of my own" (the path is stored in `serverModel`). */
export const CUSTOM_MODEL_OPTION = "__custom_gguf__";

/** Plan reason codes with their own panel text (`transl.llm_server_reason_<code>`). */
export const PLAN_REASON_CODES = [
  "fits",
  "vc_running",
  "shared_gpu",
  "small_gpu",
  "gpu_full",
  "no_gpu",
  "integrated_gpu",
  "gpu_unknown",
  "no_gpu_runtime",
  "low_memory",
  "needs_runtime_update",
  "cpu_selected",
  "user_layers",
] as const;

/** i18n key of a plan reason (unknown codes fall back to the plan's English `reason`). */
export function planReasonKey(code: string | null | undefined): string | null {
  return code && (PLAN_REASON_CODES as readonly string[]).includes(code) ? `transl.llm_server_reason_${code}` : null;
}

export function stateKey(state: RustLlmServerState | null | undefined): string {
  return `transl.llm_server_state_${state ?? "stopped"}`;
}

/** The value is a file path rather than a catalog id or `auto` (mirrors Rust `catalog::looks_like_path`). */
export function isModelPath(value: string | null | undefined): boolean {
  const v = (value ?? "").trim();
  return v.includes("\\") || v.startsWith("//") || v.charAt(1) === ":" || v.toLowerCase().endsWith(".gguf");
}

/** File name of a path (either separator). */
export function fileName(path: string): string {
  const parts = path.trim().split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

/** The select value for a stored `serverModel`. */
export function modelSelectValue(serverModel: string | null | undefined): string {
  const v = (serverModel ?? "").trim();
  if (!v) return "auto";
  return isModelPath(v) ? CUSTOM_MODEL_OPTION : v;
}

export interface ModelGroups {
  /** Translation models offered to everyone (the automatic choice picks among these). */
  general: LlmModelInfoExt[];
  /** Opt-in specialists (a language pair or a style). */
  specialists: LlmModelInfoExt[];
}

export function groupModels(models: readonly LlmModelInfoExt[]): ModelGroups {
  const general: LlmModelInfoExt[] = [];
  const specialists: LlmModelInfoExt[] = [];
  for (const m of models) (m.specialist || m.role === "specialist" ? specialists : general).push(m);
  return { general, specialists };
}

/** `5300` MiB → `5.2 GB` (binary, like the Rust reasons). */
export function formatMb(mb: number | null | undefined): string {
  if (mb == null || !Number.isFinite(mb) || mb < 0) return "?";
  if (mb < 1024) return `${Math.round(mb)} MB`;
  return `${(mb / 1024).toFixed(1)} GB`;
}

/** Whether a refused plan may be started anyway (not when the runtime cannot run the model at all). */
export function canOverride(plan: RustLlmServerPlan | null | undefined): boolean {
  return !!plan && !plan.allowed && plan.reasonCode !== "needs_runtime_update" && plan.reasonCode !== "no_gpu_runtime";
}

/** Whether the panel offers the plan's missing downloads: only for a plan that can run (allowed, or refused but
 * overridable); a hard refusal never uses the model or pack it lists. */
export function offersDownloads(plan: RustLlmServerPlan | null | undefined): boolean {
  return !!plan && (plan.missing?.length ?? 0) > 0 && (plan.allowed || canOverride(plan));
}

/** `-ngl` as shown: all layers (999), none, or a count. */
export function layersText(layers: number | null | undefined): "all" | "none" | number {
  if (layers == null || layers <= 0) return "none";
  return layers >= 999 ? "all" : layers;
}

/** Parse the GPU-layers field: empty / "auto" / negative → -1 (from the plan), else 0..999. */
export function parseGpuLayers(text: string): number {
  const t = text.trim().toLowerCase();
  if (!t || t === "auto") return -1;
  const n = Number(t);
  if (!Number.isFinite(n) || n < 0) return -1;
  return Math.min(999, Math.round(n));
}

/** Parse the context field: 512..131072, rounded; invalid → null (keep the old value). */
export function parseContext(text: string): number | null {
  const n = Number(text.trim());
  if (!Number.isFinite(n) || n <= 0) return null;
  return Math.min(131_072, Math.max(512, Math.round(n)));
}
