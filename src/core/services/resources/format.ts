/**
 * Pure helpers for the Rust `resources` payloads: typed views of the fields P1-D2 added, model
 * labels, claim summaries, budget fractions and the key that decides when recommendations must be
 * fetched again. No Tauri, DOM or valtio here (unit-tested under Node).
 */
import type {
  RustRecommendation,
  RustRecommendationTopic,
  RustResourceSnapshot,
  RustResourceSubsystem,
  RustVramBudget,
} from "@/types";

/** Rust `resources::VramBudget` including the P1-D2 fields. */
export interface ResVramBudget extends RustVramBudget {
  /** Part of `claimed_mb` held by helper processes (llama-server, TTS servers). */
  external_claimed_mb?: number;
  /** Device-wide use by every process (NVML; NVIDIA only). */
  device_used_mb?: number | null;
  device_free_mb?: number | null;
}

export type ResClaim = RustResourceSnapshot["claims"][number] & {
  /** Held by a helper process. */
  external?: boolean;
};

export interface ResSnapshot extends RustResourceSnapshot {
  budget: ResVramBudget;
  claims: ResClaim[];
  /**
   * Whether the running voice changer's models use a GPU EP (its own report, else inferred by Rust
   * from its EP outcomes); null/absent while it is not running or not known yet.
   */
  vc_gpu?: boolean | null;
}

/** Rust `resources::policy::Reason` codes. */
export type RecommendationReason =
  | "fits"
  | "vc_running"
  | "shared_gpu"
  | "small_gpu"
  | "no_gpu"
  | "integrated_gpu"
  | "gpu_full"
  | "low_memory"
  | "no_gpu_runtime"
  | "gpu_unknown";

/** Rust `resources::Recommendation` including the P1-D2 fields. */
export interface ResRecommendation extends RustRecommendation {
  reason_code?: RecommendationReason | string;
  device?: "gpu" | "cpu" | string;
  /** Estimated VRAM the choice needs (0 on the CPU). */
  vram_mb?: number;
  /** VRAM the policy found for this topic. */
  available_mb?: number;
}

export const RECOMMENDATION_TOPICS: readonly RustRecommendationTopic[] = ["mt_llm", "qwen3_tts", "stt_final"];

export const SUBSYSTEMS: readonly RustResourceSubsystem[] = ["vc", "stt_final", "llm", "tts", "other"];

/** Display names of the catalog ids the policy returns (proper nouns; never translated). */
const MODEL_LABELS: Record<string, string> = {
  "mt/milmmt-46-12b-q4_k_m": "MiLMMT-46 12B",
  "mt/hy-mt2-30b-a3b-q4_k_m": "Hy-MT2 30B-A3B",
  "mt/hy-mt2-7b-q4_k_m": "Hy-MT2 7B",
  "mt/milmmt-46-4b-q4_k_m": "MiLMMT-46 4B",
  "mt/hy-mt2-1.8b-q4_k_m": "Hy-MT2 1.8B",
  "mt/hy-mt2-1.8b-1.25bit": "Hy-MT2 1.8B (1.25-bit)",
  "tts/qwen3-tts-1.7b": "Qwen3-TTS 1.7B",
  "tts/qwen3-tts-0.6b": "Qwen3-TTS 0.6B",
  "stt/qwen3-asr-1.7b-bf16": "Qwen3-ASR 1.7B (BF16)",
  "stt/qwen3-asr-1.7b-q8_0": "Qwen3-ASR 1.7B (Q8_0)",
  "stt/cohere-transcribe-q4_k": "Cohere Transcribe (Q4_K)",
  "stt/qwen3-asr-0.6b-int8": "Qwen3-ASR 0.6B (int8)",
};

/** Display name of a catalog id (the id itself when unknown). */
export function modelLabel(id: string): string {
  return MODEL_LABELS[id] ?? id;
}

/** Live claims summed per subsystem, in `SUBSYSTEMS` order, omitting subsystems without claims. */
export function claimsBySubsystem(snapshot: ResSnapshot | null | undefined): { subsystem: RustResourceSubsystem; mb: number; count: number }[] {
  if (!snapshot) return [];
  return SUBSYSTEMS.map((subsystem) => {
    const mine = snapshot.claims.filter((c) => c.subsystem === subsystem);
    return { subsystem, mb: mine.reduce((sum, c) => sum + c.mb, 0), count: mine.length };
  }).filter((row) => row.count > 0);
}

const clamp01 = (x: number) => (Number.isFinite(x) ? Math.min(1, Math.max(0, x)) : 0);

/**
 * What Sigil holds against the OS budget (this process's usage or in-process claims, whichever is
 * larger, plus helper-process claims) and the fraction of the budget that is.
 */
export function sigilUsage(budget: ResVramBudget | null | undefined): { usedMb: number; budgetMb: number; fraction: number } {
  if (!budget || budget.budget_mb <= 0) return { usedMb: 0, budgetMb: 0, fraction: 0 };
  const usedMb = Math.max(0, budget.budget_mb - budget.free_for_claims_mb);
  return { usedMb, budgetMb: budget.budget_mb, fraction: clamp01(usedMb / budget.budget_mb) };
}

/** Device-wide use by every app (NVML), or null when unknown. */
export function deviceUsage(budget: ResVramBudget | null | undefined): { usedMb: number; totalMb: number; fraction: number } | null {
  if (!budget || budget.device_used_mb == null || budget.device_free_mb == null) return null;
  const totalMb = budget.device_used_mb + budget.device_free_mb;
  if (totalMb <= 0) return null;
  return { usedMb: budget.device_used_mb, totalMb, fraction: clamp01(budget.device_used_mb / totalMb) };
}

/**
 * The inputs of the Rust policy that can change at run time (VC flag and device, claim subsystems
 * and sizes). Recommendations are fetched again only when this key changes; live budget numbers
 * never change the answer (the policy ignores them on purpose).
 */
export function policyKey(snapshot: ResSnapshot | null | undefined): string {
  if (!snapshot) return "";
  const claims = snapshot.claims
    .map((c) => `${c.subsystem}:${c.mb}${c.external ? "e" : ""}`)
    .sort()
    .join(",");
  const vcGpu = snapshot.vc_gpu == null ? "?" : snapshot.vc_gpu ? "g" : "c";
  return `${snapshot.vc_running ? 1 : 0}${vcGpu}|${snapshot.budget.adapter_index ?? "-"}|${claims}`;
}

/**
 * EP-outcome components of voice-changer model sessions ("vc", "vc.content", "vc.pitch",
 * "vc.onnx"): Rust infers from them whether the voice changer runs on the GPU (`vc_gpu`).
 */
export function isVcComponent(component: string | null | undefined): boolean {
  return component === "vc" || (!!component && component.startsWith("vc."));
}
