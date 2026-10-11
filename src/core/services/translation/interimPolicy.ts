import type { RustLlmServerStatus } from "@/types";
import { Translation_State, Translation_Backends } from "./schema";

/**
 * Whether live partial ("interim") translations run, for the selected backend (P7 default, `interim: "auto"`).
 *
 * Measured next to a game (docs/benchmarks/SPEECH_BENCHMARKS.md, App footprint): the bundled Hy-MT2 on the
 * processor with an interim translation every 300 ms takes 4.0 cores on average (6.7 at the peak); final lines
 * only take 1.5. On a GPU variant the same translator takes 0.2-0.4 cores. So `"auto"` means on, except while the
 * bundled llama-server translator runs on the processor. A boolean stored by the user always wins, and the switch
 * stays in the inspector, so interim translations can be turned back on.
 *
 * Pure: the pipeline gets the "runs on the processor" fact through `TranslationPipelineOptions.translatorOnCpu`
 * (the service reads `llmServerUi.status`), so this file imports no Tauri code.
 */

/** The fields of the server status the rule reads. */
export type LlmServerStatusLike = Pick<RustLlmServerStatus, "state" | "variant" | "gpuLayers">;

type LlmSettingsLike = Pick<Translation_State["llm"], "provider" | "serverVariant" | "serverGpuLayers">;

/**
 * The bundled llama-server translator runs on the processor: its provider is `bundled` and the server in use is
 * the `cpu` variant or offloads no layers (`auto` resolved to the processor). A server that is not up yet is
 * judged by the settings (variant `cpu` or 0 GPU layers); with variant `auto` and no server it is unknown, so
 * `false` (interims stay on until the governor's choice shows in the status).
 */
export function bundledTranslatorOnCpu(llm: LlmSettingsLike, status: LlmServerStatusLike | null | undefined): boolean {
  if (llm.provider !== "bundled") return false;
  const up = status != null && (status.state === "ready" || status.state === "starting" || status.state === "restarting");
  if (up && status.variant) return status.variant === "cpu" || status.gpuLayers === 0;
  return llm.serverVariant === "cpu" || llm.serverGpuLayers === 0;
}

/** Whether `backend` would hand text to the bundled llama-server: the `llm` backend, or a router chain that has it. */
function usesBundledTranslator(state: Translation_State, backend: Translation_Backends): boolean {
  if (backend === Translation_Backends.llm) return true;
  if (backend !== Translation_Backends.auto) return false;
  const llm = Translation_Backends.llm;
  return state.auto.chain.includes(llm) || state.auto.pairOverrides.some((rule) => rule.chain.includes(llm));
}

/** Interim translations on for the selected `backend` (false while no backend runs). */
export function interimTranslationsOn(state: Translation_State, backend: Translation_Backends | null, translatorOnCpu: boolean): boolean {
  if (backend === null) return false;
  const value: boolean | "auto" | undefined = state[backend]?.interim;
  if (value === "auto") return !(translatorOnCpu && usesBundledTranslator(state, backend));
  return value !== false;
}

/** The user has not chosen and the processor translator holds interims off (the inspector explains it). */
export function interimHeldOffByCpu(state: Translation_State, backend: Translation_Backends, translatorOnCpu: boolean): boolean {
  const value: boolean | "auto" | undefined = state[backend]?.interim;
  return value === "auto" && translatorOnCpu && usesBundledTranslator(state, backend);
}
