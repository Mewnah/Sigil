/**
 * TypeScript mirrors of the Rust `diagnostics` plugin payloads (src-tauri/src/services/diagnostics).
 * Field names are the Rust serde names (snake_case). Keep in sync with the Rust structs.
 */
import type { RustEpOutcomeEntry } from "@/types";

export type { RustEpOutcome, RustEpOutcomeEntry } from "@/types";

/** Execution provider ids used across Sigil. */
export type EpId = "cpu" | "cuda" | "dml";

/** `diagnostics::logging::LogEntry` (event `diagnostics:log`, `get_recent_log_entries`). */
export interface RustLogEntry {
  /** Monotonic per process; 0 for synthetic summary lines. */
  seq: number;
  at_unix_ms: number;
  level: "warn" | "error";
  /** tracing target (module path), e.g. `sigil::services::voice_changer::worker`. */
  target: string;
  message: string;
}

/** `diagnostics::logging::LogInfo` (`get_log_info`). */
export interface RustLogInfo {
  dir: string;
  file: string;
  max_file_bytes: number;
  max_files: number;
  filter: string;
}

/** `gpu_mem::VramUsage`, MiB. */
export interface RustVramUsage {
  budget_mb: number;
  process_usage_mb: number;
  available_for_reservation_mb: number;
}

/** `gpu_mem::GpuAdapter`. */
export interface RustGpuAdapter {
  /** DXGI index = DirectML device id. */
  index: number;
  name: string;
  vendor: "nvidia" | "amd" | "intel" | "microsoft" | "other" | string;
  vendor_id: number;
  device_id: number;
  luid: string;
  dedicated_vram_mb: number;
  shared_memory_mb: number;
  software: boolean;
  driver_version: string | null;
  nvidia_driver: string | null;
  vram: RustVramUsage | null;
}

/** `ep_probe::DllCheck`. */
export interface RustDllCheck {
  name: string;
  path: string | null;
  source: "loaded" | "pack" | "exe_dir" | "path" | "system" | "missing" | string;
  size_bytes: number | null;
}

/** `ep_probe::EpCheck` (deep probe canaries). */
export interface RustEpCheck {
  ep: EpId | string;
  device_id: number;
  adapter: string | null;
  ok: boolean;
  millis: number | null;
  error: string | null;
}

/** `nvidia_pack::NvidiaPackStatus` (`get_nvidia_pack_status`). */
export interface RustNvidiaPackStatus {
  label: string;
  download_bytes: number;
  installed: boolean;
  dir: string | null;
  preloaded: boolean;
  preload_error: string | null;
  installing: boolean;
  pending_removal: boolean;
  last_error: string | null;
}

/** `nvidia_pack::PackProgress` (event `diagnostics:nvidia_pack`). */
export interface RustPackProgress {
  phase: "download" | "extract" | "finalize" | "done" | "error" | "cancelled";
  archive: string | null;
  archive_index: number;
  archive_count: number;
  done_bytes: number;
  total_bytes: number;
  message: string | null;
}

/** `ep_probe::EpProbeReport` (`get_ep_probe`, `run_deep_ep_probe`, event `diagnostics:probe`). */
export interface RustEpProbeReport {
  schema: number;
  deep: boolean;
  generated_at_unix_ms: number;
  app_version: string;
  os_version: string | null;
  exe_dir: string | null;
  ort_api_version: number;
  ort_compiled_eps: string[];
  provider_dlls: RustDllCheck[];
  directml_loaded_from: string | null;
  adapters: RustGpuAdapter[];
  nvidia_driver: string | null;
  cuda_driver_version: string | null;
  cuda_runtime_dlls: RustDllCheck[];
  cudart_version: string | null;
  cudnn_version: string | null;
  nvidia_pack: RustNvidiaPackStatus;
  /** Which NVIDIA runtime install CUDA uses (`model_store` = runtime/nvidia-cuda13, `interim` = Settings pack). */
  nvidia_runtime_source?: "model_store" | "interim" | null;
  /** CUDA provider library used by the deep probe: next to Sigil.exe or the registered app-data pack. */
  cuda_provider_source?: "exe_dir" | "registered" | null;
  cuda_provider_path?: string | null;
  /** Runtime-pack fingerprint at probe time (a changed pack makes a saved deep report stale). */
  runtime_fingerprint?: string;
  ep_checks: RustEpCheck[];
  recommended_ep: EpId | string;
  notes: string[];
}

/** Map of the latest outcome per component. */
export type EpOutcomeMap = Record<string, RustEpOutcomeEntry>;
