/**
 * Typed wrappers for `plugin:diagnostics|*` IPC and its events. Host (desktop) only: never import
 * from src/client. Every call rejects with the Rust error string on failure.
 */
import { invoke } from "@tauri-apps/api/core";
import type {
  RustEpOutcomeEntry,
  RustEpProbeReport,
  RustGpuAdapter,
  RustLogEntry,
  RustLogInfo,
  RustNvidiaPackStatus,
} from "./types";

export const DIAGNOSTICS_EVENTS = {
  epOutcome: "diagnostics:ep_outcome",
  log: "diagnostics:log",
  probe: "diagnostics:probe",
  nvidiaPack: "diagnostics:nvidia_pack",
} as const;

const cmd = (name: string) => `plugin:diagnostics|${name}`;

export const diagnosticsApi = {
  getEpOutcomes: () => invoke<RustEpOutcomeEntry[]>(cmd("get_ep_outcomes")),
  /** Startup (light) probe: no GPU runtime is loaded. */
  getEpProbe: () => invoke<RustEpProbeReport>(cmd("get_ep_probe")),
  /** Deep probe in a headless child process (CUDA / DirectML test runs); takes a few seconds. */
  runDeepEpProbe: () => invoke<RustEpProbeReport>(cmd("run_deep_ep_probe")),
  getLastDeepEpProbe: () => invoke<RustEpProbeReport | null>(cmd("get_last_deep_ep_probe")),
  getGpuAdapters: () => invoke<RustGpuAdapter[]>(cmd("get_gpu_adapters")),
  getLogInfo: () => invoke<RustLogInfo | null>(cmd("get_log_info")),
  getRecentLogEntries: () => invoke<RustLogEntry[]>(cmd("get_recent_log_entries")),
  logFromFrontend: (level: string, tag: string, message: string) =>
    invoke<void>(cmd("log_from_frontend"), { level, tag, message }),
  openLogDir: () => invoke<void>(cmd("open_log_dir")),
  getNvidiaPackStatus: () => invoke<RustNvidiaPackStatus>(cmd("get_nvidia_pack_status")),
  /** Downloads ~1.4 GB (one file at a time); progress arrives as `diagnostics:nvidia_pack`. */
  installNvidiaPack: () => invoke<RustNvidiaPackStatus>(cmd("install_nvidia_pack")),
  cancelNvidiaPackInstall: () => invoke<void>(cmd("cancel_nvidia_pack_install")),
  removeNvidiaPack: () => invoke<RustNvidiaPackStatus>(cmd("remove_nvidia_pack")),
};

/** Normalize an invoke rejection (string from Rust, or Error) into a message. */
export function errorMessage(e: unknown): string {
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message;
  try {
    return JSON.stringify(e);
  } catch {
    return String(e);
  }
}
