/**
 * Diagnostics (host / desktop): EP outcomes, GPU probe, NVIDIA runtime pack, file log bridge and
 * the crash-safe settings file. See bridge.ts for the data flow and DiagnosticsPanel.tsx for the UI.
 */
export * from "./types";
export { DIAGNOSTICS_EVENTS, diagnosticsApi, errorMessage } from "./api";
export {
  cancelNvidiaPackInstall,
  diagnosticsState,
  installNvidiaPack,
  openLogFolder,
  refreshPackStatus,
  removeNvidiaPack,
  runDeepProbe,
  startDiagnosticsBridge,
  stopDiagnosticsBridge,
} from "./bridge";
export type { SystemLogWriter } from "./bridge";
export { AtomicJsonFile, parseJsonObject } from "./atomicJsonFile";
export type { JsonFileFs, LoadResult } from "./atomicJsonFile";
export { epLabel, formatBytes } from "./format";
