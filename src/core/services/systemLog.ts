import { ServiceNetworkState } from "@/types";
import { isTauri } from "@tauri-apps/api/core";
import { proxy } from "valtio";
import { subscribeKey } from "valtio/utils";
import { diagnosticsApi } from "./diagnostics/api";
import { startDiagnosticsBridge } from "./diagnostics/bridge";
import { RateLimiter } from "./diagnostics/format";
import { SttMuteState } from "./stt/types";

export type SystemLogLevel = "info" | "success" | "warning" | "error";

export type SystemLogEntry = {
  id: number;
  at: number;
  tag: string;
  message: string;
  level: SystemLogLevel;
};

const MAX_ENTRIES = 300;

export const systemLogState = proxy({
  entries: [] as SystemLogEntry[],
  nextId: 1,
});

/** At most 30 system-log lines per 10 s are copied into the Rust file log (a stuck error loop must not flood it). */
const fileLogLimiter = new RateLimiter(30, 10_000);
let fileLogAvailable: boolean | null = null;

/** Copy a system-log line into Sigil's file log (desktop only, best effort, never throws). */
function writeToFileLog(tag: string, message: string, level: SystemLogLevel) {
  if (fileLogAvailable === null) fileLogAvailable = isTauri();
  if (!fileLogAvailable || !fileLogLimiter.allow()) return;
  const rustLevel = level === "error" ? "error" : level === "warning" ? "warning" : "info";
  diagnosticsApi.logFromFrontend(rustLevel, tag, message).catch(() => {
    // Older backend without the command, or the IPC is gone during shutdown.
  });
}

/**
 * Add a line to the in-app system log. Lines are also written to the file log
 * (`%LOCALAPPDATA%\com.sigil.app\logs\sigil.log`) unless `options.fileLog` is false (used for
 * lines that came from Rust and are already there).
 */
export function pushSystemLog(
  tag: string,
  message: string,
  level: SystemLogLevel = "info",
  options?: { fileLog?: boolean },
) {
  const id = systemLogState.nextId++;
  systemLogState.entries.push({ id, at: Date.now(), tag, message, level });
  if (systemLogState.entries.length > MAX_ENTRIES) {
    systemLogState.entries.splice(0, systemLogState.entries.length - MAX_ENTRIES);
  }
  if (options?.fileLog !== false) writeToFileLog(tag, message, level);
}

function humanStatus(s: ServiceNetworkState): string {
  switch (s) {
    case ServiceNetworkState.connected:
      return "Connected";
    case ServiceNetworkState.connecting:
      return "Connecting…";
    case ServiceNetworkState.disconnected:
      return "Disconnected";
    case ServiceNetworkState.error:
      return "Error";
    default:
      return String(s);
  }
}

function levelForStatus(s: ServiceNetworkState): SystemLogLevel {
  if (s === ServiceNetworkState.error) return "error";
  if (s === ServiceNetworkState.connected) return "success";
  if (s === ServiceNetworkState.connecting) return "warning";
  return "info";
}

function wireNetworkService(tag: string, state: { status: ServiceNetworkState; error?: string }) {
  const emit = () => {
    const { status, error = "" } = state;
    let msg = humanStatus(status);
    if (error) msg += ` — ${error}`;
    pushSystemLog(tag, msg, levelForStatus(status));
  };
  subscribeKey(state, "status", emit);
  subscribeKey(state, "error", emit);
  emit();
}

/** Subscribe to pipeline services after ApiServer.init() has constructed them. */
export function initSystemLogListeners() {
  const api = window.ApiServer;
  if (!api) return;

  // Rust warnings/errors, execution-provider outcomes (CUDA / DirectML / CPU in use) and the GPU
  // probe show up here too (desktop only).
  if (isTauri()) {
    startDiagnosticsBridge(pushSystemLog).catch((e) => {
      pushSystemLog("Diagnostics", `Diagnostics bridge unavailable: ${e instanceof Error ? e.message : String(e)}`, "warning");
    });
  }

  pushSystemLog(
    "System",
    "Monitoring STT, TTS, Translation, Rewrite, OBS WebSocket, and voice changer state.",
    "info"
  );

  wireNetworkService("STT", api.stt.serviceState);
  wireNetworkService("TTS", api.tts.serviceState);
  wireNetworkService("Translation", api.translation.serviceState);
  wireNetworkService("Rewrite", api.transform.serviceState);

  const obsWs = api.obs.wsState;
  const obsEmit = () => {
    pushSystemLog("OBS", humanStatus(obsWs.status), levelForStatus(obsWs.status));
  };
  subscribeKey(obsWs, "status", obsEmit);
  obsEmit();

  subscribeKey(api.voiceChanger.state, "isRunning", () => {
    const on = api.voiceChanger.state.isRunning;
    pushSystemLog("Voice changer", on ? "Running" : "Stopped", on ? "success" : "info");
  });
  pushSystemLog(
    "Voice changer",
    api.voiceChanger.state.isRunning ? "Running" : "Stopped",
    api.voiceChanger.state.isRunning ? "success" : "info"
  );

  subscribeKey(api.stt.serviceState, "muted", () => {
    const m = api.stt.serviceState.muted;
    if (m === SttMuteState.muted) pushSystemLog("STT", "Microphone muted (speech-to-text silenced)", "warning");
    else if (m === SttMuteState.unmuted) pushSystemLog("STT", "Microphone unmuted", "info");
    else pushSystemLog("STT", "Unmute pending (finishing current interim result)", "info");
  });
}
