/**
 * Diagnostics bridge: mirrors the Rust diagnostics plugin into a valtio store and the system log.
 *
 * - Rust warnings/errors (`diagnostics:log`) → system log (backfilled from `get_recent_log_entries`
 *   so nothing logged before the webview attached is lost, each entry exactly once).
 * - EP outcomes (`diagnostics:ep_outcome`) → `diagnosticsState.outcomes` and one system-log line
 *   per change ("Voice changer model: CUDA is unavailable, running on DirectML. Reason: …").
 * - Light probe (`diagnostics:probe`: at startup and again after the NVIDIA runtime pack is
 *   installed or removed), NVIDIA runtime pack status and install progress. A pack change drops
 *   the last GPU test (deep probe) because it no longer describes this PC.
 *
 * Started once by `initSystemLogListeners` (host + Tauri only). The system-log writer is injected
 * to avoid an import cycle with systemLog.ts.
 */
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { proxy } from "valtio";
import { DIAGNOSTICS_EVENTS, diagnosticsApi, errorMessage } from "./api";
import { OutcomeDeduper, SeqGate, epLabel, formatBytes, logEntryToSystemLog, outcomeToSystemLog, LogLevel } from "./format";
import type {
  EpOutcomeMap,
  RustEpOutcomeEntry,
  RustEpProbeReport,
  RustLogEntry,
  RustLogInfo,
  RustNvidiaPackStatus,
  RustPackProgress,
} from "./types";

export type SystemLogWriter = (tag: string, message: string, level: LogLevel, options?: { fileLog?: boolean }) => void;

export const diagnosticsState = proxy({
  /** Latest EP outcome per component (`vc`, `vc.content`, `vc.pitch`, …). */
  outcomes: {} as EpOutcomeMap,
  /** Startup (light) probe. */
  probe: null as RustEpProbeReport | null,
  /** Last deep probe (GPU test), persisted by Rust across restarts. */
  deepProbe: null as RustEpProbeReport | null,
  deepProbeRunning: false,
  deepProbeError: null as string | null,
  pack: null as RustNvidiaPackStatus | null,
  packProgress: null as RustPackProgress | null,
  packBusy: false,
  packError: null as string | null,
  /** Set after a successful install: the voice changer must be restarted to switch to CUDA. */
  packJustInstalled: false,
  logInfo: null as RustLogInfo | null,
});

let started: Promise<void> | null = null;
let writer: SystemLogWriter = () => {};
const unlisteners: UnlistenFn[] = [];

function logLine(tag: string, message: string, level: LogLevel) {
  // Lines that came from Rust are already in the file log.
  writer(tag, message, level, { fileLog: false });
}

const outcomeDeduper = new OutcomeDeduper();

function applyOutcome(e: RustEpOutcomeEntry) {
  diagnosticsState.outcomes[e.component] = e;
  if (!outcomeDeduper.accept(e)) return;
  const line = outcomeToSystemLog(e);
  if (line) logLine(line.tag, line.message, line.level);
}

function applyProgress(p: RustPackProgress) {
  diagnosticsState.packProgress = p;
  if (p.phase === "done" || p.phase === "error" || p.phase === "cancelled") {
    void refreshPackStatus();
  }
}

export async function refreshPackStatus(): Promise<void> {
  try {
    diagnosticsState.pack = await diagnosticsApi.getNvidiaPackStatus();
  } catch (e) {
    diagnosticsState.packError = errorMessage(e);
  }
}

function logProbeSummary(r: RustEpProbeReport) {
  const gpus = r.adapters.filter((a) => !a.software).map((a) => a.name);
  logLine(
    "GPU",
    `${gpus.length ? gpus.join(", ") : "No hardware GPU"}; best available: ${epLabel(r.recommended_ep)}.`,
    "info",
  );
  for (const note of r.notes) logLine("GPU", note, "info");
}

/**
 * Apply a light probe report. Rust re-emits it after the NVIDIA runtime pack is installed or
 * removed; the summary is logged once, later reports only when the recommendation changes.
 */
function applyProbe(r: RustEpProbeReport) {
  const previous = diagnosticsState.probe;
  diagnosticsState.probe = r;
  if (!previous) logProbeSummary(r);
  else if (previous.recommended_ep !== r.recommended_ep) {
    logLine("GPU", `Best available is now ${epLabel(r.recommended_ep)}.`, "info");
  }
}

/** The pack changed: a GPU test run before it no longer describes this PC. */
function dropStaleDeepProbe() {
  diagnosticsState.deepProbe = null;
  diagnosticsState.deepProbeError = null;
}

/** Start the bridge (idempotent). Resolves once listeners are attached and state is backfilled. */
export function startDiagnosticsBridge(write: SystemLogWriter): Promise<void> {
  writer = write;
  if (!started) started = start();
  return started;
}

async function start(): Promise<void> {
  const seq = new SeqGate();
  let backfilled = false;
  const early: RustLogEntry[] = [];
  const onLog = (e: RustLogEntry) => {
    if (!backfilled) {
      early.push(e);
      return;
    }
    if (seq.accept(e.seq)) {
      const l = logEntryToSystemLog(e);
      logLine(l.tag, l.message, l.level);
    }
  };

  unlisteners.push(
    await listen<RustLogEntry>(DIAGNOSTICS_EVENTS.log, (ev) => onLog(ev.payload)),
    await listen<RustEpOutcomeEntry>(DIAGNOSTICS_EVENTS.epOutcome, (ev) => applyOutcome(ev.payload)),
    await listen<RustEpProbeReport>(DIAGNOSTICS_EVENTS.probe, (ev) => applyProbe(ev.payload)),
    await listen<RustPackProgress>(DIAGNOSTICS_EVENTS.nvidiaPack, (ev) => applyProgress(ev.payload)),
  );

  // Backfill what happened before the webview attached, in order, each entry once.
  try {
    const recent = await diagnosticsApi.getRecentLogEntries();
    for (const e of [...recent, ...early].sort((a, b) => a.seq - b.seq)) {
      if (seq.accept(e.seq)) {
        const l = logEntryToSystemLog(e);
        logLine(l.tag, l.message, l.level);
      }
    }
  } catch {
    // Older backend or no file log: live events still arrive.
  } finally {
    backfilled = true;
    early.length = 0;
  }

  const results = await Promise.allSettled([
    diagnosticsApi.getEpOutcomes(),
    diagnosticsApi.getEpProbe(),
    diagnosticsApi.getNvidiaPackStatus(),
    diagnosticsApi.getLogInfo(),
    diagnosticsApi.getLastDeepEpProbe(),
  ]);
  const [outcomes, probe, pack, logInfo, deep] = results;
  if (outcomes.status === "fulfilled") outcomes.value.forEach(applyOutcome);
  if (probe.status === "fulfilled" && !diagnosticsState.probe) applyProbe(probe.value);
  if (pack.status === "fulfilled") diagnosticsState.pack = pack.value;
  if (logInfo.status === "fulfilled") diagnosticsState.logInfo = logInfo.value;
  if (deep.status === "fulfilled") diagnosticsState.deepProbe = deep.value;
}

/** Run the deep GPU test (headless child process); result lands in `diagnosticsState.deepProbe`. */
export async function runDeepProbe(): Promise<void> {
  if (diagnosticsState.deepProbeRunning) return;
  diagnosticsState.deepProbeRunning = true;
  diagnosticsState.deepProbeError = null;
  try {
    const r = await diagnosticsApi.runDeepEpProbe();
    diagnosticsState.deepProbe = r;
    const summary = r.ep_checks
      .map((c) => `${epLabel(c.ep)}${c.adapter ? ` (${c.adapter})` : ""}: ${c.ok ? "works" : "failed"}`)
      .join("; ");
    logLine("GPU", `GPU test: ${summary || "no GPU execution provider to test"}. Best: ${epLabel(r.recommended_ep)}.`, "info");
  } catch (e) {
    diagnosticsState.deepProbeError = errorMessage(e);
    logLine("GPU", `GPU test failed: ${diagnosticsState.deepProbeError}`, "error");
  } finally {
    diagnosticsState.deepProbeRunning = false;
  }
}

/** Download and install the NVIDIA runtime pack (explicit user action; ~1.4 GB). */
export async function installNvidiaPack(): Promise<void> {
  if (diagnosticsState.packBusy) return;
  diagnosticsState.packBusy = true;
  diagnosticsState.packError = null;
  diagnosticsState.packJustInstalled = false;
  diagnosticsState.packProgress = null;
  const size = diagnosticsState.pack ? formatBytes(diagnosticsState.pack.download_bytes) : "about 1.4 GB";
  logLine("GPU", `Downloading the NVIDIA GPU runtime pack (${size}).`, "info");
  try {
    diagnosticsState.pack = await diagnosticsApi.installNvidiaPack();
    diagnosticsState.packJustInstalled = true;
    dropStaleDeepProbe();
    logLine("GPU", "NVIDIA GPU runtime pack installed. Restart the voice changer to use CUDA.", "success");
  } catch (e) {
    const msg = errorMessage(e);
    if (msg !== "cancelled") {
      diagnosticsState.packError = msg;
      logLine("GPU", `NVIDIA GPU runtime pack install failed: ${msg}`, "error");
    } else {
      logLine("GPU", "NVIDIA GPU runtime pack download cancelled; it resumes where it stopped next time.", "info");
    }
    await refreshPackStatus();
  } finally {
    diagnosticsState.packBusy = false;
  }
}

export async function cancelNvidiaPackInstall(): Promise<void> {
  try {
    await diagnosticsApi.cancelNvidiaPackInstall();
  } catch (e) {
    diagnosticsState.packError = errorMessage(e);
  }
}

export async function removeNvidiaPack(): Promise<void> {
  if (diagnosticsState.packBusy) return;
  diagnosticsState.packBusy = true;
  diagnosticsState.packError = null;
  try {
    diagnosticsState.pack = await diagnosticsApi.removeNvidiaPack();
    diagnosticsState.packJustInstalled = false;
    dropStaleDeepProbe();
    logLine(
      "GPU",
      diagnosticsState.pack.pending_removal
        ? "The NVIDIA GPU runtime pack is in use and will be removed when Sigil restarts."
        : "NVIDIA GPU runtime pack removed.",
      "info",
    );
  } catch (e) {
    diagnosticsState.packError = errorMessage(e);
  } finally {
    diagnosticsState.packBusy = false;
  }
}

export async function openLogFolder(): Promise<void> {
  try {
    await diagnosticsApi.openLogDir();
  } catch (e) {
    logLine("Diagnostics", `Cannot open the log folder: ${errorMessage(e)}`, "warning");
  }
}

/** Detach listeners (tests / hot reload). */
export function stopDiagnosticsBridge() {
  while (unlisteners.length) unlisteners.pop()?.();
  started = null;
}
