/**
 * Pure helpers for diagnostics payloads: labels, system-log mapping, de-duplication and sizes.
 * No Tauri, DOM or valtio here (unit-tested under Node).
 */
import type { RustEpOutcomeEntry, RustLogEntry, RustPackProgress } from "./types";

export type LogLevel = "info" | "success" | "warning" | "error";

export interface SystemLogLine {
  tag: string;
  message: string;
  level: LogLevel;
}

/** Human name of an execution provider id. */
export function epLabel(ep: string | null | undefined): string {
  switch ((ep ?? "").trim().toLowerCase()) {
    case "cuda":
    case "gpu":
      return "CUDA";
    case "dml":
    case "directml":
      return "DirectML";
    case "cpu":
    case "":
      return "CPU";
    default:
      return ep ?? "CPU";
  }
}

/** i18n key suffix for a component (`diagnostics.component_<suffix>`), or null when unknown. */
export function componentKey(component: string): string | null {
  const known: Record<string, string> = {
    vc: "vc",
    "vc.content": "vc_content",
    "vc.pitch": "vc_pitch",
    "vc.onnx": "vc_onnx",
    "tts.kitten": "tts_kitten",
    "stt.vad": "stt_vad",
  };
  return known[component] ?? null;
}

/** English component name for system-log lines. */
export function componentLabel(component: string): string {
  const labels: Record<string, string> = {
    vc: "Voice changer model",
    "vc.content": "Voice changer content encoder",
    "vc.pitch": "Voice changer pitch model",
    "vc.onnx": "Voice changer ONNX model",
    "tts.kitten": "Local TTS",
    "stt.vad": "Speech detection",
  };
  return labels[component] ?? component;
}

export function outcomeFellBack(e: Pick<RustEpOutcomeEntry, "requested" | "active">): boolean {
  return e.requested !== e.active;
}

/**
 * System-log line for an EP outcome, or null when it is not worth a line (CPU-only components that
 * asked for the CPU, e.g. speech detection).
 */
export function outcomeToSystemLog(e: RustEpOutcomeEntry): SystemLogLine | null {
  const who = componentLabel(e.component);
  if (outcomeFellBack(e)) {
    const why = e.reason ? ` Reason: ${e.reason}` : "";
    return {
      tag: "GPU",
      level: "warning",
      message: `${who}: ${epLabel(e.requested)} is unavailable, running on ${epLabel(e.active)}.${why}`,
    };
  }
  if (e.active !== "cpu") {
    return { tag: "GPU", level: "success", message: `${who}: running on ${epLabel(e.active)}.` };
  }
  if (e.component === "vc" || e.component.startsWith("vc.")) {
    return { tag: "GPU", level: "info", message: `${who}: running on the CPU.` };
  }
  return null;
}

/** Drops repeats of the same outcome for a component within `windowMs` (one VC start = one line). */
export class OutcomeDeduper {
  private last = new Map<string, { key: string; at: number }>();

  constructor(private windowMs = 15_000) {}

  accept(e: RustEpOutcomeEntry, now = Date.now()): boolean {
    const key = `${e.requested}|${e.active}|${e.reason ?? ""}`;
    const prev = this.last.get(e.component);
    if (prev && prev.key === key && now - prev.at < this.windowMs) return false;
    this.last.set(e.component, { key, at: now });
    return true;
  }
}

/** Accepts each Rust log entry once (backfill and live events overlap); summary lines (seq 0) always pass. */
export class SeqGate {
  private max = 0;

  accept(seq: number): boolean {
    if (seq === 0) return true;
    if (seq <= this.max) return false;
    this.max = seq;
    return true;
  }
}

/** Short system-log tag for a Rust tracing target. */
export function tagForTarget(target: string): string {
  const t = target.toLowerCase();
  const rules: Array<[RegExp, string]> = [
    [/panic/, "Crash"],
    [/^ort(::|$)|::ort(::|$)|ort_session|diagnostics|nvidia/, "GPU"],
    [/voice_changer|neural_audio|sidecar/, "Voice changer"],
    [/whisper|vosk|moonshine|speech_core|::stt|services::audio/, "STT"],
    [/tts|kokoro|melo|chatterbox|fish_speech|uberduck/, "TTS"],
    [/translate/, "Translation"],
    [/services::web|peer|pubsub/, "Server"],
    [/osc/, "VRChat OSC"],
  ];
  for (const [re, tag] of rules) if (re.test(t)) return tag;
  return "Sigil";
}

/** System-log line for a forwarded Rust warning/error. */
export function logEntryToSystemLog(e: RustLogEntry): SystemLogLine {
  return {
    tag: tagForTarget(e.target),
    level: e.level === "error" ? "error" : "warning",
    message: e.message,
  };
}

/** `1.4 GB`, `549 MB`, `3.5 MB`, `812 KB` (decimal units, as download sizes are advertised). */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "?";
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e7) return `${Math.round(bytes / 1e6)} MB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${Math.round(bytes / 1e3)} KB`;
  return `${Math.round(bytes)} B`;
}

/** 0..1 progress of a pack install (downloads dominate; extraction counts as the last 5%). */
export function packProgressFraction(p: RustPackProgress | null | undefined): number {
  if (!p) return 0;
  const dl = p.total_bytes > 0 ? Math.min(1, Math.max(0, p.done_bytes / p.total_bytes)) : 0;
  switch (p.phase) {
    case "download":
      return dl * 0.95;
    case "extract":
      return 0.95 + (0.04 * Math.min(p.archive_index, p.archive_count)) / Math.max(1, p.archive_count);
    case "finalize":
      return 0.99;
    case "done":
      return 1;
    default:
      return dl * 0.95;
  }
}

/**
 * Token-bucket style limiter for lines written to the Rust file log from the webview: at most
 * `burst` per `windowMs`.
 */
export class RateLimiter {
  private windowStart = 0;
  private count = 0;

  constructor(
    private burst: number,
    private windowMs: number,
  ) {}

  allow(now = Date.now()): boolean {
    if (now - this.windowStart >= this.windowMs) {
      this.windowStart = now;
      this.count = 0;
    }
    if (this.count >= this.burst) return false;
    this.count++;
    return true;
  }
}
