import type { ISTTReceiver, SttStatusLevel, SttStatusOptions } from "./types";

/**
 * STT status channel: engine status (connecting, download / load progress, transient errors, rate-limit waits) is
 * kept apart from transcript text. `Service_STT` exposes it as `serviceState.statusText` / `progress` for the UI;
 * it is never passed to `publishText`, so it can never become a caption, a translation, a chatbox line or speech.
 *
 * Pure module (no `window`, no Tauri) so the routing rules are unit-tested.
 */

export interface SttStatus {
  /** One short line; `""` = no status. */
  text: string;
  /** 0-100, or `null` for no progress bar. */
  progress: number | null;
  level: SttStatusLevel;
  /** Clear the line after this many ms unless a newer status replaced it (absent = until replaced). */
  expireMs?: number;
}

/** How long a transient warning (skipped speech) stays visible after its last occurrence. */
export const TRANSIENT_WARNING_MS = 15_000;

export const EMPTY_STT_STATUS: Readonly<SttStatus> = Object.freeze({ text: "", progress: null, level: "info" });

/** Longest status line kept (server error bodies can be long). */
export const MAX_STATUS_LENGTH = 240;

/** Progress as 0-100, or `null` when unknown (negative, NaN, missing). */
export function clampProgress(progress: unknown): number | null {
  if (typeof progress !== "number" || !Number.isFinite(progress) || progress < 0) return null;
  return Math.min(100, progress);
}

/** Collapses whitespace and caps the length of a status line. */
export function normalizeStatusText(text: unknown): string {
  const t = String(text ?? "").replace(/\s+/g, " ").trim();
  return t.length <= MAX_STATUS_LENGTH ? t : `${t.slice(0, MAX_STATUS_LENGTH - 1)}…`;
}

export function normalizeSttStatus(text: unknown, options?: SttStatusOptions): SttStatus {
  const line = normalizeStatusText(text);
  if (!line) return { ...EMPTY_STT_STATUS };
  const status: SttStatus = {
    text: line,
    progress: clampProgress(options?.progress),
    level: options?.level === "warning" ? "warning" : "info",
  };
  const expireMs = options?.expireMs;
  if (typeof expireMs === "number" && Number.isFinite(expireMs) && expireMs > 0) status.expireMs = expireMs;
  return status;
}

/** Where a scoped receiver delivers each kind of engine report. */
export interface ScopedReceiverSinks {
  /** False once the session this receiver belongs to was stopped or replaced: every report is then dropped. */
  isCurrent(): boolean;
  start(): void;
  stop(error?: string): void;
  /** Transcript text only (published as an interim). */
  interim(text: string): void;
  /** Transcript text only (published as a final). */
  final(text: string): void;
  /** UI status only (never published). */
  status(status: SttStatus): void;
  /** The engine reported something (status or speech): re-arms the connecting watchdog. */
  alive(): void;
  muted(): boolean;
}

/**
 * Builds the receiver handed to one engine session. Reports from a stale session are dropped; status never reaches
 * the transcript sinks; every status or interim counts as a sign of life for the connecting watchdog.
 */
export function createScopedReceiver(sinks: ScopedReceiverSinks): ISTTReceiver {
  return {
    onStart: () => {
      if (sinks.isCurrent()) sinks.start();
    },
    onStop: (error) => {
      if (sinks.isCurrent()) sinks.stop(error);
    },
    onInterim: (value) => {
      if (!sinks.isCurrent()) return;
      sinks.alive();
      sinks.interim(typeof value === "string" ? value : String(value ?? ""));
    },
    onFinal: (value) => {
      if (!sinks.isCurrent()) return;
      sinks.final(typeof value === "string" ? value : String(value ?? ""));
    },
    onStatus: (text, options) => {
      if (!sinks.isCurrent()) return;
      sinks.alive();
      sinks.status(normalizeSttStatus(text, options));
    },
    // Muting applies to whatever session is live; a stale engine asking gets "muted" so it skips work.
    isMuted: () => !sinks.isCurrent() || sinks.muted(),
  };
}
