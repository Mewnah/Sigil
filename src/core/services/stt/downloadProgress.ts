/**
 * Download progress formatting shared by every downloader in Sigil (pure: no Tauri, DOM, valtio or
 * i18n, so it is unit-tested under Node). It accepts both progress payload shapes:
 *
 * - the legacy STT events `whisper:download_progress` / `vosk:download_progress`
 *   (`{ file, progress }`, a percentage, negative while the size is unknown), and
 * - the model store's `models:progress` (`RustModelProgress` in src/types.ts: bytes done / total,
 *   smoothed speed, serving mirror, attempt).
 *
 * `normalizeDownloadProgress` turns either into one view; the UI builds translated lines from the
 * view, and `formatDownloadProgress` gives an English line for logs.
 */
import { formatBytes } from "@/core/services/diagnostics/format";

/**
 * Payload of `whisper:download_progress` and `vosk:download_progress` (Rust, throttled to a few per second):
 * `progress` is a percentage 0-100, or negative when the total size is unknown (no Content-Length).
 */
export type SttDownloadProgressPayload = {
  file: string;
  progress: number;
};

/** Byte-based progress (the model store's `models:progress`, `RustModelProgress`). */
export type ByteDownloadProgressPayload = {
  file: string;
  bytes_done: number;
  /** null / absent when the size is unknown. */
  bytes_total?: number | null;
  bytes_per_sec?: number | null;
  /** Mirror family serving the bytes (`huggingface`, `hf-mirror`, ...). */
  mirror?: string | null;
  /** 1-based attempt across mirrors and resumes. */
  attempt?: number | null;
};

export type DownloadProgressPayload = SttDownloadProgressPayload | ByteDownloadProgressPayload;

/** One normalized view of a progress payload. Unknown values are null. */
export interface DownloadProgressView {
  /** File name (last path component; "model" when the payload has none). */
  file: string;
  /** Whole percent 0-100; never 100 while bytes are still missing. */
  percent: number | null;
  /** Exact 0..1 progress (for progress bars). */
  fraction: number | null;
  bytesDone: number | null;
  bytesTotal: number | null;
  /** Smoothed speed; null when unknown or not positive. */
  bytesPerSec: number | null;
  /** Seconds left at the current speed; null when size or speed is unknown. */
  etaSeconds: number | null;
  mirror: string | null;
  attempt: number | null;
}

/** ETAs longer than this are not worth showing ("stalled" is more honest than "31 h left"). */
const MAX_ETA_SECONDS = 24 * 3600;

const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** Last path component of a catalog path or URL-ish name (`a/b/model.onnx` → `model.onnx`). */
export function displayFileName(file: string | null | undefined): string {
  const trimmed = (file ?? "").trim().replace(/[\\/]+$/, "");
  const name = trimmed.split(/[\\/]/).pop() ?? "";
  return name || "model";
}

/** Whole percent of `done / total`, floored so a download never shows 100% before it ends. */
export function percentOf(done: number, total: number): number | null {
  if (!finite(done) || !finite(total) || total <= 0) return null;
  if (done >= total) return 100;
  return Math.max(0, Math.min(99, Math.floor((done / total) * 100)));
}

/** Seconds left for `remaining` bytes at `bytesPerSec`, or null when not meaningful. */
export function etaSeconds(remaining: number, bytesPerSec: number | null | undefined): number | null {
  if (!finite(remaining) || remaining < 0) return null;
  if (remaining === 0) return 0;
  if (!finite(bytesPerSec) || bytesPerSec <= 0) return null;
  const s = Math.ceil(remaining / bytesPerSec);
  return s > MAX_ETA_SECONDS ? null : s;
}

function isBytePayload(p: object): p is ByteDownloadProgressPayload {
  return "bytes_done" in p;
}

/** Normalize either payload shape (see the module comment). Tolerates missing or junk fields. */
export function normalizeDownloadProgress(
  payload: Partial<SttDownloadProgressPayload> | Partial<ByteDownloadProgressPayload> | null | undefined,
): DownloadProgressView {
  const view: DownloadProgressView = {
    file: displayFileName(payload?.file),
    percent: null,
    fraction: null,
    bytesDone: null,
    bytesTotal: null,
    bytesPerSec: null,
    etaSeconds: null,
    mirror: null,
    attempt: null,
  };
  if (!payload) return view;
  if (isBytePayload(payload)) {
    const done = finite(payload.bytes_done) && payload.bytes_done >= 0 ? payload.bytes_done : null;
    const total = finite(payload.bytes_total) && payload.bytes_total > 0 ? payload.bytes_total : null;
    const speed = finite(payload.bytes_per_sec) && payload.bytes_per_sec > 0 ? payload.bytes_per_sec : null;
    view.bytesDone = done;
    view.bytesTotal = total;
    view.bytesPerSec = speed;
    if (done !== null && total !== null) {
      view.fraction = Math.min(1, done / total);
      view.percent = percentOf(done, total);
      view.etaSeconds = etaSeconds(Math.max(0, total - done), speed);
    }
    view.mirror = typeof payload.mirror === "string" && payload.mirror ? payload.mirror : null;
    view.attempt = finite(payload.attempt) && payload.attempt >= 1 ? Math.floor(payload.attempt) : null;
    return view;
  }
  const progress = (payload as Partial<SttDownloadProgressPayload>).progress;
  if (finite(progress) && progress >= 0) {
    const pct = Math.min(100, progress);
    view.percent = Math.round(pct);
    view.fraction = pct / 100;
  }
  return view;
}

/**
 * File name and percentage (0-100) of a download progress payload; `percent` is null when unknown.
 * Kept for the STT status lines (`statusMessages.ts`); accepts both payload shapes.
 */
export function parseDownloadProgress(
  payload: Partial<SttDownloadProgressPayload> | Partial<ByteDownloadProgressPayload> | null | undefined,
): { file: string; percent: number | null } {
  if (payload && !isBytePayload(payload)) {
    // Legacy events: the full `file` label is shown (it may be a model name, not a path).
    const file = payload.file?.trim() || "model";
    const { percent } = normalizeDownloadProgress(payload);
    return { file, percent };
  }
  const { file, percent } = normalizeDownloadProgress(payload);
  return { file, percent };
}

/** `3.2 MB/s` (decimal units, like download sizes); null when the speed is unknown. */
export function formatSpeed(bytesPerSec: number | null | undefined): string | null {
  if (!finite(bytesPerSec) || bytesPerSec <= 0) return null;
  return `${formatBytes(bytesPerSec)}/s`;
}

/** `0:42`, `12:05`, `1:02:03` (language-neutral clock format). */
export function formatEta(seconds: number | null | undefined): string | null {
  if (!finite(seconds) || seconds < 0) return null;
  const s = Math.ceil(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

/** `68 MB / 150 MB`, or `68 MB` when the total is unknown; null before any byte count. */
export function formatByteProgress(view: Pick<DownloadProgressView, "bytesDone" | "bytesTotal">): string | null {
  if (view.bytesDone === null) return null;
  return view.bytesTotal !== null
    ? `${formatBytes(view.bytesDone)} / ${formatBytes(view.bytesTotal)}`
    : formatBytes(view.bytesDone);
}

/** English status line for a model download / local-model copy (logs; the UI uses the translated status). */
export function formatDownloadProgress(
  payload: Partial<SttDownloadProgressPayload> | Partial<ByteDownloadProgressPayload> | null | undefined,
): string {
  if (payload && isBytePayload(payload)) {
    const v = normalizeDownloadProgress(payload);
    const details = [formatByteProgress(v), formatSpeed(v.bytesPerSec)];
    const eta = formatEta(v.etaSeconds);
    if (eta && v.etaSeconds) details.push(`${eta} left`);
    if (v.mirror) details.push(`via ${v.mirror}`);
    if (v.attempt && v.attempt > 1) details.push(`attempt ${v.attempt}`);
    const extra = details.filter(Boolean).join(", ");
    const head = v.percent === null ? `Downloading ${v.file}` : `Downloading ${v.file}: ${v.percent}%`;
    return extra ? `${head} (${extra})...` : `${head}...`;
  }
  const { file, percent } = parseDownloadProgress(payload);
  return percent === null ? `Downloading ${file}...` : `Downloading ${file}: ${percent}%...`;
}
