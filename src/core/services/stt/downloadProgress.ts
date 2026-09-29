/**
 * Payload of `whisper:download_progress` and `vosk:download_progress` (Rust, throttled to a few per second):
 * `progress` is a percentage 0-100, or negative when the total size is unknown (no Content-Length).
 */
export type SttDownloadProgressPayload = {
  file: string;
  progress: number;
};

/** File name and percentage (0-100, rounded) of a download progress payload; `percent` is null when unknown. */
export function parseDownloadProgress(
  payload: Partial<SttDownloadProgressPayload> | null | undefined,
): { file: string; percent: number | null } {
  const file = payload?.file?.trim() || "model";
  const progress = payload?.progress;
  if (typeof progress !== "number" || !Number.isFinite(progress) || progress < 0) return { file, percent: null };
  return { file, percent: Math.round(Math.min(100, progress)) };
}

/** English status line for a model download / local-model copy (logs; the UI uses the translated status). */
export function formatDownloadProgress(payload: Partial<SttDownloadProgressPayload> | null | undefined): string {
  const { file, percent } = parseDownloadProgress(payload);
  return percent === null ? `Downloading ${file}...` : `Downloading ${file}: ${percent}%...`;
}
