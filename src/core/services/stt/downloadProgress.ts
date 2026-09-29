/**
 * Payload of `whisper:download_progress` and `vosk:download_progress` (Rust, throttled to a few per second):
 * `progress` is a percentage 0-100, or negative when the total size is unknown (no Content-Length).
 */
export type SttDownloadProgressPayload = {
  file: string;
  progress: number;
};

/** Interim status line for a model download / local-model copy. */
export function formatDownloadProgress(payload: Partial<SttDownloadProgressPayload> | null | undefined): string {
  const file = payload?.file?.trim() || "model";
  const progress = payload?.progress;
  if (typeof progress !== "number" || !Number.isFinite(progress) || progress < 0) {
    return `Downloading ${file}...`;
  }
  return `Downloading ${file}: ${Math.min(100, progress).toFixed(0)}%...`;
}
