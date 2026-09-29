import i18n from "i18next";
import { parseDownloadProgress, type SttDownloadProgressPayload } from "./downloadProgress";
import type { SttStatusOptions } from "./types";
import type { PipelineFatal, PipelineStatus } from "./utterancePipeline";

/** Translated STT status lines and stop messages shared by the engines (keys under `stt.status_*` / `stt.error_*`). */

export type StatusLine = [text: string, options?: SttStatusOptions];

/**
 * Resolves once i18next finished loading (host services init before `initI18n`, so a message built during init
 * would show raw keys). Resolves anyway after `timeoutMs` so a failed i18n init never blocks STT.
 */
export function i18nReady(timeoutMs = 5_000): Promise<void> {
  if (i18n.isInitialized) return Promise.resolve();
  return new Promise((resolve) => {
    const done = () => {
      clearTimeout(timer);
      i18n.off("initialized", done);
      resolve();
    };
    const timer = setTimeout(done, timeoutMs);
    i18n.on("initialized", done);
  });
}

const seconds = (ms: number) => Math.max(1, Math.ceil(ms / 1000));

export function downloadStatus(payload: Partial<SttDownloadProgressPayload> | null | undefined): StatusLine {
  const { file, percent } = parseDownloadProgress(payload);
  if (percent === null) return [i18n.t("stt.status_downloading", { file }), { progress: null }];
  return [i18n.t("stt.status_downloading_percent", { file, percent }), { progress: percent }];
}

export function pipelineStatus(status: PipelineStatus): StatusLine {
  if (status.kind === "rate_limited") {
    return [
      i18n.t("stt.status_rate_limited", { seconds: seconds(status.waitMs), count: status.queued }),
      { level: "info" },
    ];
  }
  return [
    i18n.t("stt.status_retrying", {
      detail: status.detail,
      seconds: seconds(status.waitMs),
      attempt: status.attempt,
      max: status.maxAttempts,
    }),
    { level: "warning" },
  ];
}

/** Stop message when the utterance pipeline gives up. `engine` is the user-facing engine name. */
export function pipelineFatalMessage(engine: string, fatal: PipelineFatal): string {
  switch (fatal.reason) {
    case "fatal":
      return i18n.t("stt.error_request_rejected", { engine, detail: fatal.detail });
    case "failed":
      return i18n.t("stt.error_requests_failed", { engine, attempts: fatal.attempts, detail: fatal.detail });
    case "wait_too_long":
      return i18n.t("stt.error_rate_limit_wait", { engine, minutes: Math.ceil(fatal.waitMs / 60_000) });
    case "backlog":
      return i18n.t("stt.error_backlog", { engine, seconds: Math.round(fatal.queuedSeconds) });
  }
}
