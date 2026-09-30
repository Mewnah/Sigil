import i18n from "i18next";
import { parseDownloadProgress, type SttDownloadProgressPayload } from "./downloadProgress";
import type { SttStatusOptions } from "./types";

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

export function downloadStatus(payload: Partial<SttDownloadProgressPayload> | null | undefined): StatusLine {
  const { file, percent } = parseDownloadProgress(payload);
  if (percent === null) return [i18n.t("stt.status_downloading", { file }), { progress: null }];
  return [i18n.t("stt.status_downloading_percent", { file, percent }), { progress: percent }];
}
