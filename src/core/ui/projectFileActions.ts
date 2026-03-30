import i18n from "i18next";
import { confirm as dialogConfirm } from "@tauri-apps/plugin-dialog";
import { isTauri } from "@tauri-apps/api/core";

/**
 * Clears the working template and restores defaults. Uses the native confirm dialog on Tauri
 * (window.confirm is unreliable in the WebView and may ignore Cancel).
 */
export async function resetTemplate(): Promise<void> {
  const message = i18n.t("project.confirm_reset_template");
  const title = i18n.t("project.title");
  const ok = isTauri()
    ? await dialogConfirm(message, { title, kind: "warning" })
    : window.confirm(message);
  if (!ok) return;
  await window.ApiClient.document.resetTemplate();
  window.ApiServer.changeTab({ tab: "scenes" });
}
