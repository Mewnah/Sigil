import { Translation_Backends } from "@/core/services/translation/schema";
import { ServiceNetworkState } from "@/types";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import { ModelRequirement } from "../components/ModelRequirement";
import { Hint } from "./common";
import { backendLabelKey } from "./logic";

/**
 * Engine status under the start button, from `window.ApiServer.translation.serviceState` (UI only, never a caption):
 * preparation (download / load progress, server start, a fallback in use), why the last start failed, the files an
 * engine reported missing (offered for download here, usable while translation runs) and, for automatic routing, the
 * engine that translated the last line.
 *
 * Owned by P4-F.
 */
export const TranslationStatusLine: FC = () => {
  const { t } = useTranslation();
  const s = useSnapshot(window.ApiServer.translation.serviceState);
  const backend = useSnapshot(window.ApiServer.state.services.translation.data).backend;
  const stopped = s.status === ServiceNetworkState.disconnected;
  const missing = s.missing as readonly string[];
  const showEngine = s.status === ServiceNetworkState.connected && backend === Translation_Backends.auto && s.activeEngine !== "";

  if (!s.statusText && !(stopped && s.error) && missing.length === 0 && !showEngine) return null;

  return (
    <div role="status" aria-live="polite" className="flex flex-col gap-1 text-xs">
      {s.statusText && (
        <span className={s.statusLevel === "warning" ? "text-warning" : "text-base-content/70"}>{s.statusText}</span>
      )}
      {s.progress !== null && <progress className="progress progress-primary w-full" value={s.progress} max={100} />}
      {stopped && s.error && <span className="text-error break-words">{s.error}</span>}
      {showEngine && <span className="text-base-content/70">{t("transl.status_active_engine", { engine: t(backendLabelKey(s.activeEngine as Translation_Backends)) })}</span>}
      {missing.length > 0 && (
        <>
          <ModelRequirement ids={missing} reason={t("transl.status_missing")} hideWhenReady />
          {stopped && <Hint className="px-0">{t("transl.status_missing_restart")}</Hint>}
        </>
      )}
    </div>
  );
};
