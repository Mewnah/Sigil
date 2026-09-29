/**
 * "GPU & diagnostics" section for the settings inspector (mounted by the P0 integrator):
 * execution providers actually in use, graphics adapters, the GPU test (deep probe), the NVIDIA
 * GPU runtime pack (install / cancel / remove with progress) and the log folder.
 *
 * Desktop only: renders nothing outside Tauri. State comes from `diagnosticsState` (bridge.ts).
 */
import { isTauri } from "@tauri-apps/api/core";
import classNames from "classnames";
import { FC, memo, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { RiCpuLine, RiFolderOpenLine, RiPulseLine } from "react-icons/ri";
import { useSnapshot } from "valtio";
import Inspector from "@/core/ui/inspector/components";
import {
  cancelNvidiaPackInstall,
  diagnosticsState,
  installNvidiaPack,
  openLogFolder,
  refreshPackStatus,
  removeNvidiaPack,
  runDeepProbe,
} from "./bridge";
import { componentKey, epLabel, formatBytes, outcomeFellBack, packProgressFraction } from "./format";
import type { RustEpOutcomeEntry, RustNvidiaPackStatus, RustPackProgress } from "./types";

const COMPONENT_ORDER = ["vc", "vc.content", "vc.pitch", "vc.onnx", "tts.kitten", "stt.vad"];

function sortOutcomes(rows: RustEpOutcomeEntry[]): RustEpOutcomeEntry[] {
  const rank = (c: string) => {
    const i = COMPONENT_ORDER.indexOf(c);
    return i < 0 ? COMPONENT_ORDER.length : i;
  };
  return [...rows].sort((a, b) => rank(a.component) - rank(b.component) || a.component.localeCompare(b.component));
}

const OutcomeRow: FC<{ row: RustEpOutcomeEntry }> = ({ row }) => {
  const { t } = useTranslation();
  const key = componentKey(row.component);
  const fell = outcomeFellBack(row);
  return (
    <div className="flex flex-col text-xs leading-snug">
      <div className="flex items-center justify-between gap-2">
        <span className="opacity-80 truncate">{key ? t(`diagnostics.component_${key}`) : row.component}</span>
        <span
          className={classNames("badge badge-sm font-mono shrink-0", fell ? "badge-warning" : row.active === "cpu" ? "badge-neutral" : "badge-success")}
        >
          {fell ? `${epLabel(row.requested)} → ${epLabel(row.active)}` : epLabel(row.active)}
        </span>
      </div>
      {fell && row.reason && <span className="text-warning/90 break-words mt-0.5">{row.reason}</span>}
    </div>
  );
};

const PackProgressLine: FC<{ progress: RustPackProgress }> = ({ progress }) => {
  const { t } = useTranslation();
  const text =
    progress.phase === "download"
      ? t("diagnostics.pack_downloading", {
          archive: progress.archive ?? "",
          done: formatBytes(progress.done_bytes),
          total: formatBytes(progress.total_bytes),
        })
      : progress.phase === "extract"
        ? t("diagnostics.pack_extracting", { archive: progress.archive ?? "" })
        : t("diagnostics.pack_finalizing");
  return (
    <div className="flex flex-col gap-1">
      <progress className="progress progress-primary w-full" value={Math.round(packProgressFraction(progress) * 1000)} max={1000} />
      <span className="text-xs opacity-70 break-words">{text}</span>
    </div>
  );
};

const PackSection: FC<{ pack: RustNvidiaPackStatus | null; hasNvidia: boolean }> = ({ pack, hasNvidia }) => {
  const { t } = useTranslation();
  const { packBusy, packProgress, packError, packJustInstalled } = useSnapshot(diagnosticsState);
  const installing = packBusy || !!pack?.installing;
  const showProgress = installing && packProgress && !["done", "error", "cancelled"].includes(packProgress.phase);
  const size = formatBytes(pack?.download_bytes ?? 1_434_973_784);

  let status: string;
  if (pack?.pending_removal) status = t("diagnostics.pack_pending_removal");
  else if (pack?.installed) status = t("diagnostics.pack_installed", { label: pack.label });
  else status = t("diagnostics.pack_not_installed");

  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-semibold opacity-80">{t("diagnostics.pack_title")}</span>
      <span className="text-xs opacity-60 leading-snug">{t("diagnostics.pack_desc")}</span>
      {!hasNvidia && !pack?.installed && <span className="text-xs opacity-60">{t("diagnostics.pack_no_nvidia")}</span>}
      <span className={classNames("text-xs", pack?.installed ? "text-success" : "opacity-80")}>{status}</span>
      {showProgress && packProgress && <PackProgressLine progress={packProgress} />}
      {packJustInstalled && <div className="alert alert-info alert-soft text-xs leading-snug">{t("diagnostics.pack_restart_hint")}</div>}
      {(packError || pack?.last_error) && !installing && (
        <div className="alert alert-warning alert-soft text-xs leading-snug break-words">{packError ?? pack?.last_error}</div>
      )}
      <div className="flex gap-2">
        {installing ? (
          <button className="btn btn-sm flex-grow" onClick={() => void cancelNvidiaPackInstall()}>
            {t("diagnostics.pack_cancel")}
          </button>
        ) : pack?.installed || pack?.pending_removal ? (
          <button className="btn btn-sm flex-grow" disabled={!!pack?.pending_removal} onClick={() => void removeNvidiaPack()}>
            {t("diagnostics.pack_remove")}
          </button>
        ) : (
          <button className="btn btn-sm btn-primary flex-grow" onClick={() => void installNvidiaPack()}>
            {t("diagnostics.pack_install", { size })}
          </button>
        )}
      </div>
    </div>
  );
};

export const DiagnosticsPanel: FC = memo(() => {
  const { t } = useTranslation();
  const snap = useSnapshot(diagnosticsState);
  const tauri = isTauri();

  useEffect(() => {
    if (tauri) void refreshPackStatus();
  }, [tauri]);

  if (!tauri) return null;

  const outcomes = sortOutcomes(Object.values(snap.outcomes) as RustEpOutcomeEntry[]);
  const probe = snap.deepProbe ?? snap.probe;
  const adapters = (snap.probe?.adapters ?? []).filter((a) => !a.software);
  const hasNvidia = adapters.some((a) => a.vendor === "nvidia");
  const deep = snap.deepProbe;

  return (
    <>
      <Inspector.SubHeader>
        <span className="flex items-center gap-2">
          <RiCpuLine /> {t("diagnostics.section_title")}
        </span>
      </Inspector.SubHeader>

      <span className="text-xs font-semibold opacity-80">{t("diagnostics.ep_title")}</span>
      {outcomes.length === 0 ? (
        <span className="text-xs opacity-60 leading-snug">{t("diagnostics.ep_none")}</span>
      ) : (
        <div className="flex flex-col gap-1.5">
          {outcomes.map((row) => (
            <OutcomeRow key={row.component} row={row} />
          ))}
        </div>
      )}

      <span className="text-xs font-semibold opacity-80 pt-2">{t("diagnostics.gpu_title")}</span>
      {adapters.length === 0 ? (
        <span className="text-xs opacity-60">{snap.probe ? t("diagnostics.gpu_none") : t("diagnostics.loading")}</span>
      ) : (
        <div className="flex flex-col gap-0.5">
          {adapters.map((a) => (
            <span key={a.index} className="text-xs leading-snug">
              <span className="opacity-80">{a.name}</span>{" "}
              <span className="opacity-50">
                {t("diagnostics.gpu_details", {
                  vram: formatBytes(a.dedicated_vram_mb * 1024 * 1024),
                  driver: a.nvidia_driver ?? a.driver_version ?? "?",
                })}
              </span>
            </span>
          ))}
        </div>
      )}
      {probe && (
        <span className="text-xs">
          {t("diagnostics.recommended", { ep: epLabel(probe.recommended_ep) })}
        </span>
      )}
      {probe?.notes.map((n, i) => (
        <span key={i} className="text-xs opacity-60 leading-snug break-words">
          {n}
        </span>
      ))}

      <button className="btn btn-sm gap-2" disabled={snap.deepProbeRunning} onClick={() => void runDeepProbe()}>
        <RiPulseLine /> {snap.deepProbeRunning ? t("diagnostics.deep_running") : t("diagnostics.deep_run")}
      </button>
      {snap.deepProbeError && <span className="text-xs text-error break-words">{snap.deepProbeError}</span>}
      {deep && (
        <div className="flex flex-col gap-0.5">
          <span className="text-xs opacity-50">
            {t("diagnostics.deep_last", { time: new Date(deep.generated_at_unix_ms).toLocaleString() })}
          </span>
          {deep.ep_checks.map((c, i) => (
            <span key={i} className={classNames("text-xs leading-snug break-words", c.ok ? "text-success" : "text-warning")}>
              {c.ok
                ? t("diagnostics.deep_ok", { ep: epLabel(c.ep), adapter: c.adapter ?? `#${c.device_id}`, ms: c.millis ?? 0 })
                : t("diagnostics.deep_failed", { ep: epLabel(c.ep), adapter: c.adapter ?? `#${c.device_id}`, error: c.error ?? "" })}
            </span>
          ))}
        </div>
      )}

      <div className="pt-2">
        <PackSection pack={snap.pack as RustNvidiaPackStatus | null} hasNvidia={hasNvidia} />
      </div>

      <span className="text-xs font-semibold opacity-80 pt-2">{t("diagnostics.log_title")}</span>
      {snap.logInfo ? (
        <span className="text-xs opacity-60 break-all select-text">{snap.logInfo.file}</span>
      ) : (
        <span className="text-xs opacity-60">{t("diagnostics.log_unavailable")}</span>
      )}
      <button className="btn btn-sm gap-2" disabled={!snap.logInfo} onClick={() => void openLogFolder()}>
        <RiFolderOpenLine /> {t("diagnostics.log_open")}
      </button>
    </>
  );
});
DiagnosticsPanel.displayName = "DiagnosticsPanel";

export default DiagnosticsPanel;
