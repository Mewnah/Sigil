/**
 * <ModelRequirement ids>: a reusable "this feature needs these files" card for inspectors. It
 * resolves the ids plus everything they require from the model-store catalog and shows, in one
 * place: what is missing and how big the download is, a Download button (the user's explicit
 * consent; nothing downloads on its own), live progress with cancel, failures with retry, one
 * Import button per entry that is not published for download (a Download never includes entries
 * whose requirements must be imported first, since Rust would fail them), and a link to the
 * Models & Runtimes tab. `onReady` fires once every entry is installed.
 *
 * Also exports the pieces the Models tab shares (state badge, progress block, import picker,
 * formatted sizes). Owned by P1-B. Desktop only: renders nothing in the web build.
 */
import { isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import classNames from "classnames";
import type { TFunction } from "i18next";
import { FC, memo, ReactNode, useEffect, useMemo, useRef } from "react";
import { useTranslation } from "react-i18next";
import { RiCheckboxCircleFill, RiDownload2Line, RiErrorWarningLine, RiFolderOpenLine, RiStackLine } from "react-icons/ri";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import type { RustCatalogEntry, RustModelEntryState, RustModelEntryStatus } from "@/types";
import { formatBytes } from "@/core/services/diagnostics/format";
import { displayFileName, formatEta, formatSpeed } from "@/core/services/stt/downloadProgress";
import {
  entryProgressView,
  entrySize,
  entryTitle,
  lowOnDisk,
  mirrorKey,
  planRequirement,
  readyTransition,
  requirementFraction,
  summarizeRequirement,
  trackRequirementIds,
  visibleOpErrors,
  type EntryProgress,
  type PendingOp,
} from "@/core/services/models/format";

// -------------------------------------------------------------------------------------------------
// Shared pieces (also used by inspector_models.tsx)
// -------------------------------------------------------------------------------------------------

/** `150 MB`, `≥ 150 MB` when a file size is unknown, or "size unknown". */
export function formatEntrySize(t: TFunction, size: { bytes: number; exact: boolean }): string {
  if (size.bytes <= 0) return t("models.size_unknown");
  return size.exact ? formatBytes(size.bytes) : t("models.size_at_least", { size: formatBytes(size.bytes) });
}

/** Human name of a mirror family (product names; "custom" / "direct" are translated). */
export function mirrorName(t: TFunction, mirror: string): string {
  return t(`models.mirror_name.${mirrorKey(mirror)}`, { defaultValue: mirror });
}

const STATE_BADGE: Record<RustModelEntryState, string> = {
  not_installed: "badge-ghost",
  queued: "badge-info badge-outline",
  downloading: "badge-info",
  verifying: "badge-info",
  extracting: "badge-info",
  installed: "badge-success",
  failed: "badge-error",
  cancelled: "badge-warning",
};

export const ModelStateBadge: FC<{ state: RustModelEntryState }> = ({ state }) => {
  const { t } = useTranslation();
  return (
    <span className={classNames("badge badge-sm shrink-0 whitespace-nowrap", STATE_BADGE[state] ?? "badge-ghost")}>
      {t(`models.state_${state}`, { defaultValue: state })}
    </span>
  );
};

/**
 * Progress of one entry: a byte bar with speed, time left, mirror and attempt while downloading;
 * a moving bar while checking or unpacking; a waiting line while queued.
 */
export const ModelProgress: FC<{
  entry: RustCatalogEntry;
  state: RustModelEntryState;
  progress?: EntryProgress;
}> = ({ entry, state, progress }) => {
  const { t } = useTranslation();
  const title = entryTitle(entry);
  if (state === "queued") {
    return <span className="text-xs opacity-70 leading-snug">{t("models.waiting")}</span>;
  }
  if (state === "verifying" || state === "extracting") {
    const label = t(state === "verifying" ? "models.progress_verifying" : "models.progress_extracting");
    return (
      <div className="flex flex-col gap-0.5">
        <progress className="progress progress-info w-full h-1.5" aria-label={`${title}: ${label}`} />
        <span className="text-xs opacity-70 leading-snug">{label}</span>
      </div>
    );
  }
  if (state !== "downloading") return null;
  if (!progress) {
    return (
      <div className="flex flex-col gap-0.5">
        <progress className="progress progress-primary w-full h-1.5" aria-label={`${title}: ${t("models.progress_starting")}`} />
        <span className="text-xs opacity-70 leading-snug">{t("models.progress_starting")}</span>
      </div>
    );
  }
  const v = entryProgressView(entry, progress);
  const parts: string[] = [];
  parts.push(
    v.bytesTotal !== null
      ? t("models.progress_bytes", { done: formatBytes(v.bytesDone), total: formatBytes(v.bytesTotal) })
      : formatBytes(v.bytesDone),
  );
  const speed = formatSpeed(v.bytesPerSec);
  if (speed) parts.push(speed);
  const eta = formatEta(v.etaSeconds);
  if (eta && v.etaSeconds) parts.push(t("models.progress_eta", { eta }));
  const details: string[] = [t("models.progress_via", { mirror: mirrorName(t, progress.mirror) })];
  if (progress.attempt > 1) details.push(t("models.progress_attempt", { attempt: progress.attempt }));
  if (v.fileCount > 1) {
    details.push(t("models.progress_file", { index: v.fileIndex, count: v.fileCount, file: displayFileName(progress.file) }));
  }
  const percent = v.percent !== null ? `${v.percent}%` : null;
  return (
    <div className="flex flex-col gap-0.5">
      <progress
        className="progress progress-primary w-full h-1.5"
        value={v.fraction !== null ? Math.round(v.fraction * 1000) : undefined}
        max={1000}
        aria-label={title}
        aria-valuetext={percent ?? parts[0]}
      />
      <span className="text-xs opacity-80 leading-snug tabular-nums">
        {percent && <span className="font-semibold">{percent} · </span>}
        {parts.join(" · ")}
      </span>
      <span className="text-xs opacity-60 leading-snug">{details.join(" · ")}</span>
    </div>
  );
};

/**
 * Ask for the file (single-file entries) or the folder holding the files (multi-file entries) to
 * import for `entry`. Resolves null when the user closes the dialog.
 */
export async function pickImportSource(t: TFunction, entry: RustCatalogEntry): Promise<string | null> {
  const title = entryTitle(entry);
  if (entry.files.length === 1) {
    const name = displayFileName(entry.files[0].path);
    const dot = name.lastIndexOf(".");
    const ext = dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
    const filters = ext
      ? [
          { name, extensions: [ext] },
          { name: t("models.import_any_file"), extensions: ["*"] },
        ]
      : undefined;
    const path = await open({ title: t("models.import_file_title", { title }), multiple: false, directory: false, filters });
    return typeof path === "string" ? path : null;
  }
  const path = await open({ title: t("models.import_folder_title", { title }), multiple: false, directory: true });
  return typeof path === "string" ? path : null;
}

/** Pick a local file or folder and import it for `entry`, with a toast on the outcome. */
export async function importEntryInteractive(t: TFunction, entry: RustCatalogEntry): Promise<boolean> {
  let path: string | null;
  try {
    path = await pickImportSource(t, entry);
  } catch (e) {
    toast.error(String(e));
    return false;
  }
  if (!path) return false;
  const ok = await window.ApiServer.models.importLocal(entry.id, path);
  if (ok) toast.success(t("models.import_ok", { title: entryTitle(entry) }));
  return ok;
}

/** Load the models service data once when a view mounts (desktop only). */
export function useModelsLoaded(): void {
  const tauri = isTauri();
  useEffect(() => {
    if (tauri) void window.ApiServer.models.load();
  }, [tauri]);
}

// -------------------------------------------------------------------------------------------------
// <ModelRequirement>
// -------------------------------------------------------------------------------------------------

export interface ModelRequirementProps {
  /** Catalog ids the feature needs (their `requires` are added automatically). */
  ids: readonly string[];
  /** One line saying what needs these files, shown above the list (e.g. "Speech detection needs:"). */
  reason?: ReactNode;
  /**
   * Called once each time every needed entry becomes installed (also on mount when already ready,
   * and when `ids` changes to another set that is already ready). A "Check files" of an installed
   * entry does not count as a change.
   */
  onReady?: () => void;
  /** Render nothing once everything is installed (default: a one-line "ready" note). */
  hideWhenReady?: boolean;
  className?: string;
}

const Titles: FC<{ entries: readonly RustCatalogEntry[] }> = ({ entries }) => (
  <>
    {entries.map((e, i) => (
      <span key={e.id}>
        {i > 0 && ", "}
        <span className="font-semibold" title={e.id}>
          {entryTitle(e)}
        </span>
      </span>
    ))}
  </>
);

export const ModelRequirement: FC<ModelRequirementProps> = memo(({ ids, reason, onReady, hideWhenReady, className }) => {
  const { t } = useTranslation();
  const tauri = isTauri();
  useModelsLoaded();
  const snap = useSnapshot(window.ApiServer.models.state);
  const catalog = snap.catalog as RustCatalogEntry[];
  const states = snap.states as Record<string, RustModelEntryStatus>;
  const pending = snap.pending as Record<string, PendingOp>;
  // A custom mirror with an unusable URL is not sent to Rust, so it serves nothing.
  const customMirror = useSnapshot(window.ApiServer.state).modelMirror === "custom" && !snap.configProblem;
  const idsKey = ids.join("\n");
  // `idsKey` stands in for `ids` so a new array with the same ids does not recompute.
  const summary = useMemo(
    () => summarizeRequirement(catalog, states, idsKey.split("\n"), pending),
    [catalog, states, idsKey, pending],
  );

  // onReady once per transition to ready, and again for another ready id set.
  const firedFor = useRef<string | null>(null);
  const readyNow = snap.loaded && summary.ready;
  useEffect(() => {
    const next = readyTransition(firedFor.current, readyNow, idsKey);
    firedFor.current = next.firedFor;
    if (next.fire) onReady?.();
  }, [readyNow, idsKey, onReady]);

  const busyIds = summary.busy.map((e) => e.id);
  const busy = busyIds.length > 0 || summary.missing.some((e) => pending[e.id] === "install" || pending[e.id] === "import");
  // Entries the overall bar follows during this install run (finished ones stay in, counted fully).
  // Recomputed from the previous value on every render; the update is idempotent.
  const tracked = useRef<string[]>([]);
  tracked.current = trackRequirementIds(tracked.current, summary, busy);

  if (!tauri || ids.length === 0) return null;
  if (readyNow && hideWhenReady) return null;

  const box = (tone: "neutral" | "success" | "warning", children: ReactNode) => (
    <div
      className={classNames(
        "rounded-lg border p-2.5 flex flex-col gap-1.5 text-xs leading-snug",
        tone === "success" ? "border-success/30 bg-success/5" : tone === "warning" ? "border-warning/40 bg-warning/5" : "border-base-content/10 bg-base-200/40",
        className,
      )}
    >
      {children}
    </div>
  );

  const manageLink = (
    <button
      type="button"
      className="link link-hover link-primary text-xs self-start inline-flex items-center gap-1"
      onClick={() => window.ApiServer.changeTab({ tab: "models", value: summary.missing[0]?.id ?? summary.entries[0]?.id })}
    >
      <RiStackLine aria-hidden />
      {t("models.req_manage")}
    </button>
  );

  if (!snap.loaded) {
    return box(
      snap.error ? "warning" : "neutral",
      <span className="opacity-70">{snap.error ? t("models.load_error", { error: snap.error }) : t("models.loading")}</span>,
    );
  }

  if (summary.unknown.length > 0) {
    return box(
      "warning",
      <span className="flex items-start gap-1.5">
        <RiErrorWarningLine className="text-warning shrink-0 mt-0.5" aria-hidden />
        <span className="break-words">{t("models.req_unknown", { ids: summary.unknown.join(", ") })}</span>
      </span>,
    );
  }

  if (summary.ready) {
    const onDisk = summary.entries.reduce((n, e) => n + (states[e.id]?.bytes_on_disk ?? 0), 0);
    return box(
      "success",
      <span className="flex items-start gap-1.5">
        <RiCheckboxCircleFill className="text-success shrink-0 mt-0.5" aria-hidden />
        <span className="break-words">
          {t("models.req_ready")} <Titles entries={summary.entries} />
          {onDisk > 0 && <span className="opacity-60"> · {formatBytes(onDisk)}</span>}
        </span>
      </span>,
    );
  }

  const trackedIds = new Set(tracked.current);
  const trackedEntries = summary.entries.filter((e) => trackedIds.has(e.id));
  const fraction = busy ? requirementFraction(trackedEntries, states, snap.progress as Record<string, EntryProgress>) : null;
  const current = summary.busy.find((e) => states[e.id]?.state === "downloading") ?? summary.busy[0];
  // Unpublished entries download only from a working custom mirror (or developer local assets), and
  // a published entry whose requirement is import-only waits for that import (Rust would fail both).
  const plan = planRequirement(catalog, states, summary, customMirror, pending);
  const size = formatEntrySize(t, { bytes: summary.downloadBytes, exact: summary.downloadExact });
  const dl = plan.download.reduce(
    (acc, e) => {
      const s = entrySize(e);
      return { bytes: acc.bytes + s.bytes, exact: acc.exact && s.exact };
    },
    { bytes: 0, exact: true },
  );
  const low = lowOnDisk(snap.disk, dl.bytes);
  const opErrors = visibleOpErrors(summary.missing, states, snap.opErrors as Record<string, string>);
  // One Import button per import-only entry; a lone missing entry can always be imported instead.
  const importTargets = busy
    ? []
    : summary.missing.length === 1 && !plan.importOnly.some((e) => e.id === summary.missing[0].id)
      ? [...plan.importOnly, summary.missing[0]]
      : plan.importOnly;
  const allImportOnly = plan.download.length === 0 && plan.blocked.length === 0;

  return box(
    summary.failed.length > 0 ? "warning" : "neutral",
    <>
      {reason && <span className="opacity-80">{reason}</span>}
      <span className="break-words">
        {t("models.req_needs")} <Titles entries={summary.missing} />
        {summary.downloadBytes > 0 && <span className="opacity-60"> · {size}</span>}
      </span>

      {busy && current && (
        <div className="flex flex-col gap-0.5">
          {trackedEntries.length > 1 && (
            <progress
              className="progress progress-primary w-full h-1.5"
              value={fraction !== null ? Math.round(fraction * 1000) : undefined}
              max={1000}
              aria-label={t("models.req_overall")}
            />
          )}
          <span className="opacity-70">{entryTitle(current)}</span>
          <ModelProgress
            entry={current}
            state={states[current.id]?.state ?? "queued"}
            progress={snap.progress[current.id] as EntryProgress | undefined}
          />
        </div>
      )}

      {summary.failed.map(({ entry, error }) => (
        <span key={entry.id} className="text-error break-words">
          {t("models.req_failed", { title: entryTitle(entry), error: error || t("common.error_no_details") })}
        </span>
      ))}
      {opErrors.map((e, i) => (
        <span key={i} className="text-error break-words">
          {e}
        </span>
      ))}
      {!busy &&
        plan.importOnly.length > 0 &&
        (allImportOnly ? (
          <span className="opacity-70 break-words">{t("models.unpublished_hint")}</span>
        ) : (
          <span className="opacity-70 break-words">
            {t("models.req_unpublished")} <Titles entries={plan.importOnly} />
          </span>
        ))}
      {!busy && plan.blocked.length > 0 && (
        <span className="opacity-70 break-words">
          {t("models.req_blocked", { list: plan.blocked.map((e) => entryTitle(e)).join(", ") })}
        </span>
      )}
      {!busy && low && (
        <span className="text-warning">{t("models.req_low_disk", { free: formatBytes(snap.disk?.free_bytes ?? 0) })}</span>
      )}

      <div className="flex flex-wrap items-center gap-2 pt-0.5">
        {busy ? (
          <button
            type="button"
            className="btn btn-xs"
            onClick={() => summary.busy.forEach((e) => void window.ApiServer.models.cancel(e.id))}
            disabled={busyIds.length === 0}
          >
            {t("models.btn_cancel")}
          </button>
        ) : (
          plan.download.length > 0 && (
            <button
              type="button"
              className="btn btn-xs btn-primary gap-1"
              onClick={() => void window.ApiServer.models.install(plan.download.map((e) => e.id))}
            >
              <RiDownload2Line aria-hidden />
              {plan.download.some((e) => states[e.id]?.state === "failed")
                ? t("models.btn_retry")
                : dl.bytes > 0
                  ? t("models.req_btn_download", { size: formatEntrySize(t, dl) })
                  : t("models.btn_install")}
            </button>
          )
        )}
        {importTargets.map((e) => (
          <button
            key={e.id}
            type="button"
            className={classNames("btn btn-xs gap-1", plan.download.length > 0 ? "btn-ghost" : "btn-primary")}
            onClick={() => void importEntryInteractive(t, e)}
            title={t(e.files.length > 1 ? "models.import_folder_hint" : "models.import_file_hint")}
          >
            <RiFolderOpenLine aria-hidden />
            {importTargets.length > 1 ? t("models.req_btn_import", { title: entryTitle(e) }) : t("models.btn_import")}
          </button>
        ))}
        <span className="ms-auto">{manageLink}</span>
      </div>
    </>,
  );
});
ModelRequirement.displayName = "ModelRequirement";

/** Size label of an entry for callers that list requirements themselves. */
export function entrySizeLabel(t: TFunction, entry: RustCatalogEntry): string {
  return formatEntrySize(t, entrySize(entry));
}

export default ModelRequirement;
