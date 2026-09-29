/**
 * Models & Runtimes tab (host / desktop): catalog entries grouped by kind and family with size and
 * state, download progress (mirror, speed, time left), cancel / delete / check files, offline
 * import, disk usage, the download source (mirror, custom URL, parallel downloads) and the GPU
 * runtime card (P1-D2).
 *
 * Owned by P1-B. Routed as the non-service inspector tab "models" (src/types.ts InspectorTabPath,
 * inspector/index.tsx, servicePanels.tsx EXTRA_INSPECTOR_TAB_TITLES, SigilNavigation.tsx);
 * `changeTab({ tab: "models", value: <entry id> })` opens the tab on that entry.
 *
 * Every `plugin:model-store` command is reachable here: get_catalog / get_entry_states /
 * get_disk_usage (open + reload), set_store_config (download source), install_entries (download,
 * update, retry, resume), cancel_install, remove_entry (delete), verify_entry (check files) and
 * import_local (import).
 */
import { isTauri } from "@tauri-apps/api/core";
import classNames from "classnames";
import type { TFunction } from "i18next";
import { FC, memo, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  RiArrowDownSLine,
  RiArrowRightSLine,
  RiCloseLine,
  RiDeleteBin6Line,
  RiDownload2Line,
  RiFileCopyLine,
  RiFolderOpenLine,
  RiHardDrive2Fill,
  RiRefreshLine,
  RiSearchLine,
  RiShieldCheckLine,
} from "react-icons/ri";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import { MODEL_MIRRORS } from "@/core/schema";
import { formatBytes } from "@/core/services/diagnostics/format";
import {
  activeEntries,
  canDownloadEntry,
  deleteScope,
  diskByFamily,
  entryActions,
  entrySize,
  entryTitle,
  familyFallbackLabel,
  groupCatalog,
  hasInstall,
  hasUpdate,
  importBlockers,
  type EntryProgress,
  type FamilyGroup,
  type InstallAction,
  type KindGroup,
  type PendingOp,
  versionLabel,
} from "@/core/services/models";
import { formatMb } from "@/core/services/sysinfo/format";
import { useAppUIStore } from "@/core/ui/store";
import type { ModelMirror, RustCatalogEntry, RustModelEntryStatus } from "@/types";
import Inspector from "./components";
import { GpuRuntimeCard } from "./components/GpuRuntimeCard";
import { InputSelect } from "./components/input";
import {
  formatEntrySize,
  importEntryInteractive,
  mirrorName,
  ModelProgress,
  ModelStateBadge,
  useModelsLoaded,
} from "./components/ModelRequirement";

const models = () => window.ApiServer.models;

/** Translated family name (`models.family_<id>`), else a title-cased id. */
function familyLabel(t: TFunction, family: string): string {
  return t(`models.family_${family.replace(/[^a-z0-9]+/gi, "_").toLowerCase()}`, { defaultValue: familyFallbackLabel(family) });
}

const Card: FC<{ title: ReactNode; action?: ReactNode; children: ReactNode }> = ({ title, action, children }) => (
  <div className="rounded-box border border-base-content/10 bg-base-200/40 p-3 flex flex-col gap-2">
    <div className="flex items-center gap-2 font-semibold text-sm min-h-6">
      <span className="truncate flex items-center gap-2">{title}</span>
      {action && <span className="ms-auto shrink-0">{action}</span>}
    </div>
    {children}
  </div>
);

const ErrorLine: FC<{ text: string; onDismiss?: () => void }> = ({ text, onDismiss }) => {
  const { t } = useTranslation();
  return (
    <div role="alert" className="alert alert-error alert-soft text-xs leading-snug py-1.5 px-2 flex items-start gap-1">
      <span className="break-words flex-1 min-w-0">{text}</span>
      {onDismiss && (
        <button
          type="button"
          className="btn btn-ghost btn-xs btn-square -me-1"
          onClick={onDismiss}
          title={t("models.dismiss")}
          aria-label={t("models.dismiss")}
        >
          <RiCloseLine aria-hidden />
        </button>
      )}
    </div>
  );
};

// -------------------------------------------------------------------------------------------------
// Storage
// -------------------------------------------------------------------------------------------------

async function copyText(t: TFunction, text: string) {
  try {
    await navigator.clipboard.writeText(text);
    toast.success(t("models.copied"), { autoClose: 1500 });
  } catch {
    toast.error(text);
  }
}

const StorageCard: FC = () => {
  const { t } = useTranslation();
  const snap = useSnapshot(models().state);
  const disk = snap.disk;
  const families = diskByFamily(disk);
  return (
    <Card title={t("models.storage_title")}>
      {snap.diskError && !disk && <span className="text-xs text-warning break-words">{t("models.disk_error", { error: snap.diskError })}</span>}
      {disk && (
        <>
          <div className="flex items-baseline justify-between gap-2 text-xs">
            <span className="font-semibold">
              {disk.total_bytes > 0 ? t("models.disk_used", { size: formatBytes(disk.total_bytes) }) : t("models.disk_empty")}
            </span>
            {typeof disk.free_bytes === "number" && (
              <span className="opacity-60 shrink-0">{t("models.disk_free", { size: formatBytes(disk.free_bytes) })}</span>
            )}
          </div>
          {families.length > 0 && (
            <div className="flex flex-col gap-0.5">
              {families.map((f) => (
                <div key={f.family} className="flex items-center justify-between gap-2 text-xs">
                  <span className="opacity-80 truncate">{familyLabel(t, f.family)}</span>
                  <span className="font-mono opacity-70 shrink-0">{formatBytes(f.bytes)}</span>
                </div>
              ))}
            </div>
          )}
          {disk.models_root && (
            <div className="flex items-center gap-1 text-xs min-w-0">
              <span className="opacity-60 shrink-0">{t("models.disk_folder")}</span>
              <span className="font-mono opacity-80 truncate min-w-0 flex-1" dir="ltr" title={disk.models_root}>
                {disk.models_root}
              </span>
              <button
                type="button"
                className="btn btn-ghost btn-xs btn-square shrink-0"
                onClick={() => void copyText(t, disk.models_root ?? "")}
                title={t("models.copy_path")}
                aria-label={t("models.copy_path")}
              >
                <RiFileCopyLine aria-hidden />
              </button>
            </div>
          )}
        </>
      )}
    </Card>
  );
};

// -------------------------------------------------------------------------------------------------
// Download source
// -------------------------------------------------------------------------------------------------

/**
 * The custom mirror URL. A synchronous snapshot: valtio notifies in a microtask by default, so a
 * controlled input would first re-render with the stale value and move the caret to the end on
 * every keystroke.
 */
const CustomMirrorUrlInput: FC<{ problem: boolean }> = ({ problem }) => {
  const url = useSnapshot(window.ApiServer.state, { sync: true }).modelMirrorCustomUrl;
  return (
    <input
      id="models-custom-url"
      type="url"
      dir="ltr"
      spellCheck={false}
      autoComplete="off"
      className={classNames("input input-bordered input-sm w-full font-mono text-xs", problem && "input-warning")}
      placeholder="https://mirror.example.com/sigil"
      value={url}
      onChange={(e) => {
        window.ApiServer.state.modelMirrorCustomUrl = e.target.value;
      }}
      onBlur={() => void models().pushConfig()}
      aria-invalid={problem}
    />
  );
};

const DownloadSourceCard: FC = () => {
  const { t } = useTranslation();
  const root = useSnapshot(window.ApiServer.state);
  const snap = useSnapshot(models().state);
  const mirror = root.modelMirror as ModelMirror;
  const mirrorOptions = MODEL_MIRRORS.map((m) => ({ value: m, label: mirrorName(t, m) }));
  const parallelOptions = [
    { value: "1", label: t("models.parallel_one") },
    { value: "2", label: t("models.parallel_two") },
  ];
  const desc =
    mirror === "auto"
      ? t("models.mirror_desc_auto")
      : mirror === "custom"
        ? t("models.mirror_desc_custom")
        : t("models.mirror_desc_pref", { mirror: mirrorName(t, mirror) });
  return (
    <Card title={t("models.source_title")}>
      <InputSelect
        label="models.field_mirror"
        value={mirror}
        options={mirrorOptions}
        onValueChange={(v) => {
          window.ApiServer.state.modelMirror = v as ModelMirror;
        }}
      />
      <Inspector.Description>{desc}</Inspector.Description>
      {mirror === "custom" && (
        <div className="flex flex-col gap-1">
          <label className="font-medium text-base-content/80 text-xs" htmlFor="models-custom-url">
            {t("models.field_custom_url")}
          </label>
          <CustomMirrorUrlInput problem={!!snap.configProblem} />
          {snap.configProblem && (
            <span className="text-xs text-warning leading-snug">
              {t(`models.url_problem_${snap.configProblem}`)} {t("models.config_not_applied")}
            </span>
          )}
        </div>
      )}
      {snap.configError && <ErrorLine text={t("models.config_error", { error: snap.configError })} />}
      <InputSelect
        label="models.field_parallel"
        value={String(root.modelMaxConcurrentDownloads)}
        options={parallelOptions}
        onValueChange={(v) => {
          window.ApiServer.state.modelMaxConcurrentDownloads = v === "2" ? 2 : 1;
        }}
      />
      <Inspector.Description>{t("models.parallel_desc")}</Inspector.Description>
    </Card>
  );
};

// -------------------------------------------------------------------------------------------------
// Entries
// -------------------------------------------------------------------------------------------------

const INSTALL_LABEL: Record<InstallAction, string> = {
  install: "models.btn_install",
  update: "models.btn_update",
  retry: "models.btn_retry",
  resume: "models.btn_resume",
};

const EntryRow: FC<{ entry: RustCatalogEntry; focused?: boolean }> = memo(({ entry, focused }) => {
  const { t } = useTranslation();
  const snap = useSnapshot(models().state);
  // A custom mirror with an unusable URL is not sent to Rust, so it serves nothing.
  const customMirror = useSnapshot(window.ApiServer.state).modelMirror === "custom" && !snap.configProblem;
  const status = snap.states[entry.id] as RustModelEntryStatus | undefined;
  const state = status?.state ?? "not_installed";
  const pending = snap.pending[entry.id];
  const opError = snap.opErrors[entry.id];
  const actions = entryActions(entry, status, pending);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const title = entryTitle(entry);
  const size = formatEntrySize(t, entrySize(entry));
  const unpublished = entry.status === "unpublished";
  const catalog = snap.catalog as RustCatalogEntry[];
  const states = snap.states as Record<string, RustModelEntryStatus>;
  const requires = (entry.requires ?? []).map((id) => catalog.find((e) => e.id === id)).filter(Boolean) as RustCatalogEntry[];
  // Requirements that must be imported first: Rust would queue them with this entry and fail both.
  const blockers = actions.install
    ? importBlockers(catalog, states, entry, customMirror, snap.pending as Record<string, PendingOp>)
    : [];
  // Unpublished entries download only their hash-pinned files from a custom mirror (or developer local assets).
  const downloadable = canDownloadEntry(entry, customMirror);

  useEffect(() => {
    if (!focused) return;
    // After the frame that lays out the opened family.
    const raf = requestAnimationFrame(() => ref.current?.scrollIntoView({ block: "center", behavior: "smooth" }));
    return () => cancelAnimationFrame(raf);
  }, [focused]);
  useEffect(() => {
    if (!actions.remove) setConfirmDelete(false);
  }, [actions.remove]);

  const install = () => void models().install([entry.id]);
  const verify = async () => {
    const ok = await models().verify(entry.id);
    if (ok) toast.success(t("models.verify_ok", { title }));
  };
  const remove = async () => {
    setConfirmDelete(false);
    await models().remove(entry.id);
  };

  const meta = [versionLabel(entry.version), entry.license].filter(Boolean) as string[];
  // The working install, also while a failed, cancelled or running update overrides the state.
  const installedVersion = status?.installed_version ? versionLabel(status.installed_version) : null;
  const installedLine =
    status && hasInstall(status)
      ? [
          installedVersion && (state !== "installed" || hasUpdate(entry, status))
            ? t("models.installed_version", { version: installedVersion })
            : null,
          status.bytes_on_disk > 0 ? t("models.on_disk", { size: formatBytes(status.bytes_on_disk) }) : null,
        ].filter(Boolean)
      : [];
  const failure = state === "failed" ? status?.error || t("common.error_no_details") : null;
  const scope = deleteScope(status);
  const onDiskSize = formatBytes(status?.bytes_on_disk || entrySize(entry).bytes);
  const deleteQuestion =
    scope === "installed"
      ? t("models.delete_confirm", { title, size: onDiskSize })
      : scope === "installed_and_partial"
        ? t("models.delete_confirm_installed_partial", { title, version: installedVersion ?? "", size: onDiskSize })
        : t("models.delete_confirm_partial", { title });

  return (
    <div
      ref={ref}
      role="group"
      aria-label={title}
      className={classNames(
        "rounded-lg border p-2 flex flex-col gap-1.5 transition-colors",
        focused ? "border-primary ring-1 ring-primary/40" : "border-base-content/10",
      )}
    >
      <div className="flex items-start gap-2">
        <div className="flex-1 min-w-0">
          <div className="text-sm font-semibold leading-snug break-words" title={entry.id}>
            {title}
          </div>
          <div className="text-xs opacity-60 flex flex-wrap gap-x-2 leading-snug">
            <span className="font-semibold">{size}</span>
            {meta.map((m) => (
              <span key={m}>{m}</span>
            ))}
          </div>
        </div>
        <div className="flex flex-col items-end gap-1 shrink-0">
          <ModelStateBadge state={state} />
          {hasUpdate(entry, status) && <span className="badge badge-xs badge-primary badge-outline">{t("models.badge_update")}</span>}
        </div>
      </div>

      {!!(entry.variant || unpublished || (status?.adopted && state === "installed") || entry.min_vram_mb) && (
        <div className="flex flex-wrap gap-1">
          {entry.variant && <span className="badge badge-xs badge-ghost font-mono">{entry.variant}</span>}
          {unpublished && (
            <span className="badge badge-xs badge-neutral" title={t("models.unpublished_hint")}>
              {t("models.badge_unpublished")}
            </span>
          )}
          {status?.adopted && state === "installed" && (
            <span className="badge badge-xs badge-ghost" title={t("models.adopted_hint")}>
              {t("models.badge_adopted")}
            </span>
          )}
          {!!entry.min_vram_mb && (
            <span className="badge badge-xs badge-ghost">{t("models.min_vram", { size: formatMb(entry.min_vram_mb) })}</span>
          )}
        </div>
      )}

      {entry.description && <p className="text-xs opacity-70 leading-snug line-clamp-3">{entry.description}</p>}
      {requires.length > 0 && (
        <span className="text-xs opacity-60 leading-snug break-words">
          {t("models.requires", { list: requires.map((r) => entryTitle(r)).join(", ") })}
        </span>
      )}
      {unpublished && state !== "installed" && <span className="text-xs opacity-60 leading-snug">{t("models.unpublished_hint")}</span>}
      {blockers.length > 0 && (
        <span className="text-xs opacity-70 leading-snug break-words">
          {t("models.needs_import_first", { list: blockers.map((b) => entryTitle(b)).join(", ") })}
        </span>
      )}
      {installedLine.length > 0 && (
        <span className="text-xs opacity-70 leading-snug" title={status?.installed_path ?? undefined}>
          {installedLine.join(" · ")}
        </span>
      )}

      <ModelProgress entry={entry} state={state} progress={snap.progress[entry.id] as EntryProgress | undefined} />

      {failure && <ErrorLine text={failure} />}
      {opError && opError !== failure && <ErrorLine text={opError} onDismiss={() => models().dismissError(entry.id)} />}

      {confirmDelete ? (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="flex-1 min-w-0 break-words">{deleteQuestion}</span>
          <button type="button" className="btn btn-xs btn-error" onClick={() => void remove()}>
            {t("models.btn_delete")}
          </button>
          <button type="button" className="btn btn-xs btn-ghost" onClick={() => setConfirmDelete(false)} autoFocus>
            {t("models.btn_keep")}
          </button>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-1.5">
          {actions.install && downloadable && blockers.length === 0 && (
            <button
              type="button"
              className={classNames("btn btn-xs gap-1", actions.install === "install" || actions.install === "update" ? "btn-primary" : "btn-soft btn-primary")}
              onClick={install}
            >
              <RiDownload2Line aria-hidden />
              {t(INSTALL_LABEL[actions.install])}
            </button>
          )}
          {actions.cancel && (
            <button type="button" className="btn btn-xs" onClick={() => void models().cancel(entry.id)}>
              {t("models.btn_cancel")}
            </button>
          )}
          {actions.import && (
            <button
              type="button"
              className={classNames("btn btn-xs gap-1", downloadable ? "btn-ghost" : "btn-primary")}
              onClick={() => void importEntryInteractive(t, entry)}
              title={t(entry.files.length > 1 ? "models.import_folder_hint" : "models.import_file_hint")}
            >
              <RiFolderOpenLine aria-hidden />
              {t("models.btn_import")}
            </button>
          )}
          {actions.verify && (
            <button type="button" className="btn btn-xs btn-ghost gap-1" onClick={() => void verify()} title={t("models.verify_hint")}>
              <RiShieldCheckLine aria-hidden />
              {t("models.btn_verify")}
            </button>
          )}
          {actions.remove && (
            <button
              type="button"
              className="btn btn-xs btn-ghost gap-1 text-error ms-auto"
              onClick={() => setConfirmDelete(true)}
              aria-label={t("models.btn_delete_entry", { title })}
            >
              <RiDeleteBin6Line aria-hidden />
              {t("models.btn_delete")}
            </button>
          )}
          {pending && (
            <span role="status" className="loading loading-spinner loading-xs opacity-60" aria-label={t(`models.pending_${pending}`)} />
          )}
        </div>
      )}
    </div>
  );
});
EntryRow.displayName = "ModelEntryRow";

const FamilySection: FC<{
  group: FamilyGroup;
  defaultOpen: boolean;
  focusId?: string;
}> = ({ group, defaultOpen, focusId }) => {
  const { t } = useTranslation();
  const [openOverride, setOpenOverride] = useState<boolean | null>(null);
  const containsFocus = !!focusId && group.entries.some((e) => e.id === focusId);
  useEffect(() => {
    if (containsFocus) setOpenOverride(true);
  }, [containsFocus, focusId]);
  const open = openOverride ?? defaultOpen;
  const summary = [
    t("models.family_installed", { installed: group.installed, count: group.entries.length }),
    group.bytesOnDisk > 0 ? formatBytes(group.bytesOnDisk) : null,
  ].filter(Boolean);
  const bodyId = `models-family-${group.family}`;
  return (
    <div className="flex flex-col gap-1.5">
      <button
        type="button"
        className="flex items-center gap-1.5 text-start rounded-md px-1 py-1 hover:bg-base-content/5"
        onClick={() => setOpenOverride(!open)}
        aria-expanded={open}
        aria-controls={bodyId}
      >
        {open ? <RiArrowDownSLine className="shrink-0" aria-hidden /> : <RiArrowRightSLine className="shrink-0 rtl:rotate-180" aria-hidden />}
        <span className="text-sm font-semibold truncate">{familyLabel(t, group.family)}</span>
        {group.busy > 0 && <span className="loading loading-dots loading-xs text-info shrink-0" aria-hidden />}
        <span className="ms-auto text-xs opacity-60 shrink-0">{summary.join(" · ")}</span>
      </button>
      {open && (
        <div id={bodyId} className="flex flex-col gap-1.5">
          {group.entries.map((e) => (
            <EntryRow key={e.id} entry={e} focused={e.id === focusId} />
          ))}
        </div>
      )}
    </div>
  );
};

const KindSection: FC<{ group: KindGroup; filtering: boolean; focusId?: string }> = ({ group, filtering, focusId }) => {
  const { t } = useTranslation();
  return (
    <>
      <Inspector.SubHeader>
        <span>{t(`models.kind_${group.kind}`, { defaultValue: familyFallbackLabel(group.kind) })}</span>
        <span className="text-xs font-normal text-base-content/60">
          {t("models.group_installed", { installed: group.installed, count: group.count })}
        </span>
      </Inspector.SubHeader>
      {group.families.map((f) => (
        <FamilySection
          key={f.family}
          group={f}
          focusId={focusId}
          defaultOpen={filtering || f.entries.length <= 4 || (group.families.length === 1 && f.entries.length <= 8)}
        />
      ))}
    </>
  );
};

const ActiveCard: FC = () => {
  const { t } = useTranslation();
  const snap = useSnapshot(models().state);
  const active = activeEntries(snap.catalog as RustCatalogEntry[], snap.states as Record<string, RustModelEntryStatus>);
  if (active.length === 0) return null;
  return (
    <Card title={t("models.active_title")}>
      {active.map((e) => {
        const state = snap.states[e.id]?.state ?? "queued";
        return (
          <div key={e.id} className="flex flex-col gap-1">
            <div className="flex items-center gap-2 text-xs">
              <span className="font-semibold truncate flex-1 min-w-0" title={e.id}>
                {entryTitle(e)}
              </span>
              {state !== "verifying" || snap.pending[e.id] !== "verify" ? (
                <button type="button" className="btn btn-xs btn-ghost shrink-0" onClick={() => void models().cancel(e.id)}>
                  {t("models.btn_cancel")}
                </button>
              ) : null}
            </div>
            <ModelProgress entry={e} state={state} progress={snap.progress[e.id] as EntryProgress | undefined} />
          </div>
        );
      })}
    </Card>
  );
};

const CatalogSection: FC<{ focusId?: string }> = ({ focusId }) => {
  const { t } = useTranslation();
  const snap = useSnapshot(models().state);
  const [query, setQuery] = useState("");
  const [installedOnly, setInstalledOnly] = useState(false);
  const catalog = snap.catalog as RustCatalogEntry[];
  const states = snap.states as Record<string, RustModelEntryStatus>;
  // Opening the tab on an entry: clear the filters that could hide it. Done while rendering (not in
  // an effect) so the focused row never lays out, and scrolls, inside a filtered list first.
  const [handledFocus, setHandledFocus] = useState(focusId);
  if (focusId !== handledFocus) {
    setHandledFocus(focusId);
    if (focusId) {
      setQuery("");
      setInstalledOnly(false);
    }
  }
  const groups = useMemo(() => groupCatalog(catalog, states, { query, installedOnly }), [catalog, states, query, installedOnly]);
  const filtering = query.trim().length > 0 || installedOnly;

  if (catalog.length === 0) {
    return <p className="text-xs text-base-content/60">{t("models.catalog_empty")}</p>;
  }
  return (
    <>
      <div className="flex items-center gap-2">
        <label className="input input-bordered input-sm flex items-center gap-2 flex-1 min-w-0">
          <RiSearchLine className="opacity-60 shrink-0" aria-hidden />
          <input
            type="search"
            className="grow min-w-0"
            placeholder={t("models.search_placeholder")}
            aria-label={t("models.search_placeholder")}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <label className="flex items-center gap-1.5 text-xs cursor-pointer shrink-0">
          <input
            type="checkbox"
            className="checkbox checkbox-xs checkbox-primary"
            checked={installedOnly}
            onChange={(e) => setInstalledOnly(e.target.checked)}
          />
          {t("models.installed_only")}
        </label>
      </div>
      {groups.length === 0 ? (
        <p className="text-xs text-base-content/60">{t("models.no_results")}</p>
      ) : (
        groups.map((g) => <KindSection key={g.kind} group={g} filtering={filtering} focusId={focusId} />)
      )}
    </>
  );
};

// -------------------------------------------------------------------------------------------------
// Tab
// -------------------------------------------------------------------------------------------------

const Inspector_Models: FC = memo(() => {
  const { t } = useTranslation();
  const tauri = isTauri();
  useModelsLoaded();
  // Fresh states each time the tab opens: Rust may have changed records without an event (startup
  // maintenance re-recording versions), which would otherwise leave a stale "New version" badge.
  useEffect(() => {
    if (tauri) void models().refreshStates();
  }, [tauri]);
  const snap = useSnapshot(models().state);
  const focusId = useAppUIStore((s) => (s.sidebar.tab?.tab === "models" ? s.sidebar.tab.value : undefined));

  return (
    <Inspector.Body>
      <Inspector.Header>
        <RiHardDrive2Fill />
        <span className="truncate">{t("models.title")}</span>
        {tauri && (
          <button
            type="button"
            className="btn btn-ghost btn-sm btn-square ms-auto"
            onClick={() => void models().refresh()}
            disabled={snap.loading}
            title={t("models.refresh")}
            aria-label={t("models.refresh")}
          >
            <RiRefreshLine className={classNames(snap.loading && "animate-spin")} />
          </button>
        )}
      </Inspector.Header>
      <Inspector.Content>
        <p className="text-xs text-base-content/70 leading-relaxed">{t("models.intro")}</p>
        {!tauri ? (
          <p className="text-xs text-base-content/60 leading-relaxed">{t("models.desktop_only")}</p>
        ) : (
          <>
            {snap.error && (
              <div className="alert alert-warning alert-soft text-xs leading-snug flex flex-col items-start gap-1">
                <span className="break-words">{t("models.load_error", { error: snap.error })}</span>
                <button type="button" className="btn btn-xs" onClick={() => void models().refresh()} disabled={snap.loading}>
                  {t("models.retry_load")}
                </button>
              </div>
            )}
            <ActiveCard />
            {snap.loaded ? (
              <CatalogSection focusId={focusId} />
            ) : (
              !snap.error && (
                <p className="text-xs text-base-content/60 flex items-center gap-2">
                  <span className="loading loading-spinner loading-xs" aria-hidden />
                  {t("models.loading")}
                </p>
              )
            )}
            <div className="pt-2 flex flex-col gap-2">
              <StorageCard />
              <DownloadSourceCard />
            </div>
          </>
        )}
        <GpuRuntimeCard />
      </Inspector.Content>
    </Inspector.Body>
  );
});

export default Inspector_Models;
