/**
 * Pure helpers for the Models & Runtimes tab and <ModelRequirement>: catalog grouping and search,
 * entry sizes, per-state actions, multi-file download progress, requirement closures and the
 * store config built from the root settings. No Tauri, DOM, valtio or i18n here (unit-tested
 * under Node); the UI turns the codes returned here into translated text.
 */
import type {
  ModelMirror,
  RustCatalogEntry,
  RustModelDiskUsage,
  RustModelEntryState,
  RustModelEntryStatus,
  RustModelProgress,
  RustModelStoreConfig,
} from "@/types";
import { etaSeconds, percentOf } from "../stt/downloadProgress";

export type EntryKind = RustCatalogEntry["kind"];

/** Display order of the entry kinds (unknown future kinds sort after these). */
export const ENTRY_KINDS: readonly EntryKind[] = ["model", "runtime", "template", "bench"];

/** States in which an install (or a verify) is in flight. */
const BUSY: ReadonlySet<RustModelEntryState> = new Set(["queued", "downloading", "verifying", "extracting"]);

export function isBusyState(state: RustModelEntryState | null | undefined): boolean {
  return !!state && BUSY.has(state);
}

/** A status for entries the store has not reported yet. */
export function emptyStatus(id: string): RustModelEntryStatus {
  return {
    id,
    state: "not_installed",
    installed_version: null,
    installed_path: null,
    bytes_on_disk: 0,
    adopted: false,
    error: null,
  };
}

export function entryTitle(entry: Pick<RustCatalogEntry, "id" | "title">): string {
  return entry.title?.trim() || entry.id;
}

/**
 * Download size of an entry. `exact` is false when a file's size is unknown (`size: 0` on an
 * unpinned file), so the UI can say "at least".
 */
export function entrySize(entry: Pick<RustCatalogEntry, "files">): { bytes: number; exact: boolean } {
  let bytes = 0;
  let exact = true;
  for (const f of entry.files ?? []) {
    const size = Number.isFinite(f.size) && f.size > 0 ? f.size : 0;
    if (size === 0) exact = false;
    bytes += size;
  }
  return { bytes, exact };
}

/**
 * `v1.13.8` for release-style versions; revisions such as commit hashes (`5359861c`) or dates are
 * shown as they are. null when empty.
 */
export function versionLabel(version: string | null | undefined): string | null {
  const v = (version ?? "").trim();
  if (!v) return null;
  return /^\d+(\.\d+)*([-+][0-9a-z.]+)?$/i.test(v) ? `v${v}` : v;
}

/** Whether every file of the entry is hash-pinned (`sha256` or `git_oid`). */
export function isPinned(entry: Pick<RustCatalogEntry, "files">): boolean {
  return (entry.files ?? []).every((f) => !!(f.sha256 || f.git_oid));
}

/** Installed, but the catalog now names another version. */
export function hasUpdate(entry: Pick<RustCatalogEntry, "version">, status: RustModelEntryStatus | undefined): boolean {
  return (
    status?.state === "installed" &&
    !!status.installed_version &&
    !!entry.version &&
    status.installed_version !== entry.version
  );
}

/**
 * A working install is on disk. Rust lets a transient state (queued, downloading, failed,
 * cancelled, ...) override `installed` during an update or a check while it keeps reporting the
 * installed version, so this is not the same as `state === "installed"`.
 */
export function hasInstall(status: RustModelEntryStatus | undefined): boolean {
  return !!status && (status.state === "installed" || !!status.installed_version);
}

/**
 * The entry counts as installed for a feature that needs it: installed, or installed and being
 * re-checked by the user ("Check files" sets the transient state `verifying`; the files stay usable
 * unless the check fails, which then reports `failed`).
 */
export function isSatisfied(status: RustModelEntryStatus | undefined, pending?: PendingOp | null): boolean {
  if (!status) return false;
  if (status.state === "installed") return true;
  return status.state === "verifying" && pending === "verify" && !!status.installed_version;
}

/**
 * Whether Rust can download the entry: published entries always; unpublished ones only from an
 * active custom mirror, and only when every file is hash-pinned (mirror of
 * `model_store::mirrors::sources_for`). Developer local assets (SIGIL_LOCAL_ASSETS) are not visible
 * to the UI, so they are not counted.
 */
export function canDownloadEntry(entry: Pick<RustCatalogEntry, "status" | "files">, customMirror: boolean): boolean {
  return entry.status === "published" || (customMirror && isPinned(entry));
}

/**
 * Rust re-installs the entry when it is queued (directly or as a requirement): not installed, or
 * installed at another catalog version (Rust `is_up_to_date`, approximated by the version).
 */
export function needsInstall(
  entry: Pick<RustCatalogEntry, "version">,
  status: RustModelEntryStatus | undefined,
  pending?: PendingOp | null,
): boolean {
  return !isSatisfied(status, pending) || hasUpdate(entry, status);
}

/** What the delete confirmation must describe. */
export type DeleteScope = "installed" | "installed_and_partial" | "partial";

/**
 * `remove_entry` deletes the working install whenever one exists, even while a failed or cancelled
 * update overrides the state, so the confirmation must say so ("installed_and_partial"). "partial"
 * is only for entries with nothing installed (leftover partial downloads).
 */
export function deleteScope(status: RustModelEntryStatus | undefined): DeleteScope {
  if (status?.state === "installed") return "installed";
  return hasInstall(status) ? "installed_and_partial" : "partial";
}

// -----------------------------------------------------------------------------------------------
// Actions
// -----------------------------------------------------------------------------------------------

/** A UI operation in flight for one entry (buttons are disabled meanwhile). */
export type PendingOp = "install" | "cancel" | "remove" | "verify" | "import";

/** Label of the install button: fresh install, a newer catalog version, retry, or resume. */
export type InstallAction = "install" | "update" | "retry" | "resume";

export interface EntryActions {
  install: InstallAction | null;
  cancel: boolean;
  verify: boolean;
  remove: boolean;
  import: boolean;
}

/**
 * Which actions an entry offers in its current state.
 * - installed: verify, delete, and update when the catalog version changed; import (over the
 *   installed files) when the catalog version changed or the entry is import-only, so a newer local
 *   build can replace the old one without deleting the working install first;
 * - queued / downloading / extracting (and verifying during an install): cancel;
 * - failed / cancelled: retry or resume, import, and delete (discards partial downloads, and the
 *   working install of a failed update);
 * - not installed: install and import.
 * Everything is disabled while a UI operation for the entry is still pending.
 */
export function entryActions(
  entry: Pick<RustCatalogEntry, "version" | "status">,
  status: RustModelEntryStatus | undefined,
  pending?: PendingOp | null,
): EntryActions {
  const none: EntryActions = { install: null, cancel: false, verify: false, remove: false, import: false };
  const state = status?.state ?? "not_installed";
  if (pending) {
    // Cancel stays available while the install request itself is pending.
    return { ...none, cancel: pending === "install" && isBusyState(state) };
  }
  switch (state) {
    case "installed": {
      const update = hasUpdate(entry, status);
      return {
        ...none,
        install: update ? "update" : null,
        verify: true,
        remove: true,
        import: update || entry.status === "unpublished",
      };
    }
    case "queued":
    case "downloading":
    case "extracting":
    case "verifying":
      return { ...none, cancel: true };
    case "failed":
      return { ...none, install: "retry", import: true, remove: true };
    case "cancelled":
      return { ...none, install: "resume", import: true, remove: true };
    case "not_installed":
    default:
      return { ...none, install: "install", import: true };
  }
}

// -----------------------------------------------------------------------------------------------
// Search and grouping
// -----------------------------------------------------------------------------------------------

/** Every whitespace-separated term of `query` appears in the entry's id, title, family, ... */
export function matchesQuery(entry: RustCatalogEntry, query: string): boolean {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return true;
  const haystack = [entry.id, entry.title, entry.family, entry.description, entry.variant, entry.license, entry.kind]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return terms.every((t) => haystack.includes(t));
}

export interface FamilyGroup {
  family: string;
  entries: RustCatalogEntry[];
  /** Entries of this family (among `entries`) with a working install (see `hasInstall`). */
  installed: number;
  /** Entries with an install or verify in flight. */
  busy: number;
  bytesOnDisk: number;
}

export interface KindGroup {
  kind: EntryKind;
  families: FamilyGroup[];
  count: number;
  installed: number;
}

export interface GroupOptions {
  query?: string;
  /** Only installed entries, plus anything busy, failed or cancelled (needs attention). */
  installedOnly?: boolean;
}

function kindRank(kind: string): number {
  const i = ENTRY_KINDS.indexOf(kind as EntryKind);
  return i < 0 ? ENTRY_KINDS.length : i;
}

/**
 * Catalog entries grouped by kind (display order), then by family in first-appearance order;
 * entries keep catalog order (catalog authors list sizes smallest first). Empty groups are dropped.
 */
export function groupCatalog(
  catalog: readonly RustCatalogEntry[],
  states: Readonly<Record<string, RustModelEntryStatus | undefined>>,
  options: GroupOptions = {},
): KindGroup[] {
  const query = options.query?.trim() ?? "";
  const kinds = new Map<string, Map<string, FamilyGroup>>();
  for (const entry of catalog) {
    if (query && !matchesQuery(entry, query)) continue;
    const st = states[entry.id];
    const state = st?.state ?? "not_installed";
    if (options.installedOnly && state === "not_installed") continue;
    let families = kinds.get(entry.kind);
    if (!families) kinds.set(entry.kind, (families = new Map()));
    let group = families.get(entry.family);
    if (!group) families.set(entry.family, (group = { family: entry.family, entries: [], installed: 0, busy: 0, bytesOnDisk: 0 }));
    group.entries.push(entry);
    if (hasInstall(st)) group.installed += 1;
    if (isBusyState(state)) group.busy += 1;
    group.bytesOnDisk += st?.bytes_on_disk ?? 0;
  }
  return [...kinds.entries()]
    .sort(([a], [b]) => kindRank(a) - kindRank(b) || a.localeCompare(b))
    .map(([kind, families]) => {
      const list = [...families.values()];
      return {
        kind: kind as EntryKind,
        families: list,
        count: list.reduce((n, f) => n + f.entries.length, 0),
        installed: list.reduce((n, f) => n + f.installed, 0),
      };
    });
}

/** Entries with an install or verify in flight, in catalog order. */
export function activeEntries(
  catalog: readonly RustCatalogEntry[],
  states: Readonly<Record<string, RustModelEntryStatus | undefined>>,
): RustCatalogEntry[] {
  return catalog.filter((e) => isBusyState(states[e.id]?.state));
}

/** `whisper` → `Whisper`, `stt-legacy` → `Stt legacy` (fallback when no translated family name exists). */
export function familyFallbackLabel(family: string): string {
  const words = family.replace(/[-_./]+/g, " ").trim();
  if (!words) return family;
  return words.charAt(0).toUpperCase() + words.slice(1);
}

// -----------------------------------------------------------------------------------------------
// Progress (per entry, across files)
// -----------------------------------------------------------------------------------------------

/** Latest progress of one entry, with the bytes of every file seen so far. */
export interface EntryProgress {
  id: string;
  /** Catalog path of the file downloading now. */
  file: string;
  bytesDone: number;
  bytesTotal: number | null;
  bytesPerSec: number;
  mirror: RustModelProgress["mirror"];
  attempt: number;
  /** Latest `bytes_done` per catalog file path (a mirror switch may restart a file at 0). */
  fileBytes: Record<string, number>;
}

/** Fold one `models:progress` event into the entry's progress. */
export function mergeProgress(prev: EntryProgress | undefined, ev: RustModelProgress): EntryProgress {
  const done = Number.isFinite(ev.bytes_done) && ev.bytes_done > 0 ? ev.bytes_done : 0;
  const total = typeof ev.bytes_total === "number" && Number.isFinite(ev.bytes_total) && ev.bytes_total > 0 ? ev.bytes_total : null;
  return {
    id: ev.id,
    file: ev.file,
    bytesDone: done,
    bytesTotal: total,
    bytesPerSec: Number.isFinite(ev.bytes_per_sec) && ev.bytes_per_sec > 0 ? ev.bytes_per_sec : 0,
    mirror: ev.mirror,
    attempt: Number.isFinite(ev.attempt) && ev.attempt >= 1 ? Math.floor(ev.attempt) : 1,
    fileBytes: { ...(prev?.fileBytes ?? {}), [ev.file]: done },
  };
}

export interface EntryProgressView {
  /** Bytes of the whole entry (every file) done so far. */
  bytesDone: number;
  /** Size of the whole entry; null when unknown. */
  bytesTotal: number | null;
  /** 0..1 over the whole entry; null when the size is unknown (indeterminate bar). */
  fraction: number | null;
  percent: number | null;
  bytesPerSec: number | null;
  etaSeconds: number | null;
  /** 1-based index of the file downloading now, and the file count (multi-file entries). */
  fileIndex: number;
  fileCount: number;
}

/**
 * Whole-entry progress: the catalog sizes of the entry's files against the bytes seen per file.
 * Rust acquires an entry's files one after another in catalog order, so every file before the one
 * downloading now is complete and counts at its full catalog size, including files that sent no
 * progress event (complete partials after a resume, files adopted from a legacy folder, copies from
 * local assets). Falls back to the current file's own total when the catalog size is unknown.
 */
export function entryProgressView(entry: Pick<RustCatalogEntry, "files">, p: EntryProgress): EntryProgressView {
  const files = entry.files ?? [];
  const size = entrySize(entry);
  const idx = files.findIndex((f) => f.path === p.file);
  let total: number | null = size.exact && size.bytes > 0 ? size.bytes : null;
  let done = 0;
  files.forEach((f, i) => {
    const seen = p.fileBytes[f.path] ?? 0;
    if (i < idx && f.size > 0) done += f.size;
    else done += f.size > 0 ? Math.min(seen, f.size) : seen;
  });
  if (total === null && files.length <= 1) {
    total = p.bytesTotal;
    done = p.bytesDone;
  }
  const fraction = total ? Math.min(1, done / total) : null;
  return {
    bytesDone: done,
    bytesTotal: total,
    fraction,
    percent: total ? percentOf(done, total) : null,
    bytesPerSec: p.bytesPerSec > 0 ? p.bytesPerSec : null,
    etaSeconds: total ? etaSeconds(Math.max(0, total - done), p.bytesPerSec) : null,
    fileIndex: idx >= 0 ? idx + 1 : 1,
    fileCount: Math.max(1, files.length),
  };
}

// -----------------------------------------------------------------------------------------------
// Requirements (<ModelRequirement ids>)
// -----------------------------------------------------------------------------------------------

/**
 * `ids` plus everything they require, transitively, in install order (requirements first).
 * Ids missing from the catalog are reported in `unknown`; cycles are tolerated.
 */
export function requirementClosure(
  catalog: readonly RustCatalogEntry[],
  ids: readonly string[],
): { order: RustCatalogEntry[]; unknown: string[] } {
  const byId = new Map(catalog.map((e) => [e.id, e]));
  const order: RustCatalogEntry[] = [];
  const unknown: string[] = [];
  const done = new Set<string>();
  const visiting = new Set<string>();
  const visit = (raw: string) => {
    const id = raw.trim();
    if (!id || done.has(id) || visiting.has(id)) return;
    const entry = byId.get(id);
    if (!entry) {
      if (!unknown.includes(id)) unknown.push(id);
      done.add(id);
      return;
    }
    visiting.add(id);
    for (const r of entry.requires ?? []) visit(r);
    visiting.delete(id);
    done.add(id);
    order.push(entry);
  };
  ids.forEach(visit);
  return { order, unknown };
}

export interface RequirementSummary {
  /** Every needed entry (the ids and their requirements), requirements first. */
  entries: RustCatalogEntry[];
  /** Needed entries that are not installed (including busy and failed ones; see `isSatisfied`). */
  missing: RustCatalogEntry[];
  /** Ids not in the catalog (a newer feature on an older catalog, or a typo). */
  unknown: string[];
  /** Missing entries with an install or verify in flight. */
  busy: RustCatalogEntry[];
  /** Missing entries whose last attempt failed. */
  failed: { entry: RustCatalogEntry; error: string | null }[];
  /** Missing entries that are not published for download (import or local assets only). */
  unpublished: RustCatalogEntry[];
  /** Download size of the missing entries. */
  downloadBytes: number;
  /** false when a missing file's size is unknown. */
  downloadExact: boolean;
  /** Every needed entry is installed and nothing is unknown (true for an empty id list). */
  ready: boolean;
}

/**
 * What `ids` need and what is missing. `pending` (the service's UI operations) keeps an installed
 * entry satisfied while the user re-checks its files (see `isSatisfied`), so a "Check files" never
 * looks like a download or fires `onReady` again.
 */
export function summarizeRequirement(
  catalog: readonly RustCatalogEntry[],
  states: Readonly<Record<string, RustModelEntryStatus | undefined>>,
  ids: readonly string[],
  pending: Readonly<Record<string, PendingOp | undefined>> = {},
): RequirementSummary {
  const { order, unknown } = requirementClosure(catalog, ids);
  const missing = order.filter((e) => !isSatisfied(states[e.id], pending[e.id]));
  let downloadBytes = 0;
  let downloadExact = true;
  for (const e of missing) {
    const s = entrySize(e);
    downloadBytes += s.bytes;
    downloadExact &&= s.exact;
  }
  return {
    entries: order,
    missing,
    unknown,
    busy: missing.filter((e) => isBusyState(states[e.id]?.state)),
    failed: missing
      .filter((e) => states[e.id]?.state === "failed")
      .map((e) => ({ entry: e, error: states[e.id]?.error ?? null })),
    unpublished: missing.filter((e) => e.status === "unpublished"),
    downloadBytes,
    downloadExact,
    ready: unknown.length === 0 && missing.length === 0,
  };
}

/**
 * Ids whose install the overall bar follows: every entry seen missing since the current install
 * run began, so an entry that finishes stays in the total (and counts fully) instead of dropping
 * out and moving the bar backwards. Empty once nothing is in flight; ids that left the requirement
 * (the `ids` prop changed) are dropped. Returned in requirement order.
 */
export function trackRequirementIds(
  prev: readonly string[],
  summary: Pick<RequirementSummary, "entries" | "missing">,
  busy: boolean,
): string[] {
  if (!busy) return [];
  const keep = new Set(prev);
  for (const e of summary.missing) keep.add(e.id);
  return summary.entries.filter((e) => keep.has(e.id)).map((e) => e.id);
}

/**
 * Overall 0..1 progress of the entries an install run covers (see `trackRequirementIds`): installed
 * ones count fully, the others count their downloaded bytes; null when a size is unknown.
 */
export function requirementFraction(
  tracked: readonly RustCatalogEntry[],
  states: Readonly<Record<string, RustModelEntryStatus | undefined>>,
  progress: Readonly<Record<string, EntryProgress | undefined>>,
): number | null {
  let total = 0;
  let done = 0;
  for (const e of tracked) {
    const size = entrySize(e);
    if (!size.exact || size.bytes === 0) return null;
    total += size.bytes;
    if (states[e.id]?.state === "installed") done += size.bytes;
    else {
      const p = progress[e.id];
      if (p) done += entryProgressView(e, p).bytesDone;
    }
  }
  return total > 0 ? Math.min(1, done / total) : null;
}

/**
 * Requirements of `entry` (transitively, not the entry itself) that Rust would have to install
 * with it but cannot download: queuing `entry` would enqueue them, they would fail as "not
 * published for download yet", and `entry` would then fail as "(required) could not be installed".
 * They must be imported first.
 */
export function importBlockers(
  catalog: readonly RustCatalogEntry[],
  states: Readonly<Record<string, RustModelEntryStatus | undefined>>,
  entry: Pick<RustCatalogEntry, "id">,
  customMirror: boolean,
  pending: Readonly<Record<string, PendingOp | undefined>> = {},
): RustCatalogEntry[] {
  return requirementClosure(catalog, [entry.id]).order.filter(
    (e) => e.id !== entry.id && needsInstall(e, states[e.id], pending[e.id]) && !canDownloadEntry(e, customMirror),
  );
}

export interface RequirementPlan {
  /** Missing entries the Download button queues: downloadable, and so is everything they need. */
  download: RustCatalogEntry[];
  /**
   * Entries only an import can install: missing entries that cannot be downloaded, then
   * requirements of downloadable entries that block them (see `importBlockers`).
   */
  importOnly: RustCatalogEntry[];
  /** Downloadable missing entries held back until an import-only requirement is imported. */
  blocked: RustCatalogEntry[];
}

/** Split a requirement's missing entries into what a Download can fetch and what needs an import. */
export function planRequirement(
  catalog: readonly RustCatalogEntry[],
  states: Readonly<Record<string, RustModelEntryStatus | undefined>>,
  summary: Pick<RequirementSummary, "missing">,
  customMirror: boolean,
  pending: Readonly<Record<string, PendingOp | undefined>> = {},
): RequirementPlan {
  const plan: RequirementPlan = { download: [], importOnly: [], blocked: [] };
  const importIds = new Set<string>();
  const addImport = (e: RustCatalogEntry) => {
    if (!importIds.has(e.id)) {
      importIds.add(e.id);
      plan.importOnly.push(e);
    }
  };
  for (const e of summary.missing) {
    if (!canDownloadEntry(e, customMirror)) addImport(e);
  }
  for (const e of summary.missing) {
    if (importIds.has(e.id)) continue;
    const blockers = importBlockers(catalog, states, e, customMirror, pending);
    if (blockers.length === 0) {
      plan.download.push(e);
    } else {
      plan.blocked.push(e);
      blockers.forEach(addImport);
    }
  }
  return plan;
}

/**
 * `onReady` bookkeeping of <ModelRequirement>: fire once per transition to ready, and again when
 * the requirement's id set changes to another ready set. `firedFor` is the ids key the callback
 * last fired for (null while not ready).
 */
export function readyTransition(firedFor: string | null, ready: boolean, idsKey: string): { fire: boolean; firedFor: string | null } {
  if (!ready) return { fire: false, firedFor: null };
  return { fire: firedFor !== idsKey, firedFor: idsKey };
}

/**
 * Operation errors worth showing for `entries`: an import (or verify) that fails is reported by
 * Rust twice, as the rejected call and as the entry's `failed` status, so errors equal to the
 * entry's status error are dropped (the status line already shows them). Duplicates collapse.
 */
export function visibleOpErrors(
  entries: readonly Pick<RustCatalogEntry, "id">[],
  states: Readonly<Record<string, RustModelEntryStatus | undefined>>,
  opErrors: Readonly<Record<string, string | undefined>>,
): string[] {
  const out: string[] = [];
  for (const e of entries) {
    const err = opErrors[e.id];
    if (!err || err === states[e.id]?.error || out.includes(err)) continue;
    out.push(err);
  }
  return out;
}

// -----------------------------------------------------------------------------------------------
// Disk usage
// -----------------------------------------------------------------------------------------------

/** Families by bytes on disk, largest first (families with 0 bytes are dropped). */
export function diskByFamily(disk: RustModelDiskUsage | null | undefined): { family: string; bytes: number }[] {
  if (!disk?.per_family) return [];
  return Object.entries(disk.per_family)
    .filter(([, bytes]) => Number.isFinite(bytes) && bytes > 0)
    .map(([family, bytes]) => ({ family, bytes }))
    .sort((a, b) => b.bytes - a.bytes || a.family.localeCompare(b.family));
}

/** Whether a download of `bytes` would leave less than `marginBytes` free (null: unknown). */
export function lowOnDisk(
  disk: RustModelDiskUsage | null | undefined,
  bytes: number,
  marginBytes = 1_000_000_000,
): boolean | null {
  if (typeof disk?.free_bytes !== "number" || !Number.isFinite(disk.free_bytes)) return null;
  return disk.free_bytes - bytes < marginBytes;
}

// -----------------------------------------------------------------------------------------------
// Mirror settings → StoreConfig
// -----------------------------------------------------------------------------------------------

/** Why a custom mirror URL is not usable (translated by the UI as `models.url_problem_<code>`). */
export type CustomUrlProblem = "empty" | "scheme" | "invalid" | "credentials" | "insecure";

function ipv4Local(host: string): boolean | null {
  const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(host);
  if (!m) return null;
  const [a, b] = [Number(m[1]), Number(m[2])];
  return (
    a === 127 ||
    a === 10 ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168) ||
    (a === 169 && b === 254) ||
    (a === 0 && b === 0 && m[3] === "0" && m[4] === "0") ||
    (a === 100 && b >= 64 && b <= 127)
  );
}

/**
 * Whether a URL host is this PC or the local network (mirror of Rust `http_client::is_local_endpoint`):
 * loopback, private, link-local and CGNAT/Tailscale IPv4, loopback / unique-local / link-local IPv6
 * (and IPv4-mapped local addresses), `localhost`, single-label names and the `.local`, `.lan`,
 * `.home.arpa`, `.internal`, `.localhost` suffixes.
 */
export function isLocalHost(hostname: string): boolean {
  const host = hostname.trim().toLowerCase().replace(/^\[/, "").replace(/\]$/, "").replace(/\.$/, "");
  if (!host) return false;
  const v4 = ipv4Local(host);
  if (v4 !== null) return v4;
  if (host.includes(":")) {
    if (host === "::1" || host === "::") return true;
    const mapped = /^::ffff:(?:(\d+\.\d+\.\d+\.\d+)|([0-9a-f]{1,4}):([0-9a-f]{1,4}))$/.exec(host);
    if (mapped) {
      if (mapped[1]) return ipv4Local(mapped[1]) === true;
      const hi = parseInt(mapped[2], 16);
      const lo = parseInt(mapped[3], 16);
      return ipv4Local(`${hi >> 8}.${hi & 255}.${lo >> 8}.${lo & 255}`) === true;
    }
    const first = parseInt(host.split(":")[0] || "0", 16);
    return (first & 0xfe00) === 0xfc00 || (first & 0xffc0) === 0xfe80;
  }
  return !host.includes(".") || [".localhost", ".local", ".lan", ".home.arpa", ".internal"].some((s) => host.endsWith(s));
}

/**
 * Check a custom mirror base URL like Rust `model_store::mirrors::validate_custom_url` does:
 * https anywhere, or http on this PC / the local network (files are hash-checked either way); no
 * credentials, query or fragment. Returns null when usable.
 */
export function checkCustomMirrorUrl(raw: string | null | undefined): CustomUrlProblem | null {
  const trimmed = (raw ?? "").trim();
  if (!trimmed) return "empty";
  if (!trimmed.includes("://")) return "scheme";
  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    return "invalid";
  }
  if (url.protocol !== "https:" && url.protocol !== "http:") return "scheme";
  if (!url.hostname) return "invalid";
  if (url.username || url.password || url.search || url.hash || /[?#]/.test(trimmed)) return "credentials";
  if (url.protocol === "http:" && !isLocalHost(url.hostname)) return "insecure";
  return null;
}

/** The root settings the store config is built from (src/core/schema.ts). */
export interface ModelStoreSettings {
  modelMirror: ModelMirror;
  modelMirrorCustomUrl: string;
  modelMaxConcurrentDownloads: number;
}

/**
 * `set_store_config` argument from the root settings. `problem` is set when the custom mirror is
 * selected but its URL is unusable: Rust would reject the whole config, so it is not sent.
 * A custom URL is kept (when valid) with other mirrors so switching back needs no retyping.
 */
export function buildStoreConfig(settings: ModelStoreSettings): { config: RustModelStoreConfig; problem: CustomUrlProblem | null } {
  const url = (settings.modelMirrorCustomUrl ?? "").trim();
  const urlProblem = checkCustomMirrorUrl(url);
  const n = Math.round(Number(settings.modelMaxConcurrentDownloads));
  const config: RustModelStoreConfig = {
    mirror: settings.modelMirror,
    custom_url: url && !urlProblem ? url : null,
    max_concurrent: Number.isFinite(n) ? Math.min(2, Math.max(1, n)) : 1,
  };
  return { config, problem: settings.modelMirror === "custom" ? urlProblem : null };
}

/** i18n key suffix of a mirror family (`hf-mirror` → `hf_mirror`). */
export function mirrorKey(mirror: string): string {
  return mirror.replace(/-/g, "_");
}
