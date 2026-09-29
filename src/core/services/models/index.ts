/**
 * Models & Runtimes service (host / desktop only): mirrors the Rust `model-store` plugin into a
 * valtio store for the Models tab and <ModelRequirement ids>.
 *
 * Owned by P1-B; scaffolded by P1-S0b. The IPC contract below (command names, events, payload
 * types in src/types.ts) is frozen by P1-S0b.
 *
 * - `init()` attaches the `models:progress` / `models:state` listeners first, then loads the
 *   catalog, the entry states and the disk usage, and pushes the store config. It never rejects.
 * - Root settings pushed to Rust at init and on change (`set_store_config`): `modelMirror`,
 *   `modelMirrorCustomUrl` (debounced while typing), `modelMaxConcurrentDownloads`
 *   (src/core/schema.ts). A "custom" mirror with an unusable URL is not sent (Rust would reject the
 *   whole config); `state.configProblem` says why.
 * - Entry states come from events. A bulk reload never overwrites an entry that received an event
 *   after the reload started, and an operation's returned status never overwrites a newer event.
 * - Operation methods never reject: failures land in `state.opErrors[id]` (IPC rejections) or in
 *   the entry's own `failed` status (install failures, reported by Rust). Rust's "cancelled"
 *   rejection (an import the user cancelled) is not an error and is not recorded.
 * - `install()` re-fetches the entry states when an id it queued got no `models:state` event:
 *   Rust queues nothing (and says nothing) for an entry it considers up to date, which happens
 *   when startup maintenance re-recorded an installed entry's version after the states were
 *   loaded. The Models tab also re-fetches the states each time it opens (`refreshStates`).
 */
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { proxy } from "valtio";
import { subscribeKey } from "valtio/utils";
import type {
  RustCatalogEntry,
  RustModelDiskUsage,
  RustModelEntryState,
  RustModelEntryStatus,
  RustModelProgress,
  RustModelStoreConfig,
} from "@/types";
import {
  buildStoreConfig,
  isBusyState,
  mergeProgress,
  type CustomUrlProblem,
  type EntryProgress,
  type PendingOp,
} from "./format";

export * from "./format";

/** Tauri events of the model store. */
export const MODELS_EVENTS = {
  /** `RustModelProgress`, at most 4 Hz per download. */
  progress: "models:progress",
  /** `RustModelEntryStatus`, on every state change. */
  state: "models:state",
} as const;

const cmd = (name: string) => `plugin:model-store|${name}`;

/** Typed wrappers for `plugin:model-store|*` (reject with the Rust error string). */
export const modelsApi = {
  getCatalog: () => invoke<RustCatalogEntry[]>(cmd("get_catalog")),
  getEntryStates: () => invoke<RustModelEntryStatus[]>(cmd("get_entry_states")),
  setStoreConfig: (config: RustModelStoreConfig) => invoke<void>(cmd("set_store_config"), { config }),
  /** Queues the ids (and their requirements); progress arrives as events. */
  installEntries: (ids: string[]) => invoke<void>(cmd("install_entries"), { ids }),
  /** Keeps the partial file for resume. */
  cancelInstall: (id: string) => invoke<void>(cmd("cancel_install"), { id }),
  removeEntry: (id: string) => invoke<RustModelEntryStatus>(cmd("remove_entry"), { id }),
  verifyEntry: (id: string) => invoke<RustModelEntryStatus>(cmd("verify_entry"), { id }),
  /** Offline import of a downloaded file or archive for `id` (a folder for multi-file entries). */
  importLocal: (id: string, path: string) => invoke<RustModelEntryStatus>(cmd("import_local"), { id, path }),
  getDiskUsage: () => invoke<RustModelDiskUsage>(cmd("get_disk_usage")),
};

/** Root settings keys pushed with `set_store_config`. */
export const STORE_SETTINGS_KEYS = ["modelMirror", "modelMirrorCustomUrl", "modelMaxConcurrentDownloads"] as const;
/** Debounce of config pushes while the custom mirror URL is being typed. */
export const CONFIG_TYPING_DELAY_MS = 500;
/** Coalesces disk-usage reloads after bursts of state changes. */
export const DISK_REFRESH_DELAY_MS = 750;
/** How long after `install_entries` resolves its `models:state` events may still arrive. */
export const INSTALL_EVENT_GRACE_MS = 1000;
/** Rust `model_store::queue::CANCELLED`: the rejection of an operation the user cancelled. */
export const RUST_CANCELLED = "cancelled";

/** States after which the installed files (and so the disk usage) may have changed. */
const SETTLED: ReadonlySet<RustModelEntryState> = new Set(["installed", "not_installed", "failed", "cancelled"]);
/** States in which late progress events of a finished download are ignored. */
const FINISHED: ReadonlySet<RustModelEntryState> = new Set(["installed", "failed", "cancelled"]);

function message(e: unknown): string {
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message;
  try {
    return JSON.stringify(e);
  } catch {
    return String(e);
  }
}

export default class Service_Models {
  /** UI state, filled from `get_catalog`, `get_entry_states`, `get_disk_usage` and the events. */
  readonly state = proxy({
    catalog: [] as RustCatalogEntry[],
    states: {} as Record<string, RustModelEntryStatus>,
    /** Latest download progress per entry id (kept while its install is in flight). */
    progress: {} as Record<string, EntryProgress>,
    disk: null as RustModelDiskUsage | null,
    /** The catalog and the states were loaded at least once. */
    loaded: false,
    /** A catalog / state reload is running. */
    loading: false,
    /** Why the last catalog / state load failed. */
    error: null as string | null,
    /** Why the last disk-usage load failed. */
    diskError: null as string | null,
    /** The custom mirror is selected but its URL is unusable (config not sent). */
    configProblem: null as CustomUrlProblem | null,
    /** Rust rejected the last store config. */
    configError: null as string | null,
    /** UI operation in flight per entry id. */
    pending: {} as Record<string, PendingOp>,
    /** Last failed operation per entry id (Rust's message), cleared by the next operation. */
    opErrors: {} as Record<string, string>,
  });

  private started = false;
  private unlisteners: UnlistenFn[] = [];
  private unsubscribers: (() => void)[] = [];
  /** Monotonic counter of state events / reload starts (see `lastEvent`). */
  private seq = 0;
  /** `seq` of the last `models:state` event per id. */
  private readonly lastEvent = new Map<string, number>();
  private loadInflight: Promise<void> | null = null;
  private diskInflight: Promise<void> | null = null;
  private diskAgain = false;
  private configChain: Promise<void> = Promise.resolve();
  private configTimer: ReturnType<typeof setTimeout> | null = null;
  private diskTimer: ReturnType<typeof setTimeout> | null = null;
  private statesInflight: Promise<void> | null = null;
  private statesAgain = false;
  private readonly installChecks = new Set<ReturnType<typeof setTimeout>>();

  async init(): Promise<void> {
    // Must never reject: the web build and a missing plugin are normal.
    if (!window.Config.isApp() || this.started) return;
    this.started = true;
    try {
      this.unlisteners.push(
        await listen<RustModelProgress>(MODELS_EVENTS.progress, (e) => this.applyProgress(e.payload)),
        await listen<RustModelEntryStatus>(MODELS_EVENTS.state, (e) => this.applyEvent(e.payload)),
      );
    } catch {
      // No event bridge (web preview): explicit reloads still work.
    }
    this.watchSettings();
    void this.pushConfig();
    void this.load();
  }

  /** Detach listeners and timers (tests, hot reload). */
  dispose(): void {
    this.unlisteners.forEach((u) => u());
    this.unsubscribers.forEach((u) => u());
    this.unlisteners = [];
    this.unsubscribers = [];
    if (this.configTimer) clearTimeout(this.configTimer);
    if (this.diskTimer) clearTimeout(this.diskTimer);
    this.configTimer = this.diskTimer = null;
    this.installChecks.forEach((t) => clearTimeout(t));
    this.installChecks.clear();
    this.started = false;
  }

  // ---------------------------------------------------------------------------------------------
  // Loading
  // ---------------------------------------------------------------------------------------------

  /** Load once (concurrent callers share the request). Never rejects. */
  load(): Promise<void> {
    if (this.state.loaded) return Promise.resolve();
    return this.loadInflight ?? this.reload();
  }

  /** Reload the catalog, every entry state and the disk usage. Never rejects. */
  refresh(): Promise<void> {
    return this.loadInflight ?? this.reload();
  }

  private reload(): Promise<void> {
    const since = ++this.seq;
    this.state.loading = true;
    const task = (async () => {
      try {
        const [catalog, statuses] = await Promise.all([modelsApi.getCatalog(), modelsApi.getEntryStates()]);
        this.state.catalog = Array.isArray(catalog) ? catalog : [];
        this.applyStatuses(Array.isArray(statuses) ? statuses : [], since);
        this.state.loaded = true;
        this.state.error = null;
      } catch (e) {
        this.state.error = message(e);
      }
      await this.refreshDiskUsage();
    })().finally(() => {
      this.state.loading = false;
      if (this.loadInflight === task) this.loadInflight = null;
    });
    this.loadInflight = task;
    return task;
  }

  /**
   * Re-fetch every entry state (not the catalog or the disk usage); coalesces overlapping calls.
   * Before the first load it is a plain `load()`. Never rejects.
   */
  refreshStates(): Promise<void> {
    if (!this.state.loaded) return this.load();
    if (this.statesInflight) {
      this.statesAgain = true;
      return this.statesInflight;
    }
    const task = (async () => {
      do {
        this.statesAgain = false;
        const since = ++this.seq;
        try {
          const statuses = await modelsApi.getEntryStates();
          if (Array.isArray(statuses)) this.applyStatuses(statuses, since);
        } catch {
          // Keep the current states; the next event or reload corrects them.
        }
      } while (this.statesAgain);
    })().finally(() => {
      if (this.statesInflight === task) this.statesInflight = null;
    });
    this.statesInflight = task;
    return task;
  }

  /** Reload the disk usage (coalesces overlapping calls). Never rejects. */
  refreshDiskUsage(): Promise<void> {
    if (this.diskInflight) {
      this.diskAgain = true;
      return this.diskInflight;
    }
    const task = (async () => {
      do {
        this.diskAgain = false;
        try {
          this.state.disk = await modelsApi.getDiskUsage();
          this.state.diskError = null;
        } catch (e) {
          this.state.diskError = message(e);
        }
      } while (this.diskAgain);
    })().finally(() => {
      if (this.diskInflight === task) this.diskInflight = null;
    });
    this.diskInflight = task;
    return task;
  }

  private scheduleDiskRefresh(): void {
    if (this.diskTimer) clearTimeout(this.diskTimer);
    this.diskTimer = setTimeout(() => {
      this.diskTimer = null;
      void this.refreshDiskUsage();
    }, DISK_REFRESH_DELAY_MS);
  }

  // ---------------------------------------------------------------------------------------------
  // Events and statuses
  // ---------------------------------------------------------------------------------------------

  private applyEvent(status: RustModelEntryStatus): void {
    if (!status?.id) return;
    this.lastEvent.set(status.id, ++this.seq);
    this.applyStatus(status);
  }

  /** Apply `status` unless a state event for the entry arrived after `since`. */
  private applyStatus(status: RustModelEntryStatus, since?: number): void {
    const id = status.id;
    if (since !== undefined && (this.lastEvent.get(id) ?? 0) > since) return;
    const prev = this.state.states[id];
    this.state.states[id] = status;
    if (status.state === "queued" || SETTLED.has(status.state)) delete this.state.progress[id];
    if (status.state === "installed") delete this.state.opErrors[id];
    if (SETTLED.has(status.state) && prev?.state !== status.state) this.scheduleDiskRefresh();
  }

  private applyStatuses(list: RustModelEntryStatus[], since: number): void {
    const next: Record<string, RustModelEntryStatus> = {};
    for (const s of list) if (s?.id) next[s.id] = s;
    // Entries that got an event after the reload started keep their newer status.
    for (const [id, s] of Object.entries(this.state.states)) {
      if ((this.lastEvent.get(id) ?? 0) > since) next[id] = s;
    }
    this.state.states = next;
    for (const id of Object.keys(this.state.progress)) {
      if (!isBusyState(next[id]?.state)) delete this.state.progress[id];
    }
  }

  private applyProgress(ev: RustModelProgress): void {
    if (!ev?.id || typeof ev.file !== "string") return;
    const st = this.state.states[ev.id]?.state;
    // A late event of a download that already finished must not bring its bar back.
    if (st && FINISHED.has(st)) return;
    this.state.progress[ev.id] = mergeProgress(this.state.progress[ev.id], ev);
  }

  // ---------------------------------------------------------------------------------------------
  // Operations (never reject)
  // ---------------------------------------------------------------------------------------------

  private begin(ids: string[], op: PendingOp): number {
    for (const id of ids) {
      this.state.pending[id] = op;
      delete this.state.opErrors[id];
    }
    return this.seq;
  }

  private end(ids: string[], op: PendingOp, error?: unknown): void {
    const text = error === undefined ? null : message(error);
    for (const id of ids) {
      if (this.state.pending[id] === op) delete this.state.pending[id];
      // A cancelled import is what the user asked for, not a failure.
      if (text !== null && text !== RUST_CANCELLED) this.state.opErrors[id] = text;
    }
  }

  /** Queue installs of `ids` (and what they require). Returns whether Rust accepted them. */
  async install(ids: readonly string[]): Promise<boolean> {
    const list = [...new Set(ids.map((i) => i.trim()).filter(Boolean))];
    if (list.length === 0) return true;
    const since = this.begin(list, "install");
    try {
      await modelsApi.installEntries(list);
      this.end(list, "install");
      this.checkInstallEvents(list, since);
      return true;
    } catch (e) {
      this.end(list, "install", e);
      return false;
    }
  }

  /**
   * After a short grace period, re-fetch the states when an id `install()` queued got no state
   * event since `since`: Rust found it up to date (or already queued) and our state may be stale,
   * e.g. a "New version" badge for a record that startup maintenance already re-versioned.
   */
  private checkInstallEvents(ids: string[], since: number): void {
    const timer = setTimeout(() => {
      this.installChecks.delete(timer);
      if (ids.some((id) => (this.lastEvent.get(id) ?? 0) <= since)) void this.refreshStates();
    }, INSTALL_EVENT_GRACE_MS);
    this.installChecks.add(timer);
  }

  /** Cancel a queued or running install (the partial download is kept for resume). */
  async cancel(id: string): Promise<boolean> {
    const wasInstalling = this.state.pending[id] === "install";
    if (!wasInstalling) this.begin([id], "cancel");
    try {
      await modelsApi.cancelInstall(id);
      if (!wasInstalling) this.end([id], "cancel");
      return true;
    } catch (e) {
      if (wasInstalling) this.state.opErrors[id] = message(e);
      else this.end([id], "cancel", e);
      return false;
    }
  }

  /** Delete the entry's files and partial downloads (a running install is cancelled first). */
  async remove(id: string): Promise<boolean> {
    return this.statusOp(id, "remove", () => modelsApi.removeEntry(id));
  }

  /** Re-hash the installed files; a mismatch unregisters the entry (state `failed` with the reason). */
  async verify(id: string): Promise<boolean> {
    return this.statusOp(id, "verify", () => modelsApi.verifyEntry(id));
  }

  /** Install `id` from a local file (single-file entries) or a folder holding its files. */
  async importLocal(id: string, path: string): Promise<boolean> {
    const p = path.trim();
    if (!p) return false;
    return this.statusOp(id, "import", () => modelsApi.importLocal(id, p));
  }

  private async statusOp(id: string, op: PendingOp, run: () => Promise<RustModelEntryStatus>): Promise<boolean> {
    if (this.state.pending[id]) return false;
    const since = this.begin([id], op);
    try {
      const status = await run();
      if (status?.id) this.applyStatus(status, since);
      this.end([id], op);
      if (op !== "verify") this.scheduleDiskRefresh();
      return status?.state !== "failed";
    } catch (e) {
      this.end([id], op, e);
      return false;
    }
  }

  /** Forget the last operation error of an entry (the row's close button). */
  dismissError(id: string): void {
    delete this.state.opErrors[id];
  }

  // ---------------------------------------------------------------------------------------------
  // Store config (root settings → Rust)
  // ---------------------------------------------------------------------------------------------

  private watchSettings(): void {
    const root = window.ApiServer?.state;
    if (!root) return;
    for (const key of STORE_SETTINGS_KEYS) {
      this.unsubscribers.push(
        subscribeKey(root, key, () => this.scheduleConfigPush(key === "modelMirrorCustomUrl" ? CONFIG_TYPING_DELAY_MS : 0)),
      );
    }
  }

  private scheduleConfigPush(delayMs: number): void {
    if (this.configTimer) clearTimeout(this.configTimer);
    this.configTimer = null;
    if (delayMs <= 0) {
      void this.pushConfig();
      return;
    }
    this.configTimer = setTimeout(() => {
      this.configTimer = null;
      void this.pushConfig();
    }, delayMs);
  }

  /**
   * Send the current root settings to Rust now (pushes are serialized, and each one reads the
   * settings when it runs, so the last change always wins). Never rejects.
   */
  pushConfig(): Promise<void> {
    if (this.configTimer) {
      clearTimeout(this.configTimer);
      this.configTimer = null;
    }
    this.configChain = this.configChain.then(() => this.pushConfigNow());
    return this.configChain;
  }

  private async pushConfigNow(): Promise<void> {
    const root = window.ApiServer?.state;
    if (!root) return;
    const { config, problem } = buildStoreConfig({
      modelMirror: root.modelMirror,
      modelMirrorCustomUrl: root.modelMirrorCustomUrl,
      modelMaxConcurrentDownloads: root.modelMaxConcurrentDownloads,
    });
    this.state.configProblem = problem;
    if (problem) {
      this.state.configError = null;
      return;
    }
    try {
      await modelsApi.setStoreConfig(config);
      this.state.configError = null;
    } catch (e) {
      this.state.configError = message(e);
    }
  }
}
