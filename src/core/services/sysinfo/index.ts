/**
 * System info service (host / desktop only): the TS mirror of the Rust `sysinfo` plugin (GPUs,
 * drivers, CPU features, execution-provider eligibility), used by the GPU runtime card, the Models
 * tab and later the autotuner.
 *
 * Owned by P1-D2; scaffolded by P1-S0b. The IPC contract (command, event, `RustSystemInfo` in
 * src/types.ts) is frozen by P1-S0b; the fields P1-D2 added are typed in ./format.ts.
 *
 * The first probe costs a few hundred milliseconds (DXGI, Direct3D 12, NVML), so `init()` only
 * attaches the listener and starts it in the background; it never rejects and never delays startup.
 */
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { proxy } from "valtio";
import type { RustSystemInfo } from "@/types";
import type { SysSystemInfo } from "./format";

export * from "./format";

/** Tauri event emitted after a re-probe (payload `RustSystemInfo`). */
export const SYSINFO_EVENT = "sysinfo:changed";

export const sysinfoApi = {
  /** Cached probe; `refresh` re-probes (no GPU runtime is loaded). */
  getSystemInfo: (refresh = false) => invoke<RustSystemInfo>("plugin:sysinfo|get_system_info", { refresh }),
};

function message(e: unknown): string {
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message;
  return String(e);
}

export default class Service_Sysinfo {
  readonly state = proxy({
    info: null as SysSystemInfo | null,
    /** A probe is running (first load or an explicit re-check). */
    loading: false,
    error: null as string | null,
  });

  private unlisten: UnlistenFn | null = null;
  private inflight: Promise<void> | null = null;
  private pending = 0;

  async init(): Promise<void> {
    // Must never reject (host init runs every service in parallel).
    if (!window.Config.isApp()) return;
    try {
      this.unlisten ??= await listen<RustSystemInfo>(SYSINFO_EVENT, (e) => this.apply(e.payload));
    } catch {
      // No event bridge (web preview): the explicit loads below still work.
    }
    void this.load();
  }

  private apply(info: RustSystemInfo) {
    this.state.info = info as SysSystemInfo;
    this.state.error = null;
  }

  private run(refresh: boolean): Promise<void> {
    this.pending += 1;
    this.state.loading = true;
    const task = sysinfoApi
      .getSystemInfo(refresh)
      .then((info) => this.apply(info))
      .catch((e) => {
        this.state.error = message(e);
      })
      .finally(() => {
        this.pending -= 1;
        this.state.loading = this.pending > 0;
        if (this.inflight === task) this.inflight = null;
      });
    this.inflight = task;
    return task;
  }

  /** Load the cached probe once (concurrent callers share the request). Never rejects. */
  load(): Promise<void> {
    if (this.state.info) return Promise.resolve();
    return this.inflight ?? this.run(false);
  }

  /** Probe again (after a driver update or a GPU change). Never rejects. */
  refresh(): Promise<void> {
    if (this.inflight && this.state.loading && !this.state.info) return this.inflight;
    return this.run(true);
  }

  /** Detach the event listener (tests, hot reload). */
  dispose(): void {
    this.unlisten?.();
    this.unlisten = null;
  }
}
