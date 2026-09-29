/**
 * Resource governor service (host / desktop only): the TS mirror of the Rust `resources` plugin
 * (VRAM budget, claims by subsystem, VC-running flag, deterministic model recommendations).
 *
 * Owned by P1-D2; scaffolded by P1-S0b. The IPC contract (commands, event, payload types in
 * src/types.ts) is frozen by P1-S0b; the fields P1-D2 added are typed in ./format.ts.
 *
 * - `resources:changed` (claims or the VC flag changed, throttled by Rust) updates the snapshot.
 *   Recommendations are fetched again only when a policy input changed (`policyKey`) or after a
 *   system re-probe (`sysinfo:changed`), never on budget-number noise.
 * - A voice-changer model session reporting its EP (`diagnostics:ep_outcome`, component `vc*`)
 *   while the voice changer runs refreshes the snapshot: Rust infers `vc_gpu` (a policy input)
 *   from those outcomes.
 * - The live budget (this process's usage, device-wide use) moves without events; views that show
 *   it call `watch()` while mounted, which polls every few seconds while the window is visible.
 */
import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { proxy } from "valtio";
import type { RustRecommendation, RustRecommendationTopic, RustResourceSnapshot } from "@/types";
import { DIAGNOSTICS_EVENTS } from "../diagnostics/api";
import { SYSINFO_EVENT } from "../sysinfo";
import { isVcComponent, policyKey, RECOMMENDATION_TOPICS, type ResRecommendation, type ResSnapshot } from "./format";

export * from "./format";

/** Tauri event emitted when claims or the VC flag change (payload `RustResourceSnapshot`). */
export const RESOURCES_EVENT = "resources:changed";

/** Default refresh period of `watch()`. */
export const WATCH_INTERVAL_MS = 3000;

export const resourcesApi = {
  getResourceSnapshot: () => invoke<RustResourceSnapshot>("plugin:resources|get_resource_snapshot"),
  getRecommendation: (topic: RustRecommendationTopic) =>
    invoke<RustRecommendation>("plugin:resources|get_recommendation", { topic }),
};

function message(e: unknown): string {
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message;
  return String(e);
}

function windowHidden(): boolean {
  return typeof document !== "undefined" && document.visibilityState === "hidden";
}

export default class Service_Resources {
  readonly state = proxy({
    snapshot: null as ResSnapshot | null,
    recommendations: {} as Partial<Record<RustRecommendationTopic, ResRecommendation>>,
    recommendationsLoading: false,
    error: null as string | null,
  });

  private unlisteners: UnlistenFn[] = [];
  private lastPolicyKey: string | null = null;
  private recsInflight: Promise<void> | null = null;
  private recsAgain = false;
  private watchers = 0;
  private timer: ReturnType<typeof setInterval> | null = null;

  async init(): Promise<void> {
    // Must never reject (host init runs every service in parallel).
    if (!window.Config.isApp()) return;
    try {
      this.unlisteners.push(
        await listen<RustResourceSnapshot>(RESOURCES_EVENT, (e) => this.applySnapshot(e.payload)),
        await listen(SYSINFO_EVENT, () => {
          // A re-probe can change the primary GPU: every recommendation may change.
          this.lastPolicyKey = null;
          void this.refreshSnapshot();
        }),
        await listen<{ component?: string } | null>(DIAGNOSTICS_EVENTS.epOutcome, (e) => {
          // The voice changer's models loaded on some EP: whether it runs on the GPU may have changed.
          if (this.state.snapshot?.vc_running && isVcComponent(e.payload?.component)) void this.refreshSnapshot();
        }),
      );
    } catch {
      // No event bridge (web preview): explicit refreshes still work.
    }
    void this.refreshSnapshot();
  }

  private applySnapshot(snapshot: RustResourceSnapshot) {
    const s = snapshot as ResSnapshot;
    this.state.snapshot = s;
    this.state.error = null;
    const key = policyKey(s);
    if (key !== this.lastPolicyKey) {
      this.lastPolicyKey = key;
      void this.refreshRecommendations();
    }
  }

  /** Fetch the budget, claims and VC flag now. Never rejects. */
  async refreshSnapshot(): Promise<void> {
    try {
      this.applySnapshot(await resourcesApi.getResourceSnapshot());
    } catch (e) {
      this.state.error = message(e);
    }
  }

  /**
   * Fetch every topic's recommendation. Calls while one round is running coalesce into one more
   * round afterwards (so the last policy input always wins). Never rejects.
   */
  refreshRecommendations(): Promise<void> {
    if (this.recsInflight) {
      this.recsAgain = true;
      return this.recsInflight;
    }
    this.state.recommendationsLoading = true;
    const round = async () => {
      do {
        this.recsAgain = false;
        const results = await Promise.allSettled(RECOMMENDATION_TOPICS.map((t) => resourcesApi.getRecommendation(t)));
        results.forEach((r, i) => {
          const topic = RECOMMENDATION_TOPICS[i];
          if (r.status === "fulfilled") this.state.recommendations[topic] = r.value as ResRecommendation;
          else this.state.error = message(r.reason);
        });
      } while (this.recsAgain);
    };
    this.recsInflight = round().finally(() => {
      this.recsInflight = null;
      this.state.recommendationsLoading = false;
    });
    return this.recsInflight;
  }

  /** One recommendation, straight from Rust (for consumers choosing a model now). */
  getRecommendation(topic: RustRecommendationTopic): Promise<ResRecommendation> {
    return resourcesApi.getRecommendation(topic) as Promise<ResRecommendation>;
  }

  /**
   * Keep the live budget fresh while a view shows it: refreshes now, then every `intervalMs` while
   * the window is visible. Watchers share one timer; call the returned function to stop.
   */
  watch(intervalMs = WATCH_INTERVAL_MS): () => void {
    this.watchers += 1;
    if (!this.timer) {
      void this.refreshSnapshot();
      this.timer = setInterval(() => {
        if (!windowHidden()) void this.refreshSnapshot();
      }, Math.max(500, intervalMs));
    }
    let stopped = false;
    return () => {
      if (stopped) return;
      stopped = true;
      this.watchers = Math.max(0, this.watchers - 1);
      if (this.watchers === 0 && this.timer) {
        clearInterval(this.timer);
        this.timer = null;
      }
    };
  }

  /** Detach listeners and timers (tests, hot reload). */
  dispose(): void {
    this.unlisteners.forEach((u) => u());
    this.unlisteners = [];
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.watchers = 0;
  }
}
