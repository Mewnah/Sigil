import type {
  RustHelperLogLine,
  RustLlmModelInfo,
  RustLlmServerConfig,
  RustLlmServerEndpoint,
  RustLlmServerError,
  RustLlmServerPlan,
  RustLlmServerStarted,
  RustLlmServerStatus,
} from "@/types";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { proxy } from "valtio";

/**
 * Host module of the bundled llama-server helper (`plugin:llm-server`, Rust `services/llm_server`). One server
 * serves every consumer (translation's `llm` engine, the transform service); a consumer holds a lease while it
 * needs the server and releases it when it stops. The server stops when the last lease is released (after
 * `LLM_SERVER_IDLE_STOP_MS`, so a consumer that restarts right away keeps the loaded model).
 *
 * Owned by P4-B (with `src/core/ui/inspector/translation/LlmServer.tsx`). Frozen by P4-S0: the exports below and
 * their signatures change only through INTEGRATE; exports may be added.
 *
 * - `acquireLlmServer(holder, config, signal?)`: starts (or reuses) the server for `config` and resolves with a
 *   lease once it answers `/health`. A config with a different model / variant restarts the server; earlier
 *   holders then see failed requests and re-acquire. Rejects with the Rust error object (`RustLlmServerError`:
 *   `not_installed` with `missing` ids, `refused` by the VRAM governor, `start_failed`, ...). Downloads nothing.
 *   An abort rejects with `signal.reason`.
 * - `lease.endpoint`: `{ baseUrl, apiKey, model }` for `LlmConnectionSettings.bundled`; `lease.refresh()` re-reads it
 *   after a helper restart (the port may change). `lease.release()` is idempotent. P4-B: `refresh()` resolves with the
 *   server's current endpoint, but `lease.endpoint` only follows it while it serves the acquired model: a server
 *   replaced by another model shows up as `refresh().model !== lease.endpoint.model` (re-acquire).
 * - `llmServerUi`: valtio state for panels (status from `llm_server:state`, last plan); read-only for consumers.
 * - P4-B additions: `llmServerConfigFromSettings` (the translation `llm` slice → config), `setOverBudgetConsent`
 *   (the user accepted a refused plan for this session: acquisitions of that model and variant send
 *   `allowOverBudget`), `LlmModelInfoExt` (the additive model-list fields), `ensureLlmServerHost` (event
 *   subscription and the idle stop of a server nobody holds, e.g. after a window reload).
 */

/** A consumer's hold on the running server. */
export interface LlmServerLease {
  readonly holder: string;
  /**
   * The endpoint of the model this lease acquired. It follows restarts of that model (a new port), never a
   * replacement: when another holder replaced the server with a different model, it keeps the acquired `model`.
   */
  readonly endpoint: RustLlmServerEndpoint;
  /**
   * The server's current endpoint (after a restart the port can change); rejects (`not_running`) when the server is
   * gone. When it serves another model than the lease acquired (`result.model !== lease.endpoint.model`), the server
   * was replaced: `lease.endpoint` is left as it was and the caller should re-acquire.
   */
  refresh(): Promise<RustLlmServerEndpoint>;
  /** Drop the hold; the server stops when no lease is left. */
  release(): void;
}

export interface LlmServerUiState {
  status: RustLlmServerStatus | null;
  plan: RustLlmServerPlan | null;
  /** Consumers holding the server (`translation`, `transform`, ...). */
  holders: string[];
}

export const llmServerUi = proxy<LlmServerUiState>({ status: null, plan: null, holders: [] });

/** Session state beyond the frozen `llmServerUi` shape (not persisted). */
export interface LlmServerSessionState {
  /** `consentKey(config)` → the user chose to run this model and variant although the governor refused it. */
  overBudget: Record<string, boolean>;
  /** The last start failure (for the panel), cleared by a successful start. */
  lastError: RustLlmServerError | null;
}

export const llmServerSession = proxy<LlmServerSessionState>({ overBudget: {}, lastError: null });

/** Recommended sampler of a model (Rust `LlmSamplingPreset`). */
export interface LlmSamplingPreset {
  temperature?: number | null;
  topP?: number | null;
  /** -1 = disabled. */
  topK?: number | null;
  repetitionPenalty?: number | null;
}

/** A `list_llm_models` row; the P4-B fields (role, availability, directions, sampling, ...) are on `RustLlmModelInfo`. */
export type LlmModelInfoExt = RustLlmModelInfo;

export const LLM_SERVER_COMMANDS = {
  start: "plugin:llm-server|llm_server_start",
  stop: "plugin:llm-server|llm_server_stop",
  status: "plugin:llm-server|llm_server_status",
  endpoint: "plugin:llm-server|llm_server_endpoint",
  logs: "plugin:llm-server|llm_server_logs",
  plan: "plugin:llm-server|llm_server_plan",
  models: "plugin:llm-server|list_llm_models",
} as const;

/** `LLM_SERVER_EVENTS.state` of `@/types` (kept local so this module imports `@/types` for types only). */
export const LLM_SERVER_STATE_EVENT = "llm_server:state";

/** How long the server stays loaded after the last lease is released. */
export const LLM_SERVER_IDLE_STOP_MS = 10_000;

/** The `server*` fields of the translation `llm` slice. */
export interface LlmServerSettings {
  serverModel: string;
  serverVariant: string;
  serverContext: number;
  /** -1 = from the VRAM plan. */
  serverGpuLayers: number;
  serverFlashAttn: string;
}

const VARIANTS = new Set(["auto", "vulkan", "cuda13", "cpu"]);
const FLASH = new Set(["auto", "on", "off"]);

/** The Rust config for the translation `llm` slice's server settings. */
export function llmServerConfigFromSettings(s: LlmServerSettings): RustLlmServerConfig {
  const model = (s.serverModel ?? "").trim() || "auto";
  const variant = VARIANTS.has(s.serverVariant) ? (s.serverVariant as RustLlmServerConfig["variant"]) : "auto";
  const ctx = Number.isFinite(s.serverContext) && s.serverContext > 0 ? Math.round(s.serverContext) : null;
  const layers = Number.isFinite(s.serverGpuLayers) && s.serverGpuLayers >= 0 ? Math.round(s.serverGpuLayers) : null;
  const flash = FLASH.has(s.serverFlashAttn) ? (s.serverFlashAttn as "auto" | "on" | "off") : "auto";
  return { model, variant, ctxSize: ctx, gpuLayers: layers, flashAttn: flash, allowOverBudget: false };
}

/** Key of the session over-budget consent: the model and the variant. */
export function consentKey(config: RustLlmServerConfig): string {
  return `${(config.model ?? "auto").trim() || "auto"}|${config.variant ?? "auto"}`;
}

/** Remember (or forget) that the user accepted running `config` over the governor's budget this session. */
export function setOverBudgetConsent(config: RustLlmServerConfig, allowed: boolean): void {
  const key = consentKey(config);
  if (allowed) llmServerSession.overBudget[key] = true;
  else delete llmServerSession.overBudget[key];
}

function withConsent(config: RustLlmServerConfig): RustLlmServerConfig {
  return llmServerSession.overBudget[consentKey(config)] ? { ...config, allowOverBudget: true } : config;
}

const ERROR_KINDS = new Set<RustLlmServerError["kind"]>([
  "not_installed",
  "refused",
  "start_failed",
  "not_running",
  "cancelled",
  "invalid_input",
  "network",
  "runtime",
  "not_implemented",
]);

export function isLlmServerError(e: unknown): e is RustLlmServerError {
  return (
    typeof e === "object" &&
    e !== null &&
    typeof (e as RustLlmServerError).message === "string" &&
    ERROR_KINDS.has((e as RustLlmServerError).kind)
  );
}

/** Any rejection of an `llm-server` command as a `RustLlmServerError`. */
export function toLlmServerError(e: unknown): RustLlmServerError {
  if (isLlmServerError(e)) return { kind: e.kind, message: e.message, missing: Array.isArray(e.missing) ? e.missing : [] };
  const message = typeof e === "string" ? e : e instanceof Error ? e.message : String(e);
  return { kind: "runtime", message, missing: [] };
}

// -------------------------------------------------------------------------------------------------
// Leases
// -------------------------------------------------------------------------------------------------

interface LeaseRecord {
  holder: string;
}

const leases = new Map<number, LeaseRecord>();
let nextLeaseId = 1;
let idleTimer: ReturnType<typeof setTimeout> | null = null;
let hostReady: Promise<void> | null = null;
let unlisten: UnlistenFn | null = null;

function syncHolders(): void {
  llmServerUi.holders = [...new Set([...leases.values()].map((l) => l.holder))].sort();
}

function cancelIdleStop(): void {
  if (idleTimer !== null) {
    clearTimeout(idleTimer);
    idleTimer = null;
  }
}

/** Nothing runs: a stop would only clear the failure the panel shows. Unknown counts as running. */
function serverIdle(): boolean {
  const state = llmServerUi.status?.state;
  return state === "stopped" || state === "failed";
}

/** Stop the server `LLM_SERVER_IDLE_STOP_MS` from now unless someone acquires it meanwhile. */
function scheduleIdleStop(): void {
  if (leases.size > 0 || idleTimer !== null || serverIdle()) return;
  idleTimer = setTimeout(() => {
    idleTimer = null;
    if (leases.size > 0 || serverIdle()) return;
    invoke(LLM_SERVER_COMMANDS.stop).catch((e) => console.warn("llm-server: idle stop failed", toLlmServerError(e).message));
  }, LLM_SERVER_IDLE_STOP_MS);
}

/**
 * Subscribe to the server's state events and read its status once. A server running with no lease (the
 * window was reloaded while it ran) is stopped after the idle delay. Idempotent.
 */
export function ensureLlmServerHost(): Promise<void> {
  if (hostReady) return hostReady;
  hostReady = (async () => {
    try {
      unlisten = await listen<RustLlmServerStatus>(LLM_SERVER_STATE_EVENT, (e) => {
        llmServerUi.status = e.payload;
      });
    } catch (e) {
      console.warn("llm-server: cannot subscribe to state events", e);
    }
    try {
      const status = await invoke<RustLlmServerStatus>(LLM_SERVER_COMMANDS.status);
      llmServerUi.status = status;
      if (status.state !== "stopped" && status.state !== "failed" && leases.size === 0) scheduleIdleStop();
    } catch (e) {
      console.warn("llm-server: status unavailable", toLlmServerError(e).message);
    }
  })();
  return hostReady;
}

function abortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException("The operation was aborted.", "AbortError");
}

/** `promise`, or a rejection with the signal's reason as soon as it aborts. */
function raceAbort<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(abortReason(signal));
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(abortReason(signal));
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (v) => {
        signal.removeEventListener("abort", onAbort);
        resolve(v);
      },
      (e) => {
        signal.removeEventListener("abort", onAbort);
        reject(e);
      },
    );
  });
}

export function acquireLlmServer(holder: string, config: RustLlmServerConfig, signal?: AbortSignal): Promise<LlmServerLease> {
  if (signal?.aborted) return Promise.reject(abortReason(signal));
  void ensureLlmServerHost();
  cancelIdleStop();
  const id = nextLeaseId++;
  leases.set(id, { holder });
  syncHolders();
  const drop = () => {
    if (!leases.delete(id)) return;
    syncHolders();
    scheduleIdleStop();
  };
  const start = invoke<RustLlmServerStarted>(LLM_SERVER_COMMANDS.start, { config: withConsent(config) }).then((started) => {
    llmServerUi.status = started.status;
    return started;
  });
  return raceAbort(start, signal).then(
    (started) => {
      llmServerSession.lastError = null;
      llmServerUi.status = started.status;
      let endpoint = started.endpoint;
      const acquiredModel = started.endpoint.model;
      const lease: LlmServerLease = {
        holder,
        get endpoint() {
          return endpoint;
        },
        async refresh() {
          let current: RustLlmServerEndpoint | null;
          try {
            current = await invoke<RustLlmServerEndpoint | null>(LLM_SERVER_COMMANDS.endpoint);
          } catch (e) {
            throw toLlmServerError(e);
          }
          if (!current) {
            const err: RustLlmServerError = { kind: "not_running", message: "The local model server is not running.", missing: [] };
            throw err;
          }
          // Same model (a restart may have moved the port): follow it. Another model replaced the server: keep the
          // acquired endpoint so the holder sees the change and re-acquires instead of using the other model.
          if (current.model === acquiredModel) endpoint = current;
          return current;
        },
        release: drop,
      };
      return lease;
    },
    (e: unknown) => {
      drop();
      if (signal?.aborted) {
        // Rust keeps starting: once that settles, stop the server if nobody took it meanwhile.
        start.then(scheduleIdleStop, scheduleIdleStop);
        throw e;
      }
      const err = toLlmServerError(e);
      if (err.kind !== "cancelled") llmServerSession.lastError = err;
      throw err;
    },
  );
}

/** Stops the server regardless of leases (the panel's Stop button). */
export async function stopLlmServer(): Promise<void> {
  cancelIdleStop();
  try {
    await invoke(LLM_SERVER_COMMANDS.stop);
  } catch (e) {
    throw toLlmServerError(e);
  }
  await getLlmServerStatus().catch(() => undefined);
}

export async function getLlmServerStatus(): Promise<RustLlmServerStatus> {
  try {
    const status = await invoke<RustLlmServerStatus>(LLM_SERVER_COMMANDS.status);
    llmServerUi.status = status;
    return status;
  } catch (e) {
    throw toLlmServerError(e);
  }
}

/** Dry run: what `acquireLlmServer` would start for `config`, or why the governor refuses it. */
export async function planLlmServer(config: RustLlmServerConfig): Promise<RustLlmServerPlan> {
  try {
    const plan = await invoke<RustLlmServerPlan>(LLM_SERVER_COMMANDS.plan, { config });
    llmServerUi.plan = plan;
    return plan;
  } catch (e) {
    throw toLlmServerError(e);
  }
}

export async function listLlmModels(): Promise<LlmModelInfoExt[]> {
  try {
    return await invoke<LlmModelInfoExt[]>(LLM_SERVER_COMMANDS.models);
  } catch (e) {
    throw toLlmServerError(e);
  }
}

export async function getLlmServerLogs(limit?: number): Promise<RustHelperLogLine[]> {
  try {
    return await invoke<RustHelperLogLine[]>(LLM_SERVER_COMMANDS.logs, { limit: limit ?? null });
  } catch (e) {
    throw toLlmServerError(e);
  }
}

/** Test seam: forget every lease, timer and subscription. */
export function __resetLlmServerHostForTests(): void {
  cancelIdleStop();
  leases.clear();
  nextLeaseId = 1;
  unlisten?.();
  unlisten = null;
  hostReady = null;
  llmServerUi.status = null;
  llmServerUi.plan = null;
  llmServerUi.holders = [];
  llmServerSession.overBudget = {};
  llmServerSession.lastError = null;
}
