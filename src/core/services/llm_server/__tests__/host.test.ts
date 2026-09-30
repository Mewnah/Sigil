import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LLM_SERVER_EVENTS, type RustLlmServerConfig, type RustLlmServerStatus } from "@/types";

type Handler = (ev: { payload: unknown }) => void;
const handlers = new Map<string, Handler>();
const invoke = vi.fn();

vi.mock("@tauri-apps/api/event", () => ({
  listen: async (name: string, cb: Handler) => {
    handlers.set(name, cb);
    return () => handlers.delete(name);
  },
}));
vi.mock("@tauri-apps/api/core", () => ({ invoke: (...args: unknown[]) => invoke(...args) }));

const host = await import("../index");
const {
  acquireLlmServer,
  stopLlmServer,
  llmServerUi,
  llmServerSession,
  llmServerConfigFromSettings,
  setOverBudgetConsent,
  consentKey,
  toLlmServerError,
  isLlmServerError,
  ensureLlmServerHost,
  LLM_SERVER_COMMANDS,
  LLM_SERVER_IDLE_STOP_MS,
  LLM_SERVER_STATE_EVENT,
  __resetLlmServerHostForTests,
} = host;

const status = (state: RustLlmServerStatus["state"], extra: Partial<RustLlmServerStatus> = {}): RustLlmServerStatus => ({
  state,
  model: "mt/hy-mt2-1.8b-q4_k_m",
  variant: "vulkan",
  gpuLayers: 999,
  ctxSize: 4096,
  flashAttn: "auto",
  baseUrl: state === "ready" ? "http://127.0.0.1:5123/v1" : null,
  vramMb: 1600,
  restarts: 0,
  error: null,
  ...extra,
});

const endpoint = { baseUrl: "http://127.0.0.1:5123/v1", apiKey: "k".repeat(64), model: "hy-mt2-1.8b-q4_k_m" };
const cfg: RustLlmServerConfig = { model: "mt/hy-mt2-1.8b-q4_k_m", variant: "auto" };

/** Backend stub: start succeeds after `delay` ms (or rejects with `startError`). */
function backend(opts: { delay?: number; startError?: unknown; status?: RustLlmServerStatus; endpoint?: typeof endpoint | null } = {}) {
  return (cmd: string, args?: unknown) => {
    switch (cmd) {
      case LLM_SERVER_COMMANDS.start:
        return new Promise((resolve, reject) =>
          setTimeout(() => (opts.startError ? reject(opts.startError) : resolve({ status: status("ready"), endpoint, args })), opts.delay ?? 0),
        );
      case LLM_SERVER_COMMANDS.stop:
        return Promise.resolve(null);
      case LLM_SERVER_COMMANDS.status:
        return Promise.resolve(opts.status ?? status("stopped", { model: null, vramMb: 0 }));
      case LLM_SERVER_COMMANDS.endpoint:
        return Promise.resolve(opts.endpoint === undefined ? endpoint : opts.endpoint);
      case LLM_SERVER_COMMANDS.plan:
        return Promise.resolve({ model: "mt/hy-mt2-1.8b-q4_k_m", allowed: true, missing: [] });
      default:
        return Promise.reject(new Error(`unexpected ${cmd}`));
    }
  };
}

const calls = (cmd: string) => invoke.mock.calls.filter(([c]) => c === cmd);

describe("llama-server host", () => {
  beforeEach(() => {
    handlers.clear();
    invoke.mockReset();
    __resetLlmServerHostForTests();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("uses the frozen event name", () => {
    expect(LLM_SERVER_STATE_EVENT).toBe(LLM_SERVER_EVENTS.state);
  });

  it("leases the server, shares it between holders and stops it only after the last release and the idle delay", async () => {
    vi.useFakeTimers();
    invoke.mockImplementation(backend());
    const pA = acquireLlmServer("translation", cfg);
    const pB = acquireLlmServer("transform", cfg);
    await vi.advanceTimersByTimeAsync(1);
    const [a, b] = await Promise.all([pA, pB]);
    expect(a.endpoint).toEqual(endpoint);
    expect(calls(LLM_SERVER_COMMANDS.start)[0][1]).toEqual({ config: cfg });
    expect(llmServerUi.holders).toEqual(["transform", "translation"]);
    expect(llmServerUi.status?.state).toBe("ready");

    a.release();
    a.release(); // idempotent
    expect(llmServerUi.holders).toEqual(["transform"]);
    await vi.advanceTimersByTimeAsync(LLM_SERVER_IDLE_STOP_MS * 2);
    expect(calls(LLM_SERVER_COMMANDS.stop)).toHaveLength(0);

    b.release();
    expect(llmServerUi.holders).toEqual([]);
    await vi.advanceTimersByTimeAsync(LLM_SERVER_IDLE_STOP_MS - 10);
    expect(calls(LLM_SERVER_COMMANDS.stop)).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(20);
    expect(calls(LLM_SERVER_COMMANDS.stop)).toHaveLength(1);
  });

  it("keeps the model loaded when a consumer re-acquires within the idle delay", async () => {
    vi.useFakeTimers();
    invoke.mockImplementation(backend());
    const p = acquireLlmServer("translation", cfg);
    await vi.advanceTimersByTimeAsync(1);
    (await p).release();
    await vi.advanceTimersByTimeAsync(LLM_SERVER_IDLE_STOP_MS / 2);
    const again = acquireLlmServer("translation", cfg);
    await vi.advanceTimersByTimeAsync(LLM_SERVER_IDLE_STOP_MS * 2);
    await again;
    expect(calls(LLM_SERVER_COMMANDS.stop)).toHaveLength(0);
  });

  it("rejects with the Rust error object and clears the lease", async () => {
    vi.useFakeTimers();
    const refused = { kind: "refused", message: "Hy-MT2 7B: refused: would leave too little for the voice changer", missing: [] };
    invoke.mockImplementation(backend({ startError: refused }));
    const p = acquireLlmServer("translation", { model: "mt/hy-mt2-7b-q4_k_m" });
    const assertion = expect(p).rejects.toEqual(refused);
    await vi.advanceTimersByTimeAsync(1);
    await assertion;
    expect(llmServerUi.holders).toEqual([]);
    expect(llmServerSession.lastError?.kind).toBe("refused");

    const notInstalled = { kind: "not_installed", message: "needs downloads", missing: ["mt/hy-mt2-7b-q4_k_m", "runtime/llama-server-vulkan"] };
    invoke.mockImplementation(backend({ startError: notInstalled }));
    const p2 = acquireLlmServer("translation", cfg);
    const a2 = expect(p2).rejects.toMatchObject({ kind: "not_installed", missing: notInstalled.missing });
    await vi.advanceTimersByTimeAsync(1);
    await a2;

    // A plain string rejection (a panic message, an IPC failure) becomes a runtime error.
    invoke.mockImplementation(backend({ startError: "IPC failed" }));
    const p3 = acquireLlmServer("translation", cfg);
    const a3 = expect(p3).rejects.toEqual({ kind: "runtime", message: "IPC failed", missing: [] });
    await vi.advanceTimersByTimeAsync(1);
    await a3;
  });

  it("does not stop a server that already failed or stopped (keeps the failure visible)", async () => {
    vi.useFakeTimers();
    const failed = { kind: "start_failed", message: "llama-server exited while starting", missing: [] };
    invoke.mockImplementation(backend({ startError: failed, delay: 10 }));
    await ensureLlmServerHost();
    const p = acquireLlmServer("translation", cfg);
    const assertion = expect(p).rejects.toEqual(failed);
    // Rust emits the failure before the command rejects.
    handlers.get(LLM_SERVER_STATE_EVENT)?.({ payload: status("failed", { error: failed.message }) });
    await vi.advanceTimersByTimeAsync(20);
    await assertion;
    await vi.advanceTimersByTimeAsync(LLM_SERVER_IDLE_STOP_MS * 2);
    expect(calls(LLM_SERVER_COMMANDS.stop)).toHaveLength(0);
    expect(llmServerUi.status?.error).toBe(failed.message);
  });

  it("an abort rejects with the signal's reason at once and the server is stopped when nobody holds it", async () => {
    vi.useFakeTimers();
    invoke.mockImplementation(backend({ delay: 5000 }));
    const ac = new AbortController();
    const p = acquireLlmServer("translation", cfg, ac.signal);
    const reason = new Error("stopped");
    const assertion = expect(p).rejects.toBe(reason);
    ac.abort(reason);
    await assertion;
    expect(llmServerUi.holders).toEqual([]);
    // Rust finishes loading 5 s later; nobody holds it, so it is stopped after the idle delay.
    await vi.advanceTimersByTimeAsync(5000);
    expect(llmServerUi.status?.state).toBe("ready");
    expect(calls(LLM_SERVER_COMMANDS.stop)).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(LLM_SERVER_IDLE_STOP_MS + 10);
    expect(calls(LLM_SERVER_COMMANDS.stop)).toHaveLength(1);
    // Already aborted: nothing is started.
    const before = calls(LLM_SERVER_COMMANDS.start).length;
    await expect(acquireLlmServer("x", cfg, ac.signal)).rejects.toBe(reason);
    expect(calls(LLM_SERVER_COMMANDS.start)).toHaveLength(before);
  });

  it("refresh reads the current endpoint and reports a stopped server", async () => {
    invoke.mockImplementation(backend());
    const lease = await acquireLlmServer("translation", cfg);
    const moved = { ...endpoint, baseUrl: "http://127.0.0.1:6000/v1" };
    invoke.mockImplementation(backend({ endpoint: moved }));
    await expect(lease.refresh()).resolves.toEqual(moved);
    expect(lease.endpoint).toEqual(moved);
    invoke.mockImplementation(backend({ endpoint: null }));
    await expect(lease.refresh()).rejects.toMatchObject({ kind: "not_running" });
    lease.release();
  });

  it("refresh never rebinds a lease to another model that replaced the server", async () => {
    invoke.mockImplementation(backend());
    const lease = await acquireLlmServer("translation", cfg);
    // Another holder started a different model: new alias, new key, new port.
    const replaced = { baseUrl: "http://127.0.0.1:7000/v1", apiKey: "z".repeat(64), model: "milmmt-46-4b-q4_k_m" };
    invoke.mockImplementation(backend({ endpoint: replaced }));
    const current = await lease.refresh();
    expect(current).toEqual(replaced);
    // The lease keeps the model it acquired, so the holder can tell the server was replaced and re-acquire.
    expect(lease.endpoint).toEqual(endpoint);
    expect(current.model).not.toBe(lease.endpoint.model);
    // Its own model back (restarted on another port): followed again.
    const back = { ...endpoint, baseUrl: "http://127.0.0.1:7100/v1", apiKey: "y".repeat(64) };
    invoke.mockImplementation(backend({ endpoint: back }));
    await expect(lease.refresh()).resolves.toEqual(back);
    expect(lease.endpoint).toEqual(back);
    lease.release();
  });

  it("sends allowOverBudget only for a model the user accepted this session", async () => {
    invoke.mockImplementation(backend());
    const seven = { model: "mt/hy-mt2-7b-q4_k_m", variant: "vulkan" } as const;
    (await acquireLlmServer("t", seven)).release();
    expect(calls(LLM_SERVER_COMMANDS.start).at(-1)?.[1]).toEqual({ config: seven });
    setOverBudgetConsent(seven, true);
    (await acquireLlmServer("t", seven)).release();
    expect(calls(LLM_SERVER_COMMANDS.start).at(-1)?.[1]).toEqual({ config: { ...seven, allowOverBudget: true } });
    // Another variant is another decision.
    (await acquireLlmServer("t", { ...seven, variant: "cuda13" })).release();
    expect(calls(LLM_SERVER_COMMANDS.start).at(-1)?.[1]).toEqual({ config: { ...seven, variant: "cuda13" } });
    setOverBudgetConsent(seven, false);
    (await acquireLlmServer("t", seven)).release();
    expect(calls(LLM_SERVER_COMMANDS.start).at(-1)?.[1]).toEqual({ config: seven });
    expect(consentKey({})).toBe("auto|auto");
  });

  it("follows state events and stops a server nobody holds after a window reload", async () => {
    vi.useFakeTimers();
    invoke.mockImplementation(backend({ status: status("ready") }));
    await ensureLlmServerHost();
    expect(llmServerUi.status?.state).toBe("ready");
    handlers.get(LLM_SERVER_STATE_EVENT)!({ payload: status("restarting", { restarts: 1 }) });
    expect(llmServerUi.status?.state).toBe("restarting");
    await vi.advanceTimersByTimeAsync(LLM_SERVER_IDLE_STOP_MS + 10);
    expect(calls(LLM_SERVER_COMMANDS.stop)).toHaveLength(1);
    // Idempotent: one subscription.
    await ensureLlmServerHost();
    expect(calls(LLM_SERVER_COMMANDS.status)).toHaveLength(1);
  });

  it("stop cancels a pending idle stop and refreshes the status", async () => {
    vi.useFakeTimers();
    invoke.mockImplementation(backend());
    const p = acquireLlmServer("t", cfg);
    await vi.advanceTimersByTimeAsync(1);
    (await p).release();
    await stopLlmServer();
    expect(llmServerUi.status?.state).toBe("stopped");
    await vi.advanceTimersByTimeAsync(LLM_SERVER_IDLE_STOP_MS * 2);
    expect(calls(LLM_SERVER_COMMANDS.stop)).toHaveLength(1);
  });

  it("maps the translation settings to a server config", () => {
    expect(
      llmServerConfigFromSettings({
        serverModel: "mt/hy-mt2-7b-q4_k_m",
        serverVariant: "cuda13",
        serverContext: 8192,
        serverGpuLayers: -1,
        serverFlashAttn: "on",
      }),
    ).toEqual({ model: "mt/hy-mt2-7b-q4_k_m", variant: "cuda13", ctxSize: 8192, gpuLayers: null, flashAttn: "on", allowOverBudget: false });
    expect(
      llmServerConfigFromSettings({ serverModel: " ", serverVariant: "rocm", serverContext: Number.NaN, serverGpuLayers: 20.4, serverFlashAttn: "x" }),
    ).toEqual({ model: "auto", variant: "auto", ctxSize: null, gpuLayers: 20, flashAttn: "auto", allowOverBudget: false });
  });

  it("recognizes Rust errors", () => {
    expect(isLlmServerError({ kind: "refused", message: "x", missing: [] })).toBe(true);
    expect(isLlmServerError({ kind: "other", message: "x" })).toBe(false);
    expect(isLlmServerError(new Error("x"))).toBe(false);
    expect(toLlmServerError({ kind: "network", message: "down" })).toEqual({ kind: "network", message: "down", missing: [] });
    expect(toLlmServerError(new Error("boom"))).toEqual({ kind: "runtime", message: "boom", missing: [] });
  });
});
