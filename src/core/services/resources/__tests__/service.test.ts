import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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

const snapshot = (vcRunning: boolean, processUsage = 0, claims: unknown[] = []) => ({
  budget: {
    adapter_index: 0,
    adapter_name: "NVIDIA GeForce RTX 4060",
    total_mb: 8188,
    budget_mb: 7800,
    process_usage_mb: processUsage,
    claimed_mb: 0,
    free_for_claims_mb: 7800 - processUsage,
  },
  claims,
  vc_running: vcRunning,
});

const rec = (topic: string, choice: string) => ({ topic, choice, reason: "", alternatives: [], reason_code: "fits", device: "gpu" });

/** Backend stub: the MT choice depends on the VC flag, like the Rust policy on 8 GB. */
function backend(state: { vc: boolean; usage: number }) {
  return async (cmd: string, args?: { topic?: string }) => {
    if (cmd === "plugin:resources|get_resource_snapshot") return snapshot(state.vc, state.usage);
    if (cmd === "plugin:resources|get_recommendation") {
      if (args?.topic === "mt_llm") return rec("mt_llm", state.vc ? "mt/hy-mt2-1.8b-q4_k_m" : "mt/hy-mt2-7b-q4_k_m");
      if (args?.topic === "qwen3_tts") return rec("qwen3_tts", state.vc ? "tts/qwen3-tts-0.6b" : "tts/qwen3-tts-1.7b");
      return rec("stt_final", "stt/qwen3-asr-1.7b-q8_0");
    }
    throw new Error(`unexpected ${cmd}`);
  };
}

const recCalls = () => invoke.mock.calls.filter(([c]) => c === "plugin:resources|get_recommendation").length;

describe("Service_Resources", () => {
  beforeEach(() => {
    handlers.clear();
    invoke.mockReset();
    vi.stubGlobal("window", { Config: { isApp: () => true } });
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("loads the snapshot and recommendations, and refetches them only when a policy input changes", async () => {
    const state = { vc: false, usage: 0 };
    invoke.mockImplementation(backend(state));
    const { default: Service_Resources, RESOURCES_EVENT } = await import("../index");
    const s = new Service_Resources();
    await s.init();
    await vi.waitFor(() => expect(s.state.recommendations.mt_llm?.choice).toBe("mt/hy-mt2-7b-q4_k_m"));
    expect(s.state.recommendations.qwen3_tts?.choice).toBe("tts/qwen3-tts-1.7b");
    expect(s.state.snapshot?.budget.total_mb).toBe(8188);
    expect(recCalls()).toBe(3);

    // Budget noise (usage moved) updates the snapshot but never refetches recommendations.
    handlers.get(RESOURCES_EVENT)!({ payload: snapshot(false, 2500) });
    expect(s.state.snapshot?.budget.process_usage_mb).toBe(2500);
    await Promise.resolve();
    expect(recCalls()).toBe(3);

    // The voice changer started: the policy input changed, so every topic is asked again.
    state.vc = true;
    handlers.get(RESOURCES_EVENT)!({ payload: snapshot(true, 2500) });
    await vi.waitFor(() => expect(s.state.recommendations.mt_llm?.choice).toBe("mt/hy-mt2-1.8b-q4_k_m"));
    expect(s.state.recommendations.qwen3_tts?.choice).toBe("tts/qwen3-tts-0.6b");
    expect(recCalls()).toBe(6);
    s.dispose();
  });

  it("refreshes the snapshot when the running voice changer's models report their EP", async () => {
    const state = { vc: true, usage: 0 };
    invoke.mockImplementation(backend(state));
    const { default: Service_Resources } = await import("../index");
    const s = new Service_Resources();
    await s.init();
    await vi.waitFor(() => expect(s.state.snapshot?.vc_running).toBe(true));
    const snapCalls = () => invoke.mock.calls.filter(([c]) => c === "plugin:resources|get_resource_snapshot").length;
    const before = snapCalls();
    const outcome = handlers.get("diagnostics:ep_outcome")!;
    // Other subsystems' sessions do not matter.
    outcome({ payload: { component: "tts.kitten", requested: "cuda", active: "cuda", reason: null, at_unix_ms: 1 } });
    expect(snapCalls()).toBe(before);
    // A voice-changer session does (Rust infers `vc_gpu` from it).
    outcome({ payload: { component: "vc.content", requested: "cuda", active: "cuda", reason: null, at_unix_ms: 2 } });
    await vi.waitFor(() => expect(snapCalls()).toBe(before + 1));
    // Not while the voice changer is off.
    handlers.get("resources:changed")!({ payload: snapshot(false) });
    outcome({ payload: { component: "vc.pitch", requested: "cuda", active: "cpu", reason: "x", at_unix_ms: 3 } });
    expect(snapCalls()).toBe(before + 1);
    s.dispose();
  });

  it("coalesces recommendation requests made while a round is running", async () => {
    const state = { vc: false, usage: 0 };
    const impl = backend(state);
    const held: Array<() => void> = [];
    let first = true;
    invoke.mockImplementation(async (cmd: string, args?: { topic?: string }) => {
      if (cmd === "plugin:resources|get_recommendation" && first) {
        await new Promise<void>((r) => held.push(r));
      }
      return impl(cmd, args);
    });
    const gate = () => held.splice(0).forEach((r) => r());
    const { default: Service_Resources } = await import("../index");
    const s = new Service_Resources();
    const a = s.refreshRecommendations();
    first = false;
    const b = s.refreshRecommendations();
    const c = s.refreshRecommendations();
    expect(b).toBe(a);
    expect(c).toBe(a);
    state.vc = true;
    gate();
    await a;
    // One extra round for the two requests that arrived during the first: 3 + 3 calls, last input wins.
    expect(recCalls()).toBe(6);
    expect(s.state.recommendations.mt_llm?.choice).toBe("mt/hy-mt2-1.8b-q4_k_m");
    expect(s.state.recommendationsLoading).toBe(false);
  });

  it("polls the live budget only while watched and shares one timer", async () => {
    vi.useFakeTimers();
    const state = { vc: false, usage: 0 };
    invoke.mockImplementation(backend(state));
    const { default: Service_Resources } = await import("../index");
    const s = new Service_Resources();
    const snapCalls = () => invoke.mock.calls.filter(([c]) => c === "plugin:resources|get_resource_snapshot").length;

    const stopA = s.watch(1000);
    const stopB = s.watch(1000);
    expect(snapCalls()).toBe(1); // immediate refresh, once
    await vi.advanceTimersByTimeAsync(3000);
    expect(snapCalls()).toBe(4);
    stopA();
    stopA(); // idempotent
    await vi.advanceTimersByTimeAsync(1000);
    expect(snapCalls()).toBe(5); // still watched by B
    stopB();
    await vi.advanceTimersByTimeAsync(5000);
    expect(snapCalls()).toBe(5);
  });

  it("never rejects on backend errors", async () => {
    invoke.mockRejectedValue("the resource governor is unavailable");
    const { default: Service_Resources } = await import("../index");
    const s = new Service_Resources();
    await s.init();
    await s.refreshSnapshot();
    await s.refreshRecommendations();
    expect(s.state.error).toBe("the resource governor is unavailable");
    expect(s.state.snapshot).toBeNull();
    s.dispose();
  });
});
