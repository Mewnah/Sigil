import { describe, expect, it } from "vitest";
import { claimsBySubsystem, deviceUsage, isVcComponent, modelLabel, policyKey, sigilUsage, type ResSnapshot } from "../format";

const snapshot = (over: Partial<ResSnapshot> = {}): ResSnapshot => ({
  budget: {
    adapter_index: 0,
    adapter_name: "NVIDIA GeForce RTX 4090",
    total_mb: 24_138,
    budget_mb: 23_370,
    process_usage_mb: 900,
    claimed_mb: 6_500,
    free_for_claims_mb: 23_370 - 1_200 - 5_300,
    external_claimed_mb: 5_300,
    device_used_mb: 9_000,
    device_free_mb: 15_564,
  },
  claims: [
    { id: 1, subsystem: "vc", label: "rvc: voice", mb: 700, external: false },
    { id: 2, subsystem: "llm", label: "llama-server: hy-mt2-7b", mb: 5_300, external: true },
    { id: 3, subsystem: "vc", label: "rvc: content encoder", mb: 500, external: false },
  ],
  vc_running: true,
  ...over,
});

describe("resources format helpers", () => {
  it("sums claims per subsystem in a stable order", () => {
    expect(claimsBySubsystem(snapshot())).toEqual([
      { subsystem: "vc", mb: 1_200, count: 2 },
      { subsystem: "llm", mb: 5_300, count: 1 },
    ]);
    expect(claimsBySubsystem(snapshot({ claims: [] }))).toEqual([]);
    expect(claimsBySubsystem(null)).toEqual([]);
  });

  it("computes what Sigil holds against the budget and device-wide use", () => {
    expect(sigilUsage(snapshot().budget)).toEqual({ usedMb: 6_500, budgetMb: 23_370, fraction: 6_500 / 23_370 });
    const over = sigilUsage({ ...snapshot().budget, free_for_claims_mb: 0 });
    expect(over.fraction).toBe(1);
    expect(sigilUsage({ ...snapshot().budget, budget_mb: 0 })).toEqual({ usedMb: 0, budgetMb: 0, fraction: 0 });
    expect(deviceUsage(snapshot().budget)).toEqual({ usedMb: 9_000, totalMb: 24_564, fraction: 9_000 / 24_564 });
    // Non-NVIDIA GPUs (and older backends) have no device-wide numbers.
    expect(deviceUsage({ ...snapshot().budget, device_used_mb: null })).toBeNull();
    expect(deviceUsage({ ...snapshot().budget, device_used_mb: undefined, device_free_mb: undefined })).toBeNull();
  });

  it("keys recommendations on policy inputs only", () => {
    const base = policyKey(snapshot());
    // Live budget numbers and claim ids/labels/order do not matter.
    const noisy = snapshot({ budget: { ...snapshot().budget, process_usage_mb: 3_000, device_used_mb: 12_000 } });
    expect(policyKey(noisy)).toBe(base);
    const reordered = snapshot({ claims: [...snapshot().claims].reverse().map((c, i) => ({ ...c, id: 50 + i, label: "x" })) });
    expect(policyKey(reordered)).toBe(base);
    // The VC flag, a claim size, a new claim or another primary GPU do.
    expect(policyKey(snapshot({ vc_running: false }))).not.toBe(base);
    expect(policyKey(snapshot({ claims: snapshot().claims.map((c) => (c.id === 2 ? { ...c, mb: 1_600 } : c)) }))).not.toBe(base);
    expect(policyKey(snapshot({ claims: [...snapshot().claims, { id: 9, subsystem: "tts", label: "", mb: 1_500 }] }))).not.toBe(base);
    expect(policyKey(snapshot({ budget: { ...snapshot().budget, adapter_index: 1 } }))).not.toBe(base);
    expect(policyKey(null)).toBe("");
    // Whether the voice changer runs on the GPU is a policy input too (unknown, CPU and GPU differ).
    const unknown = policyKey(snapshot({ vc_gpu: null }));
    expect(policyKey(snapshot())).toBe(unknown);
    const onCpu = policyKey(snapshot({ vc_gpu: false }));
    const onGpu = policyKey(snapshot({ vc_gpu: true }));
    expect(new Set([unknown, onCpu, onGpu]).size).toBe(3);
  });

  it("recognises voice-changer EP components", () => {
    for (const c of ["vc", "vc.content", "vc.pitch", "vc.onnx"]) expect(isVcComponent(c), c).toBe(true);
    for (const c of ["vcx", "tts.kitten", "stt.vad", "", null, undefined]) expect(isVcComponent(c), String(c)).toBe(false);
  });

  it("labels the recommended catalog ids", () => {
    expect(modelLabel("mt/hy-mt2-7b-q4_k_m")).toBe("Hy-MT2 7B");
    expect(modelLabel("tts/qwen3-tts-0.6b")).toBe("Qwen3-TTS 0.6B");
    expect(modelLabel("stt/unknown")).toBe("stt/unknown");
  });
});
