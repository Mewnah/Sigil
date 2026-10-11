import { describe, expect, it } from "vitest";
import { bundledTranslatorOnCpu, interimHeldOffByCpu, interimTranslationsOn } from "../interimPolicy";
import { Service_Translation_Schema, Translation_Backends, Translation_State } from "../schema";

// P7: interim translations are off by default while the bundled llama-server translator runs on the processor
// (4.0 cores next to a game against 1.5 with final lines only); a stored boolean always wins.

const state = (patch: (s: Translation_State) => void = () => {}): Translation_State => {
  const s = Service_Translation_Schema.parse({});
  patch(s);
  return s;
};
const llmOf = (patch: Partial<Translation_State["llm"]> = {}) => ({ ...Service_Translation_Schema.parse({}).llm, ...patch });
const up = (variant: string | null, gpuLayers: number | null, status: "ready" | "starting" | "restarting" | "stopped" | "failed" = "ready") => ({
  state: status,
  variant,
  gpuLayers,
});

describe("persisted switch (migration)", () => {
  it("defaults to the automatic value, also for settings saved before the key had one", () => {
    const fresh = Service_Translation_Schema.parse({});
    expect(fresh.auto.interim).toBe("auto");
    expect(fresh.llm.interim).toBe("auto");
    // A profile of an earlier build saved the slices without `interim`.
    const old = Service_Translation_Schema.parse({ backend: "auto", auto: { chain: ["llm", "nllb"] }, llm: { serverVariant: "cpu" } });
    expect(old.auto.interim).toBe("auto");
    expect(old.llm.interim).toBe("auto");
  });

  it("keeps a stored boolean as the user's choice", () => {
    const a = Service_Translation_Schema.parse({ auto: { interim: true }, llm: { interim: false } });
    expect([a.auto.interim, a.llm.interim]).toEqual([true, false]);
    const b = Service_Translation_Schema.parse({ auto: { interim: false }, llm: { interim: true } });
    expect([b.auto.interim, b.llm.interim]).toEqual([false, true]);
  });

  it("resets anything else to the automatic value and keeps the other engines' defaults", () => {
    for (const bad of ["yes", 1, 0, null, {}, "false"]) {
      const s = Service_Translation_Schema.parse({ auto: { interim: bad }, llm: { interim: bad } });
      expect(s.auto.interim).toBe("auto");
      expect(s.llm.interim).toBe("auto");
    }
    expect(Service_Translation_Schema.parse({ auto: { interim: "auto" } }).auto.interim).toBe("auto");
    const s = Service_Translation_Schema.parse({});
    expect(s.deepl.interim).toBe(false);
    expect(s.bergamot.interim).toBe(true);
    expect(s.nllb.interim).toBe(false);
  });
});

describe("bundledTranslatorOnCpu", () => {
  it("only the bundled server can be on the processor", () => {
    expect(bundledTranslatorOnCpu(llmOf({ provider: "gemini", serverVariant: "cpu" }), null)).toBe(false);
    expect(bundledTranslatorOnCpu(llmOf({ provider: "custom", serverGpuLayers: 0 }), up("cpu", 0))).toBe(false);
  });

  it("a running server decides: the cpu variant, or no GPU layers (auto resolved to the processor)", () => {
    expect(bundledTranslatorOnCpu(llmOf(), up("cpu", 0))).toBe(true);
    expect(bundledTranslatorOnCpu(llmOf(), up("vulkan", 0))).toBe(true);
    expect(bundledTranslatorOnCpu(llmOf(), up("vulkan", 99))).toBe(false);
    expect(bundledTranslatorOnCpu(llmOf(), up("cuda13", 99))).toBe(false);
    // The running server wins over the settings: variant cpu in the settings is stale when the server runs on the GPU.
    expect(bundledTranslatorOnCpu(llmOf({ serverVariant: "cpu" }), up("vulkan", 99))).toBe(false);
    expect(bundledTranslatorOnCpu(llmOf({ serverVariant: "vulkan" }), up("cpu", 0, "starting"))).toBe(true);
  });

  it("a server that is not up is judged by the settings; variant auto stays unknown (not on the processor)", () => {
    expect(bundledTranslatorOnCpu(llmOf({ serverVariant: "cpu" }), null)).toBe(true);
    expect(bundledTranslatorOnCpu(llmOf({ serverVariant: "cpu" }), up("vulkan", 99, "stopped"))).toBe(true);
    expect(bundledTranslatorOnCpu(llmOf({ serverGpuLayers: 0 }), up("cuda13", 99, "failed"))).toBe(true);
    expect(bundledTranslatorOnCpu(llmOf({ serverVariant: "auto" }), null)).toBe(false);
    expect(bundledTranslatorOnCpu(llmOf({ serverVariant: "vulkan" }), up("cpu", 0, "stopped"))).toBe(false);
  });
});

describe("interimTranslationsOn", () => {
  const llm = Translation_Backends.llm;
  const auto = Translation_Backends.auto;

  it("is off when no backend runs", () => {
    expect(interimTranslationsOn(state(), null, false)).toBe(false);
  });

  it("automatic: on, except on the processor with the bundled translator in use", () => {
    expect(interimTranslationsOn(state(), llm, false)).toBe(true);
    expect(interimTranslationsOn(state(), llm, true)).toBe(false);
    expect(interimTranslationsOn(state(), auto, false)).toBe(true);
    expect(interimTranslationsOn(state(), auto, true)).toBe(false);
  });

  it("the router only counts when its default chain or a pair rule has the bundled translator", () => {
    const noLlm = state((s) => (s.auto.chain = [Translation_Backends.deepl, Translation_Backends.nllb]));
    expect(interimTranslationsOn(noLlm, auto, true)).toBe(true);
    const rule = state((s) => {
      s.auto.chain = [Translation_Backends.deepl];
      s.auto.pairOverrides = [{ from: "", to: "ja", chain: [Translation_Backends.llm] }];
    });
    expect(interimTranslationsOn(rule, auto, true)).toBe(false);
  });

  it("a stored boolean wins either way", () => {
    expect(interimTranslationsOn(state((s) => (s.llm.interim = true)), llm, true)).toBe(true);
    expect(interimTranslationsOn(state((s) => (s.auto.interim = true)), auto, true)).toBe(true);
    expect(interimTranslationsOn(state((s) => (s.llm.interim = false)), llm, false)).toBe(false);
    expect(interimTranslationsOn(state((s) => (s.auto.interim = false)), auto, false)).toBe(false);
  });

  it("the other backends keep their own plain switch", () => {
    expect(interimTranslationsOn(state(), Translation_Backends.deepl, true)).toBe(false);
    expect(interimTranslationsOn(state(), Translation_Backends.nllb, true)).toBe(false);
    expect(interimTranslationsOn(state(), Translation_Backends.bergamot, true)).toBe(true);
    expect(interimTranslationsOn(state((s) => (s.bergamot.interim = false)), Translation_Backends.bergamot, false)).toBe(false);
  });
});

describe("interimHeldOffByCpu (the inspector's hint)", () => {
  it("shows only while the automatic value is what turns interims off", () => {
    expect(interimHeldOffByCpu(state(), Translation_Backends.llm, true)).toBe(true);
    expect(interimHeldOffByCpu(state(), Translation_Backends.llm, false)).toBe(false);
    expect(interimHeldOffByCpu(state((s) => (s.llm.interim = false)), Translation_Backends.llm, true)).toBe(false);
    expect(interimHeldOffByCpu(state((s) => (s.llm.interim = true)), Translation_Backends.llm, true)).toBe(false);
    expect(interimHeldOffByCpu(state((s) => (s.auto.chain = [Translation_Backends.nllb])), Translation_Backends.auto, true)).toBe(false);
  });
});
