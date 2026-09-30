import { describe, expect, it } from "vitest";
import type { RustLlmServerPlan } from "@/types";
import {
  CUSTOM_MODEL_OPTION,
  canOverride,
  fileName,
  formatMb,
  groupModels,
  isModelPath,
  layersText,
  modelSelectValue,
  offersDownloads,
  parseContext,
  parseGpuLayers,
  planReasonKey,
  stateKey,
} from "../format";
import type { LlmModelInfoExt } from "../index";

const model = (id: string, extra: Partial<LlmModelInfoExt> = {}): LlmModelInfoExt => ({
  id,
  title: id,
  promptFormat: "generic",
  sizeBytes: 1,
  vramMb: 1,
  installed: false,
  specialist: false,
  languages: [],
  recommended: false,
  ...extra,
});

const plan = (extra: Partial<RustLlmServerPlan>): RustLlmServerPlan => ({
  model: "m",
  variant: "vulkan",
  gpuLayers: 999,
  ctxSize: 4096,
  flashAttn: "auto",
  device: "gpu",
  vramMb: 1600,
  availableMb: 4532,
  allowed: true,
  reasonCode: "fits",
  reason: "",
  missing: [],
  ...extra,
});

describe("llama-server panel helpers", () => {
  it("tells file paths from catalog ids like the Rust side", () => {
    expect(isModelPath("C:\\Models\\x.gguf")).toBe(true);
    expect(isModelPath("D:/m/x.GGUF")).toBe(true);
    expect(isModelPath("\\\\nas\\share\\m")).toBe(true);
    expect(isModelPath("mt/hy-mt2-7b-q4_k_m")).toBe(false);
    expect(isModelPath("auto")).toBe(false);
    expect(modelSelectValue("C:\\m\\x.gguf")).toBe(CUSTOM_MODEL_OPTION);
    expect(modelSelectValue("")).toBe("auto");
    expect(modelSelectValue("mt/hy-mt2-7b-q4_k_m")).toBe("mt/hy-mt2-7b-q4_k_m");
    expect(fileName("C:\\m\\My Model.gguf")).toBe("My Model.gguf");
    expect(fileName("/x/y.gguf")).toBe("y.gguf");
  });

  it("groups specialists apart", () => {
    const g = groupModels([model("a"), model("b", { specialist: true }), model("c", { role: "specialist" }), model("d", { role: "ab" })]);
    expect(g.general.map((m) => m.id)).toEqual(["a", "d"]);
    expect(g.specialists.map((m) => m.id)).toEqual(["b", "c"]);
  });

  it("keys reasons and states", () => {
    expect(planReasonKey("vc_running")).toBe("transl.llm_server_reason_vc_running");
    expect(planReasonKey("needs_runtime_update")).toBe("transl.llm_server_reason_needs_runtime_update");
    expect(planReasonKey("something_new")).toBeNull();
    expect(planReasonKey(undefined)).toBeNull();
    expect(stateKey("ready")).toBe("transl.llm_server_state_ready");
    expect(stateKey(null)).toBe("transl.llm_server_state_stopped");
  });

  it("formats and parses the numbers", () => {
    expect(formatMb(5300)).toBe("5.2 GB");
    expect(formatMb(700)).toBe("700 MB");
    expect(formatMb(undefined)).toBe("?");
    expect(layersText(999)).toBe("all");
    expect(layersText(0)).toBe("none");
    expect(layersText(12)).toBe(12);
    expect(parseGpuLayers("")).toBe(-1);
    expect(parseGpuLayers("auto")).toBe(-1);
    expect(parseGpuLayers("-3")).toBe(-1);
    expect(parseGpuLayers("20")).toBe(20);
    expect(parseGpuLayers("5000")).toBe(999);
    expect(parseContext("8192")).toBe(8192);
    expect(parseContext("100")).toBe(512);
    expect(parseContext("1e9")).toBe(131_072);
    expect(parseContext("abc")).toBeNull();
  });

  it("offers an override only for budget refusals", () => {
    expect(canOverride(plan({ allowed: false, reasonCode: "vc_running" }))).toBe(true);
    expect(canOverride(plan({ allowed: false, reasonCode: "needs_runtime_update" }))).toBe(false);
    expect(canOverride(plan({ allowed: false, reasonCode: "no_gpu_runtime" }))).toBe(false);
    expect(canOverride(plan({}))).toBe(false);
    expect(canOverride(null)).toBe(false);
  });

  it("offers downloads only for plans that can run", () => {
    const missing = ["mt/hy-mt2-7b-q4_k_m", "runtime/llama-server-vulkan"];
    expect(offersDownloads(plan({ allowed: true, missing }))).toBe(true);
    expect(offersDownloads(plan({ allowed: false, reasonCode: "vc_running", missing }))).toBe(true);
    expect(offersDownloads(plan({ allowed: false, reasonCode: "no_gpu_runtime", missing: ["runtime/llama-server-cuda13"] }))).toBe(false);
    expect(offersDownloads(plan({ allowed: false, reasonCode: "needs_runtime_update", missing: ["mt/hy-mt2-1.8b-1.25bit"] }))).toBe(false);
    expect(offersDownloads(plan({ allowed: true, missing: [] }))).toBe(false);
    expect(offersDownloads(null)).toBe(false);
  });
});
