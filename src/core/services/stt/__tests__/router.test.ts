import { describe, expect, it } from "vitest";
import type { RustSpeechEngineSpec, RustSpeechRoutePlan, RustSpeechTier } from "@/types";
import fixture from "../../../../../src-tauri/test-fixtures/routing.json";
import routingDocument from "../../../../../src-tauri/model-catalog/stt-routing.json";
import {
  GOVERNOR_REASONS,
  isAutoLanguage,
  isTwoPassPlan,
  normalizeLanguage,
  parseRoutingTable,
  planHasEngine,
  planRoles,
  planRoute,
  referencedIds,
  requiredModelIds,
  ROUTE_NOTES,
  routeContext,
  routedLanguages,
  ROUTING_TABLE,
  sameLanguage,
  SMART_TURN_MODEL_ID,
  specOnGpu,
  STREAM_LANGUAGES_OPTION,
  type RouteContext,
  type RoutingTable,
} from "../router";

// The TS mirror of speech::router (P2-C) runs the shared fixture the Rust test runs
// (`speech::router::tests::shared_fixture_cases_hold`), with the same expectation keys.

const NEMOTRON = "stt/nemotron-3.5-streaming-560ms-int8";
const PARAKEET_ULTRA = "stt/parakeet-ultra-int8";
const QWEN3_0_6B = "stt/qwen3-asr-0.6b-int8";
const SENSE_VOICE = "stt/sense-voice-small-int8";
const QWEN3_Q8 = "stt/qwen3-asr-1.7b-q8_0";
const QWEN3_BF16 = "stt/qwen3-asr-1.7b-bf16";
const COHERE = "stt/cohere-transcribe-q4_k";
const LID = "stt/lid-whisper-base-int8";

function table(): RoutingTable {
  expect(ROUTING_TABLE).not.toBeNull();
  return ROUTING_TABLE as RoutingTable;
}

/** Every model a plan names (specs and the id list). */
function planModels(plan: RustSpeechRoutePlan): string[] {
  return [plan.streaming, plan.final, plan.fallback, plan.languageId]
    .filter((s): s is RustSpeechEngineSpec => !!s)
    .map((s) => s.model ?? "")
    .concat(plan.models);
}

const model = (spec: RustSpeechEngineSpec | null) => spec?.model ?? null;

type Expect = Record<string, unknown>;
interface FixtureCase {
  name: string;
  request: { language?: string; tier?: RustSpeechTier; twoPass?: boolean };
  context: Partial<RouteContext>;
  expect: Expect;
}

describe("shared routing fixture (src-tauri/test-fixtures/routing.json)", () => {
  const cases = (fixture as { version: number; cases: FixtureCase[] }).cases;

  it("has the frozen format and covers every rule", () => {
    expect((fixture as { version: number }).version).toBe(1);
    expect(cases.length).toBeGreaterThanOrEqual(30);
    expect(new Set(cases.map((c) => c.name)).size).toBe(cases.length);
  });

  for (const c of cases) {
    it(c.name, () => {
      const plan = planRoute(c.request, c.context);
      const models = planModels(plan);
      const optional = (key: string, got: string | null) => {
        const want = c.expect[key];
        if (want === null) expect(got, key).toBeNull();
        else {
          expect(typeof want, `${key} must be a string or null`).toBe("string");
          expect(got, key).toBe(want);
        }
      };
      for (const key of Object.keys(c.expect)) {
        const want = c.expect[key];
        switch (key) {
          case "language":
            expect(plan.language).toBe(want);
            break;
          case "tier":
            expect(plan.tier).toBe(want);
            break;
          case "streamingModel":
            optional(key, model(plan.streaming));
            break;
          case "finalModel":
            optional(key, model(plan.final));
            break;
          case "fallbackModel":
            optional(key, model(plan.fallback));
            break;
          case "languageIdModel":
            optional(key, model(plan.languageId));
            break;
          case "streamingLanguage": {
            const lang = plan.streaming?.options?.language;
            optional(key, typeof lang === "string" ? lang : null);
            break;
          }
          case "neverModels":
            for (const never of want as string[]) expect(models, `plan uses ${never}`).not.toContain(never);
            break;
          case "notesInclude":
            for (const n of want as string[]) expect(plan.notes, `note ${n}`).toContain(n);
            break;
          case "missingInclude":
            for (const id of want as string[]) expect(plan.missing, `${id} missing`).toContain(id);
            break;
          default:
            // Rust rejects unknown keys too: a new key means the mirror needs the same check.
            throw new Error(`unknown expect key ${key}`);
        }
      }
    });
  }
});

describe("language normalization (mirror of speech::lang)", () => {
  it("normalizes common tags", () => {
    expect(normalizeLanguage("en-US")).toBe("en");
    expect(normalizeLanguage("zh-Hant-TW")).toBe("zh");
    expect(normalizeLanguage("pt_BR")).toBe("pt");
    expect(normalizeLanguage("yue")).toBe("yue");
    expect(normalizeLanguage("zh-yue")).toBe("yue");
    expect(normalizeLanguage(" JA ")).toBe("ja");
    expect(normalizeLanguage("auto")).toBeNull();
    expect(normalizeLanguage("")).toBeNull();
    expect(normalizeLanguage(undefined)).toBeNull();
    expect(normalizeLanguage("e")).toBeNull();
    expect(normalizeLanguage("12-x")).toBeNull();
    expect(isAutoLanguage("Auto") && isAutoLanguage(" ") && !isAutoLanguage("en")).toBe(true);
  });

  it("maps Cantonese, aliases and rejects private use", () => {
    expect(normalizeLanguage("zh-Hant-yue")).toBe("yue");
    expect(normalizeLanguage("zh-HK")).toBe("zh");
    expect(normalizeLanguage("yue-Hant-HK")).toBe("yue");
    expect(normalizeLanguage("iw")).toBe("he");
    expect(normalizeLanguage("in-ID")).toBe("id");
    expect(normalizeLanguage("jpn")).toBe("ja");
    expect(normalizeLanguage("cmn-Hans-CN")).toBe("zh");
    expect(normalizeLanguage("tgl")).toBe("fil");
    expect(normalizeLanguage("fil")).toBe("fil");
    expect(normalizeLanguage("nb-NO")).toBe("nb");
    expect(normalizeLanguage("x-klingon")).toBeNull();
    expect(normalizeLanguage("i-klingon")).toBeNull();
    expect(normalizeLanguage("english")).toBeNull();
    expect(sameLanguage("en-GB", "EN") && sameLanguage("zh-yue", "yue")).toBe(true);
    expect(sameLanguage("zh", "yue") || sameLanguage("auto", "auto")).toBe(false);
  });
});

describe("routing table", () => {
  it("parses the shipped table and every row language is normalized and unique", () => {
    const t = table();
    const seen = new Map<string, string>();
    for (const row of t.routes)
      for (const l of row.languages) {
        expect(seen.get(l), `language ${l} in two rows`).toBeUndefined();
        seen.set(l, row.name);
      }
    expect(routedLanguages()).toContain("ja");
    expect(routedLanguages()).not.toContain("auto");
    expect(routedLanguages()).not.toContain("*");
    expect(referencedIds(t)).toEqual(expect.arrayContaining([NEMOTRON, "runtime/sherpa-onnx", "runtime/nvidia-cuda13", LID]));
  });

  it("refuses broken documents with a message", () => {
    expect(() => parseRoutingTable({})).toThrow(/routing/);
    expect(() => parseRoutingTable({ routing: { version: 2 } })).toThrow(/version 2/);
    const doc = JSON.parse(JSON.stringify(routingDocument));
    doc.routing.routes = doc.routing.routes.filter((r: { languages: string[] }) => !r.languages.includes("*"));
    expect(() => parseRoutingTable(doc)).toThrow(/'auto' and '\*'/);
    const gpuInCpu = JSON.parse(JSON.stringify(routingDocument));
    gpuInCpu.routing.routes[0].cpuFinal = [QWEN3_Q8];
    expect(() => parseRoutingTable(gpuInCpu)).toThrow(/GPU model/);
    const unknown = JSON.parse(JSON.stringify(routingDocument));
    unknown.routing.routes[0].streaming = ["stt/nope"];
    expect(() => parseRoutingTable(unknown)).toThrow(/unknown model/);
    const badLang = JSON.parse(JSON.stringify(routingDocument));
    badLang.routing.routes[0].languages = ["en-US"];
    expect(() => parseRoutingTable(badLang)).toThrow(/non-normalized/);
  });

  it("encodes the plan's rules as data", () => {
    const t = table();
    const row = (l: string) => t.routes.find((r) => r.languages.includes(l)) ?? t.routes.find((r) => r.languages.includes("*"))!;
    for (const l of ["zh", "yue"]) {
      expect(row(l).cpuFinal[0]).toBe(SENSE_VOICE);
      expect(row(l).streaming).toHaveLength(0);
    }
    for (const l of ["ja", "ko", "ru"]) expect(row(l).cpuFinal[0]).toBe(QWEN3_0_6B);
    const cohere = t.models[COHERE];
    for (const l of ["ja", "zh", "ko", "ru", "yue"]) {
      expect(cohere.languages).not.toContain(l);
      expect(cohere.neverLanguages).toContain(l);
    }
    expect(cohere.requiresLanguage).toBe(true);
    expect(t.models[NEMOTRON].needsLanguage).toBe(true);
    expect(t.languageId).toBe(LID);
  });
});

describe("planRoute invariants on every language × tier × PC", () => {
  const t = table();
  const pc = (vramTotal: number, vramAvailable: number, cores: number, extra: Partial<RouteContext> = {}): RouteContext =>
    routeContext({
      gpu: vramTotal > 0,
      gpuVariant: vramTotal > 0 ? "cuda13" : null,
      vramAvailableMb: vramAvailable,
      vramTotalMb: vramTotal,
      physicalCores: cores,
      installed: referencedIds(t),
      gpuReason: vramTotal === 0 ? "no_gpu" : null,
      ...extra,
    });
  const pcs: [string, RouteContext][] = [
    ["cpu 8 cores", pc(0, 0, 8)],
    ["cpu 4 cores", pc(0, 0, 4)],
    ["cpu unknown cores", pc(0, 0, 0)],
    ["8 GB", pc(8_188, 3_600, 6)],
    ["12 GB", pc(12_288, 6_600, 8)],
    ["16 GB", pc(16_376, 11_400, 8)],
    ["24 GB", pc(24_576, 15_000, 16)],
    ["vc on cpu", pc(0, 0, 8, { vcRunning: true, vcGpu: false })],
    ["vc on gpu", pc(12_288, 5_400, 8, { vcRunning: true, vcGpu: true })],
    ["no transcribe pack", pc(24_576, 15_000, 16, { unavailable: ["runtime/sigil-transcribe-cuda13"] })],
    ["cpu, vc device unknown", pc(0, 0, 8, { vcRunning: true })],
    ["24 GB, vc device unknown", pc(24_576, 15_000, 16, { vcRunning: true })],
    ["nothing installed", pc(24_576, 15_000, 16, { installed: [] })],
  ];
  const languages = [...routedLanguages(), "auto", "sw", "he", "zh-HK", "pt-BR", "zh-yue", "", "xx-invalid-1"];
  const isGpu = (s: RustSpeechEngineSpec) => !!t.models[s.model ?? ""]?.gpu;
  const known = new Set<string>([...Object.values(ROUTE_NOTES), ...GOVERNOR_REASONS]);

  it("keeps the never-route rules and the bookkeeping", () => {
    for (const [pcName, ctx] of pcs)
      for (const l of languages)
        for (const tier of ["auto", "cpu", "gpu"] as const)
          for (const twoPass of [true, false]) {
            const plan = planRoute({ language: l, tier, twoPass }, ctx);
            const what = `${pcName} / ${JSON.stringify(l)} / ${tier} / two-pass ${twoPass}`;
            expect(plan, `deterministic: ${what}`).toEqual(planRoute({ language: l, tier, twoPass }, ctx));
            const norm = normalizeLanguage(l);
            expect(plan.language, what).toBe(norm ?? "auto");
            expect(planHasEngine(plan), `an engine: ${what}`).toBe(true);
            expect(plan.tier, what).not.toBe("auto");
            const models = planModels(plan);
            if (norm === null || ["ja", "zh", "ko", "ru", "yue"].includes(norm))
              expect(models.some((m) => m.startsWith("stt/cohere-")), what).toBe(false);
            // Auto streams carry their languages (never zh / yue); explicit ones carry none.
            const limit = plan.streaming?.options?.[STREAM_LANGUAGES_OPTION] as string[] | undefined;
            if (norm === null && plan.streaming) {
              expect(limit?.length, what).toBeGreaterThan(0);
              for (const l of limit ?? []) {
                expect(["zh", "yue"], what).not.toContain(l);
                const row = t.routes.find((r) => r.languages.includes(l)) ?? t.routes.find((r) => r.languages.includes("*"))!;
                expect(row.streaming, `${l}: ${what}`).toContain(plan.streaming.model);
              }
            } else expect(limit, what).toBeUndefined();
            // A second CPU engine (CPU two-pass, or a GPU final's CPU fallback next to the stream) only where the CPU
            // may run two.
            const cpuTwo =
              (ctx.physicalCores === 0 || ctx.physicalCores >= t.cpuTwoPassMinCores) && !(ctx.vcRunning && ctx.vcGpu !== true);
            if (plan.streaming && (plan.fallback || (plan.tier === "cpu" && plan.final))) expect(cpuTwo, what).toBe(true);
            if (plan.streaming?.model?.startsWith("stt/nemotron-")) {
              if (norm) expect(plan.streaming.options?.language, what).toBe(norm);
              else expect(t.models[plan.final?.model ?? ""]?.detectsLanguage || !!plan.languageId, what).toBe(true);
            }
            for (const s of [plan.streaming, plan.fallback, plan.languageId])
              if (s) expect(isGpu(s) || specOnGpu(s), what).toBe(false);
            if (plan.tier === "gpu") {
              expect(plan.final && isGpu(plan.final) && plan.final.variant === "cuda13" && plan.vramMb > 0, what).toBe(true);
              expect(plan.vramMb, what).toBeLessThanOrEqual(ctx.vramAvailableMb);
              expect(plan.models, what).toContain("runtime/sigil-transcribe-cuda13");
              expect(tier, what).not.toBe("cpu");
            } else {
              expect(!plan.final || !isGpu(plan.final), what).toBe(true);
              expect(plan.vramMb, what).toBe(0);
            }
            for (const s of [plan.streaming, plan.final, plan.fallback, plan.languageId])
              if (s) expect(plan.models, what).toContain(s.model);
            for (const m of plan.missing) expect(plan.models.includes(m) && !ctx.installed.includes(m), what).toBe(true);
            for (const m of plan.models) expect(ctx.unavailable, what).not.toContain(m);
            if (ctx.installed.length === 0) expect(plan.missing, what).toEqual(plan.models);
            if (plan.tier === "cpu") {
              const fin = model(plan.final);
              if (norm === "zh" || norm === "yue") expect(fin, what).toBe(SENSE_VOICE);
              if (norm === "ja" || norm === "ko" || norm === "ru") expect(fin, what).toBe(QWEN3_0_6B);
              if (norm && norm !== "en" && t.models[PARAKEET_ULTRA].languages.includes(norm) && norm !== "ru")
                expect(fin, what).toBe(PARAKEET_ULTRA);
            }
            if (norm === null) expect(plan.notes, what).toContain(ROUTE_NOTES.autoLanguage);
            for (const n of plan.notes) expect(known.has(n), `unknown note ${n}: ${what}`).toBe(true);
          }
  });

  it("lists runtimes and missing ids in order", () => {
    const ctx = pc(12_288, 6_600, 8, { installed: [NEMOTRON, "runtime/sherpa-onnx"] });
    const plan = planRoute({ language: "ja", tier: "gpu", twoPass: true }, ctx);
    expect(plan.models).toEqual([
      NEMOTRON,
      "runtime/sherpa-onnx",
      QWEN3_Q8,
      "runtime/sigil-transcribe-cuda13",
      "runtime/nvidia-cuda13",
      QWEN3_0_6B,
    ]);
    expect(plan.missing).toEqual([QWEN3_Q8, "runtime/sigil-transcribe-cuda13", "runtime/nvidia-cuda13", QWEN3_0_6B]);
    expect(plan.vramMb).toBe(3500);
    expect(plan.final).toMatchObject({ engine: "transcribe", variant: "cuda13", options: null });
    expect(plan.fallback).toMatchObject({ engine: "sherpa-offline", model: QWEN3_0_6B, options: { language: "ja" } });
    expect(model(planRoute({ language: "ja", tier: "gpu", twoPass: true }, pc(16_376, 11_400, 8)).final)).toBe(QWEN3_BF16);
  });

  it("uses the language-ID engine when an auto final cannot detect the language", () => {
    const doc = {
      routing: {
        version: 1,
        cpuTwoPassMinCores: 6,
        vcShareGpuMinTotalMb: 15000,
        gpuRuntimes: {},
        languageId: LID,
        models: {
          [NEMOTRON]: { engine: "sherpa-online", languages: [], needsLanguage: true },
          [PARAKEET_ULTRA]: { engine: "sherpa-offline", languages: [] },
          [LID]: { engine: "sherpa-offline", languages: [], detectsLanguage: true },
        },
        routes: [
          { name: "auto", languages: ["auto"], streaming: [NEMOTRON], cpuFinal: [PARAKEET_ULTRA] },
          // The `*` row streams with Nemotron too: auto mode streams only where some language's own row does.
          { name: "other", languages: ["*"], streaming: [NEMOTRON], cpuFinal: [PARAKEET_ULTRA] },
        ],
      },
    };
    const lidTable = parseRoutingTable(doc);
    const plan = planRoute({ language: "auto" }, pc(0, 0, 8), lidTable);
    expect(model(plan.languageId)).toBe(LID);
    expect(model(plan.streaming)).toBe(NEMOTRON);
    expect(plan.streaming?.options).toBeNull();
    expect(plan.notes).toContain(ROUTE_NOTES.languageId);
    const noLid = planRoute({ language: "auto" }, pc(0, 0, 8, { unavailable: [LID] }), lidTable);
    expect(noLid.streaming).toBeNull();
    expect(noLid.languageId).toBeNull();
    expect(model(noLid.final)).toBe(PARAKEET_ULTRA);
    expect(noLid.notes).toContain(ROUTE_NOTES.nemotronNeedsLanguage);
  });

  it("overrides a table that breaks a never-route rule", () => {
    const bad = parseRoutingTable({
      routing: {
        version: 1,
        cpuTwoPassMinCores: 6,
        vcShareGpuMinTotalMb: 15000,
        gpuRuntimes: { cuda13: ["runtime/sigil-transcribe-cuda13"] },
        models: {
          [COHERE]: { engine: "transcribe", gpu: true, languages: [], vramMb: 1000 },
          [NEMOTRON]: { engine: "sherpa-online", languages: [] },
          [QWEN3_0_6B]: { engine: "sherpa-offline", languages: [] },
        },
        routes: [
          { name: "ja", languages: ["ja"], streaming: [], cpuFinal: [QWEN3_0_6B], gpuFinal: [COHERE] },
          { name: "auto", languages: ["auto"], streaming: [NEMOTRON], streamFinal: true, cpuFinal: [], gpuFinal: [COHERE] },
          { name: "other", languages: ["*"], streaming: [NEMOTRON], streamFinal: true },
        ],
      },
    });
    const ctx = pc(24_576, 15_000, 8, { installed: [] });
    const ja = planRoute({ language: "ja", tier: "gpu" }, ctx, bad);
    expect(model(ja.final)).toBe(QWEN3_0_6B);
    expect(ja.tier).toBe("cpu");
    expect(ja.notes).toContain(ROUTE_NOTES.neverRoute);
    const auto = planRoute({ language: "auto", tier: "gpu" }, ctx, bad);
    expect(auto.streaming === null && auto.final === null).toBe(true);
    expect(auto.notes).toEqual(expect.arrayContaining([ROUTE_NOTES.nemotronNeedsLanguage, ROUTE_NOTES.noEngine]));
    // An explicit language reaches Nemotron even when the table forgot `needsLanguage`.
    expect(planRoute({ language: "de", tier: "cpu" }, ctx, bad).streaming?.options).toEqual({ language: "de" });
  });

  it("streams in auto mode only for languages whose own route streams (never zh / yue)", () => {
    const plan = planRoute({ language: "auto", tier: "auto", twoPass: true }, pc(0, 0, 8));
    expect(model(plan.streaming)).toBe(NEMOTRON);
    expect(plan.streaming?.options?.language).toBeUndefined();
    const expected = [...new Set(t.models[NEMOTRON].languages.filter((l) => l !== "zh" && l !== "yue"))].sort();
    expect(plan.streaming?.options?.[STREAM_LANGUAGES_OPTION]).toEqual(expected);
    expect(planRoute({ language: "ja", tier: "auto", twoPass: true }, pc(0, 0, 8)).streaming?.options).toEqual({ language: "ja" });
  });

  it("returns an empty plan without a table", () => {
    const plan = planRoute({ language: "ja-JP" }, pc(0, 0, 8), null);
    expect(plan).toMatchObject({ language: "ja", tier: "cpu", streaming: null, final: null, notes: [ROUTE_NOTES.routingTableInvalid] });
  });
});

describe("plan helpers", () => {
  const plan = planRoute({ language: "en", tier: "auto", twoPass: true }, { physicalCores: 8, gpuReason: "no_gpu" });

  it("describes the passes and roles", () => {
    expect(isTwoPassPlan(plan)).toBe(true);
    expect(planRoles(plan).map((r) => r.role)).toEqual(["streaming", "final"]);
    expect(isTwoPassPlan(planRoute({ language: "en", twoPass: false }, { physicalCores: 8 }))).toBe(false);
    expect(planRoles(null)).toEqual([]);
  });

  it("adds Smart Turn to the required ids only when it is on", () => {
    expect(requiredModelIds(plan, false)).toEqual(plan.models);
    expect(requiredModelIds(plan, true)).toEqual([...plan.models, SMART_TURN_MODEL_ID]);
    expect(requiredModelIds(null, true)).toEqual([SMART_TURN_MODEL_ID]);
  });

  it("recognizes GPU variants", () => {
    expect(specOnGpu({ variant: "cuda13" }) && specOnGpu({ variant: "vulkan" })).toBe(true);
    expect(specOnGpu({ variant: null }) || specOnGpu({ variant: "cpu-avx2" }) || specOnGpu(null)).toBe(false);
  });

  it("fills a context the way Rust's serde defaults do", () => {
    expect(routeContext({ gpu: true, vramAvailableMb: -5, installed: ["a", 3 as unknown as string] })).toMatchObject({
      gpu: true,
      gpuVariant: null,
      vramAvailableMb: 0,
      installed: ["a"],
      vcGpu: null,
      unavailable: [],
    });
  });
});
