/**
 * Benchmark and autotune helpers (`tuning.ts`) and the params patch the autotuner's result applies
 * (`params.ts` `applyRustParamsPatch`). The "Benchmark this PC" spec is checked field by field against
 * the Rust spec structs (`bench/spec.rs` denies unknown fields, so a typo would fail the run).
 */
import { describe, expect, it } from "vitest";

import autotuneRs from "../../../../../src-tauri/src/services/bench/autotune.rs?raw";
import specRs from "../../../../../src-tauri/src/services/bench/spec.rs?raw";
import { VC_LEVERS } from "../contract";
import { applyRustParamsPatch, RUST_PARAM_KEYS } from "../params";
import { VoiceChangerStateSchema } from "../schema";
import {
    AUTOTUNE_PARAM_KEYS,
    autotuneParamRows,
    benchmarkSpec,
    normalizeAutotuneResult,
    normalizeLiveRecordSummary,
    summarizeBenchReport,
} from "../tuning";

const defaults = () => VoiceChangerStateSchema.parse({});

/** Field names of a `pub struct <name> { ... }` in a Rust source. */
function structFields(src: string, name: string): string[] {
    const start = src.indexOf(`pub struct ${name} {`);
    expect(start, `struct ${name}`).toBeGreaterThanOrEqual(0);
    let depth = 0;
    let end = start;
    for (let i = src.indexOf("{", start); i < src.length; i++) {
        if (src[i] === "{") depth++;
        else if (src[i] === "}" && --depth === 0) {
            end = i;
            break;
        }
    }
    return [...src.slice(start, end).matchAll(/^\s*pub (\w+):/gm)].map((m) => m[1]);
}

describe("Benchmark this PC", () => {
    it("needs a voice", () => {
        expect(benchmarkSpec(defaults())).toBeNull();
    });

    it("builds a paced spec of the selected voice with the current settings, using only Rust's field names", () => {
        const vc = {
            ...defaults(),
            mode: "rvc" as const,
            rvcSlotDir: "C:\\models\\alto",
            rvcExecutionProvider: "dml_hp" as const,
            rvcProfile: "quality" as const,
            pitchExtractor: "fusion" as const,
            precision: "fp16" as const,
            cudaGraphs: "off" as const,
            blockMs: 40,
            outputPrefillMs: 300,
            outputMaxQueueMs: 100,
            rmsMix: 0.25,
            f0Range: "singing" as const,
        };
        const spec = benchmarkSpec(vc)! as Record<string, Record<string, unknown>>;
        expect(spec.schema).toBe("sigil-bench-spec/1");
        for (const key of Object.keys(spec)) expect(structFields(specRs, "BenchSpec"), `BenchSpec.${key}`).toContain(key);
        for (const key of Object.keys(spec.stream)) expect(structFields(specRs, "StreamSpec"), `StreamSpec.${key}`).toContain(key);
        const engine = spec.engine as Record<string, unknown>;
        for (const key of Object.keys(engine).filter((k) => k !== "kind")) expect(structFields(specRs, "RvcSpec"), `RvcSpec.${key}`).toContain(key);
        const runs = spec.runs as unknown as Record<string, unknown>[];
        for (const key of Object.keys(runs[0]).filter((k) => k !== "mode")) expect(structFields(specRs, "PacedRun"), `PacedRun.${key}`).toContain(key);
        for (const key of Object.keys(spec.output)) expect(structFields(specRs, "OutputSpec"), `OutputSpec.${key}`).toContain(key);

        expect(engine).toMatchObject({
            kind: "rvc",
            slot_dir: "C:\\models\\alto",
            execution_provider: "dml_hp",
            preset: "quality",
            pitch_extractor: "fusion",
            precision: "fp16",
            f0_range: "singing",
            block_ms: 40,
        });
        expect(Object.keys(engine.levers as object)).toEqual([...VC_LEVERS]);
        expect((engine.levers as Record<string, string>).cuda_graphs).toBe("off");
        // The spec's own validation: max queue at least prefill + 5.
        expect(spec.stream.output_prefill_ms).toBe(300);
        expect(spec.stream.output_max_queue_ms).toBe(305);
        expect(spec.stream.rms_mix).toBe(0.25);
        expect(spec.stream.block_frames).toBe(512);
        expect(runs[0]).toMatchObject({ mode: "paced", name: "paced" });
    });

    it("summarizes the first run of a report", () => {
        const s = summarizeBenchReport({
            engine: { execution_provider: "cuda", execution_provider_active: "cpu", execution_provider_reason: "no cuDNN", warnings: ["w"] },
            runs: [
                {
                    status: "passed",
                    metrics: { block_ms_p50: 3, block_ms_p95: 4.5, block_ms_p99: 6, rtf_p99: 0.4, latency_ms_p50: 70, latency_ms_p99: 90, underruns: 0, dropped_ms: 0, cpu_cores: 0.2, vram_engine_mb: 512, junk: "x" },
                },
            ],
        })!;
        expect(s).toMatchObject({ status: "passed", blockP95: 4.5, rtfP99: 0.4, latencyP99: 90, vramMb: 512, epActive: "cpu", epRequested: "cuda", epReason: "no cuDNN", warnings: ["w"] });
        expect(summarizeBenchReport({ runs: [] })).toBeNull();
        expect(summarizeBenchReport({ runs: [{ status: "error", error: "slot missing", metrics: {} }] })).toMatchObject({ status: "error", error: "slot missing", blockP50: null });
    });
});

describe("autotune", () => {
    it("reads a result (additive fields optional) and refuses a reply without params", () => {
        expect(normalizeAutotuneResult(null)).toBeNull();
        expect(normalizeAutotuneResult({ rtf_p99: 0.3 })).toBeNull();
        const r = normalizeAutotuneResult({
            gpu: "RTX 4090",
            created_unix_ms: 5,
            params: { rvc_execution_provider: "cuda", block_ms: 40, partial_decode: "on" },
            rtf_p99: 0.31,
            block_ms_p95: 3.2,
            latency_ms_p50: 61,
            fidelity: 0.98,
            notes: ["n", 3],
            met_limit: true,
            fidelity_metrics: { mcd_db: 0.3, bad: "x" },
        })!;
        expect(r).toMatchObject({ gpu: "RTX 4090", driver: null, rtf_p99: 0.31, notes: ["n"], met_limit: true, fidelity_metrics: { mcd_db: 0.3 } });
        // A tune Rust found stale keeps its reasons (and the conditions it was measured with).
        expect(r.stale).toEqual([]);
        expect(r.conditions).toBeNull();
        const stale = normalizeAutotuneResult({
            params: {},
            stale: ["another voice is selected", "crossfade_ms: 20 -> 30", 7],
            conditions: { crossfade_ms: 20, preset: "low_latency" },
        })!;
        expect(stale.stale).toEqual(["another voice is selected", "crossfade_ms: 20 -> 30"]);
        expect(stale.conditions).toEqual({ crossfade_ms: 20, preset: "low_latency" });
        expect(normalizeAutotuneResult({ params: {}, conditions: [1] })!.conditions).toBeNull();
        expect(autotuneParamRows(r.params)).toEqual([
            { key: "rvc_execution_provider", value: "cuda" },
            { key: "block_ms", value: "40" },
            { key: "partial_decode", value: "on" },
        ]);
    });

    it("the keys it may apply cover what the Rust autotuner writes, and are params", () => {
        const written = [...autotuneRs.matchAll(/p\.insert\("(\w+)"/g)].map((m) => m[1]);
        expect(written.length).toBeGreaterThan(0);
        for (const k of written) expect(AUTOTUNE_PARAM_KEYS, k).toContain(k);
        for (const k of AUTOTUNE_PARAM_KEYS) expect(RUST_PARAM_KEYS, k).toContain(k);
        // Never devices, voices, the mode or the consent.
        for (const k of ["mode", "rvc_slot_dir", "slot_id", "input_device", "output_device", "rvc_consent_acknowledged", "enabled"]) {
            expect(AUTOTUNE_PARAM_KEYS).not.toContain(k);
        }
    });

    it("applies a patch to the named keys only, normalized like a hydration", () => {
        const prev = { ...defaults(), rvcProfile: "quality" as const, rvcExecutionProvider: "cuda" as const, pitchExtractor: "rmvpe" as const };
        const { next, applied } = applyRustParamsPatch(
            prev,
            {
                rvc_execution_provider: "DML_HP",
                block_ms: 60,
                precision: "fp16",
                io_binding: "on",
                mode: "classic",
                rvc_slot_dir: "C:/elsewhere",
                unknown_field: 1,
                pitch_extractor: null,
            },
            AUTOTUNE_PARAM_KEYS,
        );
        expect(applied.sort()).toEqual(["block_ms", "io_binding", "precision", "rvc_execution_provider"]);
        expect(next).toMatchObject({ rvcExecutionProvider: "dml_hp", blockMs: 60, precision: "fp16", ioBinding: "on", pitchExtractor: "rmvpe" });
        expect(next.mode).toBe(prev.mode);
        expect(next.rvcSlotDir).toBe(prev.rvcSlotDir);
        // An unset EP with a new preset shows that preset's default (the preset comes from the patch).
        const { next: n2 } = applyRustParamsPatch(prev, { rvc_preset: "streamer", rvc_execution_provider: "" }, AUTOTUNE_PARAM_KEYS);
        expect(n2).toMatchObject({ rvcProfile: "streamer", rvcExecutionProvider: "cpu" });
    });
});

describe("live record", () => {
    it("reads a summary with its report folder", () => {
        expect(normalizeLiveRecordSummary({ seconds: 3 })).toBeNull();
        expect(normalizeLiveRecordSummary({ seconds: 12.5, ticks: 300, report_dir: "C:/r", metrics: { latency_ms_p50: 70, x: "y" }, notes: ["a"] })).toEqual({
            seconds: 12.5,
            ticks: 300,
            report_dir: "C:/r",
            metrics: { latency_ms_p50: 70 },
            notes: ["a"],
        });
    });
});
