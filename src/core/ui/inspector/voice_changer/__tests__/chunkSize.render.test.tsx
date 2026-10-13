/**
 * Server-side render tests of the Chunk size control against a mocked host: "Auto (about N ms on this PC)" first (exact
 * once the engine reports its chunk), then the sizes the engine plans; the short-chunk warnings; the running chunk; the
 * control disabled with a reason where the setting does not reach the voice; and that it sits in the main RVC settings,
 * not inside a collapsed section.
 */
import { renderToStaticMarkup } from "react-dom/server";
import { proxy } from "valtio";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("react-i18next", () => ({
    useTranslation: () => ({ t: (key: string, opts?: Record<string, unknown>) => (opts ? `${key}${JSON.stringify(opts)}` : key) }),
}));
vi.mock("@tauri-apps/api/core", () => ({ isTauri: () => true, invoke: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: vi.fn() }));
vi.mock("react-toastify", () => ({ toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warning: vi.fn(), success: vi.fn() }) }));
vi.mock("@/core/ui/inspector/components", () => ({ default: { SubHeader: ({ children }: { children: unknown }) => children } }));
vi.mock("@/core/ui/inspector/components/input", () => ({ InputSelect: () => null, InputContainer: () => null }));
vi.mock("@/core/ui/dropdown/Tooltip", () => ({ default: ({ children }: { children: unknown }) => children }));
vi.mock("@/core/ui/inspector/components/ModelRequirement", () => ({ ModelRequirement: () => null }));
vi.mock("@/core/services/voice_library", () => ({ voiceLibraryUi: {}, ensureVoiceLibraryHost: vi.fn(), readClip: vi.fn() }));
vi.mock("../VoiceLibrary", () => ({ VoiceLibrary: () => <div data-mock="voice-library" /> }));
vi.mock("../FamilySection", () => ({ FamilySection: () => <div data-mock="family-section" /> }));
vi.mock("../BenchmarkPanel", () => ({ BenchmarkPanel: () => <div data-mock="benchmark" /> }));
vi.mock("../EngineSection", () => ({
    EngineSection: () => <div data-mock="engine-section" />,
    LookaheadMsSelect: () => <div data-mock="lookahead" />,
}));

import { normalizeCapabilities, type VcSlotInfo } from "@/core/services/voice_changer/ipc";
import type { VoiceChangerMetrics } from "@/core/services/voice_changer/metrics";
import { VoiceChangerStateSchema, type VoiceChangerState } from "@/core/services/voice_changer/schema";
import { ChunkSizeSelect } from "../ChunkSize";
import { VcMetricsPanel } from "../Metrics";
import { RvcSection } from "../RvcSection";

function host(over: { vc?: Partial<VoiceChangerState>; cores?: number | null; running?: boolean; slots?: VcSlotInfo[]; families?: string[] } = {}) {
    const voiceChanger = {
        ...VoiceChangerStateSchema.parse({}),
        mode: "rvc" as const,
        rvcProfile: "streamer" as const,
        rvcExecutionProvider: "cpu" as const,
        rvcSlotDir: "C:/m/v",
        ...over.vc,
    };
    const state = proxy({
        voiceChanger,
        isRunning: over.running ?? false,
        hydrated: true,
        rvcConsentPending: false,
        capabilities: {
            ...normalizeCapabilities({ rvc_execution_providers: ["cpu", "cuda", "dml"] }),
            families: normalizeCapabilities({
                families: (over.families ?? ["ddsp_svc", "sigil_zs", "meanvc", "sovits", "sidecar_v2"]).map((family) => ({
                    family,
                    available: true,
                    execution_providers: ["cpu", "cuda", "dml"],
                    controls: ["pitch"],
                })),
            }).families,
        },
        capabilitiesLoaded: true,
        trt: [],
    });
    const library = {
        state: proxy({ slots: over.slots ?? [], listing: false, listError: null, slotStore: true, jobs: [], busy: {} as Record<string, string>, optionsReloadPending: {} }),
    };
    const sysinfo = { state: proxy({ info: over.cores === null ? null : { cpu: { physical_cores: over.cores ?? 8 }, gpus: [] } }) };
    vi.stubGlobal("window", { ApiServer: { voiceChanger: { state, library, update: vi.fn(), applyRvcProfile: vi.fn(), commitHarnessParams: vi.fn() }, sysinfo } });
    return state;
}


const voice = (kind: "template" | "legacy" | "sovits" | "ddsp_svc" | "sigil_zs" | "meanvc" | "sidecar_v2", options: Record<string, unknown> | null = null): VcSlotInfo => {
    const family = kind === "template" || kind === "legacy" ? null : kind;
    return {
        id: "v",
        name: "V",
        path: "C:/m/v",
        format: family ? normalizeCapabilities({ families: [{ family, available: true }] }).families[0].format : kind === "legacy" ? "rvc" : "sigil-rvc-1",
        template: null,
        version: null,
        sample_rate: 44100,
        f0: null,
        embedder: null,
        vocoder: null,
        has_index: false,
        speaker_count: null,
        fp16_ok: null,
        source_sha256: null,
        created_unix_ms: null,
        problem: null,
        family,
        family_options: options,
    };
};

const metrics = (over: Partial<VoiceChangerMetrics> = {}): VoiceChangerMetrics => ({
    buffer_ms: 40,
    process_ms: 4,
    chunk_ms: 10.67,
    process_p99_ms: null,
    backlog_ms: null,
    dropped_ms: null,
    underruns: null,
    ep: null,
    process_p50_ms: null,
    process_p95_ms: null,
    e2e_ms: null,
    vram_mb: null,
    governor_level: null,
    levers: null,
    cushion_ms: null,
    buffer_prefill_ms: null,
    buffer_max_queue_ms: null,
    engine_block_ms: null,
    call_p50_ms: null,
    call_p99_ms: null,
    ...over,
});

/** The markup with the quotes of the mocked translator's JSON unescaped. */
const unescape = (markup: string) => markup.replace(/&quot;/g, '"');
const html = (v: VcSlotInfo | undefined, m: VoiceChangerMetrics | null = null) => unescape(renderToStaticMarkup(<ChunkSizeSelect voice={v} metrics={m} />));
/** The `<option>` rows of the markup, as `[value, selected, label]`. */
const options = (markup: string) => [...markup.matchAll(/<option value="([^"]*)"([^>]*)>([^<]*)<\/option>/g)].map((m) => [m[1], m[2].includes("selected"), m[3]] as const);

afterEach(() => vi.unstubAllGlobals());

describe("the Chunk size control", () => {
    it("lists Auto with what it runs on this PC first, then every size the engine plans", () => {
        host({ cores: 4 });
        const rows = options(html(voice("template")));
        expect(rows[0]).toEqual(["0", true, 'voice_changer.chunk_size_auto_about_ms{"ms":200}']);
        expect(rows.slice(1).map((r) => r[0])).toEqual(["20", "40", "60", "80", "100", "120", "140", "160", "180", "200"]);
        expect(rows[3][2]).toBe('voice_changer.chunk_size_value{"ms":60}');
        // Its label is Chunk size, with an (i), and it is enabled.
        const markup = html(voice("template"));
        expect(markup).toContain(">voice_changer.chunk_size<");
        expect(markup).not.toContain("disabled");
    });

    it("tracks the PC's cores and the preset in the Auto label", () => {
        host({ cores: 8 });
        expect(options(html(undefined))[0][2]).toBe('voice_changer.chunk_size_auto_about_ms{"ms":120}');
        host({ cores: 6 });
        expect(options(html(undefined))[0][2]).toBe('voice_changer.chunk_size_auto_about_ms{"ms":160}');
        host({ cores: 4, vc: { rvcProfile: "quality", rvcExecutionProvider: "cuda" } });
        expect(options(html(undefined))[0][2]).toBe('voice_changer.chunk_size_auto_about_ms{"ms":20}');
        // The probe has not answered: Auto without a number.
        host({ cores: null });
        expect(options(html(undefined))[0][2]).toBe("voice_changer.chunk_size_auto");
    });

    it("reads Auto from the running engine: exact, whatever the estimate says, and an estimate again when a size is set", () => {
        // 4 cores: the estimate is 200 ms; the engine says it runs 20 ms (a developer switch turns the rule off).
        host({ cores: 4, running: true });
        expect(options(html(voice("template"), metrics({ engine_block_ms: 20 })))[0][2]).toBe('voice_changer.chunk_size_auto_ms{"ms":20}');
        // Nothing reported yet (the engine is starting): the estimate, marked as one.
        expect(options(html(voice("template"), metrics({ engine_block_ms: null })))[0][2]).toBe('voice_changer.chunk_size_auto_about_ms{"ms":200}');
        // A size set: Auto is not what runs, so it stays an estimate.
        host({ cores: 4, running: true, vc: { blockMs: 120 } });
        expect(options(html(voice("template"), metrics({ engine_block_ms: 120 })))[0][2]).toBe('voice_changer.chunk_size_auto_about_ms{"ms":200}');
    });

    it("warns about a short size on a big PC and while the cores are unknown, and not about a size it keeps up with", () => {
        host({ cores: 8, vc: { blockMs: 40 } });
        const markup = html(voice("template"));
        expect(markup).toContain('voice_changer.chunk_size_short_note{"block":120,"current":40}');
        expect(markup).not.toContain("few_cores");
        host({ cores: 8, vc: { blockMs: 120 } });
        expect(html(voice("template"))).not.toContain("chunk_size_short_note");
        host({ cores: null, vc: { blockMs: 40 } });
        expect(html(voice("template"))).toContain('voice_changer.chunk_size_short_note{"block":120,"current":40}');
        // The preset the size was set under is not the Streamer on the processor: no note until it is.
        host({ cores: 8, vc: { blockMs: 40, rvcProfile: "quality", rvcExecutionProvider: "cuda" } });
        expect(html(voice("template"))).not.toContain("_note{");
    });

    it("selects the stored size, and warns when it is shorter than this PC keeps up with", () => {
        host({ cores: 4, vc: { blockMs: 120 } });
        const markup = html(voice("template"));
        expect(options(markup).find((r) => r[1])).toEqual(["120", true, 'voice_changer.chunk_size_value{"ms":120}']);
        expect(markup).toContain('voice_changer.chunk_size_few_cores_note{"cores":4,"block":200,"current":120}');
        host({ cores: 4, vc: { blockMs: 200 } });
        expect(html(voice("template"))).not.toContain("few_cores");
        host({ cores: 4, vc: { blockMs: 0 } });
        expect(html(voice("template"))).not.toContain("few_cores");
    });

    it("keeps a size the engine would plan to another visible, marked with what runs", () => {
        host({ vc: { blockMs: 30 } });
        const row = options(html(voice("template"))).find((r) => r[1]);
        expect(row).toEqual(["30", true, 'voice_changer.chunk_size_value_runs_as{"set":30,"ms":40}']);
    });

    it("says which chunk the running engine converts, when it reports one", () => {
        host({ running: true, vc: { blockMs: 160 } });
        expect(html(voice("template"), metrics({ engine_block_ms: 160 }))).toContain('voice_changer.chunk_size_running{"ms":"160"}');
        // A short chunk reads as the whole number it is, like the choices ("20 ms", not "20.0 ms").
        host({ running: true, vc: { blockMs: 20, rvcProfile: "quality", rvcExecutionProvider: "cuda" } });
        expect(html(voice("template"), metrics({ engine_block_ms: 20 }))).toContain('voice_changer.chunk_size_running{"ms":"20"}');
        expect(html(voice("template"), metrics({ engine_block_ms: 80 }))).toContain('voice_changer.chunk_size_running{"ms":"80"}');
        host({ running: true, vc: { blockMs: 160 } });
        // Stopped, or a backend without the metric: nothing claimed.
        expect(html(voice("template"), metrics({ engine_block_ms: null }))).not.toContain("chunk_size_running");
        host({ running: false, vc: { blockMs: 160 } });
        expect(html(voice("template"), metrics({ engine_block_ms: 160 }))).not.toContain("chunk_size_running");
    });

    it("is disabled with the reason for a bring-your-own ONNX voice", () => {
        host({ vc: { blockMs: 160 } });
        const markup = html(voice("legacy"));
        expect(markup).toContain("disabled");
        // Not "Set by this voice": a bring-your-own voice is converted one Audio device buffer at a time, and says which.
        expect(options(markup)).toEqual([["fixed", true, 'voice_changer.chunk_size_fixed_legacy{"ms":"10.7"}']]);
        expect(markup).toContain("voice_changer.chunk_size_reason_legacy");
        expect(markup).not.toContain("few_cores");
        // The running engine's own device buffer, when it reports one; the setting at 48 kHz otherwise.
        host({ running: true, vc: { chunkFrames: 2048 } });
        expect(options(html(voice("legacy"), metrics({ chunk_ms: 42.67 })))[0][2]).toBe('voice_changer.chunk_size_fixed_legacy{"ms":"42.7"}');
        host({ vc: { chunkFrames: 1024 } });
        expect(options(html(voice("legacy")))[0][2]).toBe('voice_changer.chunk_size_fixed_legacy{"ms":"21.3"}');
    });

    it("is disabled with its reason for the families that set their own chunk, and works for the others", () => {
        host({});
        for (const family of ["sigil_zs", "meanvc", "sidecar_v2"] as const) {
            const markup = html(voice(family));
            expect(markup, family).toContain("disabled");
            expect(markup, family).toContain(`voice_changer.chunk_size_reason_${family}`);
            expect(options(markup), family).toEqual([["fixed", true, "voice_changer.chunk_size_fixed"]]);
        }
        // so-vits-svc takes the size (device-derived Auto, no per-PC rule); DDSP-SVC from 60 ms with the voice's preset as Auto.
        host({ cores: 4 });
        const sovits = options(html(voice("sovits")));
        expect(sovits[0][2]).toBe('voice_changer.chunk_size_auto_about_ms{"ms":20}');
        expect(sovits).toHaveLength(11);
        const ddsp = options(html(voice("ddsp_svc", { preset: "upstream_gui" })));
        expect(ddsp[0][2]).toBe('voice_changer.chunk_size_auto_about_ms{"ms":500}');
        expect(ddsp.slice(1).map((r) => r[0])).toEqual(["60", "80", "100", "120", "140", "160", "180", "200"]);
        expect(html(voice("ddsp_svc"))).toContain("voice_changer.chunk_size_ddsp_note");
    });

    it("reports the engine's own chunk of a voice that sets it, next to the reason", () => {
        host({ running: true });
        const markup = html(voice("sigil_zs"), metrics({ engine_block_ms: 100 }));
        expect(markup).toContain("voice_changer.chunk_size_reason_sigil_zs");
        expect(markup).toContain('voice_changer.chunk_size_running{"ms":"100"}');
    });
});

describe("where the control sits", () => {
    it("is in the main RVC settings under the preset, before any collapsed section", () => {
        host({});
        const markup = unescape(renderToStaticMarkup(<RvcSection crossfadeMaxMs={20} crossfadeShownMs={10} metrics={null} />));
        const control = markup.indexOf("voice_changer.chunk_size<");
        const preset = markup.indexOf("voice_changer.rvc_profile<");
        const firstDetails = markup.indexOf("<details");
        expect(control).toBeGreaterThan(-1);
        expect(preset).toBeGreaterThan(-1);
        expect(control).toBeGreaterThan(preset);
        expect(firstDetails).toBeGreaterThan(-1);
        expect(control, "the Chunk size is not inside a collapsed section").toBeLessThan(firstDetails);
        // Exactly one such control: the old Block size select is gone.
        expect(markup.match(/voice_changer\.chunk_size</g)).toHaveLength(1);
        expect(markup).not.toContain("voice_changer.block_ms");
    });
});

describe("the metrics panel", () => {
    const panel = (m: VoiceChangerMetrics) =>
        unescape(renderToStaticMarkup(<VcMetricsPanel metrics={m} prev={null} budgetMs={300} />));

    it("shows the chunk the engine converts and the device buffer as two rows", () => {
        host({});
        const markup = panel(metrics({ engine_block_ms: 160, chunk_ms: 10.67, process_ms: 3, process_p99_ms: 100, process_p95_ms: 90, process_p50_ms: 0.4 }));
        expect(markup).toContain("voice_changer.metrics_chunk_label");
        expect(markup).toContain("160");
        expect(markup).toContain("voice_changer.metrics_device_buffer_label");
        // No engine block (a bring-your-own voice, a backend without it): only the device buffer.
        const without = panel(metrics({ engine_block_ms: null, process_ms: 3, process_p99_ms: 5 }));
        expect(without).not.toContain("voice_changer.metrics_chunk_label");
        expect(without).toContain("voice_changer.metrics_device_buffer_label");
    });

    it("shows the model calls of a long chunk and judges the voice by them", () => {
        host({});
        // 10 % of the 200 ms calls overrun: the per-tick numbers read healthy, the calls row carries the warning.
        const m = metrics({ engine_block_ms: 200, chunk_ms: 10.67, process_ms: 8, process_p50_ms: 0.3, process_p95_ms: 40, process_p99_ms: 150, call_p50_ms: 150, call_p99_ms: 230 });
        const markup = panel(m);
        expect(markup).toContain("voice_changer.metrics_call_label");
        expect(markup).toContain("150 / 230");
        expect(markup).toContain("metrics_spikes_warn");
        const without = panel({ ...m, call_p50_ms: null, call_p99_ms: null });
        expect(without).not.toContain("voice_changer.metrics_call_label");
        expect(without).not.toContain("metrics_spikes_warn");
        // The chunk row reads as a whole number, and the waiting audio is shown against the budget in force.
        expect(panel(metrics({ engine_block_ms: 20, chunk_ms: 10.67, process_ms: 3, process_p99_ms: 5 }))).toMatch(/metrics_chunk_label[^]*?>20 /);
        // A stored 100 ms budget against 120 ms chunks: the engine applies 141 ms, and the panel shows that one.
        const waiting = unescape(
            renderToStaticMarkup(
                <VcMetricsPanel metrics={metrics({ engine_block_ms: 120, chunk_ms: 10.67, process_ms: 3, process_p99_ms: 5, backlog_ms: 100 })} prev={null} budgetMs={100} />,
            ),
        );
        expect(waiting).toContain("100 / 141");
    });

    it("does not call a healthy long-chunk voice too slow, and does when its calls outrun the chunk", () => {
        host({});
        const healthy = panel(metrics({ engine_block_ms: 160, chunk_ms: 10.67, process_ms: 22, process_p50_ms: 0.4, process_p95_ms: 98, process_p99_ms: 112 }));
        expect(healthy).not.toContain("metrics_slow_warn");
        expect(healthy).not.toContain("metrics_spikes_warn");
        const behind = panel(metrics({ engine_block_ms: 160, chunk_ms: 10.67, process_ms: 30, process_p50_ms: 0.4, process_p95_ms: 175, process_p99_ms: 190 }));
        expect(behind).toContain("metrics_slow_warn");
    });
});
