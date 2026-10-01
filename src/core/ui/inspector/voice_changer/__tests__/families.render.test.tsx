/**
 * Server-side render smoke tests of the P5 family UI (P5-G) against a mocked host: every family's "new voice"
 * form, options form and selected-voice section renders without throwing, and shows what the family needs (a
 * reference, a file, a sidecar address), its labels, the EP switch and the singing suggestion. The web build has
 * no Tauri and no families, so this is where the components run in CI.
 */
import { renderToStaticMarkup } from "react-dom/server";
import { proxy } from "valtio";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("react-i18next", () => ({
    useTranslation: () => ({ t: (key: string, opts?: Record<string, unknown>) => (opts ? `${key}${JSON.stringify(opts)}` : key) }),
}));
vi.mock("@tauri-apps/api/core", () => ({ isTauri: () => true, invoke: vi.fn() }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: vi.fn() }));
vi.mock("react-toastify", () => ({ toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warning: vi.fn(), success: vi.fn() }) }));
vi.mock("@/core/ui/inspector/components", () => ({ default: { SubHeader: ({ children }: { children: unknown }) => children } }));
vi.mock("@/core/ui/inspector/components/input", () => ({ InputSelect: () => null }));
vi.mock("@/core/ui/dropdown/Tooltip", () => ({ default: ({ children }: { children: unknown }) => children }));
vi.mock("@/core/ui/inspector/components/ModelRequirement", () => ({
    ModelRequirement: ({ ids }: { ids: string[] }) => <div data-requirement={ids.join(",")} />,
}));
vi.mock("@/core/ui/inspector/components/ReferenceVoiceRecorder", () => ({ ReferenceVoiceRecorder: () => <div>recorder</div> }));
vi.mock("@/core/services/voice_library", async () => {
    const { proxy: p } = await import("valtio");
    return {
        voiceLibraryUi: p({
            voices: [{ id: "alice", name: "Alice", durationMs: 8200, transcript: "hello there" }],
            loaded: true,
            recording: false,
            progress: null,
            error: null,
        }),
        ensureVoiceLibraryHost: vi.fn(),
        readClip: vi.fn(),
    };
});

import { normalizeCapabilities, type VcFamilyCapability, type VcFamilyId, type VcSlotInfo } from "@/core/services/voice_changer/ipc";
import { familyOptionsForm, type FamilyFormOf } from "@/core/services/voice_changer/families";
import { VoiceChangerStateSchema } from "@/core/services/voice_changer/schema";
import { FamilyCreator } from "../FamilyCreator";
import { FamilyOptionsFields } from "../FamilyOptions";
import { FamilySection } from "../FamilySection";
import { SoundSection } from "../SoundSection";

const FAMILIES = normalizeCapabilities({
    families: [
        { family: "ddsp_svc", available: true, singing: true, execution_providers: ["cuda", "dml"], needs_pitch: true, needs_content_encoder: true, creation: "import", import_extensions: [".pt", ".zip"], controls: ["pitch", "formant", "speaker", "silence_db"], requires: ["vc/embedder-contentvec"] },
        { family: "sigil_zs", available: true, zero_shot: true, execution_providers: ["cuda", "dml"], creation: "reference", controls: ["silence_db"] },
        { family: "meanvc", available: true, experimental: true, zero_shot: true, execution_providers: ["cpu", "cuda", "dml"], creation: "reference", controls: ["formant", "rms_mix", "silence_db"] },
        { family: "sovits", available: true, singing: true, execution_providers: ["cuda", "dml", "cpu"], needs_pitch: true, needs_content_encoder: true, creation: "import", import_extensions: [".onnx", ".zip"], controls: ["pitch", "rms_mix"] },
        { family: "sidecar_v2", available: false, experimental: true, zero_shot: true, execution_providers: ["cuda", "dml", "cpu"], creation: "sidecar", controls: ["pitch"] },
    ],
}).families;

function host(
    over: {
        families?: VcFamilyCapability[];
        ep?: string;
        f0Range?: "speech" | "singing";
        referenceVoiceId?: string;
        running?: boolean;
        slots?: VcSlotInfo[];
        formantSemitones?: number;
        optionsReloadPending?: Record<string, true>;
    } = {},
) {
    const voiceChanger = {
        ...VoiceChangerStateSchema.parse({}),
        mode: "rvc" as const,
        rvcExecutionProvider: (over.ep ?? "cpu") as "cpu",
        f0Range: over.f0Range ?? "speech",
        referenceVoiceId: over.referenceVoiceId ?? "",
        rvcSlotDir: "C:/m/v",
        formantSemitones: over.formantSemitones ?? 0,
    };
    const state = proxy({
        voiceChanger,
        isRunning: over.running ?? false,
        hydrated: true,
        rvcConsentPending: false,
        capabilities: { ...normalizeCapabilities({ rvc_execution_providers: ["cpu", "cuda", "dml"] }), families: over.families ?? FAMILIES },
        capabilitiesLoaded: true,
        trt: [],
    });
    const library = {
        state: proxy({
            slots: over.slots ?? [],
            listing: false,
            listError: null,
            slotStore: true,
            jobs: [],
            busy: {} as Record<string, string>,
            optionsReloadPending: over.optionsReloadPending ?? {},
        }),
        createFamilyVoice: vi.fn(),
        setFamilyOptions: vi.fn(),
    };
    vi.stubGlobal("window", { ApiServer: { voiceChanger: { state, library, update: vi.fn(), setReferenceVoice: vi.fn(), restartIfRunning: vi.fn() } } });
}

const voice = (family: VcFamilyId | null, over: Partial<VcSlotInfo> = {}): VcSlotInfo => ({
    id: "v",
    name: "V",
    path: "C:/m/v",
    format: family ? FAMILIES.find((f) => f.family === family)!.format : "sigil-rvc-1",
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
    family_options: {},
    reference_voice_id: null,
    ...over,
});

/** React warnings other than the server renderer's note about `useLayoutEffect` (InspectorInfoButton). */
let warnings: string[] = [];

beforeEach(() => {
    host();
    warnings = [];
    vi.spyOn(console, "error").mockImplementation((msg: unknown, ...rest: unknown[]) => {
        const text = [msg, ...rest].map(String).join(" ");
        if (!text.includes("useLayoutEffect does nothing on the server")) warnings.push(text);
    });
});
afterEach(() => {
    vi.unstubAllGlobals();
    // No missing keys, invalid DOM props or uncontrolled / controlled switches.
    expect(warnings).toEqual([]);
});

describe("new family voice form", () => {
    it("lists the families this build runs and starts on the first one (an import: it asks for the file)", () => {
        const html = renderToStaticMarkup(<FamilyCreator onClose={() => {}} />);
        for (const f of ["ddsp_svc", "sigil_zs", "meanvc", "sovits"]) expect(html).toContain(`voice_changer.family_name_${f}`);
        // The unavailable sidecar family is only named, as not available.
        expect(html).toContain("voice_changer.family_unavailable_list");
        expect(html).not.toContain('value="sidecar_v2"');
        expect(html).toContain("voice_changer.family_source_hint_ddsp_svc");
        expect(html).toContain("voice_changer.family_problem_source_required");
        expect(html).toContain('data-requirement="vc/embedder-contentvec"');
        // DDSP-SVC runs on CUDA only and the voice changer is on the CPU.
        expect(html).toContain("voice_changer.family_create_ep_note");
        expect(html).toMatch(/<button[^>]*disabled=""[^>]*>voice_changer\.family_create</);
    });

    it("a zero-shot family asks for a library reference; a dropped file starts its family", () => {
        const zs = renderToStaticMarkup(<FamilyCreator initial={{ family: "sigil_zs" }} onClose={() => {}} />);
        expect(zs).toContain("voice_changer.family_reference");
        expect(zs).toContain("Alice (8.2 s)");
        expect(zs).toContain("voice_changer.family_problem_reference_required");
        expect(zs).toContain("voice_changer.family_zs_preset_sigil_low");
        const file = renderToStaticMarkup(<FamilyCreator initial={{ family: "sovits", source: "C:/v/singer.onnx" }} onClose={() => {}} />);
        expect(file).toContain('value="C:/v/singer.onnx"');
        expect(file).not.toContain("voice_changer.family_problem_");
        expect(file).toMatch(/<button[^>]*>voice_changer\.family_create</);
        expect(file).not.toMatch(/<button[^>]*disabled=""[^>]*>voice_changer\.family_create</);
    });

    it("a family whose rungs this build does not offer cannot be created; no family at all says so", () => {
        host({ families: [FAMILIES[1]] });
        window.ApiServer.voiceChanger.state.capabilities.rvc_execution_providers = ["cpu"];
        const html = renderToStaticMarkup(<FamilyCreator onClose={() => {}} />);
        expect(html).toContain("voice_changer.family_create_no_ep");
        host({ families: [] });
        expect(renderToStaticMarkup(<FamilyCreator onClose={() => {}} />)).toContain("voice_changer.family_none_available");
    });
});

describe("family options forms", () => {
    it("render for every family", () => {
        for (const family of ["ddsp_svc", "sigil_zs", "meanvc", "sovits", "sidecar_v2"] as const) {
            const value = { family, form: familyOptionsForm(family, {}) } as FamilyFormOf;
            expect(() => renderToStaticMarkup(<FamilyOptionsFields value={value} onChange={() => {}} />), family).not.toThrow();
        }
        const sidecar = renderToStaticMarkup(
            <FamilyOptionsFields value={{ family: "sidecar_v2", form: { ...familyOptionsForm("sidecar_v2", {}), url: "ws://10.0.0.2:1" } }} onChange={() => {}} />,
        );
        expect(sidecar).toContain("voice_changer.family_problem_sidecar_url_not_loopback");
        const recipe = renderToStaticMarkup(<FamilyOptionsFields value={{ family: "sidecar_v2", form: { ...familyOptionsForm("sidecar_v2", {}), recipe: "xvc" } }} onChange={() => {}} />);
        expect(recipe).toContain("voice_changer.family_sidecar_python");
        expect(recipe).not.toContain("voice_changer.family_sidecar_url");
        // A recipe may attach to a server already running: its address is optional, and never hidden.
        expect(recipe).toContain("voice_changer.family_sidecar_attach_url_hint");
    });

    it("a recipe's attach address and refused settings are shown where they are typed", () => {
        const base = { ...familyOptionsForm("sidecar_v2", {}), recipe: "seedvc" as const };
        const attached = renderToStaticMarkup(<FamilyOptionsFields value={{ family: "sidecar_v2", form: { ...base, url: "ws://127.0.0.1:9000/vc" } }} onChange={() => {}} />);
        expect(attached).toContain('value="ws://127.0.0.1:9000/vc"');
        expect(attached).toContain("voice_changer.family_sidecar_attach_url_active");
        const remote = renderToStaticMarkup(<FamilyOptionsFields value={{ family: "sidecar_v2", form: { ...base, url: "ws://10.0.0.2:1" } }} onChange={() => {}} />);
        expect(remote).toContain("voice_changer.family_problem_sidecar_url_not_loopback");
        const relative = renderToStaticMarkup(<FamilyOptionsFields value={{ family: "sidecar_v2", form: { ...base, python: "venv/python" } }} onChange={() => {}} />);
        expect(relative).toContain("voice_changer.family_problem_sidecar_python_relative");
        const reserved = renderToStaticMarkup(<FamilyOptionsFields value={{ family: "sidecar_v2", form: { ...base, args: ["--port=1"] } }} onChange={() => {}} />);
        expect(reserved).toContain("voice_changer.family_problem_sidecar_args_reserved");
        // A custom server never sends the recipe settings, so they raise nothing.
        const custom = renderToStaticMarkup(
            <FamilyOptionsFields value={{ family: "sidecar_v2", form: { ...base, recipe: "custom", url: "ws://127.0.0.1:1", python: "venv/python" } }} onChange={() => {}} />,
        );
        expect(custom).not.toContain("voice_changer.family_problem_");
    });
});

describe("the new voice form, reopened", () => {
    it("a failed family voice reopens with what was sent", () => {
        const request = {
            family: "sovits" as const,
            name: "Singer",
            source: "C:\\models\\so-vits-svc-4.1",
            options: { cluster_ratio: 0.4 },
        };
        const html = renderToStaticMarkup(<FamilyCreator initial={{ family: "sovits", request }} onClose={() => {}} />);
        expect(html).toContain('value="C:\\models\\so-vits-svc-4.1"');
        expect(html).toContain('value="Singer"');
        // A folder with a dot in its name is not a file of the wrong type: the voice can be created.
        expect(html).not.toContain("voice_changer.family_problem_");
        expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>voice_changer\.family_create</);
    });
});

describe("the selected voice's family section", () => {
    it("renders nothing for RVC voices or without a voice", () => {
        expect(renderToStaticMarkup(<FamilySection voice={voice(null)} />)).toBe("");
        expect(renderToStaticMarkup(<FamilySection voice={undefined} />)).toBe("");
    });

    it("a zero-shot voice: labels, the EP switch, its reference and its options", () => {
        host({ ep: "cpu", referenceVoiceId: "alice", running: true });
        const html = renderToStaticMarkup(<FamilySection voice={voice("sigil_zs", { family_options: { preset: "sigil_lean" } })} />);
        expect(html).toContain("voice_changer.family_name_sigil_zs");
        expect(html).toContain("voice_changer.family_badge_zero_shot");
        expect(html).toContain("voice_changer.family_runs_on_other");
        expect(html).toContain("voice_changer.family_use_ep");
        expect(html).toMatch(/<option value="alice" selected="">Alice \(8\.2 s\)<\/option>/);
        expect(html).toContain("voice_changer.family_reference_switch_note");
        expect(html).toMatch(/<option value="sigil_lean" selected="">/);
    });

    it("a missing reference and an experimental family say so; singing families suggest singing mode", () => {
        host({ ep: "cpu", referenceVoiceId: "gone" });
        const mean = renderToStaticMarkup(<FamilySection voice={voice("meanvc")} />);
        expect(mean).toContain("voice_changer.family_experimental_note");
        expect(mean).toContain("voice_changer.family_reference_missing");
        expect(mean).not.toContain("voice_changer.family_runs_on_other");
        host({ ep: "cuda", f0Range: "speech" });
        const ddsp = renderToStaticMarkup(<FamilySection voice={voice("ddsp_svc")} />);
        expect(ddsp).toContain("voice_changer.family_singing_suggest");
        expect(ddsp).not.toContain("voice_changer.family_reference");
        host({ ep: "cuda", f0Range: "singing" });
        expect(renderToStaticMarkup(<FamilySection voice={voice("ddsp_svc")} />)).not.toContain("voice_changer.family_singing_suggest");
    });

    it("a family this build does not run says so and offers no options", () => {
        const html = renderToStaticMarkup(<FamilySection voice={voice("sidecar_v2")} />);
        expect(html).toContain("voice_changer.family_not_in_build");
        expect(html).not.toContain("voice_changer.family_options");
    });

    it("options saved while running offer the restart for that voice until the engine stops, also after switching voices", () => {
        host({ ep: "cuda", running: true, optionsReloadPending: { v: true } });
        expect(renderToStaticMarkup(<FamilySection voice={voice("ddsp_svc")} />)).toContain("voice_changer.family_options_reload_note");
        expect(renderToStaticMarkup(<FamilySection voice={voice("ddsp_svc", { id: "other" })} />)).not.toContain("voice_changer.family_options_reload_note");
        host({ ep: "cuda", running: false, optionsReloadPending: { v: true } });
        expect(renderToStaticMarkup(<FamilySection voice={voice("ddsp_svc")} />)).not.toContain("voice_changer.family_options_reload_note");
    });
});

describe("pitch and formant of a family voice", () => {
    it("a Formant value the family ignores still turns Brightness off: it says so and offers a reset", () => {
        host({ formantSemitones: 3, slots: [voice("sigil_zs")] });
        const zs = renderToStaticMarkup(<SoundSection />);
        expect(zs).not.toContain('aria-label="voice_changer.formant_semitones"');
        expect(zs).toContain("voice_changer.family_formant_hidden_note");
        expect(zs).toContain("voice_changer.family_formant_reset");
        // A family that answers Formant shows the slider and the usual note.
        host({ formantSemitones: 3, slots: [voice("meanvc")] });
        const mean = renderToStaticMarkup(<SoundSection />);
        expect(mean).toContain('aria-label="voice_changer.formant_semitones"');
        expect(mean).toContain("voice_changer.formant_tilt_rvc_note");
        expect(mean).not.toContain("voice_changer.family_formant_hidden_note");
        // Formant off: Brightness works, nothing to say.
        host({ formantSemitones: 0, slots: [voice("sigil_zs")] });
        const off = renderToStaticMarkup(<SoundSection />);
        expect(off).not.toContain("voice_changer.family_formant_hidden_note");
        expect(off).not.toContain("voice_changer.formant_tilt_rvc_note");
    });
});
