/**
 * P5 voice families in the UI (`families.ts`, P5-G): the options forms against the Rust options structs and
 * presets / recipes (`vc/families/*`), the checks before `voice_changer_create_family_slot`, which controls,
 * EP rungs and model-store entries a family voice gets, family slot rows, and that every string the family
 * UI uses exists (English or a staged i18n file).
 */
import { describe, expect, it } from "vitest";

import en from "../../../../../public/i18n/en/translation.json";
import { familyVoiceControls } from "../families";
import {
    availableFamilies,
    clipSecondsLabel,
    DDSP_SVC_PRESETS,
    extensionOf,
    FAMILY_DRAFT_PROBLEMS,
    FAMILY_OPTIONS_PROBLEMS,
    familyCapability,
    familyDraftFromRequest,
    familyDraftProblem,
    familyEpChoice,
    familyForImportFile,
    familyOf,
    familyOptionsForm,
    familyOptionsPayload,
    familyOptionsProblem,
    familyPickerExtensions,
    familyRunsOnEp,
    familySlotRequest,
    FORMANT_SEMITONES_EPSILON,
    formantShiftOn,
    hiddenFormantBlocksTilt,
    importSourceProblem,
    isAbsolutePath,
    isLoopbackWsUrl,
    KNOWN_FILE_EXTENSIONS,
    MEANVC_STEPS,
    MEANVC_TICK_MS,
    newFamilyDraft,
    parseSidecarArgs,
    referenceName,
    referenceState,
    sameFamilyOptions,
    showsControl,
    sidecarProblemField,
    SIDECAR_RECIPES,
    SIDECAR_RESERVED_ARGS,
    SIGIL_ZS_PRESETS,
    singingRecommended,
    switchDraftFamily,
    switchSidecarRecipe,
    voiceFamilyCapability,
    voiceKind,
    type FamilyDraft,
} from "../families";
import {
    BASELINE_CAPABILITIES,
    normalizeCapabilities,
    VC_FAMILIES,
    VC_FAMILY_CONTROLS,
    type VcCapabilities,
    type VcFamilyCapability,
    type VcFamilyId,
    type VcSlotInfo,
} from "../ipc";
import { voiceModelIds } from "../requirements";
import { normalizeSlotInfo } from "../slots";

// ---- Rust sources (the families' options contract) -------------------------------------------------

const rustFamilies = import.meta.glob("../../../../../src-tauri/src/services/neural_audio/vc/families/**/*.rs", {
    eager: true,
    query: "?raw",
    import: "default",
}) as Record<string, string>;

/** The Rust source of a family module (`<x>.rs` plus `<x>/**`). */
function rustOf(family: VcFamilyId): string {
    return Object.entries(rustFamilies)
        .filter(([path]) => path.endsWith(`/families/${family}.rs`) || path.includes(`/families/${family}/`))
        .map(([, src]) => src)
        .join("\n");
}

function rustStrConst(src: string, name: string): string[] {
    const m = new RegExp(`pub const ${name}: \\[&str; \\d+\\] = \\[([^\\]]*)\\];`).exec(src);
    if (!m) throw new Error(`no ${name}`);
    return [...m[1].matchAll(/"([^"]*)"/g)].map((x) => x[1]);
}

function rustStructFields(src: string, name: string): string[] {
    const m = new RegExp(`pub struct ${name} \\{([\\s\\S]*?)\\n\\}`).exec(src);
    if (!m) throw new Error(`no struct ${name}`);
    return [...m[1].matchAll(/^\s*pub ([a-z0-9_]+):/gm)].map((x) => x[1]).sort();
}

const OPTION_STRUCTS: Record<VcFamilyId, string> = {
    ddsp_svc: "DdspSvcOptions",
    sigil_zs: "SigilZsOptions",
    meanvc: "MeanVcOptions",
    sovits: "SovitsOptions",
    sidecar_v2: "SidecarV2Options",
};

// ---- fixtures ---------------------------------------------------------------------------------------

const famCap = (family: VcFamilyId, over: Partial<VcFamilyCapability> = {}): VcFamilyCapability =>
    normalizeCapabilities({ families: [{ family, available: true, ...over }] }).families[0];

/** Each family's facts as its Rust `info()` reports them (P5-I0), all available. */
const FAMILIES: VcFamilyCapability[] = [
    famCap("ddsp_svc", { singing: true, execution_providers: ["cuda", "dml"], needs_pitch: true, needs_content_encoder: true, creation: "import", import_extensions: [".pt", ".zip"], controls: ["pitch", "formant", "speaker", "silence_db"], requires: ["vc/embedder-contentvec", "vc/ddsp-svc-6.3", "vc/vocoder-pc-nsf-hifigan-44k"] }),
    famCap("sigil_zs", { zero_shot: true, execution_providers: ["cuda", "dml"], creation: "reference", controls: ["silence_db"], requires: ["vc/zs-seed-vc"] }),
    famCap("meanvc", { experimental: true, zero_shot: true, execution_providers: ["cpu", "cuda", "dml"], creation: "reference", controls: ["formant", "rms_mix", "silence_db"], requires: ["vc/meanvc2"] }),
    famCap("sovits", { singing: true, execution_providers: ["cuda", "dml", "cpu"], needs_pitch: true, needs_content_encoder: true, creation: "import", import_extensions: [".onnx", ".zip"], controls: ["pitch", "formant", "rms_mix", "speaker", "noise_scale", "silence_db"], requires: ["vc/embedder-contentvec"] }),
    famCap("sidecar_v2", { experimental: true, zero_shot: true, execution_providers: ["cuda", "dml", "cpu"], creation: "sidecar", controls: ["pitch"] }),
];

const caps = (over: Partial<VcCapabilities> = {}): VcCapabilities => ({
    ...BASELINE_CAPABILITIES,
    rvc_execution_providers: ["cpu", "cuda", "dml"],
    families: FAMILIES,
    ...over,
});

const slot = (over: Partial<VcSlotInfo> = {}): VcSlotInfo => ({
    id: "v",
    name: "V",
    path: "C:/m/v",
    format: "sigil-rvc-1",
    template: null,
    version: "v2",
    sample_rate: 40000,
    f0: true,
    embedder: null,
    vocoder: null,
    has_index: false,
    speaker_count: null,
    fp16_ok: null,
    source_sha256: null,
    created_unix_ms: null,
    problem: null,
    ...over,
});

const famSlot = (family: VcFamilyId, over: Partial<VcSlotInfo> = {}) =>
    slot({ format: famCap(family).format, family, version: null, f0: null, ...over });

// ---- tests ------------------------------------------------------------------------------------------

describe("family options against the Rust options structs", () => {
    it("presets and recipes are Rust's, in Rust's order (the first is the default)", () => {
        expect([...DDSP_SVC_PRESETS]).toEqual(rustStrConst(rustOf("ddsp_svc"), "PRESETS"));
        expect([...SIGIL_ZS_PRESETS]).toEqual(rustStrConst(rustOf("sigil_zs"), "PRESETS"));
        expect([...SIDECAR_RECIPES]).toEqual(rustStrConst(rustOf("sidecar_v2"), "RECIPES"));
    });

    it("every family's payload has exactly the Rust struct's fields", () => {
        for (const family of VC_FAMILIES) {
            const payload = familyOptionsPayload(family, familyOptionsForm(family, {}));
            expect(Object.keys(payload).sort(), family).toEqual(rustStructFields(rustOf(family), OPTION_STRUCTS[family]));
        }
    });

    it("forms normalize like Rust: unknown presets take the default, numbers are clamped, unset stays unset", () => {
        expect(familyOptionsForm("ddsp_svc", { preset: "nope", steps: 99.4, t_start: -1 })).toEqual({ preset: "sigil_singing", steps: 50, t_start: 0 });
        expect(familyOptionsForm("ddsp_svc", { preset: " UPSTREAM_GUI ", steps: "5" })).toEqual({ preset: "upstream_gui", steps: null, t_start: null });
        expect(familyOptionsForm("sigil_zs", { preset: "sigil_lean", steps: 0, cfg_rate: 2, prompt_s: 0.1, fp16_dit: true, fp16_vocoder: false })).toEqual({
            preset: "sigil_lean",
            steps: 1,
            cfg_rate: 1,
            prompt_s: 0.5,
            fp16_xlsr: false,
            fp16_dit: true,
            fp16_vocoder: false,
        });
        expect(familyOptionsForm("sigil_zs", null)).toMatchObject({ preset: "sigil_low", fp16_dit: false, fp16_vocoder: null });
        expect(familyOptionsForm("meanvc", {})).toEqual({ steps: 2, tick_ms: 80 });
        expect(familyOptionsForm("meanvc", { steps: 0, tick_ms: 120 })).toEqual({ steps: 1, tick_ms: 160 });
        expect(familyOptionsForm("meanvc", { steps: 7, tick_ms: 119 })).toEqual({ steps: 2, tick_ms: 80 });
        expect([...MEANVC_STEPS, ...MEANVC_TICK_MS]).toEqual([1, 2, 80, 160]);
        expect(familyOptionsForm("sovits", { cluster_ratio: 3 })).toEqual({ cluster_ratio: 1 });
        expect(familyOptionsForm("sovits", { cluster_ratio: Number.NaN })).toEqual({ cluster_ratio: 0 });
        expect(familyOptionsForm("sidecar_v2", { recipe: " XVC ", url: " ws://127.0.0.1:1 ", args: ["a", 3], timeout_ms: 1 })).toEqual({
            recipe: "xvc",
            url: "ws://127.0.0.1:1",
            python: "",
            model_dir: "",
            args: ["a"],
            timeout_ms: 20,
        });
    });

    it("payloads trim sidecar fields and drop blank arguments; equal payloads are the same options", () => {
        const form = { recipe: "xvc" as const, url: " ws://localhost:9/vc ", python: " C:/v/python.exe ", model_dir: "", args: [" --x ", "", "  "], timeout_ms: null };
        expect(familyOptionsPayload("sidecar_v2", form)).toEqual({ recipe: "xvc", url: "ws://localhost:9/vc", python: "C:/v/python.exe", model_dir: "", args: ["--x"], timeout_ms: null });
        expect(sameFamilyOptions("sidecar_v2", form, { ...form, url: "ws://localhost:9/vc", args: ["--x"] })).toBe(true);
        expect(sameFamilyOptions("ddsp_svc", familyOptionsForm("ddsp_svc", {}), { preset: "sigil_singing", steps: 5, t_start: null })).toBe(false);
    });

    it("sidecar URLs stay on this PC (the Rust test's cases) and custom servers need one", () => {
        for (const ok of ["ws://127.0.0.1:18888/vc", "ws://localhost:9", "WS://[::1]:80/x", "ws://127.0.0.1"]) expect(isLoopbackWsUrl(ok), ok).toBe(true);
        for (const bad of ["wss://127.0.0.1:1", "ws://192.168.1.2:1", "ws://127.0.0.1.evil.com:1", "http://127.0.0.1:1", "ws://user@127.0.0.1:1", ""]) {
            expect(isLoopbackWsUrl(bad), bad).toBe(false);
        }
        const base = familyOptionsForm("sidecar_v2", {});
        expect(familyOptionsProblem("sidecar_v2", base)).toBe("sidecar_url_required");
        expect(familyOptionsProblem("sidecar_v2", { ...base, url: "ws://10.0.0.1:1" })).toBe("sidecar_url_not_loopback");
        expect(familyOptionsProblem("sidecar_v2", { ...base, recipe: "xvc", url: "ws://10.0.0.1:1" })).toBe("sidecar_url_not_loopback");
        expect(familyOptionsProblem("sidecar_v2", { ...base, recipe: "xvc" })).toBeNull();
        expect(familyOptionsProblem("sidecar_v2", { ...base, url: "ws://127.0.0.1:1/vc" })).toBeNull();
        expect(familyOptionsProblem("sidecar_v2", { ...base, recipe: "seedvc", args: Array(65).fill("x") })).toBe("sidecar_args_too_long");
        expect(familyOptionsProblem("sidecar_v2", { ...base, recipe: "seedvc", args: ["x".repeat(1025)] })).toBe("sidecar_args_too_long");
        // Rust counts bytes: 512 two-byte characters fit, 513 do not.
        expect(familyOptionsProblem("sidecar_v2", { ...base, recipe: "seedvc", args: ["é".repeat(512)] })).toBeNull();
        expect(familyOptionsProblem("sidecar_v2", { ...base, recipe: "seedvc", args: ["é".repeat(513)] })).toBe("sidecar_args_too_long");
        expect(familyOptionsProblem("ddsp_svc", familyOptionsForm("ddsp_svc", {}))).toBeNull();
        expect(parseSidecarArgs(" --a 1 \r\n\n--b\n  ")).toEqual(["--a 1", "--b"]);
    });

    it("recipe settings are checked like Rust's normalize_options (the Rust test's cases), only when a recipe sends them", () => {
        const seed = { ...familyOptionsForm("sidecar_v2", {}), recipe: "seedvc" as const };
        // The bind and the key stay Sigil's (`--host`, `--PORT=1`); an argument that only contains one is fine.
        expect(familyOptionsProblem("sidecar_v2", { ...seed, args: ["--host", "0.0.0.0"] })).toBe("sidecar_args_reserved");
        expect(familyOptionsProblem("sidecar_v2", { ...seed, args: ["--PORT=9000"] })).toBe("sidecar_args_reserved");
        expect(familyOptionsProblem("sidecar_v2", { ...seed, args: ["--api-key=x"] })).toBe("sidecar_args_reserved");
        expect(familyOptionsProblem("sidecar_v2", { ...seed, args: ["--steps", "6", "--port-range"] })).toBeNull();
        expect(familyOptionsProblem("sidecar_v2", { ...seed, args: ["a\0b"] })).toBe("sidecar_args_invalid");
        // Paths are full paths; a bare interpreter name is looked up on PATH.
        expect(familyOptionsProblem("sidecar_v2", { ...seed, python: "venv/python" })).toBe("sidecar_python_relative");
        expect(familyOptionsProblem("sidecar_v2", { ...seed, python: "venv\\Scripts\\python.exe" })).toBe("sidecar_python_relative");
        expect(familyOptionsProblem("sidecar_v2", { ...seed, python: "python" })).toBeNull();
        expect(familyOptionsProblem("sidecar_v2", { ...seed, python: "x\u0007" })).toBe("sidecar_python_invalid");
        expect(familyOptionsProblem("sidecar_v2", { ...seed, model_dir: "models" })).toBe("sidecar_model_dir_relative");
        expect(familyOptionsProblem("sidecar_v2", { ...seed, model_dir: "D:/m".padEnd(1025, "m") })).toBe("sidecar_model_dir_invalid");
        expect(
            familyOptionsProblem("sidecar_v2", { ...seed, python: " C:/v/python.exe ", model_dir: "D:\\m", args: ["--steps", "6"], url: "ws://127.0.0.1:9/vc" }),
        ).toBeNull();
        for (const p of ["C:\\v\\python.exe", "c:/v/python.exe", "\\\\server\\share\\python.exe", "\\\\?\\C:\\v\\python.exe", "/usr/bin/python3"]) {
            expect(isAbsolutePath(p), p).toBe(true);
        }
        for (const p of ["venv/python", "C:python.exe", ".\\python.exe", "models"]) expect(isAbsolutePath(p), p).toBe(false);
        // A custom server never sends them, so they cannot block it either.
        const custom = { ...seed, recipe: "custom" as const, url: "ws://127.0.0.1:9/vc", python: "venv/python", model_dir: "models", args: ["--port"] };
        expect(familyOptionsProblem("sidecar_v2", custom)).toBeNull();
        expect(familyOptionsPayload("sidecar_v2", custom)).toEqual({ recipe: "custom", url: "ws://127.0.0.1:9/vc", python: "", model_dir: "", args: [], timeout_ms: null });
        // Each problem is shown at its field.
        expect(FAMILY_OPTIONS_PROBLEMS.map(sidecarProblemField)).toEqual(["url", "url", "python", "python", "model_dir", "model_dir", "args", "args", "args"]);
        expect(sidecarProblemField(null)).toBeNull();
    });

    it("reserved sidecar arguments are Rust's", () => {
        const src = rustOf("sidecar_v2");
        const m = /const RESERVED_ARGS: \[&str; \d+\] = \[([^\]]*)\];/.exec(src);
        expect(m).not.toBeNull();
        expect([...SIDECAR_RESERVED_ARGS]).toEqual([...m![1].matchAll(/"([^"]*)"/g)].map((x) => x[1]));
    });

    it("switching the recipe never sends a value the form hides", () => {
        const custom = { ...familyOptionsForm("sidecar_v2", {}), url: "ws://127.0.0.1:8765", python: "C:/v/python.exe" };
        // A custom server's address is not the recipe's helper: switching clears it, so Sigil starts the helper.
        const seed = switchSidecarRecipe(custom, "seedvc");
        expect(seed).toMatchObject({ recipe: "seedvc", url: "", python: "C:/v/python.exe" });
        expect(familyOptionsPayload("sidecar_v2", seed).url).toBe("");
        // Between recipes (and back to custom) an attach address stays: it is shown for every recipe.
        const attached = { ...seed, url: "ws://127.0.0.1:9000/vc" };
        expect(switchSidecarRecipe(attached, "xvc").url).toBe("ws://127.0.0.1:9000/vc");
        const back = switchSidecarRecipe(attached, "custom");
        expect(back).toMatchObject({ recipe: "custom", url: "ws://127.0.0.1:9000/vc", python: "C:/v/python.exe" });
        expect(familyOptionsPayload("sidecar_v2", back)).toMatchObject({ python: "", model_dir: "", args: [] });
        expect(switchSidecarRecipe(seed, "seedvc")).toBe(seed);
    });
});

describe("family voices in the inspector", () => {
    it("tells family voices from template and bring-your-own ONNX voices", () => {
        expect(voiceKind(undefined)).toBe("none");
        expect(voiceKind(slot())).toBe("template");
        expect(voiceKind(slot({ format: "rvc" }))).toBe("legacy");
        expect(voiceKind(famSlot("sigil_zs"))).toBe("family");
        // A row without the P5 field still names its family by format.
        expect(familyOf(slot({ format: "ddsp-svc-6.3" }))).toBe("ddsp_svc");
        expect(familyOf(slot())).toBeNull();
        expect(voiceFamilyCapability(caps(), famSlot("meanvc"))?.experimental).toBe(true);
        expect(familyCapability(caps({ families: [] }), "meanvc")).toBeNull();
        expect(availableFamilies(caps({ families: [famCap("sovits"), famCap("meanvc", { available: false })] })).map((f) => f.family)).toEqual(["sovits"]);
    });

    it("a family voice shows only its family's controls; RVC voices show every one", () => {
        const zs = familyVoiceControls(caps(), famSlot("sigil_zs"))!;
        expect(zs.needsPitch).toBe(false);
        expect(VC_FAMILY_CONTROLS.filter((c) => showsControl(zs, c))).toEqual(["silence_db"]);
        const so = familyVoiceControls(caps(), famSlot("sovits"))!;
        expect(so.needsPitch).toBe(true);
        expect(showsControl(so, "rms_mix") && showsControl(so, "noise_scale") && !showsControl(so, "index")).toBe(true);
        expect(familyVoiceControls(caps(), slot())).toBeNull();
        expect(VC_FAMILY_CONTROLS.every((c) => showsControl(null, c))).toBe(true);
    });

    it("EP rungs: accelerators run on their base rung; the switch goes to the family's best offered rung", () => {
        const zs = familyCapability(caps(), "sigil_zs")!;
        expect(familyRunsOnEp(zs, "cpu")).toBe(false);
        expect(familyRunsOnEp(zs, "dml_hp")).toBe(true);
        expect(familyRunsOnEp(zs, "tensorrt_rtx")).toBe(true);
        expect(familyEpChoice(zs, caps())).toBe("cuda");
        expect(familyEpChoice(zs, caps({ rvc_execution_providers: ["cpu", "dml"] }))).toBe("dml");
        expect(familyEpChoice(zs, caps({ rvc_execution_providers: ["cpu"] }))).toBeNull();
        // MeanVC2 computes on the CPU on every rung, so a GPU setup keeps its rung.
        expect(familyRunsOnEp(familyCapability(caps(), "meanvc")!, "cuda")).toBe(true);
        expect(familyRunsOnEp({ execution_providers: [] }, "cuda")).toBe(true);
    });

    it("singing mode is suggested for singing families that track pitch", () => {
        expect(singingRecommended(familyCapability(caps(), "ddsp_svc"))).toBe(true);
        expect(singingRecommended(familyCapability(caps(), "sigil_zs"))).toBe(false);
        expect(singingRecommended(null)).toBe(false);
    });

    it("a Formant value left from another voice still turns Brightness off for a family that ignores Formant (Rust rvc_output_tilt)", () => {
        expect(formantShiftOn(0)).toBe(false);
        expect(formantShiftOn(0.005)).toBe(false);
        expect(formantShiftOn(-0.01)).toBe(true);
        expect(formantShiftOn(3)).toBe(true);
        expect(formantShiftOn(Number.NaN)).toBe(false);
        const zs = familyVoiceControls(caps(), famSlot("sigil_zs"));
        expect(hiddenFormantBlocksTilt(zs, 3)).toBe(true);
        expect(hiddenFormantBlocksTilt(zs, 0)).toBe(false);
        // Families that answer Formant, and RVC voices, show the slider itself.
        expect(hiddenFormantBlocksTilt(familyVoiceControls(caps(), famSlot("meanvc")), 3)).toBe(false);
        expect(hiddenFormantBlocksTilt(null, 3)).toBe(false);
        // The epsilon is Rust's.
        const rust = Object.entries(import.meta.glob("../../../../../src-tauri/src/services/neural_audio/chunk_processor.rs", { eager: true, query: "?raw", import: "default" }))[0][1] as string;
        expect(Number(/const FORMANT_SEMITONES_EPSILON: f32 = ([0-9.]+);/.exec(rust)?.[1])).toBe(FORMANT_SEMITONES_EPSILON);
    });

    it("model-store entries: the family's, its content encoder and pitch models only when it uses them", () => {
        const base = {
            mode: "rvc" as const,
            pitchExtractor: "rmvpe" as const,
            rvcExecutionProvider: "cuda" as const,
            fcpePath: "",
            rmvpePath: "",
            capabilities: caps(),
        };
        expect(voiceModelIds({ ...base, voice: famSlot("sigil_zs") })).toEqual(["vc/zs-seed-vc"]);
        expect(voiceModelIds({ ...base, voice: famSlot("meanvc") })).toEqual(["vc/meanvc2"]);
        expect(voiceModelIds({ ...base, voice: famSlot("ddsp_svc") })).toEqual(["vc/embedder-contentvec", "vc/ddsp-svc-6.3", "vc/vocoder-pc-nsf-hifigan-44k", "vc/pitch-rmvpe"]);
        // A voice that names another content encoder needs only that one (not the family's usual ContentVec).
        expect(voiceModelIds({ ...base, voice: famSlot("sovits", { embedder: "hubert-ja" }) })).toEqual(["vc/embedder-hubert-ja", "vc/pitch-rmvpe"]);
        expect(voiceModelIds({ ...base, rvcExecutionProvider: "cpu", voice: famSlot("sovits") })).toEqual(["vc/embedder-contentvec", "vc/pitch-fcpe"]);
        // A family this build does not run (or does not list) needs nothing yet: the voice shows why.
        expect(voiceModelIds({ ...base, capabilities: caps({ families: [] }), voice: famSlot("sigil_zs") })).toEqual([]);
        expect(voiceModelIds({ ...base, mode: "classic", voice: famSlot("ddsp_svc") })).toEqual([]);
    });

    it("slot rows carry the family, its options, the experimental label and the reference voice", () => {
        const row = normalizeSlotInfo({
            id: "zs",
            path: "C:/m/zs",
            format: "sigil-zs-1",
            family: "sigil_zs",
            experimental: false,
            family_options: { preset: "sigil_lean" },
            reference_voice_id: " alice ",
        })!;
        expect(row).toMatchObject({ family: "sigil_zs", experimental: false, family_options: { preset: "sigil_lean" }, reference_voice_id: "alice" });
        // Older backends: the family comes from the format; RVC rows never get family fields.
        expect(normalizeSlotInfo({ id: "m", path: "C:/m/m", format: "meanvc-2", experimental: true, family_options: [1] })).toMatchObject({
            family: "meanvc",
            experimental: true,
            family_options: null,
        });
        expect(normalizeSlotInfo({ id: "r", path: "C:/m/r", format: "sigil-rvc-1", experimental: true, family_options: { a: 1 } })).toMatchObject({
            family: null,
            experimental: false,
            family_options: null,
            reference_voice_id: null,
        });
    });
});

describe("the zero-shot reference", () => {
    const voices = [
        { id: "alice", name: "Alice" },
        { id: "bob", name: "" },
    ];
    it("is none, listed, missing, or unknown until the library was read", () => {
        expect(referenceState("", voices, true)).toBe("none");
        expect(referenceState(" alice ", voices, true)).toBe("ok");
        expect(referenceState("carol", voices, true)).toBe("missing");
        expect(referenceState("carol", [], false)).toBe("unknown");
        expect(referenceName("alice", voices)).toBe("Alice");
        expect(referenceName("bob", voices)).toBe("bob");
        expect(referenceName("zed", voices)).toBe("zed");
        expect(clipSecondsLabel(8240)).toBe("8.2 s");
        expect(clipSecondsLabel(0)).toBeNull();
    });
});

describe("new family voices", () => {
    const fam = (id: VcFamilyId) => familyCapability(caps(), id);
    const draft = (family: VcFamilyId, over: Partial<FamilyDraft> = {}): FamilyDraft => ({ ...newFamilyDraft(family), ...over });

    it("checks what each kind of family needs before it is sent", () => {
        expect(familyDraftProblem(draft("sigil_zs"), fam("sigil_zs"))).toBe("reference_required");
        expect(familyDraftProblem(draft("sigil_zs", { referenceVoiceId: "alice" }), fam("sigil_zs"))).toBeNull();
        expect(familyDraftProblem(draft("ddsp_svc"), fam("ddsp_svc"))).toBe("source_required");
        expect(familyDraftProblem(draft("ddsp_svc", { source: "C:/v/model_100.PT" }), fam("ddsp_svc"))).toBeNull();
        expect(familyDraftProblem(draft("ddsp_svc", { source: "C:/v/voice.onnx" }), fam("ddsp_svc"))).toBe("source_type");
        expect(familyDraftProblem(draft("ddsp_svc", { source: "C:/v/my voice" }), fam("ddsp_svc"))).toBeNull(); // a folder
        expect(familyDraftProblem(draft("sidecar_v2"), fam("sidecar_v2"))).toBe("sidecar_url_required");
        expect(familyDraftProblem(draft("sidecar_v2"), null)).toBe("family_unavailable");
        expect(familyDraftProblem(draft("meanvc", { referenceVoiceId: "a" }), famCap("meanvc", { available: false, creation: "reference" }))).toBe(
            "family_unavailable",
        );
        expect(familyDraftProblem(draft("meanvc", { referenceVoiceId: "a" }), fam("sigil_zs"))).toBe("family_unavailable");
        // Sidecar recipe settings Rust would refuse are caught before the form closes.
        const xvc = { ...familyOptionsForm("sidecar_v2", {}), recipe: "xvc" as const };
        expect(familyDraftProblem(draft("sidecar_v2", { form: { ...xvc, python: "venv\\Scripts\\python.exe" } }), fam("sidecar_v2"))).toBe("sidecar_python_relative");
        expect(familyDraftProblem(draft("sidecar_v2", { form: { ...xvc, args: ["--port=9000"] } }), fam("sidecar_v2"))).toBe("sidecar_args_reserved");
    });

    it("a folder with a dot in its name is a folder, not a file of the wrong type", () => {
        const sovits = fam("sovits");
        const ddsp = fam("ddsp_svc");
        // Picked with "Folder…": never checked by extension.
        expect(familyDraftProblem(draft("sovits", { source: "C:\\models\\so-vits-svc-4.1", sourceIsDir: true }), sovits)).toBeNull();
        expect(familyDraftProblem(draft("ddsp_svc", { source: "D:\\voices\\singer.pt", sourceIsDir: true }), ddsp)).toBeNull();
        // Typed: an "extension" that is no known file type may be a folder name; Rust checks it.
        expect(familyDraftProblem(draft("sovits", { source: "C:\\models\\so-vits-svc-4.1" }), sovits)).toBeNull();
        expect(familyDraftProblem(draft("ddsp_svc", { source: "C:/v/DDSP-SVC 6.3/" }), ddsp)).toBeNull();
        expect(familyDraftProblem(draft("ddsp_svc", { source: "D:/voices/singer.v2" }), ddsp)).toBeNull();
        // A known file type the family does not import is still refused.
        expect(familyDraftProblem(draft("sovits", { source: "C:/v/model.pt" }), sovits)).toBe("source_type");
        expect(familyDraftProblem(draft("ddsp_svc", { source: "C:/v/take.wav" }), ddsp)).toBe("source_type");
        expect(importSourceProblem("  ", true, ddsp!)).toBe("source_required");
        expect(KNOWN_FILE_EXTENSIONS).toEqual(expect.arrayContaining([".pt", ".onnx", ".zip", ".pth", ".index"]));
        expect(newFamilyDraft("sovits", "", true).sourceIsDir).toBe(false);
    });

    it("switching family keeps the name, a zero-shot reference, and a file or folder the new family imports too", () => {
        const folder = draft("ddsp_svc", { name: "Singer", source: "D:\\voices\\singer.v2", sourceIsDir: true });
        expect(switchDraftFamily(folder, fam("sovits")!)).toMatchObject({ family: "sovits", name: "Singer", source: "D:\\voices\\singer.v2", sourceIsDir: true });
        const typedFolder = draft("ddsp_svc", { source: "C:\\models\\so-vits-svc-4.1" });
        expect(switchDraftFamily(typedFolder, fam("sovits")!).source).toBe("C:\\models\\so-vits-svc-4.1");
        // A .pt is DDSP-SVC's only; a .zip both import.
        expect(switchDraftFamily(draft("ddsp_svc", { source: "C:/v/model.pt" }), fam("sovits")!).source).toBe("");
        expect(switchDraftFamily(draft("ddsp_svc", { source: "C:/v/voice.zip" }), fam("sovits")!).source).toBe("C:/v/voice.zip");
        // Non-import families take no file; zero-shot ones keep the reference.
        const zs = switchDraftFamily(draft("meanvc", { referenceVoiceId: "alice", source: "C:/v/voice.zip", sourceIsDir: true }), fam("sigil_zs")!);
        expect(zs).toMatchObject({ family: "sigil_zs", referenceVoiceId: "alice", source: "", sourceIsDir: false, form: familyOptionsForm("sigil_zs", {}) });
        expect(switchDraftFamily(draft("sigil_zs", { referenceVoiceId: "alice" }), fam("ddsp_svc")!).referenceVoiceId).toBe("");
    });

    it("a failed voice's request reopens as the same form", () => {
        const form = { ...familyOptionsForm("sidecar_v2", {}), recipe: "xvc" as const, python: "C:/v/python.exe", args: ["--steps", "6"] };
        const sent = familySlotRequest(draft("sidecar_v2", { name: "X", referenceVoiceId: "alice", form }), fam("sidecar_v2")!);
        const again = familyDraftFromRequest(sent);
        expect(again).toEqual({ family: "sidecar_v2", name: "X", referenceVoiceId: "alice", source: "", sourceIsDir: false, form });
        expect(familySlotRequest(again, fam("sidecar_v2")!)).toEqual(sent);
        const ddsp = familyDraftFromRequest({ family: "ddsp_svc", source: " C:/v/model.pt ", options: null });
        expect(ddsp).toMatchObject({ source: "C:/v/model.pt", name: "", form: familyOptionsForm("ddsp_svc", {}) });
    });

    it("builds the request: options always, reference only for zero-shot families, the file only for imports, never activate", () => {
        expect(familySlotRequest(draft("sigil_zs", { name: "  Alto ", referenceVoiceId: "alice", source: "C:/x.pt" }), fam("sigil_zs")!)).toEqual({
            family: "sigil_zs",
            name: "Alto",
            reference_voice_id: "alice",
            options: familyOptionsPayload("sigil_zs", familyOptionsForm("sigil_zs", {})),
            activate: false,
        });
        expect(familySlotRequest(draft("ddsp_svc", { referenceVoiceId: "alice", source: " C:/v/model.pt " }), fam("ddsp_svc")!)).toEqual({
            family: "ddsp_svc",
            source: "C:/v/model.pt",
            options: { preset: "sigil_singing", steps: null, t_start: null },
            activate: false,
        });
        const sidecar = familySlotRequest(
            draft("sidecar_v2", { referenceVoiceId: "bob", form: { ...familyOptionsForm("sidecar_v2", {}), url: "ws://127.0.0.1:9/vc" } }),
            fam("sidecar_v2")!,
        );
        expect(sidecar).toMatchObject({ family: "sidecar_v2", reference_voice_id: "bob", options: { recipe: "custom", url: "ws://127.0.0.1:9/vc" } });
        expect(sidecar.source).toBeUndefined();
    });

    it("a dropped file goes to the one available family that imports it; RVC files and ambiguous ones do not", () => {
        expect(familyForImportFile("C:\\v\\model_100.pt", caps())).toBe("ddsp_svc");
        expect(familyForImportFile("C:/v/singer.ONNX", caps())).toBe("sovits");
        expect(familyForImportFile("C:/v/voice.zip", caps())).toBeNull();
        expect(familyForImportFile("C:/v/voice.pth", caps())).toBeNull();
        expect(familyForImportFile("C:/v/readme", caps())).toBeNull();
        expect(familyForImportFile("C:/v/model.pt", caps({ families: [famCap("ddsp_svc", { available: false, creation: "import", import_extensions: [".pt"] })] }))).toBeNull();
        const two = caps({
            families: [
                famCap("ddsp_svc", { creation: "import", import_extensions: [".pt"] }),
                famCap("sovits", { creation: "import", import_extensions: [".pt"] }),
            ],
        });
        expect(familyForImportFile("C:/v/model.pt", two)).toBeNull();
        expect(extensionOf("C:/v/a.b/model.PT")).toBe(".pt");
        expect(extensionOf("C:/v/folder/")).toBe("");
        expect(familyPickerExtensions(fam("ddsp_svc")!)).toEqual(["pt", "zip"]);
    });
});

// ---- strings ----------------------------------------------------------------------------------------

const staged = import.meta.glob("../../../../../scripts/i18n-staging/*.json", { eager: true, import: "default" }) as Record<string, unknown>;
const uiSources = import.meta.glob("../../../ui/inspector/voice_changer/*.tsx", { eager: true, query: "?raw", import: "default" }) as Record<string, string>;

/** English keys of `voice_changer`, plus every staged one (the INTEGRATE merge adds them to English). */
function knownKeys(): Set<string> {
    const keys = new Set(Object.keys((en as { voice_changer: Record<string, string> }).voice_changer));
    for (const doc of Object.values(staged)) {
        const vc = (doc as { voice_changer?: Record<string, unknown> }).voice_changer;
        if (vc) for (const k of Object.keys(vc)) keys.add(k);
    }
    return keys;
}

describe("family UI strings", () => {
    it("every static voice_changer key the inspector uses exists", () => {
        expect(Object.keys(uiSources).some((p) => p.endsWith("/FamilySection.tsx"))).toBe(true);
        expect(Object.keys(uiSources).length).toBeGreaterThan(10);
        const known = knownKeys();
        const missing: string[] = [];
        for (const [path, src] of Object.entries(uiSources)) {
            for (const m of src.matchAll(/["'`]voice_changer\.([a-z0-9_]+)["'`]/g)) {
                if (!known.has(m[1])) missing.push(`${path.split("/").pop()}: ${m[1]}`);
            }
        }
        expect(missing).toEqual([]);
    });

    it("every key built from a family, preset, recipe or problem exists", () => {
        const known = knownKeys();
        const want: string[] = [];
        for (const id of VC_FAMILIES) want.push(`family_name_${id}`, `family_short_${id}`, `family_about_${id}`, `family_info_${id}`);
        for (const c of ["reference", "import", "sidecar"]) want.push(`family_name_placeholder_${c}`);
        for (const p of DDSP_SVC_PRESETS) want.push(`family_ddsp_preset_${p}`);
        for (const p of SIGIL_ZS_PRESETS) want.push(`family_zs_preset_${p}`);
        for (const r of SIDECAR_RECIPES) want.push(`family_sidecar_recipe_${r}`, `family_sidecar_recipe_${r}_about`);
        for (const s of MEANVC_STEPS) want.push(`family_meanvc_steps_${s}`);
        for (const ms of MEANVC_TICK_MS) want.push(`family_meanvc_tick_${ms}`);
        for (const p of FAMILY_DRAFT_PROBLEMS) want.push(`family_problem_${p}`);
        for (const f of ["ddsp_svc", "sovits"]) want.push(`family_source_hint_${f}`);
        for (const op of ["rename", "duplicate", "delete", "merge", "options"]) want.push(`voice_busy_${op}`);
        for (const ep of ["cpu", "cuda", "dml"]) want.push(`ep_name_${ep}`);
        for (const m of ["auto", "on", "off"]) want.push(`lever_mode_${m}`);
        expect(want.filter((k) => !known.has(k))).toEqual([]);
    });
});
