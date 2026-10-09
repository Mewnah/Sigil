/**
 * P5 voice families in the voice-changer UI (P5-G): which family a voice is, what the inspector
 * shows for it (its hot controls, its EP rungs, its model-store entries), the typed options form of
 * each family (`manifest.family`, mirroring Rust `vc::families::<family>::normalize_options`), the
 * checks a new family voice passes before `voice_changer_create_family_slot`, and which family a
 * dropped file belongs to.
 *
 * Everything is driven by `VcCapabilities.families` (Rust `FamilyInfo` + `available`): a family is
 * offered only when this build runs it, and its controls, EP rungs, creation kind, import
 * extensions and requirements come from Rust, never from a copy here. The only family facts kept
 * here are the options contract (preset / recipe ids and ranges), which `__tests__/families.test.ts`
 * checks against the Rust sources.
 *
 * Pure module: no Tauri, DOM or `window`.
 */
import { rvcBaseEp, type ExecutionProviderId, type RvcExecutionProviderId } from "./contract";
import {
    familyOfFormat,
    VC_FAMILIES,
    type VcCapabilities,
    type VcDdspSvcOptions,
    type VcFamilyCapability,
    type VcFamilyControl,
    type VcFamilyId,
    type VcFamilyOptions,
    type VcFamilySlotRequest,
    type VcMeanVcOptions,
    type VcSidecarV2Options,
    type VcSigilZsOptions,
    type VcSlotInfo,
    type VcSovitsOptions,
} from "./ipc";
import { SIGIL_RVC_FORMAT } from "./slots";

// ---------------------------------------------------------------------------------------------
// Options contract (Rust `families::<family>`; checked against the Rust sources by the tests)
// ---------------------------------------------------------------------------------------------

/** DDSP-SVC presets (Rust `ddsp_svc::PRESETS`; the first is the default). */
export const DDSP_SVC_PRESETS = ["sigil_singing", "sigil_singing_fast", "upstream_gui"] as const;
export type DdspSvcPreset = (typeof DDSP_SVC_PRESETS)[number];

/** Sigil-ZS presets (Rust `sigil_zs::PRESETS`; the first is the default). */
export const SIGIL_ZS_PRESETS = ["sigil_low", "sigil_lean", "upstream_readme"] as const;
export type SigilZsPreset = (typeof SIGIL_ZS_PRESETS)[number];

/** Sidecar recipes (Rust `sidecar_v2::RECIPES`; the first is the default). */
export const SIDECAR_RECIPES = ["custom", "xvc", "streamvoiceanon", "seedvc"] as const;
export type SidecarRecipe = (typeof SIDECAR_RECIPES)[number];

/** MeanVC2 mean-flow steps and tick lengths (Rust `MeanVcOptions`; the last step count and the first tick are the defaults). */
export const MEANVC_STEPS = [1, 2] as const;
export const MEANVC_TICK_MS = [80, 160] as const;

/** Ranges of the numeric options (Rust `normalize_options` clamps). */
export const FAMILY_OPTION_RANGES = {
    ddspSteps: { min: 1, max: 50 },
    /** Upstream's flow start, used when unset. */
    ddspTStart: { min: 0, max: 1, default: 0.7 },
    zsSteps: { min: 1, max: 30 },
    zsCfgRate: { min: 0, max: 1 },
    zsPromptS: { min: 0.5, max: 10 },
    sovitsClusterRatio: { min: 0, max: 1 },
    sidecarTimeoutMs: { min: 20, max: 5000 },
    /** At most this many sidecar arguments, each at most `maxLength` UTF-8 bytes (Rust `recipes::validate_args`). */
    sidecarArgs: { maxCount: 64, maxLength: 1024 },
    /** The sidecar's Python interpreter and model folder: at most this many UTF-8 bytes (Rust `recipes::validate_python` / `validate_model_dir`). */
    sidecarPathMaxBytes: 1024,
} as const;

/** Sidecar arguments Sigil sets itself (Rust `recipes::RESERVED_ARGS`; compared before any `=`, ignoring case). */
export const SIDECAR_RESERVED_ARGS = ["--host", "--port", "--key", "--api-key", "--allow-remote"] as const;

/** The options form of each family: every field present, `null` = the preset's / Rust's default. */
export interface DdspSvcForm {
    preset: DdspSvcPreset;
    steps: number | null;
    t_start: number | null;
}
export interface SigilZsForm {
    preset: SigilZsPreset;
    steps: number | null;
    cfg_rate: number | null;
    prompt_s: number | null;
    fp16_xlsr: boolean;
    fp16_dit: boolean;
    fp16_vocoder: boolean | null;
}
export interface MeanVcForm {
    steps: 1 | 2;
    tick_ms: 80 | 160;
}
export interface SovitsForm {
    cluster_ratio: number;
}
export interface SidecarV2Form {
    recipe: SidecarRecipe;
    url: string;
    python: string;
    model_dir: string;
    args: string[];
    timeout_ms: number | null;
}
export interface FamilyForms {
    ddsp_svc: DdspSvcForm;
    sigil_zs: SigilZsForm;
    meanvc: MeanVcForm;
    sovits: SovitsForm;
    sidecar_v2: SidecarV2Form;
}
/** A family id with its form (a discriminated union, so a component can switch on `family`). */
export type FamilyFormOf = { [F in VcFamilyId]: { family: F; form: FamilyForms[F] } }[VcFamilyId];

const obj = (raw: unknown): Record<string, unknown> => (raw && typeof raw === "object" && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {});
const finite = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v));
/** An optional integer option clamped to its range (`null` when unset or not a number), like Rust's `Option<u32>` clamp. */
const optInt = (v: unknown, r: { min: number; max: number }): number | null => {
    const n = finite(v);
    return n === null ? null : clamp(Math.round(n), r.min, r.max);
};
const optFloat = (v: unknown, r: { min: number; max: number }): number | null => {
    const n = finite(v);
    return n === null ? null : clamp(n, r.min, r.max);
};
const oneOf = <T extends string>(all: readonly T[], v: unknown, fallback: T): T =>
    typeof v === "string" && (all as readonly string[]).includes(v.trim().toLowerCase()) ? (v.trim().toLowerCase() as T) : fallback;
const str = (v: unknown): string => (typeof v === "string" ? v.trim() : "");

const R = FAMILY_OPTION_RANGES;

/**
 * The form of `family` from a stored or partial options object (`manifest.family`), normalized like
 * Rust does: unknown presets take the default, numbers are clamped, unset stays unset. (Rust
 * refuses an unknown sidecar recipe; the form shows it as `custom`, the default.)
 */
export function familyOptionsForm<F extends VcFamilyId>(family: F, raw: unknown): FamilyForms[F] {
    const o = obj(raw);
    const forms: { [K in VcFamilyId]: () => FamilyForms[K] } = {
        ddsp_svc: () => ({
            preset: oneOf(DDSP_SVC_PRESETS, o.preset, DDSP_SVC_PRESETS[0]),
            steps: optInt(o.steps, R.ddspSteps),
            t_start: optFloat(o.t_start, R.ddspTStart),
        }),
        sigil_zs: () => ({
            preset: oneOf(SIGIL_ZS_PRESETS, o.preset, SIGIL_ZS_PRESETS[0]),
            steps: optInt(o.steps, R.zsSteps),
            cfg_rate: optFloat(o.cfg_rate, R.zsCfgRate),
            prompt_s: optFloat(o.prompt_s, R.zsPromptS),
            fp16_xlsr: o.fp16_xlsr === true,
            fp16_dit: o.fp16_dit === true,
            fp16_vocoder: typeof o.fp16_vocoder === "boolean" ? o.fp16_vocoder : null,
        }),
        meanvc: () => {
            const steps = finite(o.steps);
            const tick = finite(o.tick_ms);
            return {
                steps: steps === null ? 2 : steps <= 1 ? 1 : 2,
                tick_ms: tick !== null && tick >= 120 ? 160 : 80,
            };
        },
        sovits: () => ({ cluster_ratio: optFloat(o.cluster_ratio, R.sovitsClusterRatio) ?? 0 }),
        sidecar_v2: () => ({
            recipe: oneOf(SIDECAR_RECIPES, o.recipe, SIDECAR_RECIPES[0]),
            url: str(o.url),
            python: str(o.python),
            model_dir: str(o.model_dir),
            args: Array.isArray(o.args) ? o.args.filter((a): a is string => typeof a === "string") : [],
            timeout_ms: optInt(o.timeout_ms, R.sidecarTimeoutMs),
        }),
    };
    return forms[family]() as FamilyForms[F];
}

/** The options object sent to Rust for `form` (`voice_changer_create_family_slot` / `_set_family_options`). */
export function familyOptionsPayload<F extends VcFamilyId>(family: F, form: FamilyForms[F]): VcFamilyOptions[F] {
    switch (family) {
        case "ddsp_svc": {
            const f = form as DdspSvcForm;
            return { preset: f.preset, steps: f.steps, t_start: f.t_start } satisfies VcDdspSvcOptions as VcFamilyOptions[F];
        }
        case "sigil_zs": {
            const f = form as SigilZsForm;
            return {
                preset: f.preset,
                steps: f.steps,
                cfg_rate: f.cfg_rate,
                prompt_s: f.prompt_s,
                fp16_xlsr: f.fp16_xlsr,
                fp16_dit: f.fp16_dit,
                fp16_vocoder: f.fp16_vocoder,
            } satisfies VcSigilZsOptions as VcFamilyOptions[F];
        }
        case "meanvc": {
            const f = form as MeanVcForm;
            return { steps: f.steps, tick_ms: f.tick_ms } satisfies VcMeanVcOptions as VcFamilyOptions[F];
        }
        case "sovits": {
            const f = form as SovitsForm;
            return { cluster_ratio: f.cluster_ratio } satisfies VcSovitsOptions as VcFamilyOptions[F];
        }
        case "sidecar_v2": {
            const f = form as SidecarV2Form;
            // A custom server is never started by Sigil: its form hides the interpreter, model folder and
            // arguments, so they are not sent (Rust would still validate them).
            const helper = sidecarRecipeStartsHelper(f.recipe);
            return {
                recipe: f.recipe,
                url: f.url.trim(),
                python: helper ? f.python.trim() : "",
                model_dir: helper ? f.model_dir.trim() : "",
                args: helper ? cleanSidecarArgs(f.args) : [],
                timeout_ms: f.timeout_ms,
            } satisfies VcSidecarV2Options as VcFamilyOptions[F];
        }
        default:
            return {} as VcFamilyOptions[F];
    }
}

/** Whether two forms send the same options (to tell an edited form from the stored one). */
export function sameFamilyOptions<F extends VcFamilyId>(family: F, a: FamilyForms[F], b: FamilyForms[F]): boolean {
    return JSON.stringify(familyOptionsPayload(family, a)) === JSON.stringify(familyOptionsPayload(family, b));
}

/**
 * Whether `url` is a WebSocket URL on this PC (Rust `sidecar_v2::loopback_ws_url`): `ws://` to
 * 127.0.0.1, localhost or [::1], without credentials. Sidecars never leave loopback.
 */
export function isLoopbackWsUrl(url: string): boolean {
    const u = url.trim().toLowerCase();
    if (!u.startsWith("ws://")) return false;
    const hostPort = u.slice("ws://".length).split(/[/?#]/)[0] ?? "";
    if (hostPort.includes("@")) return false;
    const host = hostPort.startsWith("[") ? `[${hostPort.slice(1).split("]")[0] ?? ""}]` : (hostPort.split(":")[0] ?? "");
    return host === "127.0.0.1" || host === "localhost" || host === "[::1]";
}

/** Why a family's options would be refused by Rust (an i18n suffix: `voice_changer.family_problem_<code>`). */
export const FAMILY_OPTIONS_PROBLEMS = [
    "sidecar_url_required",
    "sidecar_url_not_loopback",
    "sidecar_python_invalid",
    "sidecar_python_relative",
    "sidecar_model_dir_invalid",
    "sidecar_model_dir_relative",
    "sidecar_args_too_long",
    "sidecar_args_invalid",
    "sidecar_args_reserved",
] as const;
export type FamilyOptionsProblem = (typeof FAMILY_OPTIONS_PROBLEMS)[number];

/** The sidecar form field a problem belongs to (where the form shows it). */
export function sidecarProblemField(problem: FamilyOptionsProblem | null): "url" | "python" | "model_dir" | "args" | null {
    if (!problem) return null;
    if (problem.startsWith("sidecar_url_")) return "url";
    if (problem.startsWith("sidecar_python_")) return "python";
    if (problem.startsWith("sidecar_model_dir_")) return "model_dir";
    return "args";
}

/** UTF-8 length of `s` (Rust's `str::len`). */
function utf8Length(s: string): number {
    let n = 0;
    for (const ch of s) {
        const c = ch.codePointAt(0) ?? 0;
        n += c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
    }
    return n;
}

/** Rust's `char::is_control` (Unicode Cc). */
const CONTROL_CHARS = /[\u0000-\u001f\u007f-\u009f]/;

/**
 * Whether `path` is a full path: `C:\…` / `C:/…`, a UNC or verbatim path (`\\server\share`, `\\?\C:\…`), or a
 * POSIX `/…` path. A path this accepts may still be refused by Rust (e.g. `/x` on Windows); it never refuses
 * one Rust accepts.
 */
export function isAbsolutePath(path: string): boolean {
    const p = path.trim();
    return /^[a-zA-Z]:[\\/]/.test(p) || /^[\\/]{2}[^\\/]/.test(p) || p.startsWith("/");
}

/** Sidecar arguments as they are sent: trimmed, blank ones dropped. */
function cleanSidecarArgs(args: readonly string[]): string[] {
    return args.map((a) => a.trim()).filter((a) => a !== "");
}

/** The problem of the sidecar's interpreter (Rust `recipes::validate_python`): empty, a bare name on PATH, or a full path. */
function sidecarPythonProblem(python: string): FamilyOptionsProblem | null {
    const p = python.trim();
    if (utf8Length(p) > R.sidecarPathMaxBytes || CONTROL_CHARS.test(p)) return "sidecar_python_invalid";
    if (/[\\/]/.test(p) && !isAbsolutePath(p)) return "sidecar_python_relative";
    return null;
}

/** The problem of the sidecar's model folder (Rust `recipes::validate_model_dir`): empty or a full path. */
function sidecarModelDirProblem(dir: string): FamilyOptionsProblem | null {
    const d = dir.trim();
    if (utf8Length(d) > R.sidecarPathMaxBytes || CONTROL_CHARS.test(d)) return "sidecar_model_dir_invalid";
    if (d !== "" && !isAbsolutePath(d)) return "sidecar_model_dir_relative";
    return null;
}

/** The problem of the sidecar's extra arguments (Rust `recipes::validate_args`). */
function sidecarArgsProblem(raw: readonly string[]): FamilyOptionsProblem | null {
    const args = cleanSidecarArgs(raw);
    if (args.length > R.sidecarArgs.maxCount || args.some((a) => utf8Length(a) > R.sidecarArgs.maxLength)) return "sidecar_args_too_long";
    if (args.some((a) => /[\0\r\n]/.test(a))) return "sidecar_args_invalid";
    const reserved: readonly string[] = SIDECAR_RESERVED_ARGS;
    if (args.some((a) => reserved.includes((a.split("=")[0] ?? "").toLowerCase()))) return "sidecar_args_reserved";
    return null;
}

/**
 * The first problem of `form` that Rust's `normalize_options` would refuse, else `null` (the cheap checks:
 * Rust still checks that the files exist when the voice loads). Checks only what the payload sends.
 */
export function familyOptionsProblem<F extends VcFamilyId>(family: F, form: FamilyForms[F]): FamilyOptionsProblem | null {
    if (family !== "sidecar_v2") return null;
    const f = form as SidecarV2Form;
    const url = f.url.trim();
    if (url !== "" && !isLoopbackWsUrl(url)) return "sidecar_url_not_loopback";
    if (f.recipe === "custom" && url === "") return "sidecar_url_required";
    if (!sidecarRecipeStartsHelper(f.recipe)) return null;
    return sidecarPythonProblem(f.python) ?? sidecarModelDirProblem(f.model_dir) ?? sidecarArgsProblem(f.args);
}

/** Sidecar recipes that start a helper (every one but `custom`, which connects to the user's server). */
export function sidecarRecipeStartsHelper(recipe: SidecarRecipe): boolean {
    return recipe !== "custom";
}

/**
 * The sidecar form after the user picks another recipe. A custom server's address is not where a recipe's
 * helper listens, so it is cleared when switching to a recipe (which then starts its helper; the form shows an
 * optional address to attach to one already running). The interpreter, model folder and arguments stay in the
 * form for a later switch back, but a custom server never sends them (`familyOptionsPayload`).
 */
export function switchSidecarRecipe(form: SidecarV2Form, recipe: SidecarRecipe): SidecarV2Form {
    if (recipe === form.recipe) return form;
    const fromCustom = !sidecarRecipeStartsHelper(form.recipe);
    return { ...form, recipe, url: fromCustom && sidecarRecipeStartsHelper(recipe) ? "" : form.url };
}

/**
 * Sidecar arguments as edited in a text box: one per line (blank lines dropped). Lines keep their
 * inner spaces (an argument may contain them); quoting is not interpreted.
 */
export function parseSidecarArgs(text: string): string[] {
    return text
        .split(/\r?\n/)
        .map((l) => l.trim())
        .filter((l) => l !== "");
}

// ---------------------------------------------------------------------------------------------
// Which family a voice is, and what the inspector shows for it
// ---------------------------------------------------------------------------------------------

/** The family of a voice (its `family` field, else its `manifest.format`), else `null` (an RVC voice). */
export function familyOf(slot: Pick<VcSlotInfo, "format" | "family"> | null | undefined): VcFamilyId | null {
    if (!slot) return null;
    return familyOfFormat(slot.family ?? null) ?? familyOfFormat(slot.format);
}

/** The capability row of `family` (`null` when this backend does not list it). */
export function familyCapability(caps: Pick<VcCapabilities, "families">, family: VcFamilyId | null): VcFamilyCapability | null {
    if (!family) return null;
    return caps.families.find((f) => f.family === family) ?? null;
}

/** The capability row of a voice's family (`null` for RVC voices, or a family this backend does not list). */
export function voiceFamilyCapability(
    caps: Pick<VcCapabilities, "families">,
    slot: Pick<VcSlotInfo, "format" | "family"> | null | undefined,
): VcFamilyCapability | null {
    return familyCapability(caps, familyOf(slot));
}

/**
 * What runs a voice: no listed voice selected (`none`), an imported template voice on the Sigil
 * Voice Engine (`template`), a bring-your-own ONNX voice (`legacy`), or a P5 family voice (`family`).
 */
export type VoiceKind = "none" | "template" | "legacy" | "family";

export function voiceKind(slot: Pick<VcSlotInfo, "format" | "family"> | null | undefined): VoiceKind {
    if (!slot) return "none";
    if (familyOf(slot)) return "family";
    return slot.format === SIGIL_RVC_FORMAT ? "template" : "legacy";
}

/**
 * Whether the Sigil Voice Engine's own controls apply (engine block, lookahead, content context, device
 * block): a template voice, or no listed voice (the default setup). Not a bring-your-own ONNX voice, and
 * not a family voice either (a family sets its own block).
 */
export function runsOnTemplateEngine(kind: VoiceKind): boolean {
    return kind !== "legacy" && kind !== "family";
}

/** Families this build runs, in `VC_FAMILIES` order. */
export function availableFamilies(caps: Pick<VcCapabilities, "families">): VcFamilyCapability[] {
    return VC_FAMILIES.flatMap((id) => {
        const f = caps.families.find((x) => x.family === id);
        return f && f.available ? [f] : [];
    });
}

/** Whether the family runs on the EP ladder rung of `ep` (`tensorrt_rtx` runs on CUDA, `dml_hp` on DirectML). */
export function familyRunsOnEp(fam: Pick<VcFamilyCapability, "execution_providers">, ep: RvcExecutionProviderId | string): boolean {
    if (fam.execution_providers.length === 0) return true;
    return fam.execution_providers.includes(rvcBaseEp(ep));
}

/**
 * The execution provider to switch to for a family that does not run on the current rung: its best
 * rung (the family lists them best first) that this build offers, else `null`.
 */
export function familyEpChoice(
    fam: Pick<VcFamilyCapability, "execution_providers">,
    caps: Pick<VcCapabilities, "rvc_execution_providers">,
): ExecutionProviderId | null {
    for (const ep of fam.execution_providers) {
        if (ep === "cpu" || ep === "cuda" || ep === "dml") {
            if (caps.rvc_execution_providers.includes(ep)) return ep;
        }
    }
    return null;
}

/**
 * Which per-voice controls the inspector shows for a voice: every one for RVC voices (`null`; the
 * existing template / bring-your-own rules apply), else only the family's (`VcFamilyCapability.controls`),
 * plus whether it takes an F0 method and the F0 range (`needs_pitch`).
 */
export interface FamilyVoiceControls {
    controls: ReadonlySet<VcFamilyControl>;
    needsPitch: boolean;
}

export function familyVoiceControls(
    caps: Pick<VcCapabilities, "families">,
    slot: Pick<VcSlotInfo, "format" | "family"> | null | undefined,
): FamilyVoiceControls | null {
    const fam = voiceFamilyCapability(caps, slot);
    if (!fam) return null;
    return { controls: new Set(fam.controls), needsPitch: fam.needs_pitch };
}

/** Whether a control is shown for a voice (`controls` = `familyVoiceControls(...)`; RVC voices show every one). */
export function showsControl(controls: FamilyVoiceControls | null, control: VcFamilyControl): boolean {
    return controls === null || controls.controls.has(control);
}

/** Rust `chunk_processor::FORMANT_SEMITONES_EPSILON`: a smaller `formant_semitones` is off. */
export const FORMANT_SEMITONES_EPSILON = 0.01;

/**
 * Whether the true formant shift is on (Rust `rvc_output_tilt`). In RVC mode it turns the classic Brightness
 * tilt off for every voice, whatever its family, including a family that ignores the shift itself.
 */
export function formantShiftOn(formantSemitones: number): boolean {
    return Number.isFinite(formantSemitones) && Math.abs(formantSemitones) >= FORMANT_SEMITONES_EPSILON;
}

/**
 * RVC mode: the voice's family ignores Formant (so the inspector hides it), yet a Formant value is set (e.g.
 * inherited from the previous voice, since a new family voice's `slot.json` does not hold one) and still turns
 * Brightness off. The inspector then offers to reset it.
 */
export function hiddenFormantBlocksTilt(controls: FamilyVoiceControls | null, formantSemitones: number): boolean {
    return !showsControl(controls, "formant") && formantShiftOn(formantSemitones);
}

/** A singing family that tracks pitch: the inspector suggests singing mode (F0 range `singing`) for it. */
export function singingRecommended(fam: Pick<VcFamilyCapability, "singing" | "needs_pitch"> | null): boolean {
    return !!fam && fam.singing && fam.needs_pitch;
}

// ---------------------------------------------------------------------------------------------
// The zero-shot reference
// ---------------------------------------------------------------------------------------------

/** A zero-shot voice's reference against the voice library: none chosen, listed, missing (deleted), or not known yet (list not read). */
export type ReferenceState = "none" | "ok" | "missing" | "unknown";

export function referenceState(id: string | null | undefined, voices: readonly { id: string }[], loaded: boolean): ReferenceState {
    const v = (id ?? "").trim();
    if (v === "") return "none";
    if (!loaded) return "unknown";
    return voices.some((x) => x.id === v) ? "ok" : "missing";
}

/** A voice-library voice's display name, else its id. */
export function referenceName(id: string | null | undefined, voices: readonly { id: string; name: string }[]): string {
    const v = (id ?? "").trim();
    return voices.find((x) => x.id === v)?.name || v;
}

/** `8.2 s` style duration of a reference clip. */
export function clipSecondsLabel(ms: number | null | undefined): string | null {
    if (typeof ms !== "number" || !Number.isFinite(ms) || ms <= 0) return null;
    return `${(ms / 1000).toFixed(1)} s`;
}

// ---------------------------------------------------------------------------------------------
// New family voices (`voice_changer_create_family_slot`)
// ---------------------------------------------------------------------------------------------

/** What the "new voice" form holds. */
export interface FamilyDraft {
    family: VcFamilyId;
    /** Display name (`""` = Rust names it from the reference voice, the file or the family). */
    name: string;
    /** The library voice a zero-shot voice converts toward (`""` = none). */
    referenceVoiceId: string;
    /** The file or folder to import (import families). */
    source: string;
    /** `source` came from the folder picker (a typed path is not known to be one). */
    sourceIsDir: boolean;
    form: FamilyForms[VcFamilyId];
}

export function newFamilyDraft(family: VcFamilyId, source = "", sourceIsDir = false): FamilyDraft {
    return { family, name: "", referenceVoiceId: "", source, sourceIsDir: sourceIsDir && source.trim() !== "", form: familyOptionsForm(family, {}) };
}

/**
 * The form of a request sent before (e.g. to edit and retry a family voice that failed). Whether its source
 * was a folder is not kept; a folder whose name has a dot still passes (`importSourceProblem`).
 */
export function familyDraftFromRequest(request: VcFamilySlotRequest): FamilyDraft {
    return {
        family: request.family,
        name: request.name?.trim() ?? "",
        referenceVoiceId: request.reference_voice_id?.trim() ?? "",
        source: request.source?.trim() ?? "",
        sourceIsDir: false,
        form: familyOptionsForm(request.family, request.options ?? {}),
    };
}

/** Why the form cannot be sent yet (an i18n suffix: `voice_changer.family_problem_<code>`). */
export type FamilyDraftProblem = "family_unavailable" | "reference_required" | "source_required" | "source_type" | FamilyOptionsProblem;

/** Every `FamilyDraftProblem` shown to the user (`family_unavailable` disables the form instead). */
export const FAMILY_DRAFT_PROBLEMS = ["reference_required", "source_required", "source_type", ...FAMILY_OPTIONS_PROBLEMS] as const;

/** The extension of a path's file name (lower case, with the dot; `""` for none or a folder-like path). */
export function extensionOf(path: string): string {
    const name = path.trim().replace(/\\/g, "/").replace(/\/+$/, "").split("/").pop() ?? "";
    const dot = name.lastIndexOf(".");
    return dot > 0 ? name.slice(dot).toLowerCase() : "";
}

/**
 * Extensions that name a file for sure (model, archive, audio, config, text, program). A typed path whose
 * "extension" is something else (`so-vits-svc-4.1`, `singer.v2`) may be a folder with a dot in its name, which
 * the importers accept: that is left to Rust.
 */
export const KNOWN_FILE_EXTENSIONS: readonly string[] = [
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin", ".index", ".npy", ".pkl",
    ".zip", ".7z", ".rar", ".tar", ".gz",
    ".wav", ".mp3", ".flac", ".ogg", ".m4a",
    ".json", ".yaml", ".yml", ".txt", ".md",
    ".exe", ".py",
];

/**
 * Why an import family cannot take `source`, else `null`. A folder (picked as one, or a path without an
 * extension) is allowed: the family looks inside it. A file is refused only when its extension is a known file
 * type the family does not import.
 */
export function importSourceProblem(
    source: string,
    isDir: boolean,
    fam: Pick<VcFamilyCapability, "import_extensions">,
): "source_required" | "source_type" | null {
    const src = source.trim();
    if (src === "") return "source_required";
    if (isDir) return null;
    const ext = extensionOf(src);
    if (ext === "" || fam.import_extensions.length === 0 || fam.import_extensions.includes(ext)) return null;
    return KNOWN_FILE_EXTENSIONS.includes(ext) ? "source_type" : null;
}

/** The first problem of `draft`, else `null` (Rust checks the same and more). */
export function familyDraftProblem(draft: FamilyDraft, fam: VcFamilyCapability | null): FamilyDraftProblem | null {
    if (!fam || fam.family !== draft.family || !fam.available) return "family_unavailable";
    if (fam.creation === "reference" && draft.referenceVoiceId.trim() === "") return "reference_required";
    if (fam.creation === "import") {
        const problem = importSourceProblem(draft.source, draft.sourceIsDir, fam);
        if (problem) return problem;
    }
    return familyOptionsProblem(draft.family, draft.form as never);
}

/**
 * The draft after the user picks another family (`target`): the name stays, the reference when the target is
 * zero-shot, and the file or folder when the target imports it too. The options start at the target's defaults.
 */
export function switchDraftFamily(draft: FamilyDraft, target: VcFamilyCapability): FamilyDraft {
    if (draft.family === target.family) return draft;
    const keepSource = target.creation === "import" && draft.source.trim() !== "" && importSourceProblem(draft.source, draft.sourceIsDir, target) === null;
    return {
        ...newFamilyDraft(target.family),
        name: draft.name,
        referenceVoiceId: target.zero_shot ? draft.referenceVoiceId : "",
        source: keepSource ? draft.source : "",
        sourceIsDir: keepSource && draft.sourceIsDir,
    };
}

/** The `voice_changer_create_family_slot` request of a valid draft (never `activate`: the UI selects through the params sync). */
export function familySlotRequest(draft: FamilyDraft, fam: VcFamilyCapability): VcFamilySlotRequest {
    const request: VcFamilySlotRequest = {
        family: draft.family,
        options: familyOptionsPayload(draft.family, draft.form as never),
        activate: false,
    };
    const name = draft.name.trim();
    if (name) request.name = name;
    const reference = draft.referenceVoiceId.trim();
    if (fam.zero_shot && reference) request.reference_voice_id = reference;
    const source = draft.source.trim();
    if (fam.creation === "import" && source) request.source = source;
    return request;
}

/**
 * The import family a dropped or picked file belongs to: the one available family whose import
 * extensions include it. `.zip` (and `.pth` / `.index`) stay with the RVC importer, so they never
 * route here; an extension two families import is ambiguous (`null`).
 */
export function familyForImportFile(path: string, caps: Pick<VcCapabilities, "families">): VcFamilyId | null {
    const ext = extensionOf(path);
    if (ext === "" || ext === ".zip" || ext === ".pth" || ext === ".index") return null;
    const matches = availableFamilies(caps).filter((f) => f.creation === "import" && f.import_extensions.includes(ext));
    return matches.length === 1 ? matches[0].family : null;
}

/** File-picker extensions of an import family (without the dot). */
export function familyPickerExtensions(fam: Pick<VcFamilyCapability, "import_extensions">): string[] {
    return fam.import_extensions.map((e) => e.replace(/^\./, "")).filter((e) => e !== "");
}
