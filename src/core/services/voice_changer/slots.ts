/**
 * RVC voice slots under `app_data/voice-changer/models/`: the slot store's rows
 * (`voice_changer_list_slots`, Rust `vc::import::slot::SlotInfo`, TS `VcSlotInfo`), the legacy
 * folder list (`list_voice_changer_model_slots`), what a dropped file is, which voices can be
 * merged, and the content encoder / vocoder names shown read-only.
 *
 * Pure module: no Tauri, DOM or `window`.
 */
import type { VcSlotInfo } from "./ipc";

/** Slot format of voices imported from `.pth` / `.zip` (template voices, the Sigil Voice Engine). */
export const SIGIL_RVC_FORMAT = "sigil-rvc-1";
/** Slot format of bring-your-own ONNX voices. */
export const LEGACY_RVC_FORMAT = "rvc";

/** One slot folder, as the legacy `list_voice_changer_model_slots` lists it. */
export interface VoiceChangerModelSlot {
    /** Folder name (shown as the voice name). */
    id: string;
    /** Absolute folder path (what `rvc_slot_dir` stores). */
    path: string;
    /** The folder has a `manifest.json` (otherwise Sigil infers it from the model). */
    has_manifest: boolean;
}

/** Keep only well-formed rows (a stale or older backend must not break the picker). */
export function normalizeSlots(raw: unknown): VoiceChangerModelSlot[] {
    if (!Array.isArray(raw)) return [];
    const out: VoiceChangerModelSlot[] = [];
    for (const r of raw) {
        if (!r || typeof r !== "object") continue;
        const s = r as Partial<VoiceChangerModelSlot>;
        if (typeof s.id !== "string" || typeof s.path !== "string" || !s.path.trim()) continue;
        out.push({ id: s.id, path: s.path, has_manifest: s.has_manifest === true });
    }
    return out;
}

function optStr(v: unknown): string | null {
    return typeof v === "string" && v.trim() !== "" ? v.trim() : null;
}

function optNum(v: unknown): number | null {
    return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function optBool(v: unknown): boolean | null {
    return typeof v === "boolean" ? v : null;
}

/** One slot-store row, or `null` when it has no id or path (every other field is optional on the wire). */
export function normalizeSlotInfo(raw: unknown): VcSlotInfo | null {
    if (!raw || typeof raw !== "object") return null;
    const r = raw as Record<string, unknown>;
    const id = optStr(r.id);
    const path = optStr(r.path);
    if (!id || !path) return null;
    return {
        id,
        name: optStr(r.name) ?? id,
        path,
        format: optStr(r.format) ?? LEGACY_RVC_FORMAT,
        template: optStr(r.template),
        version: optStr(r.version),
        sample_rate: optNum(r.sample_rate),
        f0: optBool(r.f0),
        embedder: optStr(r.embedder),
        vocoder: optStr(r.vocoder),
        has_index: r.has_index === true,
        speaker_count: optNum(r.speaker_count),
        fp16_ok: optBool(r.fp16_ok),
        source_sha256: optStr(r.source_sha256),
        created_unix_ms: optNum(r.created_unix_ms),
        problem: optStr(r.problem),
    };
}

/** Slot-store rows, well-formed only, one per id (Rust sorts them by name). */
export function normalizeSlotInfos(raw: unknown): VcSlotInfo[] {
    if (!Array.isArray(raw)) return [];
    const seen = new Set<string>();
    const out: VcSlotInfo[] = [];
    for (const r of raw) {
        const s = normalizeSlotInfo(r);
        if (!s || seen.has(s.id)) continue;
        seen.add(s.id);
        out.push(s);
    }
    return out;
}

/** A legacy folder row as a slot-store row (an older backend without the slot store). */
export function slotInfoFromLegacy(s: VoiceChangerModelSlot): VcSlotInfo {
    return {
        id: s.id,
        name: s.id,
        path: s.path,
        format: LEGACY_RVC_FORMAT,
        template: null,
        version: null,
        sample_rate: null,
        f0: null,
        embedder: null,
        vocoder: null,
        has_index: false,
        speaker_count: null,
        fp16_ok: null,
        source_sha256: null,
        created_unix_ms: null,
        problem: null,
    };
}

/** Folder path key: separators unified, trailing separators dropped, case-insensitive (Windows paths). */
export function slotPathKey(path: string): string {
    return path.trim().replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
}

/** Whether two folder strings name the same folder (separator, trailing-slash and case insensitive). */
export function sameSlotPath(a: string, b: string): boolean {
    const ka = slotPathKey(a);
    return ka !== "" && ka === slotPathKey(b);
}

/** The listed slot the selected folder is, if any. */
export function findSlot<T extends { path: string }>(slots: readonly T[], slotDir: string): T | undefined {
    return slots.find((s) => sameSlotPath(s.path, slotDir));
}

/** An imported template voice (runs on the Sigil Voice Engine). */
export function isTemplateSlot(s: Pick<VcSlotInfo, "format">): boolean {
    return s.format === SIGIL_RVC_FORMAT;
}

/**
 * Why a slot cannot be used in this build, if it cannot: Rust's own `problem`, or a format the
 * engine factory does not run (`capabilities.slot_formats`).
 */
export function slotProblem(s: Pick<VcSlotInfo, "format" | "problem">, slotFormats: readonly string[]): string | null {
    if (s.problem) return s.problem;
    if (!slotFormats.includes(s.format)) return `format ${s.format}`;
    return null;
}

/**
 * Voices `a` can be merged with (Rust `slot::merge_slots`): other imported voices of the same
 * template and content encoder, without problems.
 */
export function mergePartners(slots: readonly VcSlotInfo[], a: VcSlotInfo): VcSlotInfo[] {
    if (!isTemplateSlot(a) || a.problem || !a.template) return [];
    return slots.filter(
        (s) =>
            s.id !== a.id &&
            isTemplateSlot(s) &&
            !s.problem &&
            s.template === a.template &&
            (s.embedder ?? "") === (a.embedder ?? ""),
    );
}

/** The slot `step` places after (`+1`) or before (`-1`) the selected folder, wrapping; runnable ones only. */
export function adjacentSlot(
    slots: readonly VcSlotInfo[],
    selectedDir: string,
    step: 1 | -1,
    slotFormats: readonly string[],
): VcSlotInfo | undefined {
    const usable = slots.filter((s) => slotProblem(s, slotFormats) === null);
    if (usable.length === 0) return undefined;
    const i = usable.findIndex((s) => sameSlotPath(s.path, selectedDir));
    if (i < 0) return step === 1 ? usable[0] : usable[usable.length - 1];
    return usable[(i + step + usable.length) % usable.length];
}

// ---------------------------------------------------------------------------------------------
// Import sources (drag and drop, file picker)
// ---------------------------------------------------------------------------------------------

/** File extensions the importer reads (`.pth` checkpoints, voice `.zip` archives). */
export const VOICE_FILE_EXTENSIONS = ["pth", "zip"] as const;
/** Retrieval index files (adopted next to a `.pth`). */
export const INDEX_FILE_EXTENSIONS = ["index"] as const;

function extensionOf(path: string): string {
    const name = path.trim().replace(/\\/g, "/").split("/").pop() ?? "";
    const dot = name.lastIndexOf(".");
    return dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
}

/** The file name of a path (either separator). */
export function fileNameOf(path: string): string {
    return path.trim().replace(/\\/g, "/").replace(/\/+$/, "").split("/").pop() ?? path;
}

export interface ImportPlanItem {
    /** The `.pth` or `.zip` to import. */
    source: string;
    /** A retrieval index to adopt with it (only when the choice is unambiguous). */
    index: string | null;
}

export interface ImportPlan {
    items: ImportPlanItem[];
    /** Dropped `.index` files that could not be paired with exactly one `.pth`. */
    unpairedIndexes: string[];
    /** Anything else that was dropped (folders, other files). */
    ignored: string[];
}

/**
 * What to import from dropped / picked paths: every `.pth` / `.zip` once; a single `.index` dropped
 * together with a single `.pth` is imported with it (otherwise Rust adopts a `.zip`'s own index, and
 * the `added_*.index` next to a `.pth` when that choice is unambiguous).
 */
export function planImport(paths: readonly string[]): ImportPlan {
    const voices: string[] = [];
    const indexes: string[] = [];
    const ignored: string[] = [];
    const seen = new Set<string>();
    for (const raw of paths) {
        if (typeof raw !== "string" || raw.trim() === "") continue;
        const path = raw.trim();
        const key = slotPathKey(path);
        if (seen.has(key)) continue;
        seen.add(key);
        const ext = extensionOf(path);
        if ((VOICE_FILE_EXTENSIONS as readonly string[]).includes(ext)) voices.push(path);
        else if ((INDEX_FILE_EXTENSIONS as readonly string[]).includes(ext)) indexes.push(path);
        else ignored.push(path);
    }
    const pths = voices.filter((v) => extensionOf(v) === "pth");
    const pairIndex = pths.length === 1 && indexes.length === 1 ? indexes[0] : null;
    return {
        items: voices.map((source) => ({ source, index: pairIndex !== null && source === pths[0] ? pairIndex : null })),
        unpairedIndexes: pairIndex !== null ? [] : indexes,
        ignored,
    };
}

// ---------------------------------------------------------------------------------------------
// Content encoder and vocoder (read-only facts of a voice)
// ---------------------------------------------------------------------------------------------

/** Model-store id of each content encoder (Rust `vc::assets::embedder_id`). */
export const EMBEDDER_CATALOG_IDS = {
    contentvec: "vc/embedder-contentvec",
    "contentvec-256": "vc/embedder-contentvec-256",
    "spin-v2": "vc/embedder-spin-v2",
    spin: "vc/embedder-spin",
    "hubert-ja": "vc/embedder-hubert-ja",
    "hubert-ko": "vc/embedder-hubert-ko",
    "hubert-zh": "vc/embedder-hubert-zh",
} as const;
export type EmbedderName = keyof typeof EMBEDDER_CATALOG_IDS;

/** Display names (product names; not translated). */
const EMBEDDER_NAMES: Record<EmbedderName, string> = {
    contentvec: "ContentVec",
    "contentvec-256": "ContentVec 256 (v1)",
    "spin-v2": "SPIN v2",
    spin: "SPIN",
    "hubert-ja": "Japanese HuBERT",
    "hubert-ko": "Korean HuBERT",
    "hubert-zh": "Chinese HuBERT",
};

/**
 * The content encoder a voice needs (Rust `EmbedderKind::from_manifest`, name part): its manifest
 * name, or ContentVec by width when unnamed; ContentVec on a v1 voice is the 256-channel encoder.
 * `null` for a name Sigil has no encoder for (the slot's `problem` says why).
 */
export function embedderOf(embedder: string | null | undefined, version: string | null | undefined): EmbedderName | null {
    const v1 = (version ?? "").trim().toLowerCase() === "v1";
    const raw = (embedder ?? "").trim().toLowerCase().replace(/[_ ]/g, "-");
    switch (raw) {
        case "":
        case "contentvec":
        case "content-vec":
        case "hubert":
        case "hubert-base":
        case "contentvec-768":
        case "vec-768-layer-12":
        case "contentvec-base":
            return v1 ? "contentvec-256" : "contentvec";
        case "contentvec-256":
        case "vec-256-layer-9":
            return "contentvec-256";
        case "spin-v2":
        case "spinv2":
            return "spin-v2";
        case "spin":
        case "spin-v1":
        case "spinv1":
            return "spin";
        case "hubert-ja":
        case "japanese-hubert-base":
        case "rinna-hubert-base":
        case "japanese-hubert":
            return "hubert-ja";
        case "hubert-ko":
        case "korean-hubert-base":
        case "korean-hubert":
            return "hubert-ko";
        case "hubert-zh":
        case "chinese-hubert-base":
        case "chinese-hubert":
            return "hubert-zh";
        default:
            return null;
    }
}

/** Display name of a voice's content encoder (the raw name when Sigil does not know it). */
export function embedderDisplayName(embedder: string | null | undefined, version: string | null | undefined): string {
    const kind = embedderOf(embedder, version);
    if (kind) return EMBEDDER_NAMES[kind];
    return (embedder ?? "").trim() || "?";
}

/** Display name of a vocoder (`hifigan`, `refinegan`, `mrf-hifigan`, ...; product names). */
export function vocoderDisplayName(vocoder: string | null | undefined): string {
    const v = (vocoder ?? "").trim().toLowerCase().replace(/[_ ]/g, "-");
    switch (v) {
        case "":
        case "hifigan":
        case "hifi-gan":
        case "hifigan-nsf":
        case "nsf-hifigan":
            return "HiFi-GAN (NSF)";
        case "refinegan":
        case "refine-gan":
            return "RefineGAN";
        case "mrf-hifigan":
        case "mrf":
        case "hifigan-mrf":
            return "MRF HiFi-GAN";
        default:
            return (vocoder ?? "").trim();
    }
}

/** `40 kHz` style label of a model sample rate. */
export function sampleRateLabel(hz: number | null | undefined): string | null {
    if (typeof hz !== "number" || !Number.isFinite(hz) || hz <= 0) return null;
    const k = hz / 1000;
    return `${Number.isInteger(k) ? k.toFixed(0) : k.toFixed(1)} kHz`;
}
