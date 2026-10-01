import { invoke } from "@tauri-apps/api/core";

import { listen, UnlistenFn } from "@tauri-apps/api/event";

import i18n from "i18next";

import { proxy } from "valtio";

import { toast } from "react-toastify";

import type { RustEpOutcomeEntry } from "@/types";
import { validVoiceId } from "@/core/services/voice_library/contract";
import { devLog } from "@/utils/devLog";

import { VoiceBench } from "./bench";
import { effectiveCrossfadeMs } from "./chunkFrames";
import {
    type AuxMicGate,
    clampLatencyBudgetMs,
    normalizeAuxMicGate,
    normalizeWorkerPriority,
    OUTPUT_BUFFER_LOW_LATENCY,
    OUTPUT_BUFFER_STABLE,
    type WorkerPriority,
} from "./contract";
import { isVcShortcut, normalizeHotkeyVoices, vcShortcutAction } from "./hotkeys";
import { BASELINE_CAPABILITIES, normalizeCapabilities, VC_COMMANDS, VC_EVENTS, type VcCapabilities, type VcTrtProgress } from "./ipc";
import { VoiceLibrary } from "./library";
import { epDisplayName, isVoiceChangerEpComponent, normalizeMetrics, type VoiceChangerMetrics } from "./metrics";
import {
    adoptAdjustedParams,
    applyRustParams,
    applyRustParamsPatch,
    toRustParams,
    type VoiceChangerParamsRust,
} from "./params";
import { rvcProfilePatch } from "./presets";
import { changedParamKeys, restartNeeded } from "./restart";
import { RvcProfile, VoiceChangerState, VoiceChangerStateSchema } from "./schema";
import { adjacentSlot, normalizeSlots, sameSlotPath, slotProblem, type VoiceChangerModelSlot } from "./slots";
import { AUTOTUNE_PARAM_KEYS } from "./tuning";

export type { VoiceChangerMetrics } from "./metrics";
export type { VoiceChangerModelSlot } from "./slots";
export type { VoiceChangerParamsRust } from "./params";
/** Template output buffer (ms) — matches Rust `PREFILL_MS_*` / `MAX_QUEUE_MS_*`. */
export { OUTPUT_BUFFER_LOW_LATENCY, OUTPUT_BUFFER_STABLE };

export type VocoderOversample = 1 | 2 | 4 | 8 | 16 | 32;

/**
 * Payload of `voice_changer:error`. A failed conversion (`onnx_infer`, `sidecar_length`, `sidecar_exchange`,
 * `rvc_convert`) mutes that block (never dry input); load, device and worker errors stop the engine.
 */
export interface VoiceChangerConversionErrorPayload {
    code: string;
    message: string;
}

/** Rust `diagnostics::EP_OUTCOME_EVENT` (frozen P0 contract). */
const EP_OUTCOME_EVENT = "diagnostics:ep_outcome";
/** Rust `commands::PARAMS_EVENT`: effective params when Rust changed what the UI sent. */
const PARAMS_EVENT = "voice_changer:params";
/** Where the webview kept the RVC notice acknowledgement before Rust persisted it. */
export const LEGACY_CONSENT_STORAGE_KEY = "sigil.vc.consent";
/** At most one "settings were not applied" toast per this interval. */
const SYNC_FAILURE_TOAST_INTERVAL_MS = 10_000;
/** A finished TensorRT engine build (or cache hit) stays visible this long. */
const TRT_DONE_VISIBLE_MS = 6_000;

interface VoiceChangerServiceState {
    voiceChanger: VoiceChangerState;
    isRunning: boolean;
    /** The settings shown are Rust's (false: schema defaults, e.g. no desktop backend). */
    hydrated: boolean;
    /**
     * A start in RVC mode was refused because the voice-permission notice is not acknowledged
     * (also a restart into RVC, e.g. switching the mode while running). The inspector shows the
     * notice; `acknowledgeRvcConsent` or `dismissRvcConsent` clears it.
     */
    rvcConsentPending: boolean;
    /** What this build runs (`voice_changer_get_capabilities`; the baseline until the reply). */
    capabilities: VcCapabilities;
    capabilitiesLoaded: boolean;
    /** TensorRT engine builds of the current start, one row per session component. */
    trt: VcTrtProgress[];
}

/** Reply of one `set_voice_changer_params`: null when Rust refused the params. */
type SyncResult = { effective: VoiceChangerParamsRust | null } | null;

/**
 * Values the RVC harness reads only at construction; changing them needs an engine restart. Stored as set (not
 * clamped): the crossfade clamp depends on the capture rate, which is only known once the engine runs.
 */
interface HarnessParams {
    extraContextMs: number;
    crossfadeMs: number;
    chunkFrames: number;
}

function harnessParamsOf(vc: VoiceChangerState): HarnessParams {
    return {
        extraContextMs: vc.extraContextMs,
        crossfadeMs: vc.crossfadeMs,
        chunkFrames: vc.chunkFrames,
    };
}

function errorText(e: unknown): string {
    if (typeof e === "string") return e;
    if (e instanceof Error) return e.message;
    try {
        return JSON.stringify(e);
    } catch {
        return String(e);
    }
}

function readLegacyConsent(): boolean {
    try {
        return globalThis.localStorage?.getItem(LEGACY_CONSENT_STORAGE_KEY) === "1";
    } catch {
        return false;
    }
}

function clearLegacyConsent() {
    try {
        globalThis.localStorage?.removeItem(LEGACY_CONSENT_STORAGE_KEY);
    } catch {
        /* private mode / no storage */
    }
}

/**
 * Host service for the real-time voice changer. Not a persisted service (no `patchService`): Rust
 * owns the settings (`app_data/voice-changer/settings.json`, per-voice `slot.json`) and this service
 * hydrates from `get_voice_changer_params`, sends every change with `set_voice_changer_params` and
 * takes over whatever Rust changed in its reply.
 */
export class VoiceChangerService {
    public state = proxy<VoiceChangerServiceState>({
        voiceChanger: VoiceChangerStateSchema.parse({}),
        isRunning: false,
        hydrated: false,
        rvcConsentPending: false,
        capabilities: { ...BASELINE_CAPABILITIES },
        capabilitiesLoaded: false,
        trt: [],
    });

    /** The voice library: slots, imports, rename / duplicate / delete / merge. */
    public readonly library = new VoiceLibrary({
        selectedVoiceDir: () => this.state.voiceChanger.rvcSlotDir,
        selectVoice: (v) => this.selectVoice(v),
        clearVoice: () => this.update({ rvcSlotDir: "", slotId: "" }),
        isRunning: () => this.state.isRunning,
    });

    /** Autotune, "Benchmark this PC" and live recording. */
    public readonly bench = new VoiceBench({
        settings: () => this.state.voiceChanger,
        isRunning: () => this.state.isRunning,
        applyTuning: (params) => this.applyTuning(params),
    });

    private unlisteners: UnlistenFn[] = [];
    private initialized = false;

    /** Harness params the running engine was last started (or restarted) with; null when unknown. */
    private appliedHarness: HarnessParams | null = null;
    /**
     * The params the running engine was started with (after Rust's adjustments), to tell whether a
     * later change needs a restart (`restart.ts`); null when stopped or unknown.
     */
    private startedParams: VoiceChangerParamsRust | null = null;
    /** A restart started by `update` (restarts are serialized; resolves whether it worked, never rejects). */
    private restartInflight: Promise<boolean> | null = null;
    private trtTimers = new Map<string, ReturnType<typeof setTimeout>>();

    /**
     * Params sync is serialized: at most one `set` is in flight. A change made meanwhile only marks
     * the state dirty, and the next payload is built after the previous reply was adopted. Rust
     * applies a voice's `slot.json` only on the sync that enters the voice, and saves changed
     * per-voice values of any later sync into it, so a payload built before that reply would carry
     * the previous voice's values and overwrite the new voice's file.
     */
    private syncLoop: Promise<SyncResult> | null = null;
    private syncDirty = false;
    /** Bumped per `set` sent; a `get` issued before a later `set` is stale. */
    private sendGen = 0;
    /**
     * What Rust holds as far as this service knows: the last payload sent (or hydrated), with the
     * values adopted from Rust since. A field that differs from it in the state was edited locally.
     */
    private lastSent: VoiceChangerParamsRust | null = null;
    private lastSyncFailureToastAt = 0;
    /** A `voice_changer:params` arrived while a `set` was in flight: re-read once the sync is done. */
    private paramsEventPending = false;

    /** A `start()` call is between its params sync and Rust's reply. */
    private starting = false;
    /** Bumped on every `voice_changer:started`; an EP fallback toasts at most once per run. */
    private runId = 0;
    private epToastRun = -1;
    /** Per EP component, the fallback (`requested>active`) the user was already told about. */
    private epFallbackKnown = new Map<string, string>();

    async init() {
        if (this.initialized) return;
        this.initialized = true;
        devLog("[VoiceChanger] Initializing service");

        this.unlisteners.push(
            await listen("voice_changer:started", () => {
                this.runId++;
                this.state.isRunning = true;
                this.state.voiceChanger.enabled = true;
                this.bench.onEngineStarted();
            }),
            await listen("voice_changer:stopped", () => {
                this.state.isRunning = false;
                this.state.voiceChanger.enabled = false;
                this.startedParams = null;
                this.clearTrt();
                this.bench.onEngineStopped();
                this.library.onEngineStopped();
            }),
            await listen<VoiceChangerConversionErrorPayload>("voice_changer:error", (ev) => this.onError(ev.payload)),
            await listen<unknown>(PARAMS_EVENT, (ev) => void this.onParamsEvent(ev.payload)),
            await listen<RustEpOutcomeEntry>(EP_OUTCOME_EVENT, (ev) => this.onEpOutcome(ev.payload)),
            await listen<VcTrtProgress>(VC_EVENTS.trtProgress, (ev) => this.onTrtProgress(ev.payload)),
        );
        await this.library.init();
        await this.bench.init();

        try {
            this.state.isRunning = await invoke<boolean>("plugin:voice-changer|is_voice_changer_running");
        } catch {
            this.state.isRunning = false;
        }

        await this.hydrate();
        await this.migrateLegacyConsent();
        await this.refreshCapabilities();
    }

    /**
     * Re-read what this build runs. Some values depend on installed models (e.g. PESTO), so the
     * inspector refreshes this when it opens and after model installs.
     */
    async refreshCapabilities(): Promise<VcCapabilities> {
        try {
            const caps = normalizeCapabilities(await invoke<unknown>(VC_COMMANDS.getCapabilities));
            this.state.capabilities = caps;
            this.state.capabilitiesLoaded = true;
            return caps;
        } catch (e) {
            devLog("[VoiceChanger] Could not read the capabilities:", e);
            return this.state.capabilities;
        }
    }

    /** Replace the shown settings with Rust's (every field, including the input device). */
    async hydrate(): Promise<boolean> {
        try {
            const p = await invoke<VoiceChangerParamsRust | null>("plugin:voice-changer|get_voice_changer_params");
            if (!p || typeof p !== "object") return false;
            const vc = this.state.voiceChanger;
            Object.assign(vc, applyRustParams({ ...vc }, p));
            this.lastSent = toRustParams(vc);
            this.state.hydrated = true;
            // Already running (e.g. webview reload): the live harness was built from these params.
            if (this.state.isRunning) {
                this.appliedHarness = harnessParamsOf(vc);
                this.startedParams = this.lastSent;
            }
            return true;
        } catch (e) {
            devLog("[VoiceChanger] Could not read settings:", e);
            return false;
        }
    }

    /**
     * One-time move of the RVC notice acknowledgement from webview localStorage into Rust's settings.
     * The old key is removed only once Rust reports the value stored.
     */
    private async migrateLegacyConsent() {
        if (!this.state.hydrated || !readLegacyConsent()) return;
        const vc = this.state.voiceChanger;
        if (!vc.rvcConsentAcknowledged) {
            vc.rvcConsentAcknowledged = true;
            const reply = await this.sendParams();
            if (reply?.effective?.rvc_consent_acknowledged !== true) return; // retry on the next launch
        }
        clearLegacyConsent();
    }

    private onError(payload: VoiceChangerConversionErrorPayload) {
        const code = payload?.code ?? "";
        const msg = payload?.message?.trim() || "";
        switch (code) {
            case "onnx_infer":
                toast.warning(i18n.t("voice_changer.toast_error_onnx", { message: msg }));
                break;
            case "sidecar_length":
                toast.warning(i18n.t("voice_changer.toast_error_sidecar_length", { message: msg }));
                break;
            case "sidecar_exchange":
                toast.warning(i18n.t("voice_changer.toast_error_sidecar", { message: msg }));
                break;
            case "rvc_load":
            case "rvc_convert":
                toast.warning(i18n.t("voice_changer.toast_error_rvc", { message: msg }));
                break;
            case "rvc_warning":
                toast.warning(i18n.t("voice_changer.toast_warning_rvc", { message: msg }));
                break;
            case "worker_panic":
                toast.error(i18n.t("voice_changer.toast_error_worker_panic", { message: msg }));
                break;
            case "device":
                toast.error(i18n.t("voice_changer.toast_error_device", { message: msg }));
                break;
            case "monitor_unavailable":
                toast.warning(i18n.t("voice_changer.toast_warning_monitor", { message: msg }));
                break;
            case "sidecar_connect":
                toast.error(i18n.t("voice_changer.toast_error_sidecar_connect", { message: msg }));
                break;
            case "onnx_load":
                toast.error(i18n.t("voice_changer.toast_error_onnx_load", { message: msg }));
                break;
            case "worker_error":
                toast.error(i18n.t("voice_changer.toast_error_stopped", { message: msg }));
                break;
            case "device_lost":
                toast.warning(i18n.t("voice_changer.toast_warning_device_lost", { message: msg }));
                break;
            case "device_restored":
                toast.info(i18n.t("voice_changer.toast_info_device_restored", { message: msg }));
                break;
            case "denoiser_warning":
                toast.warning(i18n.t("voice_changer.toast_warning_denoiser", { message: msg }));
                break;
            default:
                toast.warning(i18n.t("voice_changer.toast_error_generic", { message: msg || code }));
        }
    }

    /**
     * `voice_changer:params` (Rust applied a voice's own settings, or another window changed them).
     * The payload is only a trigger: Tauri may deliver it after the reply of a later `set`, when it
     * is stale. While a `set` of ours is in flight its reply brings Rust's values; otherwise re-read
     * them, and drop that read if a `set` was sent meanwhile (its reply is newer).
     */
    private async onParamsEvent(payload: unknown) {
        if (!payload || typeof payload !== "object") return;
        if (!this.lastSent) return;
        if (this.syncLoop) {
            // The reply of that `set` is newer only when Rust handled it after this change (e.g. a
            // failed voice switch going back to the voice that plays): read once more afterwards.
            this.paramsEventPending = true;
            return;
        }
        const gen = this.sendGen;
        let current: unknown;
        try {
            current = await invoke<unknown>("plugin:voice-changer|get_voice_changer_params");
        } catch {
            return;
        }
        if (gen !== this.sendGen || this.syncLoop || !this.lastSent) return;
        if (!current || typeof current !== "object") return;
        this.adopt(this.lastSent, current as VoiceChangerParamsRust);
    }

    /** Tell the user once when the voice changer's model cannot use the requested execution provider. */
    private onEpOutcome(e: RustEpOutcomeEntry) {
        if (!e || typeof e.component !== "string" || !isVoiceChangerEpComponent(e.component)) return;
        if (!this.state.isRunning && !this.starting) return;
        const requested = String(e.requested ?? "").trim().toLowerCase();
        const active = String(e.active ?? "").trim().toLowerCase();
        if (requested === active) {
            // Working again: a later fallback of this component is news.
            this.epFallbackKnown.delete(e.component);
            return;
        }
        const key = `${requested}>${active}`;
        const isNew = this.epFallbackKnown.get(e.component) !== key;
        this.epFallbackKnown.set(e.component, key);
        if (!isNew || this.epToastRun === this.runId) return;
        this.epToastRun = this.runId;
        toast.warning(
            i18n.t("voice_changer.toast_ep_fallback", {
                requested: epDisplayName(requested),
                active: epDisplayName(active),
                reason: e.reason?.trim() || i18n.t("voice_changer.ep_reason_unknown"),
            }),
        );
    }

    /** A TensorRT engine build of the starting engine (first start: up to a minute; later: the cache). */
    private onTrtProgress(p: VcTrtProgress) {
        if (!p || typeof p !== "object" || typeof p.component !== "string" || !p.component) return;
        // The event is process-wide: the autotuner and "Benchmark this PC" (both only while the voice
        // changer is stopped) build engines too, and a stopped engine's worker can still be building.
        if (!this.starting && !this.state.isRunning) return;
        if (this.bench.state.autotune.running || this.bench.state.benchmark.running) return;
        const row: VcTrtProgress = {
            component: p.component,
            stage: p.stage,
            fraction: typeof p.fraction === "number" && Number.isFinite(p.fraction) ? Math.min(1, Math.max(0, p.fraction)) : 0,
            message: typeof p.message === "string" ? p.message : "",
            elapsed_ms: typeof p.elapsed_ms === "number" && Number.isFinite(p.elapsed_ms) ? p.elapsed_ms : 0,
        };
        const list = this.state.trt;
        const i = list.findIndex((r) => r.component === row.component);
        if (i >= 0) list[i] = row;
        else list.push(row);
        const timer = this.trtTimers.get(row.component);
        if (timer) clearTimeout(timer);
        this.trtTimers.delete(row.component);
        // A failure stays until the next start (it explains the EP fallback); the rest fade out.
        if (row.stage === "done" || row.stage === "cache_hit") {
            this.trtTimers.set(
                row.component,
                setTimeout(() => {
                    this.trtTimers.delete(row.component);
                    const j = this.state.trt.findIndex((r) => r.component === row.component && r.stage === row.stage);
                    if (j >= 0) this.state.trt.splice(j, 1);
                }, TRT_DONE_VISIBLE_MS),
            );
        }
    }

    private clearTrt() {
        for (const t of this.trtTimers.values()) clearTimeout(t);
        this.trtTimers.clear();
        this.state.trt = [];
    }

    async getMetrics(): Promise<VoiceChangerMetrics | null> {
        try {
            return normalizeMetrics(await invoke<unknown>("plugin:voice-changer|get_voice_changer_metrics"));
        } catch {
            return null;
        }
    }

    /**
     * Start the engine. Resolves `false` without starting when RVC mode still needs the
     * voice-permission notice (`rvcConsentPending` is set for the inspector to show it); this also
     * covers a restart into RVC. Rejects (after telling the user) when Rust refuses the start.
     */
    async start(): Promise<boolean> {
        if (this.state.isRunning) return true;

        const vc = this.state.voiceChanger;
        if (vc.mode === "rvc" && !vc.rvcConsentAcknowledged) {
            vc.enabled = false;
            this.state.rvcConsentPending = true;
            devLog("[VoiceChanger] RVC start waits for the voice-permission notice");
            return false;
        }
        vc.enabled = true;
        // Snapshot taken together with the params sent below, so later edits compare against what the harness got.
        const harness = harnessParamsOf(vc);
        this.starting = true;
        this.clearTrt();
        try {
            await this.syncParams();
            // As Rust holds them after that sync (its adjustments adopted): what the engine starts with.
            const started = toRustParams(vc);

            // The input device also travels in the params (and is persisted there); passing it here
            // keeps start independent of whether that sync reached Rust.
            await invoke("plugin:voice-changer|start_voice_changer", { inputDevice: vc.inputDevice });
            this.appliedHarness = harness;
            this.startedParams = started;
            devLog("[VoiceChanger] Started");
            return true;
        } catch (error) {
            vc.enabled = false;
            console.error("[VoiceChanger] Failed to start:", error);
            // Preflight errors (no model file, no WebSocket URL, missing RVC base files) must reach the user.
            toast.error(i18n.t("voice_changer.toast_start_failed", { message: errorText(error) }));
            throw error;
        } finally {
            this.starting = false;
        }
    }

    async stop() {
        if (!this.state.isRunning) return;

        try {
            await invoke("plugin:voice-changer|stop_voice_changer");
            this.state.isRunning = false;
            this.state.voiceChanger.enabled = false;
            this.startedParams = null;
            // Rust counts the engine as stopped from now on; its `stopped` event can come much later
            // (a worker still loading a model) or never (a restart supersedes that worker).
            this.clearTrt();
            this.bench.onEngineStopped();
            this.library.onEngineStopped();
            devLog("[VoiceChanger] Stopped");
        } catch (error) {
            console.error("[VoiceChanger] Failed to stop:", error);
        }
    }

    async toggle() {
        if (this.state.isRunning) {
            await this.stop();
        } else {
            await this.start();
        }
    }

    /** Apply low_latency / stable buffer template and sync. */
    async applyLatencyPreset(preset: "low_latency" | "stable") {
        const vc = this.state.voiceChanger;
        vc.latencyPreset = preset;
        const template = preset === "stable" ? OUTPUT_BUFFER_STABLE : OUTPUT_BUFFER_LOW_LATENCY;
        vc.outputPrefillMs = template.prefillMs;
        vc.outputMaxQueueMs = template.maxQueueMs;
        await this.syncParams();
    }

    /**
     * Apply an RVC preset (Streamer / Low latency / Quality / Studio-TRT: EP, F0, index, extra
     * context, VRAM cap; `presets.ts`). A running RVC engine restarts when its shape changed.
     */
    async applyRvcProfile(profile: RvcProfile) {
        await this.update(rvcProfilePatch(profile));
    }

    /**
     * Change settings and sync them; a running engine restarts only when a changed param is read at
     * start in the running mode (`restart.ts`): voice, F0 method and F0 range switch without a restart
     * in RVC mode, the voice controls apply live. Resolves whether Rust accepted the params and a
     * restart they needed worked; never rejects (a failure was already shown to the user).
     */
    async update(patch: Partial<VoiceChangerState>): Promise<boolean> {
        const vc = this.state.voiceChanger;
        const before = toRustParams(vc);
        Object.assign(vc, patch);
        const ok = await this.syncParams();
        const restarted = await this.restartForChanges(before);
        return ok && restarted;
    }

    /**
     * Restart a running engine when the settings differ from what it started with in a way it cannot
     * apply live (edits made earlier without a restart count too). Restarts are serialized: a change
     * that arrives while one is in flight is judged against the engine that restart starts. Resolves
     * `false` only when a needed restart failed; never rejects.
     */
    private async restartForChanges(before: VoiceChangerParamsRust): Promise<boolean> {
        // A failed restart belongs to the change that caused it, not to the ones waiting here.
        while (this.restartInflight) await this.restartInflight.catch(() => false);
        if (!this.state.isRunning) return true;
        const vc = this.state.voiceChanger;
        const changed = changedParamKeys(this.startedParams ?? before, toRustParams(vc));
        if (!restartNeeded(changed, vc.mode)) return true;
        const run = this.restartIfRunning().finally(() => {
            if (this.restartInflight === run) this.restartInflight = null;
        });
        this.restartInflight = run;
        return run;
    }

    /**
     * Use a voice (a slot-store row, or any folder). While RVC runs this is a hot switch (the voice
     * loads in the background and is crossfaded in). `ensureRvc` also switches the mode to RVC.
     */
    async selectVoice(
        voice: { id?: string; path: string; reference_voice_id?: string | null },
        opts: { ensureRvc?: boolean } = {},
    ): Promise<boolean> {
        const patch: Partial<VoiceChangerState> = { rvcSlotDir: voice.path, slotId: voice.id ?? "" };
        if (opts.ensureRvc && this.state.voiceChanger.mode !== "rvc") patch.mode = "rvc";
        // P5: Rust applies only the fields a voice's slot.json holds when it enters the voice, so a voice
        // without a stored reference would keep the previous voice's. Another voice therefore starts from
        // the reference its listing holds (`""` = none); its slot.json, when it has one, still wins.
        if ("reference_voice_id" in voice && !sameSlotPath(voice.path, this.state.voiceChanger.rvcSlotDir)) {
            const ref = (voice.reference_voice_id ?? "").trim();
            patch.referenceVoiceId = validVoiceId(ref) ? ref : "";
        }
        return this.update(patch);
    }

    /**
     * The selected zero-shot voice's reference (a voice-library id, `""` = none). Saved with the voice
     * (`slot.json`); while RVC runs the voice reloads with it in the background and is crossfaded in
     * (`restart.ts`: switch). An id the library would never hold is refused (resolves `false`).
     */
    async setReferenceVoice(id: string): Promise<boolean> {
        const next = id.trim();
        if (next !== "" && !validVoiceId(next)) return false;
        if (next === this.state.voiceChanger.referenceVoiceId) return true;
        const dir = this.state.voiceChanger.rvcSlotDir;
        const ok = await this.update({ referenceVoiceId: next });
        // Rust saved it to the voice's slot.json (no `slots_changed` follows): the voice's row shows it too,
        // unless another voice was selected meanwhile (its reply carried that voice's reference).
        if (ok && sameSlotPath(this.state.voiceChanger.rvcSlotDir, dir)) {
            this.library.noteReference(dir, this.state.voiceChanger.referenceVoiceId);
        }
        return ok;
    }

    /**
     * Apply the autotuner's params patch (engine-shape fields only, `tuning.ts` `AUTOTUNE_PARAM_KEYS`),
     * restarting a running engine when its shape changed.
     */
    async applyTuning(params: Readonly<Record<string, unknown>>): Promise<boolean> {
        const vc = this.state.voiceChanger;
        const { next, applied } = applyRustParamsPatch(vc, params, AUTOTUNE_PARAM_KEYS);
        if (applied.length === 0) return false;
        const patch: Partial<VoiceChangerState> = {};
        for (const [k, v] of Object.entries(next) as [keyof VoiceChangerState, unknown][]) {
            if (!Object.is(v, vc[k])) (patch as Record<string, unknown>)[k] = v;
        }
        return this.update(patch);
    }

    /**
     * A voice-changer hotkey (`hotkeys.ts`; dispatched by the keyboard service): start / stop, the
     * next / previous voice, or voice hotkey N. A voice hotkey also switches to RVC mode.
     */
    async runShortcut(key: string): Promise<void> {
        if (!isVcShortcut(key)) return;
        const action = vcShortcutAction(key);
        if (action.kind === "toggle") {
            await this.toggle().catch(() => {});
            // The notice is shown in the voice-changer panel, which may not be open while a game has focus.
            if (this.state.rvcConsentPending) toast.info(i18n.t("voice_changer.hotkey_consent_needed"));
            return;
        }
        const slots = await this.library.refresh();
        const formats = this.state.capabilities.slot_formats;
        const assigned = action.kind === "voice" ? hotkeyVoices()[action.index] : "";
        const target =
            action.kind === "step"
                ? adjacentSlot(slots, this.state.voiceChanger.rvcSlotDir, action.step, formats)
                : assigned
                  ? slots.find((s) => s.id === assigned)
                  : undefined;
        if (!target) {
            toast.info(
                i18n.t(action.kind === "step" ? "voice_changer.hotkey_no_voices" : "voice_changer.hotkey_no_voice_assigned"),
            );
            return;
        }
        const problem = slotProblem(target, formats);
        if (problem) {
            toast.warning(i18n.t("voice_changer.hotkey_voice_unusable", { name: target.name, problem }));
            return;
        }
        // A refused sync or a failed restart was already shown to the user: no "selected" toast then.
        const ok = await this.selectVoice(target, { ensureRvc: true });
        if (this.state.rvcConsentPending) toast.info(i18n.t("voice_changer.hotkey_consent_needed"));
        if (ok) toast.info(i18n.t("voice_changer.hotkey_voice_selected", { name: target.name }), { autoClose: 1500 });
    }

    /** Voice folders under `app_data/voice-changer/models/` (sorted by name; empty on error). */
    async listModelSlots(): Promise<VoiceChangerModelSlot[]> {
        try {
            return normalizeSlots(await invoke<unknown>("plugin:voice-changer|list_voice_changer_model_slots"));
        } catch (error) {
            console.error("[VoiceChanger] Could not list voices:", error);
            return [];
        }
    }

    /**
     * Switch the RVC voice folder. Rust applies that voice's saved `slot.json` (pitch, index, protect,
     * F0, …) and the reply brings those values into the UI; a running RVC engine switches to the voice
     * without a restart.
     */
    async selectModelSlot(path: string) {
        const slot = this.library.slotFor(path);
        await this.selectVoice(slot ? { id: slot.id, path, reference_voice_id: slot.reference_voice_id ?? null } : { id: "", path });
    }

    async captureDebug(): Promise<Record<string, unknown> | null> {
        try {
            return await invoke("plugin:voice-changer|capture_voice_changer_debug");
        } catch (error) {
            console.error("[VoiceChanger] Debug capture failed:", error);
            return null;
        }
    }

    /** Mark buffer as custom (call when user edits ms fields). */
    markBufferCustom() {
        this.state.voiceChanger.latencyPreset = "custom";
    }

    /**
     * Restart if running so chunk size / buffer caps take effect. A restart into RVC without the
     * voice-permission acknowledgement leaves the engine stopped with `rvcConsentPending` set.
     * Resolves `false` when the start failed (`start()` already told the user); never rejects, so
     * fire-and-forget callers (inspector controls, hotkeys) leave no unhandled rejection.
     */
    async restartIfRunning(): Promise<boolean> {
        if (!this.state.isRunning) return true;
        await this.stop();
        await new Promise((r) => setTimeout(r, 80));
        try {
            await this.start();
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Sync Rust params then restart the engine when running (devices, mode, chunk size, bridges).
     * Resolves whether both worked; never rejects.
     */
    async syncParamsAndRestartIfRunning(): Promise<boolean> {
        const ok = await this.syncParams();
        const restarted = await this.restartIfRunning();
        return ok && restarted;
    }

    /**
     * Commit extra context / crossfade (call on slider release or field blur). The RVC harness reads them only
     * at construction, so restart once when they differ from what the running engine was started with; a
     * repeated commit of the same value is a no-op. When stopped, only sync (the next start applies them).
     */
    async commitHarnessParams() {
        const cur = harnessParamsOf(this.state.voiceChanger);
        const applied = this.appliedHarness;
        if (
            this.state.isRunning &&
            applied &&
            applied.extraContextMs === cur.extraContextMs &&
            applied.chunkFrames === cur.chunkFrames
        ) {
            // Compared as the harness sees them: crossfade beyond one chunk is clamped in Rust, so e.g. 20 -> 30 ms
            // at 512 frames is not a change and must not restart the engine. The chunk length comes from the
            // running engine's capture rate (`chunk_ms`), falling back to 48 kHz when metrics are unavailable.
            const chunkMs = (await this.getMetrics())?.chunk_ms;
            if (
                effectiveCrossfadeMs(applied.crossfadeMs, cur.chunkFrames, chunkMs) ===
                effectiveCrossfadeMs(cur.crossfadeMs, cur.chunkFrames, chunkMs)
            ) {
                // The running engine is equivalent to the new value: later changes must not count it
                // as a pending restart (`update`).
                if (this.startedParams) {
                    this.startedParams = { ...this.startedParams, crossfade_ms: toRustParams(this.state.voiceChanger).crossfade_ms };
                }
                return;
            }
        }
        await this.syncParamsAndRestartIfRunning();
    }

    /**
     * Send every param to Rust and take over what it changed (clamps, a voice's own settings).
     * Resolves `false` when Rust did not accept them (the user is told).
     */
    async syncParams(): Promise<boolean> {
        return (await this.sendParams()) !== null;
    }

    /**
     * Sync the current state with `set_voice_changer_params` (serialized, see `syncLoop`). Resolves
     * once a payload built from the state at call time (or later) was answered: null when Rust
     * refused it; otherwise `effective` is Rust's effective params, or null when the backend
     * replied without them (older builds).
     */
    private sendParams(): Promise<SyncResult> {
        this.syncDirty = true;
        if (!this.syncLoop) this.syncLoop = this.runSyncLoop();
        return this.syncLoop;
    }

    private async runSyncLoop(): Promise<SyncResult> {
        try {
            let result: SyncResult = null;
            do {
                this.syncDirty = false;
                result = await this.sendOnce();
            } while (this.syncDirty);
            return result;
        } finally {
            // Same turn as the last `syncDirty` check: a later `sendParams` starts a new loop.
            this.syncLoop = null;
            if (this.paramsEventPending) {
                this.paramsEventPending = false;
                void this.onParamsEvent({});
            }
        }
    }

    /** One `set` from the current state; its reply is adopted before the next payload is built. */
    private async sendOnce(): Promise<SyncResult> {
        const sent = toRustParams(this.state.voiceChanger);
        this.lastSent = sent;
        this.sendGen++;
        try {
            const reply = await invoke<unknown>("plugin:voice-changer|set_voice_changer_params", { params: sent });
            const effective = reply && typeof reply === "object" ? (reply as VoiceChangerParamsRust) : null;
            if (effective) this.adopt(sent, effective);
            return { effective };
        } catch (error) {
            console.error("[VoiceChanger] Failed to sync params:", error);
            this.reportSyncFailure(error);
            return null;
        }
    }

    /**
     * Take over what Rust changed relative to `sent` (fields edited locally since are kept), and
     * record the adopted values as Rust's in `lastSent`.
     */
    private adopt(sent: VoiceChangerParamsRust, effective: Partial<VoiceChangerParamsRust>) {
        const vc = this.state.voiceChanger;
        const { patch, adjusted } = adoptAdjustedParams(vc, sent, effective);
        if (adjusted.length === 0) return;
        Object.assign(vc, patch);
        if (adjusted.includes("rvc_slot_dir") && !adjusted.includes("slot_id")) void this.alignSlotId();
        if (this.lastSent === sent) {
            // As the state now sends them (e.g. an unset RVC EP shows as the profile default).
            const now = toRustParams(vc);
            const known: VoiceChangerParamsRust = { ...sent };
            for (const key of adjusted) Object.assign(known, { [key]: now[key] });
            this.lastSent = known;
        }
        devLog("[VoiceChanger] Rust adjusted:", adjusted.join(", "));
    }

    /**
     * Rust moved the selection to another folder without naming its slot (a failed voice switch went
     * back to the voice that plays): the slot id follows the folder, as `selectVoice` sets them.
     */
    private async alignSlotId() {
        if (!this.library.state.slots) await this.library.refresh();
        const vc = this.state.voiceChanger;
        const id = this.library.slotFor(vc.rvcSlotDir)?.id ?? "";
        if (id === vc.slotId) return;
        vc.slotId = id;
        await this.sendParams();
    }

    private reportSyncFailure(error: unknown) {
        // Without a desktop backend (web build) there is nothing to report.
        if (!this.state.hydrated) return;
        const now = Date.now();
        if (now - this.lastSyncFailureToastAt < SYNC_FAILURE_TOAST_INTERVAL_MS) return;
        this.lastSyncFailureToastAt = now;
        toast.error(i18n.t("voice_changer.toast_params_failed", { message: errorText(error) }));
    }

    async setPitch(pitch: number) {
        this.state.voiceChanger.pitch = Math.max(-24, Math.min(24, pitch));
        await this.syncParams();
    }

    async setFormant(formant: number) {
        this.state.voiceChanger.formant = Math.max(-1, Math.min(1, formant));
        await this.syncParams();
    }

    async setVocoderWindowMs(ms: number) {
        this.state.voiceChanger.vocoderWindowMs = Math.min(60, Math.max(30, Math.round(ms)));
        await this.syncParams();
    }

    async setVocoderOversample(o: VocoderOversample) {
        this.state.voiceChanger.vocoderOversample = o;
        await this.syncParams();
    }

    /** Latency budget (100-2000 ms). Applies while running, in every mode. */
    async setLatencyBudgetMs(ms: number) {
        this.state.voiceChanger.latencyBudgetMs = clampLatencyBudgetMs(ms);
        await this.syncParams();
    }

    /** The microphone while TTS plays through the voice changer. Applies while running. */
    async setAuxMicGate(gate: AuxMicGate) {
        this.state.voiceChanger.auxMicGate = normalizeAuxMicGate(gate);
        await this.syncParams();
    }

    /** Worker thread priority. Applies while running. */
    async setWorkerPriority(priority: WorkerPriority) {
        this.state.voiceChanger.workerPriority = normalizeWorkerPriority(priority);
        await this.syncParams();
    }

    /** Input device (`""` = system default); a running engine reopens its devices. */
    async setInputDevice(device: string) {
        this.state.voiceChanger.inputDevice = device;
        await this.syncParamsAndRestartIfRunning();
    }

    /** Store the RVC voice-permission acknowledgement in Rust's settings. Resolves whether Rust stored it. */
    async acknowledgeRvcConsent(): Promise<boolean> {
        this.state.voiceChanger.rvcConsentAcknowledged = true;
        this.state.rvcConsentPending = false;
        const reply = await this.sendParams();
        if (reply?.effective?.rvc_consent_acknowledged === true) clearLegacyConsent();
        return reply !== null;
    }

    /** The user closed the voice-permission notice without acknowledging it: RVC stays stopped. */
    dismissRvcConsent() {
        this.state.rvcConsentPending = false;
    }

    dispose() {
        for (const u of this.unlisteners.splice(0)) u();
        this.library.dispose();
        this.bench.dispose();
        this.clearTrt();
        this.initialized = false;
    }
}

/**
 * Slot ids of voice hotkeys 1-4 (`BackendState.vcHotkeyVoices`, added by the P3 INTEGRATE step;
 * empty until then).
 */
export function hotkeyVoices(): string[] {
    const state = typeof window !== "undefined" ? (window.ApiServer?.state as unknown as Record<string, unknown> | undefined) : undefined;
    return normalizeHotkeyVoices(state?.vcHotkeyVoices);
}
