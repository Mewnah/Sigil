import { invoke } from "@tauri-apps/api/core";

import { listen, UnlistenFn } from "@tauri-apps/api/event";

import i18n from "i18next";

import { proxy } from "valtio";

import { toast } from "react-toastify";

import type { RustEpOutcomeEntry } from "@/types";
import { devLog } from "@/utils/devLog";

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
import { epDisplayName, isVoiceChangerEpComponent, normalizeMetrics, type VoiceChangerMetrics } from "./metrics";
import { adoptAdjustedParams, applyRustParams, toRustParams, type VoiceChangerParamsRust } from "./params";
import { RvcProfile, VoiceChangerState, VoiceChangerStateSchema } from "./schema";
import { normalizeSlots, type VoiceChangerModelSlot } from "./slots";

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
    });

    private unlisteners: UnlistenFn[] = [];
    private initialized = false;

    /** Harness params the running engine was last started (or restarted) with; null when unknown. */
    private appliedHarness: HarnessParams | null = null;

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
            }),
            await listen("voice_changer:stopped", () => {
                this.state.isRunning = false;
                this.state.voiceChanger.enabled = false;
            }),
            await listen<VoiceChangerConversionErrorPayload>("voice_changer:error", (ev) => this.onError(ev.payload)),
            await listen<unknown>(PARAMS_EVENT, (ev) => void this.onParamsEvent(ev.payload)),
            await listen<RustEpOutcomeEntry>(EP_OUTCOME_EVENT, (ev) => this.onEpOutcome(ev.payload)),
        );

        try {
            this.state.isRunning = await invoke<boolean>("plugin:voice-changer|is_voice_changer_running");
        } catch {
            this.state.isRunning = false;
        }

        await this.hydrate();
        await this.migrateLegacyConsent();
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
            if (this.state.isRunning) this.appliedHarness = harnessParamsOf(vc);
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
        if (this.syncLoop || !this.lastSent) return;
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
        try {
            await this.syncParams();

            // The input device also travels in the params (and is persisted there); passing it here
            // keeps start independent of whether that sync reached Rust.
            await invoke("plugin:voice-changer|start_voice_changer", { inputDevice: vc.inputDevice });
            this.appliedHarness = harness;
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
     * Apply RVC Streamer / Low latency / Quality profile (RVC EP + F0 + index + extra context + VRAM).
     * Extra context follows the plan's minimal / small / higher split (HuBERT re-encodes it every tick).
     * Structural fields restart the engine when running.
     */
    async applyRvcProfile(profile: RvcProfile) {
        const vc = this.state.voiceChanger;
        vc.rvcProfile = profile;
        if (profile === "streamer") {
            vc.rvcExecutionProvider = "cpu";
            vc.pitchExtractor = "fcpe";
            vc.indexRate = 0;
            vc.extraContextMs = 64;
        } else if (profile === "low_latency") {
            vc.rvcExecutionProvider = "cuda";
            vc.pitchExtractor = "rmvpe";
            vc.indexRate = 0;
            vc.extraContextMs = 96;
            vc.vramCapMb = 768;
        } else {
            vc.rvcExecutionProvider = "cuda";
            vc.pitchExtractor = "rmvpe";
            vc.indexRate = 0.5;
            vc.extraContextMs = 192;
            vc.vramCapMb = 1536;
        }
        await this.syncParamsAndRestartIfRunning();
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
     * F0, …) and the reply brings those values into the UI; a running engine restarts with the voice.
     */
    async selectModelSlot(path: string) {
        this.state.voiceChanger.rvcSlotDir = path;
        await this.syncParamsAndRestartIfRunning();
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
     */
    async restartIfRunning() {
        if (!this.state.isRunning) return;
        await this.stop();
        await new Promise((r) => setTimeout(r, 80));
        await this.start();
    }

    /** Sync Rust params then restart the engine when running (devices, mode, chunk size, bridges). */
    async syncParamsAndRestartIfRunning() {
        await this.syncParams();
        await this.restartIfRunning();
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
        if (this.lastSent === sent) {
            // As the state now sends them (e.g. an unset RVC EP shows as the profile default).
            const now = toRustParams(vc);
            const known: VoiceChangerParamsRust = { ...sent };
            for (const key of adjusted) Object.assign(known, { [key]: now[key] });
            this.lastSent = known;
        }
        devLog("[VoiceChanger] Rust adjusted:", adjusted.join(", "));
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
        this.initialized = false;
    }
}
