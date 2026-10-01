import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (ev: { payload: unknown }) => void;
const handlers = new Map<string, Handler>();
const invoke = vi.fn();

vi.mock("@tauri-apps/api/core", () => ({ invoke: (...args: unknown[]) => invoke(...args), isTauri: () => true }));
vi.mock("@tauri-apps/api/event", () => ({
    listen: async (name: string, cb: Handler) => {
        handlers.set(name, cb);
        return () => handlers.delete(name);
    },
}));
vi.mock("react-toastify", () => ({
    toast: Object.assign(vi.fn(), { error: vi.fn(), info: vi.fn(), warning: vi.fn(), success: vi.fn() }),
}));
vi.mock("i18next", () => ({
    default: { t: (key: string, opts?: Record<string, unknown>) => (opts ? `${key} ${JSON.stringify(opts)}` : key) },
}));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import { toast } from "react-toastify";
import { LEGACY_CONSENT_STORAGE_KEY, VoiceChangerService } from "../index";
import { toRustParams, type VoiceChangerParamsRust } from "../params";
import { VoiceChangerStateSchema } from "../schema";

const emit = (name: string, payload: unknown) => handlers.get(name)?.({ payload });

/** Rust side: stored params plus a hook to change what `set` returns. */
let stored: VoiceChangerParamsRust;
let running = false;
/** `voice_changer_get_capabilities` reply (null = an older backend without it). */
let capsReply: unknown = null;
/** `voice_changer_list_slots` reply. */
let slotRows: Record<string, unknown>[] = [];
let onSet: (p: VoiceChangerParamsRust) => VoiceChangerParamsRust | Promise<VoiceChangerParamsRust> = (p) => p;

function rustDefaults(): VoiceChangerParamsRust {
    return toRustParams(VoiceChangerStateSchema.parse({}));
}

function installBackend() {
    invoke.mockImplementation(async (cmd: string, args?: Record<string, unknown>) => {
        switch (cmd) {
            case "plugin:voice-changer|is_voice_changer_running":
                return running;
            case "plugin:voice-changer|get_voice_changer_params":
                return structuredClone(stored);
            case "plugin:voice-changer|set_voice_changer_params": {
                const out = await onSet(args!.params as VoiceChangerParamsRust);
                stored = structuredClone(out);
                return structuredClone(out);
            }
            case "plugin:voice-changer|start_voice_changer":
                return null;
            case "plugin:voice-changer|stop_voice_changer":
                running = false;
                return null;
            case "plugin:voice-changer|get_voice_changer_metrics":
                return { buffer_ms: 1, process_ms: 2, chunk_ms: 10 };
            case "plugin:voice-changer|voice_changer_get_capabilities":
                if (capsReply === null) throw new Error("not allowed");
                return structuredClone(capsReply);
            case "plugin:voice-changer|voice_changer_list_slots":
                return structuredClone(slotRows);
            case "plugin:voice-changer|voice_changer_get_autotune":
                return null;
            default:
                throw new Error(`unexpected ${cmd}`);
        }
    });
}

class MemStorage {
    map = new Map<string, string>();
    getItem(k: string) {
        return this.map.has(k) ? this.map.get(k)! : null;
    }
    setItem(k: string, v: string) {
        this.map.set(k, v);
    }
    removeItem(k: string) {
        this.map.delete(k);
    }
}

let storage: MemStorage;

beforeEach(() => {
    handlers.clear();
    invoke.mockReset();
    vi.mocked(toast.warning).mockClear();
    vi.mocked(toast.error).mockClear();
    stored = rustDefaults();
    running = false;
    capsReply = null;
    slotRows = [];
    onSet = (p) => p;
    storage = new MemStorage();
    vi.stubGlobal("localStorage", storage);
    vi.spyOn(console, "error").mockImplementation(() => {});
    installBackend();
});

afterEach(() => {
    vi.unstubAllGlobals();
});

async function started(): Promise<VoiceChangerService> {
    const s = new VoiceChangerService();
    await s.init();
    return s;
}

const setCalls = () => invoke.mock.calls.filter(([c]) => c === "plugin:voice-changer|set_voice_changer_params");

describe("VoiceChangerService hydration", () => {
    it("hydrates every setting from Rust, including the input device and the P0 settings", async () => {
        stored = {
            ...rustDefaults(),
            mode: "rvc",
            input_device: "Microphone (Yeti)",
            output_device: "CABLE Input",
            latency_budget_ms: 650,
            aux_mic_gate: "mute",
            worker_priority: "protect_game",
            pitch_extractor: "crepe",
            rvc_consent_acknowledged: true,
            chunk_frames: 1024,
        };
        const s = await started();
        const vc = s.state.voiceChanger;
        expect(s.state.hydrated).toBe(true);
        expect(vc.mode).toBe("rvc");
        expect(vc.inputDevice).toBe("Microphone (Yeti)");
        expect(vc.outputDevice).toBe("CABLE Input");
        expect(vc.latencyBudgetMs).toBe(650);
        expect(vc.auxMicGate).toBe("mute");
        expect(vc.workerPriority).toBe("protect_game");
        // Since P3 `crepe` is a value of its own (CREPE-tiny), no longer an alias of rmvpe.
        expect(vc.pitchExtractor).toBe("crepe");
        expect(vc.rvcConsentAcknowledged).toBe(true);
        expect(vc.chunkFrames).toBe(1024);
        // Hydration alone writes nothing back.
        expect(setCalls()).toHaveLength(0);
    });

    it("keeps defaults and stays quiet without a backend", async () => {
        invoke.mockRejectedValue(new Error("no tauri"));
        const s = await started();
        expect(s.state.hydrated).toBe(false);
        expect(await s.syncParams()).toBe(false);
        expect(toast.error).not.toHaveBeenCalled();
    });
});

describe("RVC notice acknowledgement", () => {
    it("moves the old webview flag into Rust and removes it once stored", async () => {
        storage.setItem(LEGACY_CONSENT_STORAGE_KEY, "1");
        const s = await started();
        expect(s.state.voiceChanger.rvcConsentAcknowledged).toBe(true);
        expect(stored.rvc_consent_acknowledged).toBe(true);
        expect(storage.getItem(LEGACY_CONSENT_STORAGE_KEY)).toBeNull();
    });

    it("keeps the old flag when Rust did not store it (retried next launch)", async () => {
        storage.setItem(LEGACY_CONSENT_STORAGE_KEY, "1");
        onSet = (p) => {
            const { rvc_consent_acknowledged: _drop, ...rest } = p;
            return rest as VoiceChangerParamsRust; // an older backend without the field
        };
        await started();
        expect(storage.getItem(LEGACY_CONSENT_STORAGE_KEY)).toBe("1");
    });

    it("keeps the old flag when the reply carries no params at all", async () => {
        storage.setItem(LEGACY_CONSENT_STORAGE_KEY, "1");
        const base = invoke.getMockImplementation()!;
        invoke.mockImplementation(async (cmd: string, args?: Record<string, unknown>) =>
            cmd === "plugin:voice-changer|set_voice_changer_params" ? null : base(cmd, args),
        );
        const s = await started();
        expect(s.state.voiceChanger.rvcConsentAcknowledged).toBe(true); // this session
        expect(storage.getItem(LEGACY_CONSENT_STORAGE_KEY)).toBe("1");
        expect(await s.syncParams()).toBe(true); // a body-less reply is not a failure
    });

    it("drops a stale webview flag when Rust already has it", async () => {
        storage.setItem(LEGACY_CONSENT_STORAGE_KEY, "1");
        stored = { ...rustDefaults(), rvc_consent_acknowledged: true };
        await started();
        expect(setCalls()).toHaveLength(0);
        expect(storage.getItem(LEGACY_CONSENT_STORAGE_KEY)).toBeNull();
    });

    it("acknowledging stores it in Rust", async () => {
        const s = await started();
        expect(await s.acknowledgeRvcConsent()).toBe(true);
        expect(stored.rvc_consent_acknowledged).toBe(true);
    });
});

describe("RVC voice-permission gate", () => {
    const called = (cmd: string) => invoke.mock.calls.some(([c]) => c === `plugin:voice-changer|${cmd}`);

    it("start in RVC mode without the acknowledgement does not start and asks for it", async () => {
        stored = { ...rustDefaults(), mode: "rvc" };
        const s = await started();
        expect(await s.start()).toBe(false);
        expect(called("start_voice_changer")).toBe(false);
        expect(s.state.rvcConsentPending).toBe(true);
        expect(s.state.voiceChanger.enabled).toBe(false);
        expect(toast.error).not.toHaveBeenCalled();

        expect(await s.acknowledgeRvcConsent()).toBe(true);
        expect(s.state.rvcConsentPending).toBe(false);
        expect(await s.start()).toBe(true);
        expect(called("start_voice_changer")).toBe(true);
    });

    it("switching a running engine into RVC without the acknowledgement stops it and asks", async () => {
        running = true;
        const s = await started(); // Classic, running
        s.state.voiceChanger.mode = "rvc";
        await s.syncParamsAndRestartIfRunning();
        expect(called("stop_voice_changer")).toBe(true);
        expect(called("start_voice_changer")).toBe(false);
        expect(s.state.isRunning).toBe(false);
        expect(s.state.rvcConsentPending).toBe(true);
        expect(stored.rvc_consent_acknowledged).toBe(false);

        s.dismissRvcConsent();
        expect(s.state.rvcConsentPending).toBe(false);
        expect(called("start_voice_changer")).toBe(false);
    });

    it("other modes and an acknowledged RVC start without asking", async () => {
        const s = await started();
        expect(await s.start()).toBe(true);
        expect(s.state.rvcConsentPending).toBe(false);

        invoke.mockClear();
        stored = { ...rustDefaults(), mode: "rvc", rvc_consent_acknowledged: true };
        const r = await started();
        expect(await r.start()).toBe(true);
        expect(called("start_voice_changer")).toBe(true);
        expect(r.state.rvcConsentPending).toBe(false);
    });
});

describe("params sync", () => {
    it("adopts what Rust changed (a voice's own settings)", async () => {
        const s = await started();
        onSet = (p) => (p.rvc_slot_dir === "C:\\voices\\alto" ? { ...p, pitch_semitones: 5, protect: 0.2 } : p);
        s.state.voiceChanger.mode = "rvc";
        await s.selectModelSlot("C:\\voices\\alto");
        expect(s.state.voiceChanger.pitch).toBe(5);
        expect(s.state.voiceChanger.protect).toBe(0.2);
    });

    it("syncs are serialized: the next payload is built after the previous reply was adopted", async () => {
        const s = await started();
        let release: (() => void) | null = null;
        onSet = (p) =>
            p.pitch_semitones === 1
                ? new Promise((r) => (release = () => r({ ...p, protect: 0.1 })))
                : { ...p, protect: 0.45 };
        const first = s.setPitch(1);
        await Promise.resolve();
        const second = s.setPitch(2);
        const third = s.setPitch(3);
        await Promise.resolve();
        // The later changes wait for the first reply (and are coalesced into one payload).
        expect(setCalls()).toHaveLength(1);
        release!();
        await Promise.all([first, second, third]);
        const sent = setCalls().map(([, a]) => (a as { params: VoiceChangerParamsRust }).params);
        expect(sent).toHaveLength(2);
        // Built after the first reply: it carries the protect Rust returned, and the newest pitch.
        expect(sent[1]).toMatchObject({ pitch_semitones: 3, protect: 0.1 });
        expect(s.state.voiceChanger.pitch).toBe(3);
        expect(s.state.voiceChanger.protect).toBe(0.45);
    });

    it("entering a voice then moving a slider right away keeps the voice's settings and its slot.json", async () => {
        // Rust `update_params`, processed in order as each `set` arrives; the reply is delivered later.
        // Entering a voice (another slot, or into RVC) applies its slot.json; any later sync for the
        // same voice whose per-voice values differ saves them to that slot.json.
        const PER_VOICE = [
            "pitch_semitones",
            "formant_shift",
            "index_rate",
            "protect",
            "pitch_extractor",
            "speaker_id",
        ] as const;
        type VoiceFile = Pick<VoiceChangerParamsRust, (typeof PER_VOICE)[number]>;
        const voiceOf = (p: VoiceChangerParamsRust) =>
            Object.fromEntries(PER_VOICE.map((k) => [k, p[k]])) as VoiceFile;
        const OLD = "C:\\voices\\old";
        const NEW = "C:\\voices\\new";
        stored = {
            ...rustDefaults(),
            mode: "rvc",
            rvc_slot_dir: OLD,
            rvc_consent_acknowledged: true,
            pitch_semitones: 0,
            formant_shift: 0,
            index_rate: 0.3,
            protect: 0.5,
            pitch_extractor: "fcpe",
            speaker_id: 0,
        };
        const oldVoice = voiceOf(stored);
        const newVoice: VoiceFile = {
            pitch_semitones: 5,
            formant_shift: 0,
            index_rate: 0.75,
            protect: 0.2,
            pitch_extractor: "rmvpe",
            speaker_id: 1,
        };
        const files = new Map<string, VoiceFile>([
            [OLD, oldVoice],
            [NEW, newVoice],
        ]);
        const s = await started();

        let enteringReply: VoiceChangerParamsRust | null = null;
        const base = invoke.getMockImplementation()!;
        invoke.mockImplementation(async (cmd: string, args?: Record<string, unknown>) => {
            if (cmd !== "plugin:voice-changer|set_voice_changer_params") return base(cmd, args);
            const p = args!.params as VoiceChangerParamsRust;
            const before = stored;
            const entering = p.mode === "rvc" && (before.mode !== "rvc" || before.rvc_slot_dir !== p.rvc_slot_dir);
            let next: VoiceChangerParamsRust = { ...p };
            if (entering) {
                next = { ...next, ...files.get(p.rvc_slot_dir) };
                enteringReply = structuredClone(next);
            } else if (p.mode === "rvc" && p.rvc_slot_dir && PER_VOICE.some((k) => p[k] !== before[k])) {
                files.set(p.rvc_slot_dir, voiceOf(next));
            }
            stored = next;
            await new Promise((r) => setTimeout(r, 5));
            return structuredClone(next);
        });

        const a = s.selectModelSlot(NEW); // e.g. the voice folder field's blur
        const b = s.setPitch(3); // the pitch slider pressed before A's reply arrives
        await Promise.all([a, b]);

        const sent = setCalls().map(([, x]) => (x as { params: VoiceChangerParamsRust }).params);
        expect(sent).toHaveLength(2);
        expect(sent[1]).toMatchObject({ ...newVoice, rvc_slot_dir: NEW, pitch_semitones: 3 });
        expect(files.get(NEW)).toEqual({ ...newVoice, pitch_semitones: 3 });
        expect(files.get(OLD)).toEqual(oldVoice);
        expect(s.state.voiceChanger).toMatchObject({
            rvcSlotDir: NEW,
            pitch: 3,
            protect: 0.2,
            indexRate: 0.75,
            pitchExtractor: "rmvpe",
            speakerId: 1,
        });

        // Tauri delivers the entering sync's `voice_changer:params` after both replies: stale, not adopted.
        expect(enteringReply).not.toBeNull();
        emit("voice_changer:params", enteringReply);
        await new Promise((r) => setTimeout(r, 0));
        expect(s.state.voiceChanger.pitch).toBe(3);
        expect(files.get(NEW)).toEqual({ ...newVoice, pitch_semitones: 3 });
    });

    it("a params event re-reads Rust's params when no sync of ours is pending", async () => {
        const s = await started();
        // Another window changed the index rate.
        stored = { ...toRustParams(s.state.voiceChanger), index_rate: 0.7 };
        emit("voice_changer:params", structuredClone(stored));
        await new Promise((r) => setTimeout(r, 0));
        expect(s.state.voiceChanger.indexRate).toBe(0.7);
        // And again: the adopted value is Rust's now, so a later change is not mistaken for a local edit.
        stored = { ...stored, index_rate: 0.4 };
        emit("voice_changer:params", structuredClone(stored));
        await new Promise((r) => setTimeout(r, 0));
        expect(s.state.voiceChanger.indexRate).toBe(0.4);
    });

    it("a params event's payload is not adopted; while a sync is in flight Rust is read after it, and a read older than a sync is dropped", async () => {
        const s = await started();
        let releaseSet: (() => void) | null = null;
        onSet = (p) => new Promise((r) => (releaseSet = () => r(p)));
        const pending = s.setPitch(4);
        await Promise.resolve();
        emit("voice_changer:params", { ...toRustParams(s.state.voiceChanger), index_rate: 0.9 });
        expect(invoke.mock.calls.filter(([c]) => c === "plugin:voice-changer|get_voice_changer_params")).toHaveLength(1);
        releaseSet!();
        await pending;
        // Hydration, then one read once the sync was done (Rust may have changed something after it
        // handled that set); the stale payload itself is never adopted.
        expect(invoke.mock.calls.filter(([c]) => c === "plugin:voice-changer|get_voice_changer_params")).toHaveLength(2);
        await new Promise((r) => setTimeout(r, 0));
        expect(s.state.voiceChanger.indexRate).not.toBe(0.9);

        // The event arrives with nothing pending; its re-read is still in flight when the user changes
        // the index rate (the `set` goes after the `get`, so the read is older than that reply).
        onSet = (p) => p;
        let releaseGet: (() => void) | null = null;
        const base = invoke.getMockImplementation()!;
        invoke.mockImplementation(async (cmd: string, args?: Record<string, unknown>) => {
            if (cmd !== "plugin:voice-changer|get_voice_changer_params") return base(cmd, args);
            const snapshot = structuredClone({ ...stored, index_rate: 0.7 });
            await new Promise<void>((r) => (releaseGet = r));
            return snapshot;
        });
        emit("voice_changer:params", { ...stored, index_rate: 0.7 });
        await Promise.resolve();
        s.state.voiceChanger.indexRate = 0.2;
        await s.syncParams();
        releaseGet!();
        await new Promise((r) => setTimeout(r, 0));
        expect(s.state.voiceChanger.indexRate).toBe(0.2);
        expect(stored.index_rate).toBe(0.2);
    });

    it("tells the user (throttled) when Rust rejects the params", async () => {
        const s = await started();
        onSet = () => {
            throw new Error("invalid type: floating point `1.5`, expected u32");
        };
        expect(await s.syncParams()).toBe(false);
        expect(await s.syncParams()).toBe(false);
        expect(toast.error).toHaveBeenCalledTimes(1);
    });

    it("hot settings only sync; the input device restarts a running engine", async () => {
        running = true;
        const s = await started();
        expect(s.state.isRunning).toBe(true);
        await s.setLatencyBudgetMs(5000);
        expect(stored.latency_budget_ms).toBe(2000);
        await s.setAuxMicGate("duck");
        await s.setWorkerPriority("protect_voice");
        expect(stored.aux_mic_gate).toBe("duck");
        expect(stored.worker_priority).toBe("protect_voice");
        expect(invoke.mock.calls.some(([c]) => c === "plugin:voice-changer|stop_voice_changer")).toBe(false);
    });

    it("start sends the params, then the input device", async () => {
        const s = await started();
        s.state.voiceChanger.inputDevice = "Mic 2";
        await s.start();
        const order = invoke.mock.calls.map(([c]) => c);
        const iSet = order.lastIndexOf("plugin:voice-changer|set_voice_changer_params");
        const iStart = order.indexOf("plugin:voice-changer|start_voice_changer");
        expect(iSet).toBeGreaterThanOrEqual(0);
        expect(iStart).toBeGreaterThan(iSet);
        expect(invoke.mock.calls[iStart][1]).toEqual({ inputDevice: "Mic 2" });
        expect(stored.input_device).toBe("Mic 2");
    });

    it("a start that Rust refuses is shown to the user and leaves the switch off", async () => {
        const s = await started();
        const base = invoke.getMockImplementation()!;
        invoke.mockImplementation(async (cmd: string, args?: Record<string, unknown>) => {
            if (cmd === "plugin:voice-changer|start_voice_changer") throw "Choose an ONNX model file for ONNX voice mode.";
            return base(cmd, args);
        });
        await expect(s.start()).rejects.toBe("Choose an ONNX model file for ONNX voice mode.");
        expect(s.state.voiceChanger.enabled).toBe(false);
        expect(toast.error).toHaveBeenCalledTimes(1);
        expect(String(vi.mocked(toast.error).mock.calls[0][0])).toContain("Choose an ONNX model file");
    });
});

describe("execution-provider fallback toast", () => {
    const outcome = (component: string, requested: string, active: string, reason: string | null = null) => ({
        component,
        requested,
        active,
        reason,
        at_unix_ms: Date.now(),
    });

    it("warns once per run, not again for a known fallback, and again after it worked", async () => {
        await started();
        // Not running: ignored.
        emit("diagnostics:ep_outcome", outcome("vc", "cuda", "cpu", "cudnn missing"));
        expect(toast.warning).not.toHaveBeenCalled();

        emit("voice_changer:started", null);
        emit("diagnostics:ep_outcome", outcome("vc.content", "cuda", "cpu", "cudnn missing"));
        emit("diagnostics:ep_outcome", outcome("vc.pitch", "cuda", "cpu", "cudnn missing"));
        emit("diagnostics:ep_outcome", outcome("vc", "cuda", "cpu", "cudnn missing"));
        emit("diagnostics:ep_outcome", outcome("tts.kitten", "cuda", "cpu"));
        expect(toast.warning).toHaveBeenCalledTimes(1);
        expect(String(vi.mocked(toast.warning).mock.calls[0][0])).toContain("cudnn missing");

        // Restart with the same fallback: already known.
        emit("voice_changer:stopped", null);
        emit("voice_changer:started", null);
        emit("diagnostics:ep_outcome", outcome("vc", "cuda", "cpu", "cudnn missing"));
        expect(toast.warning).toHaveBeenCalledTimes(1);

        // CUDA works, then breaks again: that is news.
        emit("voice_changer:started", null);
        emit("diagnostics:ep_outcome", outcome("vc", "cuda", "cuda"));
        emit("voice_changer:started", null);
        emit("diagnostics:ep_outcome", outcome("vc", "cuda", "dml", "driver too old"));
        expect(toast.warning).toHaveBeenCalledTimes(2);
    });
});

describe("voice_changer:error", () => {
    it("maps codes to their toasts and unknown codes to a generic one", async () => {
        await started();
        emit("voice_changer:error", { code: "rvc_load", message: "gen.onnx not found" });
        emit("voice_changer:error", { code: "sidecar_exchange", message: "timeout" });
        emit("voice_changer:error", { code: "something_new", message: "boom" });
        const keys = vi.mocked(toast.warning).mock.calls.map(([m]) => String(m).split(" ")[0]);
        expect(keys).toEqual([
            "voice_changer.toast_error_rvc",
            "voice_changer.toast_error_sidecar",
            "voice_changer.toast_error_generic",
        ]);
    });
});

describe("P3: settings changes while running (restart only when the engine shape changed)", () => {
    const calls = (cmd: string) => invoke.mock.calls.filter(([c]) => c === `plugin:voice-changer|${cmd}`).length;

    async function runningRvc() {
        stored = { ...rustDefaults(), mode: "rvc", rvc_consent_acknowledged: true, rvc_slot_dir: "C:/m/alto", slot_id: "alto" };
        running = true;
        const s = await started();
        expect(s.state.isRunning).toBe(true);
        return s;
    }

    it("voice controls apply live, and a voice, F0 method or range change switches without a restart", async () => {
        const s = await runningRvc();
        await s.update({ rmsMix: 0.4, formantSemitones: -2, noiseScale: 0.5, silenceDb: -50, indexRate: 0.6 });
        expect(stored.rms_mix).toBe(0.4);
        await s.selectVoice({ id: "bass", path: "C:/m/bass" });
        expect(stored).toMatchObject({ rvc_slot_dir: "C:/m/bass", slot_id: "bass" });
        await s.update({ pitchExtractor: "dio", f0Range: "singing" });
        expect(calls("stop_voice_changer")).toBe(0);
    });

    it("an engine-shape change restarts the running engine once", async () => {
        const s = await runningRvc();
        await s.update({ precision: "fp16" });
        expect(calls("stop_voice_changer")).toBe(1);
        expect(calls("start_voice_changer")).toBe(1);
        emit("voice_changer:started", null);
        // The engine now runs fp16: a hot change afterwards does not restart again.
        await s.update({ rmsMix: 0.9 });
        expect(calls("stop_voice_changer")).toBe(1);
    });

    it("concurrent shape changes restart the engine once", async () => {
        const s = await runningRvc();
        await Promise.all([s.update({ precision: "fp16" }), s.update({ hotSlots: 3 }), s.update({ blockMs: 60 })]);
        expect(calls("stop_voice_changer")).toBe(1);
        expect(calls("start_voice_changer")).toBe(1);
        expect(stored).toMatchObject({ precision: "fp16", hot_slots: 3, block_ms: 60 });
    });

    it("a shape edit made without a restart is applied by the next change", async () => {
        const s = await runningRvc();
        s.state.voiceChanger.blockMs = 60; // e.g. typed, not committed yet
        await s.update({ rmsMix: 0.2 });
        expect(calls("stop_voice_changer")).toBe(1);
    });

    it("RVC engine settings do not restart another mode; a voice hotkey switches into RVC (restart)", async () => {
        stored = { ...rustDefaults(), mode: "classic", rvc_consent_acknowledged: true };
        running = true;
        const s = await started();
        await s.update({ precision: "fp16", cudaGraphs: "on" });
        expect(calls("stop_voice_changer")).toBe(0);
        await s.selectVoice({ id: "alto", path: "C:/m/alto" }, { ensureRvc: true });
        expect(stored.mode).toBe("rvc");
        expect(calls("stop_voice_changer")).toBe(1);
        expect(calls("start_voice_changer")).toBe(1);
    });

    it("presets and tuned settings go through the same rules", async () => {
        const s = await runningRvc();
        // Re-applying the preset the engine already runs changes nothing, so it does not restart.
        await s.applyRvcProfile("streamer");
        expect(calls("stop_voice_changer")).toBe(0);
        await s.applyRvcProfile("quality");
        expect(stored).toMatchObject({ rvc_preset: "quality", rvc_execution_provider: "cuda", pitch_extractor: "rmvpe", extra_context_ms: 192 });
        expect(calls("stop_voice_changer")).toBe(1);
        emit("voice_changer:started", null);
        expect(await s.applyTuning({ rvc_execution_provider: "cuda", block_ms: 40, mode: "classic", rvc_slot_dir: "C:/x" })).toBe(true);
        expect(stored).toMatchObject({ rvc_execution_provider: "cuda", block_ms: 40, mode: "rvc", rvc_slot_dir: "C:/m/alto" });
        expect(calls("stop_voice_changer")).toBe(2);
        expect(await s.applyTuning({ unknown: 1 })).toBe(false);
    });
});

describe("P3: capabilities, TensorRT progress, new error codes and hotkeys", () => {
    it("reads the capabilities (normalized) and keeps the baseline without the command", async () => {
        const s = await started();
        expect(s.state.capabilitiesLoaded).toBe(false);
        expect(s.state.capabilities.pitch_extractors).toEqual(["fcpe", "rmvpe", "dio"]);
        capsReply = { pitch_extractors: ["swiftf0", "bogus"], rvc_execution_providers: ["dml_hp"], importer: true, max_hot_slots: 4 };
        const caps = await s.refreshCapabilities();
        expect(caps.pitch_extractors).toEqual(["fcpe", "rmvpe", "dio", "swiftf0"]);
        expect(s.state.capabilities).toMatchObject({ importer: true, max_hot_slots: 4 });
        expect(s.state.capabilitiesLoaded).toBe(true);
    });

    it("tracks TensorRT engine builds per component and clears them on stop", async () => {
        vi.useFakeTimers();
        try {
            running = true;
            const s = await started();
            emit("voice_changer:trt_progress", { component: "vc.flowdec", stage: "building", fraction: 0.4, message: "", elapsed_ms: 9000 });
            emit("voice_changer:trt_progress", { component: "vc.enc", stage: "cache_hit", fraction: 1, message: "", elapsed_ms: 30 });
            emit("voice_changer:trt_progress", { component: "vc.flowdec", stage: "building", fraction: 7, message: "", elapsed_ms: 12000 });
            expect(s.state.trt.map((r) => [r.component, r.stage, r.fraction])).toEqual([
                ["vc.flowdec", "building", 1],
                ["vc.enc", "cache_hit", 1],
            ]);
            await vi.advanceTimersByTimeAsync(7000);
            expect(s.state.trt.map((r) => r.component)).toEqual(["vc.flowdec"]);
            emit("voice_changer:trt_progress", { component: "vc.flowdec", stage: "failed", fraction: 0, message: "no TensorRT-RTX", elapsed_ms: 13000 });
            await vi.advanceTimersByTimeAsync(7000);
            expect(s.state.trt.map((r) => r.stage)).toEqual(["failed"]);
            emit("voice_changer:stopped", null);
            expect(s.state.trt).toEqual([]);
        } finally {
            vi.useRealTimers();
        }
    });

    it("tells the user about lost / restored devices and the denoiser", async () => {
        await started();
        emit("voice_changer:error", { code: "device_lost", message: "The microphone disconnected." });
        emit("voice_changer:error", { code: "device_restored", message: "The microphone is back." });
        emit("voice_changer:error", { code: "denoiser_warning", message: "gtcrn: model missing" });
        expect(String(vi.mocked(toast.warning).mock.calls[0][0])).toContain("voice_changer.toast_warning_device_lost");
        expect(String(vi.mocked(toast.info).mock.calls[0][0])).toContain("voice_changer.toast_info_device_restored");
        expect(String(vi.mocked(toast.warning).mock.calls[1][0])).toContain("voice_changer.toast_warning_denoiser");
    });

    it("hotkeys toggle the engine and step through the voices", async () => {
        stored = { ...rustDefaults(), mode: "rvc", rvc_consent_acknowledged: true, rvc_slot_dir: "C:/m/a" };
        slotRows = [
            { id: "a", name: "A", path: "C:/m/a", format: "sigil-rvc-1" },
            { id: "b", name: "B", path: "C:/m/b", format: "sigil-rvc-1", problem: "broken" },
            { id: "c", name: "C", path: "C:/m/c", format: "rvc" },
        ];
        capsReply = { slot_formats: ["rvc", "sigil-rvc-1"] };
        const s = await started();
        await s.runShortcut("vcNextVoice");
        expect(stored.rvc_slot_dir).toBe("C:/m/c");
        await s.runShortcut("vcNextVoice");
        expect(stored.rvc_slot_dir).toBe("C:/m/a");
        await s.runShortcut("vcPrevVoice");
        expect(stored.rvc_slot_dir).toBe("C:/m/c");
        // No voice assigned to hotkey 1 (the app settings do not carry the assignments here).
        await s.runShortcut("vcVoice1");
        expect(stored.rvc_slot_dir).toBe("C:/m/c");
        expect(String(vi.mocked(toast.info).mock.calls.at(-1)?.[0])).toContain("hotkey_no_voice_assigned");
        await s.runShortcut("vcToggle");
        expect(invoke.mock.calls.some(([c]) => c === "plugin:voice-changer|start_voice_changer")).toBe(true);
        await s.runShortcut("muteMic");
    });

    it("a voice hotkey selects its assigned voice", async () => {
        stored = { ...rustDefaults(), mode: "classic" };
        slotRows = [
            { id: "kira", name: "Kira", path: "C:/m/kira", format: "sigil-rvc-1" },
            { id: "gone", name: "Gone", path: "C:/m/gone", format: "sigil-rvc-1", problem: "the template is missing" },
        ];
        capsReply = { slot_formats: ["rvc", "sigil-rvc-1"] };
        vi.stubGlobal("window", { ApiServer: { state: { vcHotkeyVoices: ["gone", "kira"] } } });
        const s = await started();
        await s.runShortcut("vcVoice1");
        expect(stored.rvc_slot_dir).toBe("");
        expect(String(vi.mocked(toast.warning).mock.calls.at(-1)?.[0])).toContain("hotkey_voice_unusable");
        await s.runShortcut("vcVoice2");
        expect(stored).toMatchObject({ mode: "rvc", rvc_slot_dir: "C:/m/kira", slot_id: "kira" });
        // RVC without the permission notice: the hotkey says where to confirm it.
        await s.runShortcut("vcToggle");
        expect(s.state.rvcConsentPending).toBe(true);
        expect(String(vi.mocked(toast.info).mock.calls.at(-1)?.[0])).toContain("hotkey_consent_needed");
    });
});

describe("P3-G review: restarts, failed voice switches, TensorRT rows and live recording", () => {
    const START = "plugin:voice-changer|start_voice_changer";
    const calls = (cmd: string) => invoke.mock.calls.filter(([c]) => c === cmd).length;

    /** The mocked backend, with `override` answering first (`undefined`: fall through). */
    function overrideBackend(override: (cmd: string, args?: Record<string, unknown>) => unknown) {
        const base = invoke.getMockImplementation()!;
        invoke.mockImplementation(async (cmd: string, args?: Record<string, unknown>) => {
            const r = await override(cmd, args);
            return r === undefined ? base(cmd, args) : r;
        });
    }

    async function runningRvc() {
        stored = { ...rustDefaults(), mode: "rvc", rvc_consent_acknowledged: true, rvc_slot_dir: "C:/m/alto", slot_id: "alto", pitch_semitones: 3 };
        slotRows = [
            { id: "alto", name: "Alto", path: "C:/m/alto", format: "sigil-rvc-1" },
            { id: "bass", name: "Bass", path: "C:/m/bass", format: "sigil-rvc-1" },
        ];
        capsReply = { slot_formats: ["rvc", "sigil-rvc-1"] };
        running = true;
        const s = await started();
        expect(s.state.isRunning).toBe(true);
        return s;
    }

    it("a failed restart resolves false (no rejection) and does not fail a concurrent change", async () => {
        const s = await runningRvc();
        overrideBackend((cmd) => {
            if (cmd === START) throw "missing base model: rmvpe";
            return undefined;
        });
        const results = await Promise.all([s.update({ precision: "fp16" }), s.update({ hotSlots: 3 })]);
        expect(results).toEqual([false, true]);
        expect(calls(START)).toBe(1);
        expect(String(vi.mocked(toast.error).mock.calls.at(-1)?.[0])).toContain("voice_changer.toast_start_failed");
        expect(s.state.isRunning).toBe(false);
        expect(await s.syncParamsAndRestartIfRunning()).toBe(true); // stopped: nothing to restart
    });

    it("a voice hotkey whose restart fails does not report the voice as selected", async () => {
        stored = { ...rustDefaults(), mode: "classic", rvc_consent_acknowledged: true };
        slotRows = [{ id: "kira", name: "Kira", path: "C:/m/kira", format: "sigil-rvc-1" }];
        capsReply = { slot_formats: ["rvc", "sigil-rvc-1"] };
        running = true;
        const s = await started();
        overrideBackend((cmd) => {
            if (cmd === START) throw "no voice model";
            return undefined;
        });
        vi.mocked(toast.info).mockClear();
        await expect(s.runShortcut("vcNextVoice")).resolves.toBeUndefined();
        expect(stored).toMatchObject({ mode: "rvc", rvc_slot_dir: "C:/m/kira" });
        expect(vi.mocked(toast.info).mock.calls.some(([m]) => String(m).includes("hotkey_voice_selected"))).toBe(false);
    });

    it("a failed voice switch: the selection follows Rust back to the voice that plays, with its slot id", async () => {
        const s = await runningRvc();
        await s.selectVoice({ id: "bass", path: "C:/m/bass" });
        expect(s.state.voiceChanger).toMatchObject({ rvcSlotDir: "C:/m/bass", slotId: "bass" });
        // Rust: the load failed, the worker went back to alto (its values, but not its slot id).
        stored = { ...stored, rvc_slot_dir: "C:/m/alto", pitch_semitones: 3 };
        emit("voice_changer:error", { code: "rvc_load", message: "bass: content encoder not installed" });
        emit("voice_changer:params", structuredClone(stored));
        await vi.waitFor(() => expect(stored.slot_id).toBe("alto"));
        expect(s.state.voiceChanger).toMatchObject({ rvcSlotDir: "C:/m/alto", slotId: "alto", pitch: 3 });
        expect(calls("plugin:voice-changer|stop_voice_changer")).toBe(0);
    });

    it("a params event while a set is in flight is read again once the sync is done", async () => {
        const s = await runningRvc();
        await s.selectVoice({ id: "bass", path: "C:/m/bass" });
        // Rust handles the next set at once but its reply is delivered later.
        let deliver: () => void = () => {};
        const gate = new Promise<void>((r) => (deliver = r));
        overrideBackend(async (cmd, args) => {
            if (cmd !== "plugin:voice-changer|set_voice_changer_params") return undefined;
            stored = structuredClone(args!.params as VoiceChangerParamsRust);
            const reply = structuredClone(stored);
            await gate;
            return reply;
        });
        const pending = s.update({ rmsMix: 0.3 });
        await Promise.resolve();
        // Meanwhile the switch to bass failed: Rust went back to alto and said so.
        stored = { ...stored, rvc_slot_dir: "C:/m/alto", pitch_semitones: 3 };
        emit("voice_changer:params", structuredClone(stored));
        deliver();
        await pending;
        await vi.waitFor(() => expect(s.state.voiceChanger.rvcSlotDir).toBe("C:/m/alto"));
        expect(s.state.voiceChanger).toMatchObject({ pitch: 3, rmsMix: 0.3 });
    });

    it("TensorRT rows only for the voice changer's own run", async () => {
        const row = { component: "vc.flowdec", stage: "building", fraction: 0.2, message: "", elapsed_ms: 900 };
        const s = await started(); // stopped
        // An autotune or benchmark build (only while stopped) is not the voice changer's.
        emit("voice_changer:trt_progress", row);
        expect(s.state.trt).toEqual([]);
        emit("voice_changer:started", null);
        emit("voice_changer:trt_progress", row);
        expect(s.state.trt).toHaveLength(1);
        running = true;
        await s.stop();
        expect(s.state.trt).toEqual([]);
        // The stopped worker's build keeps reporting until it ends.
        emit("voice_changer:trt_progress", { ...row, fraction: 0.6 });
        expect(s.state.trt).toEqual([]);
    });

    it("a live recording runs through a settings restart", async () => {
        vi.useFakeTimers();
        try {
            const s = await runningRvc();
            overrideBackend((cmd) => {
                if (cmd === VC_LIVE_START) return null;
                if (cmd === VC_LIVE_STOP) return { seconds: 120, ticks: 1, report_dir: "C:/r/live", metrics: {} };
                return undefined;
            });
            expect(await s.bench.startLiveRecord(120)).toBe(true);
            const restart = s.update({ precision: "fp16" }); // stop, 80 ms, start
            await vi.advanceTimersByTimeAsync(40);
            emit("voice_changer:stopped", null);
            await vi.advanceTimersByTimeAsync(60);
            expect(await restart).toBe(true);
            emit("voice_changer:started", null);
            await vi.advanceTimersByTimeAsync(10_000);
            expect(calls(VC_LIVE_STOP)).toBe(0);
            expect(s.bench.state.liveRecord.recording).toBe(true);
            // A real stop ends it (Rust after 2 s): collected then.
            await s.stop();
            await vi.advanceTimersByTimeAsync(3_000);
            expect(calls(VC_LIVE_STOP)).toBe(1);
            expect(s.bench.state.liveRecord).toMatchObject({ recording: false, summary: { report_dir: "C:/r/live" } });
            s.dispose();
        } finally {
            vi.useRealTimers();
        }
    });
});

const VC_LIVE_START = "plugin:voice-changer|voice_changer_live_record_start";
const VC_LIVE_STOP = "plugin:voice-changer|voice_changer_live_record_stop";
