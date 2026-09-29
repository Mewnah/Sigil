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
        expect(vc.pitchExtractor).toBe("rmvpe");
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

    it("a params event is ignored while a sync is in flight, and its re-read is dropped when a sync follows", async () => {
        const s = await started();
        let releaseSet: (() => void) | null = null;
        onSet = (p) => new Promise((r) => (releaseSet = () => r(p)));
        const pending = s.setPitch(4);
        await Promise.resolve();
        emit("voice_changer:params", { ...toRustParams(s.state.voiceChanger), index_rate: 0.9 });
        releaseSet!();
        await pending;
        expect(invoke.mock.calls.filter(([c]) => c === "plugin:voice-changer|get_voice_changer_params")).toHaveLength(1);
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
