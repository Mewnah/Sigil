/**
 * The voice-library host module against a stubbed Rust plugin: the shared take state, the automatic fetch of a
 * take when the recorder stops by itself, de-duplicated stops, and the voices list every picker reads.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (ev: { payload: unknown }) => void;
const handlers = new Map<string, Handler>();
const invoke = vi.fn();
let tauri = true;

vi.mock("@tauri-apps/api/event", () => ({
    listen: async (name: string, cb: Handler) => {
        handlers.set(name, cb);
        return () => handlers.delete(name);
    },
}));
vi.mock("@tauri-apps/api/core", () => ({
    invoke: (...args: unknown[]) => invoke(...args),
    isTauri: () => tauri,
}));

const lib = await import("../index");
const { VL_COMMANDS, VL_EVENTS, voiceLibraryUi } = lib;

const take = (id: string, ok = true) => ({
    takeId: id,
    quality: { durationMs: 5000, trimmedMs: 800, snrDb: 35, peakDbfs: -8, clippedRatio: 0, lufsBefore: -31, lufsAfter: -20, ok, issues: [] },
    source: "recorded" as const,
    suggestedName: null,
    sampleRate: 48000,
});

const meta = (id: string, name: string) => ({
    id,
    name,
    createdUnixMs: 1,
    durationMs: 5000,
    sampleRate: 48000,
    lufs: -20,
    snrDb: 35,
    peakDbfs: -1.5,
    clipped: false,
    transcript: null,
    language: null,
    source: "recorded" as const,
    sha256: "ab",
    cachedBackends: [],
    transcribing: true,
});

async function flush() {
    for (let i = 0; i < 5; i++) await Promise.resolve();
}

beforeEach(() => {
    tauri = true;
    handlers.clear();
    invoke.mockReset();
    lib.__resetVoiceLibraryHostForTests();
});

afterEach(() => {
    vi.useRealTimers();
});

describe("voice library host", () => {
    it("does nothing outside the desktop app", () => {
        tauri = false;
        lib.ensureVoiceLibraryHost();
        expect(invoke).not.toHaveBeenCalled();
        expect(handlers.size).toBe(0);
    });

    it("loads the list once and re-reads it on every change", async () => {
        invoke.mockResolvedValue([meta("alice-000001", "Alice")]);
        lib.ensureVoiceLibraryHost();
        lib.ensureVoiceLibraryHost();
        await flush();
        expect(invoke).toHaveBeenCalledTimes(1);
        expect(invoke).toHaveBeenCalledWith(VL_COMMANDS.list);
        expect(voiceLibraryUi.loaded).toBe(true);
        expect(voiceLibraryUi.voices.map((v) => v.name)).toEqual(["Alice"]);
        invoke.mockResolvedValue([meta("alice-000001", "Alice"), meta("bob-000002", "Bob")]);
        handlers.get(VL_EVENTS.changed)!({ payload: null });
        await flush();
        expect(voiceLibraryUi.voices).toHaveLength(2);
    });

    it("fetches the take by itself when the recorder stops at its limit", async () => {
        invoke.mockImplementation(async (cmd: string) => {
            if (cmd === VL_COMMANDS.list) return [];
            if (cmd === VL_COMMANDS.recordStart) return null;
            if (cmd === VL_COMMANDS.recordStop) return take("take-00000000000a");
            throw new Error(`unexpected ${cmd}`);
        });
        lib.ensureVoiceLibraryHost();
        await flush();
        await lib.startRecording({ device: "default", maxSeconds: 12 });
        expect(invoke).toHaveBeenCalledWith(VL_COMMANDS.recordStart, { request: { device: "default", maxSeconds: 12 } });
        expect(voiceLibraryUi.recording).toBe(true);
        const progress = handlers.get(VL_EVENTS.recordProgress)!;
        progress({ payload: { elapsedMs: 500, levelDb: -30, peakDb: -12, clipping: false, voicedMs: 300 } });
        expect(voiceLibraryUi.progress?.voicedMs).toBe(300);
        progress({ payload: { elapsedMs: 12000, levelDb: -30, peakDb: -12, clipping: false, voicedMs: 9000, stopped: true, stopReason: "max_length" } });
        // A click on Stop at the same moment shares the fetch.
        const clicked = lib.stopRecording();
        await clicked;
        await flush();
        expect(invoke.mock.calls.filter(([c]) => c === VL_COMMANDS.recordStop)).toHaveLength(1);
        expect(voiceLibraryUi.take?.takeId).toBe("take-00000000000a");
        expect(voiceLibraryUi.recording).toBe(false);
        expect(voiceLibraryUi.busy).toBe(null);
    });

    it("ignores progress of a recording that was cancelled", async () => {
        invoke.mockResolvedValue(null);
        lib.ensureVoiceLibraryHost();
        await flush();
        await lib.startRecording();
        await lib.cancelRecording();
        expect(invoke).toHaveBeenCalledWith(VL_COMMANDS.recordCancel);
        handlers.get(VL_EVENTS.recordProgress)!({ payload: { elapsedMs: 1, levelDb: -1, peakDb: -1, clipping: false, voicedMs: 0, stopped: true } });
        await flush();
        expect(invoke.mock.calls.some(([c]) => c === VL_COMMANDS.recordStop)).toBe(false);
        expect(voiceLibraryUi.progress).toBe(null);
        expect(voiceLibraryUi.recording).toBe(false);
    });

    it("a new recording or import discards the take that was waiting", async () => {
        invoke.mockImplementation(async (cmd: string) => {
            if (cmd === VL_COMMANDS.importClip) return { ...take("take-00000000000c"), source: "imported", suggestedName: "Clip" };
            return null;
        });
        voiceLibraryUi.take = take("take-00000000000b");
        await lib.startRecording();
        expect(invoke).toHaveBeenCalledWith(VL_COMMANDS.discardTake, { takeId: "take-00000000000b" });
        // Only once the recording started.
        const order = invoke.mock.calls.map(([c]) => c);
        expect(order.indexOf(VL_COMMANDS.recordStart)).toBeLessThan(order.indexOf(VL_COMMANDS.discardTake));
        expect(voiceLibraryUi.take).toBe(null);
        voiceLibraryUi.take = take("take-00000000000d");
        const imported = await lib.importClip("C:/clips/clip.wav");
        expect(invoke).toHaveBeenCalledWith(VL_COMMANDS.importClip, { request: { path: "C:/clips/clip.wav" } });
        expect(invoke).toHaveBeenCalledWith(VL_COMMANDS.discardTake, { takeId: "take-00000000000d" });
        expect(voiceLibraryUi.take?.takeId).toBe(imported.takeId);
    });

    it("a recording that cannot start keeps the take that was waiting", async () => {
        invoke.mockImplementation(async (cmd: string) => {
            if (cmd === VL_COMMANDS.recordStart) throw "the microphone could not be opened: no such device";
            return null;
        });
        voiceLibraryUi.take = take("take-00000000001a");
        await expect(lib.startRecording({ device: "USB Mic" })).rejects.toContain("could not be opened");
        expect(invoke.mock.calls.some(([c]) => c === VL_COMMANDS.discardTake)).toBe(false);
        expect(voiceLibraryUi.take?.takeId).toBe("take-00000000001a");
        expect(voiceLibraryUi.recording).toBe(false);
        expect(voiceLibraryUi.error).toContain("could not be opened");
        expect(voiceLibraryUi.busy).toBe(null);
    });

    it("replacing a voice's clip is the tracked save; other edits are not", async () => {
        voiceLibraryUi.voices = [meta("amy-000004", "Amy")];
        voiceLibraryUi.take = take("take-00000000001b");
        let resolve: (v: unknown) => void = () => {};
        invoke.mockImplementationOnce(() => new Promise((r) => (resolve = r)));
        const replacing = lib.updateVoice({ id: "amy-000004", takeId: "take-00000000001b" });
        // Busy at once (before the IPC answers): a second Replace click or Enter finds it busy.
        expect(voiceLibraryUi.busy).toBe("saving");
        resolve({ ...meta("amy-000004", "Amy"), sha256: "cd" });
        await replacing;
        expect(voiceLibraryUi.busy).toBe(null);
        expect(voiceLibraryUi.take).toBe(null);
        expect(voiceLibraryUi.voices[0].sha256).toBe("cd");
        // A failed replace says why in the state, like a failed save.
        voiceLibraryUi.take = take("take-00000000001c");
        invoke.mockRejectedValueOnce("take \"take-00000000001c\" is gone (record or import it again)");
        await expect(lib.updateVoice({ id: "amy-000004", takeId: "take-00000000001c" })).rejects.toContain("gone");
        expect(voiceLibraryUi.error).toContain("gone");
        expect(voiceLibraryUi.busy).toBe(null);
        // Renaming is not the recorder's operation.
        voiceLibraryUi.error = null;
        invoke.mockImplementationOnce(() => new Promise((r) => (resolve = r)));
        const renaming = lib.updateVoice({ id: "amy-000004", name: "Amelia" });
        expect(voiceLibraryUi.busy).toBe(null);
        resolve(meta("amy-000004", "Amelia"));
        await renaming;
        expect(voiceLibraryUi.voices.map((v) => v.name)).toEqual(["Amelia"]);
    });

    it("saving releases the take and lists the voice at once; errors land in the state", async () => {
        voiceLibraryUi.voices = [meta("zed-000003", "Zed")];
        voiceLibraryUi.take = take("take-00000000000e");
        invoke.mockResolvedValueOnce(meta("amy-000004", "amy"));
        const saved = await lib.saveTake({ takeId: "take-00000000000e", name: "amy" });
        expect(invoke).toHaveBeenCalledWith(VL_COMMANDS.saveTake, { request: { takeId: "take-00000000000e", name: "amy" } });
        expect(saved.id).toBe("amy-000004");
        expect(voiceLibraryUi.take).toBe(null);
        expect(voiceLibraryUi.voices.map((v) => v.id)).toEqual(["amy-000004", "zed-000003"]);
        invoke.mockRejectedValueOnce("this take cannot be saved (too_short)");
        await expect(lib.saveTake({ takeId: "take-00000000000f", name: "x" })).rejects.toBe("this take cannot be saved (too_short)");
        expect(voiceLibraryUi.error).toBe("this take cannot be saved (too_short)");
        expect(voiceLibraryUi.busy).toBe(null);
    });

    it("updates and deletes keep the list current", async () => {
        voiceLibraryUi.voices = [meta("amy-000004", "Amy"), meta("zed-000003", "Zed")];
        invoke.mockResolvedValueOnce({ ...meta("amy-000004", "Zoe"), transcript: "hi" });
        await lib.updateVoice({ id: "amy-000004", name: "Zoe", transcript: "hi" });
        expect(voiceLibraryUi.voices.map((v) => v.name)).toEqual(["Zed", "Zoe"]);
        invoke.mockResolvedValueOnce(null);
        await lib.deleteVoice("zed-000003");
        expect(invoke).toHaveBeenLastCalledWith(VL_COMMANDS.delete, { id: "zed-000003" });
        expect(voiceLibraryUi.voices.map((v) => v.id)).toEqual(["amy-000004"]);
    });

    it("reads WAV bytes for previews through binary IPC", async () => {
        const bytes = new ArrayBuffer(8);
        invoke.mockResolvedValue(bytes);
        expect(await lib.readTake("take-000000000001")).toBe(bytes);
        expect(invoke).toHaveBeenLastCalledWith(VL_COMMANDS.readTake, { takeId: "take-000000000001" });
        expect(await lib.readClip("amy-000004")).toBe(bytes);
        expect(invoke).toHaveBeenLastCalledWith(VL_COMMANDS.readClip, { id: "amy-000004" });
    });
});
