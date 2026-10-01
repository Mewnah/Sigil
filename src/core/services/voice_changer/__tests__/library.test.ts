/**
 * The voice library (`library.ts`) against a mocked slot store: listing (with the legacy fallback),
 * the one-at-a-time import queue and its progress events, cancel, auto-selecting the first voice, and
 * the slot operations (the selection always goes through the host, never Rust's own activate).
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

type Handler = (ev: { payload: unknown }) => void;
const handlers = new Map<string, Handler>();
const invoke = vi.fn();

vi.mock("@tauri-apps/api/core", () => ({ invoke: (...args: unknown[]) => invoke(...args) }));
vi.mock("@tauri-apps/api/event", () => ({
    listen: async (name: string, cb: Handler) => {
        handlers.set(name, cb);
        return () => handlers.delete(name);
    },
}));
vi.mock("@/utils/devLog", () => ({ devLog: () => {} }));

import { VC_COMMANDS, VC_EVENTS } from "../ipc";
import { VoiceLibrary, type VoiceLibraryHost } from "../library";

const emit = (name: string, payload: unknown) => handlers.get(name)?.({ payload });
const row = (id: string, over: Record<string, unknown> = {}) => ({ id, name: id.toUpperCase(), path: `C:/m/${id}`, format: "sigil-rvc-1", ...over });

let slots: ReturnType<typeof row>[];
let selected: string;
let host: VoiceLibraryHost & { selectVoice: ReturnType<typeof vi.fn>; clearVoice: ReturnType<typeof vi.fn> };

/** Deferred import replies, in call order. */
let imports: { request: unknown; resolve: (v: unknown) => void; reject: (e: unknown) => void }[];

beforeEach(() => {
    handlers.clear();
    invoke.mockReset();
    slots = [];
    selected = "";
    imports = [];
    host = {
        selectedVoiceDir: () => selected,
        selectVoice: vi.fn(async (v: { path: string }) => {
            selected = v.path;
            return true;
        }),
        clearVoice: vi.fn(async () => {
            selected = "";
            return true;
        }),
        isRunning: () => false,
    };
    invoke.mockImplementation(async (cmd: string, args?: Record<string, unknown>) => {
        switch (cmd) {
            case VC_COMMANDS.listSlots:
                return structuredClone(slots);
            case VC_COMMANDS.importVoice:
            case VC_COMMANDS.importDefaultVoice:
                return new Promise((resolve, reject) => imports.push({ request: args?.request ?? "default", resolve, reject }));
            case VC_COMMANDS.cancelImport:
                return null;
            case VC_COMMANDS.renameSlot:
                slots = slots.map((s) => (s.id === args!.id ? { ...s, name: String(args!.name) } : s));
                return slots.find((s) => s.id === args!.id);
            case VC_COMMANDS.duplicateSlot: {
                const copy = row(`${args!.id}-2`, { name: "copy" });
                slots.push(copy);
                return copy;
            }
            case VC_COMMANDS.deleteSlot:
                slots = slots.filter((s) => s.id !== args!.id);
                return null;
            case VC_COMMANDS.mergeSlots: {
                const m = row("merged");
                slots.push(m);
                return m;
            }
            default:
                throw new Error(`unexpected ${cmd}`);
        }
    });
});

async function library() {
    const lib = new VoiceLibrary(host);
    await lib.init();
    return lib;
}

const flush = () => new Promise((r) => setTimeout(r, 0));

describe("voice library", () => {
    it("lists the slot store, and falls back to the legacy folder list without it", async () => {
        slots = [row("alto"), row("bass")];
        const lib = await library();
        expect((await lib.refresh()).map((s) => s.id)).toEqual(["alto", "bass"]);
        expect(lib.state.slotStore).toBe(true);
        invoke.mockImplementation(async (cmd: string) => {
            if (cmd === VC_COMMANDS.listSlots) throw new Error("not allowed");
            if (cmd === VC_COMMANDS.listModelSlots) return [{ id: "old", path: "C:/m/old", has_manifest: false }];
            throw new Error(cmd);
        });
        expect((await lib.refresh()).map((s) => [s.id, s.format])).toEqual([["old", "rvc"]]);
        expect(lib.state.slotStore).toBe(false);
        // The slots-changed event re-lists.
        invoke.mockImplementation(async () => [row("new")]);
        emit(VC_EVENTS.slotsChanged, null);
        await flush();
        expect(lib.state.slots?.map((s) => s.id)).toEqual(["new"]);
    });

    it("imports one file at a time, tracks the running job's events, and selects the first voice", async () => {
        const lib = await library();
        const plan = lib.importFiles(["C:/v/a.pth", "C:/v/a.index", "C:/v/b.zip", "C:/v/c.png"]);
        expect(plan.ignored).toEqual(["C:/v/c.png"]);
        await flush();
        expect(imports).toHaveLength(1);
        expect(imports[0].request).toEqual({ source: "C:/v/a.pth", activate: false, index: "C:/v/a.index" });
        expect(lib.state.jobs.map((j) => j.stage)).toEqual(["reading", "queued"]);

        emit(VC_EVENTS.importProgress, { job_id: "j1", stage: "converting", fraction: 0.4, message: "Converting" });
        emit(VC_EVENTS.importProgress, { job_id: "other", stage: "writing", fraction: 0.9, message: "x" });
        expect(lib.state.jobs[0]).toMatchObject({ jobId: "j1", stage: "converting", fraction: 0.4, message: "Converting" });

        slots = [row("a")];
        imports[0].resolve({ job_id: "j1", slot: row("a"), cached: false, warnings: ["w"], elapsed_ms: 900 });
        await flush();
        await flush();
        expect(lib.state.jobs[0]).toMatchObject({ stage: "done", warnings: ["w"] });
        expect(host.selectVoice).toHaveBeenCalledWith({ id: "a", path: "C:/m/a" });
        // The next file starts only now; with a voice selected it is not selected automatically.
        expect(imports).toHaveLength(2);
        imports[1].reject("pickle_rejected: global os.system is not allowed");
        await flush();
        expect(lib.state.jobs[1]).toMatchObject({ stage: "failed", errorCode: "pickle_rejected" });
        expect(host.selectVoice).toHaveBeenCalledTimes(1);
        lib.dismissFinished();
        expect(lib.state.jobs).toEqual([]);
    });

    it("an import stays done when selecting it afterwards fails (that failure was shown already)", async () => {
        host.selectVoice.mockRejectedValueOnce("start failed: missing base model");
        const lib = await library();
        lib.importFiles(["C:/v/a.pth"]);
        await flush();
        slots = [row("a")];
        imports[0].resolve({ job_id: "j1", slot: row("a"), cached: false, warnings: [], elapsed_ms: 900 });
        await flush();
        await flush();
        expect(host.selectVoice).toHaveBeenCalledTimes(1);
        expect(lib.state.jobs[0]).toMatchObject({ stage: "done", error: null, slot: { id: "a" } });
    });

    it("cancels a queued job locally and a running one once its Rust job id is known", async () => {
        const lib = await library();
        lib.importFiles(["C:/v/a.pth", "C:/v/b.pth"]);
        await flush();
        const [running, queued] = lib.state.jobs;
        await lib.cancel(queued.key);
        expect(lib.state.jobs[1]).toMatchObject({ stage: "failed", errorCode: "job" });
        await lib.cancel(running.key);
        expect(invoke).not.toHaveBeenCalledWith(VC_COMMANDS.cancelImport, expect.anything());
        emit(VC_EVENTS.importProgress, { job_id: "j9", stage: "reading", fraction: 0.05, message: "" });
        await flush();
        expect(invoke).toHaveBeenCalledWith(VC_COMMANDS.cancelImport, { jobId: "j9" });
        imports[0].reject("job: the import was cancelled");
        await flush();
        expect(imports).toHaveLength(1);
    });

    it("the default voice is one job at a time", async () => {
        const lib = await library();
        const a = lib.importDefaultVoice();
        const b = lib.importDefaultVoice();
        expect(a.key).toBe(b.key);
        await flush();
        expect(imports[0].request).toBe("default");
    });

    it("renames, duplicates, merges and deletes; deleting the selected voice clears the selection", async () => {
        slots = [row("alto"), row("bass")];
        selected = "C:/m/alto";
        const lib = await library();
        await lib.refresh();
        expect((await lib.rename("alto", "Alto 2"))?.name).toBe("Alto 2");
        expect(invoke).toHaveBeenCalledWith(VC_COMMANDS.renameSlot, { id: "alto", name: "Alto 2" });
        await lib.duplicate("bass", "  ");
        expect(invoke).toHaveBeenCalledWith(VC_COMMANDS.duplicateSlot, { id: "bass" });
        await lib.merge({ a: "alto", b: "bass", weight_a: 0.3, name: "Mix" });
        expect(invoke).toHaveBeenCalledWith(VC_COMMANDS.mergeSlots, { request: { a: "alto", b: "bass", weight_a: 0.3, name: "Mix" } });
        await lib.remove("bass");
        expect(host.clearVoice).not.toHaveBeenCalled();
        await lib.remove("alto");
        expect(host.clearVoice).toHaveBeenCalledTimes(1);
        expect(lib.state.busy).toEqual({});
        expect(lib.state.slots?.map((s) => s.id)).toEqual(["bass-2", "merged"]);
    });
});
