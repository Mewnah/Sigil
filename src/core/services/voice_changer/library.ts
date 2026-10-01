/**
 * The voice library of the voice changer (P3-G): the slot store's voices (`voice_changer_list_slots`),
 * `.pth` / `.zip` import with progress and cancel (drag and drop or the file picker), the MIT
 * default voice, rename, duplicate, delete and the weighted merge of two voices (Rust
 * `voice_changer/cmd_import.rs`, P3-B).
 *
 * - Imports run one at a time (Rust runs them one at a time anyway: imports are memory-hungry), so
 *   the `voice_changer:import_progress` events of the job in flight belong to it: its Rust job id is
 *   the first one seen while it runs.
 * - Selecting a voice always goes through the voice-changer service's params sync (the only writer
 *   of the params, see `index.ts`), never `voice_changer_set_active_slot` or an import's `activate`,
 *   which would race a sync in flight.
 * - Errors keep the importer's stable code (`"<code>: <message>"`, `ipc.ts` `importErrorCode`) for a
 *   translated explanation; the message itself is shown too.
 */
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { proxy } from "valtio";

import { devLog } from "@/utils/devLog";

import {
    importErrorCode,
    VC_COMMANDS,
    VC_EVENTS,
    type ImportErrorCode,
    type VcImportProgress,
    type VcImportRequest,
    type VcImportResult,
    type VcImportStage,
    type VcMergeRequest,
    type VcSlotInfo,
} from "./ipc";
import {
    fileNameOf,
    findSlot,
    normalizeSlotInfo,
    normalizeSlotInfos,
    normalizeSlots,
    planImport,
    sameSlotPath,
    slotInfoFromLegacy,
    type ImportPlan,
} from "./slots";

/** What the library needs from the voice-changer service. */
export interface VoiceLibraryHost {
    /** The selected voice folder (`rvc_slot_dir`). */
    selectedVoiceDir(): string;
    /** Select a voice through the params sync (hot switch while RVC runs). */
    selectVoice(voice: { id: string; path: string }): Promise<boolean>;
    /** Clear the voice selection (after its folder was deleted). */
    clearVoice(): Promise<boolean>;
    isRunning(): boolean;
}

export type ImportJobKind = "file" | "default_voice";

/** One import in the library's queue (queued, running or finished). */
export interface VcImportJob {
    /** Local key (stable for the UI). */
    key: string;
    kind: ImportJobKind;
    /** Rust's job id, once its first progress event arrived. */
    jobId: string | null;
    /** The file imported (`""` for the default voice). */
    source: string;
    fileName: string;
    index: string | null;
    stage: VcImportStage;
    fraction: number;
    message: string;
    /** The error text (`"<code>: <message>"`) when the import failed. */
    error: string | null;
    errorCode: ImportErrorCode | null;
    /** The imported (or reused) voice. */
    slot: VcSlotInfo | null;
    cached: boolean;
    warnings: string[];
    /** A cancel was asked for (sent once the Rust job id is known). */
    cancelRequested: boolean;
}

export type SlotOperation = "rename" | "duplicate" | "delete" | "merge";

export interface VoiceLibraryState {
    /** The voices (Rust order: by name); `null` until listed once. */
    slots: VcSlotInfo[] | null;
    listing: boolean;
    listError: string | null;
    /** The backend has the slot store (else only the legacy folder list: no import or management). */
    slotStore: boolean;
    jobs: VcImportJob[];
    /** A rename / duplicate / delete / merge in flight, by slot id. */
    busy: Record<string, SlotOperation>;
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

/** Finished jobs kept in the list (the oldest successes are dropped first). */
const MAX_FINISHED_JOBS = 8;

let jobCounter = 0;

export class VoiceLibrary {
    readonly state = proxy<VoiceLibraryState>({
        slots: null,
        listing: false,
        listError: null,
        slotStore: true,
        jobs: [],
        busy: {},
    });

    private unlisteners: UnlistenFn[] = [];
    private listInflight: Promise<VcSlotInfo[]> | null = null;
    private listAgain = false;
    /** The import queue runs one job at a time. */
    private runner: Promise<void> | null = null;
    private running: VcImportJob | null = null;

    constructor(private readonly host: VoiceLibraryHost) {}

    async init() {
        this.unlisteners.push(
            await listen<VcImportProgress>(VC_EVENTS.importProgress, (ev) => this.onProgress(ev.payload)),
            await listen<unknown>(VC_EVENTS.slotsChanged, () => void this.refresh()),
        );
    }

    dispose() {
        for (const u of this.unlisteners.splice(0)) u();
    }

    /** The listed voice in `dir`, if any. */
    slotFor(dir: string): VcSlotInfo | undefined {
        return this.state.slots ? findSlot(this.state.slots, dir) : undefined;
    }

    slotById(id: string): VcSlotInfo | undefined {
        return this.state.slots?.find((s) => s.id === id);
    }

    /** Re-list the voices (coalesced: a call while listing lists once more afterwards). */
    refresh(): Promise<VcSlotInfo[]> {
        if (this.listInflight) {
            this.listAgain = true;
            return this.listInflight;
        }
        this.listInflight = (async () => {
            let slots: VcSlotInfo[] = this.state.slots ?? [];
            try {
                do {
                    this.listAgain = false;
                    this.state.listing = true;
                    slots = await this.listOnce();
                } while (this.listAgain);
            } finally {
                this.state.listing = false;
                this.listInflight = null;
            }
            return slots;
        })();
        return this.listInflight;
    }

    private async listOnce(): Promise<VcSlotInfo[]> {
        try {
            const slots = normalizeSlotInfos(await invoke<unknown>(VC_COMMANDS.listSlots));
            this.state.slots = slots;
            this.state.slotStore = true;
            this.state.listError = null;
            return slots;
        } catch (e) {
            // An older backend without the slot store: the legacy folder list (no management).
            try {
                const legacy = normalizeSlots(await invoke<unknown>(VC_COMMANDS.listModelSlots)).map(slotInfoFromLegacy);
                this.state.slots = legacy;
                this.state.slotStore = false;
                this.state.listError = null;
                return legacy;
            } catch {
                devLog("[VoiceChanger] Could not list voices:", e);
                this.state.listError = errorText(e);
                if (this.state.slots === null) this.state.slots = [];
                return this.state.slots;
            }
        }
    }

    // ---- import ------------------------------------------------------------------------------------

    /**
     * Queue imports of dropped / picked files (`planImport`: `.pth` / `.zip`, a lone `.index` paired
     * with a lone `.pth`). Returns the plan so the caller can report what was skipped.
     */
    importFiles(paths: readonly string[]): ImportPlan {
        const plan = planImport(paths);
        for (const item of plan.items) this.enqueue(this.newJob("file", item.source, item.index));
        return plan;
    }

    /** Install (one download through the model store) and import the MIT default voice. */
    importDefaultVoice(): VcImportJob {
        const pending = this.state.jobs.find((j) => j.kind === "default_voice" && !isFinished(j.stage));
        if (pending) return pending;
        return this.enqueue(this.newJob("default_voice", "", null));
    }

    /** Cancel a queued or running import. */
    async cancel(key: string) {
        const job = this.jobs(key);
        if (!job || isFinished(job.stage)) return;
        job.cancelRequested = true;
        if (job !== this.running) {
            job.stage = "failed";
            job.error = "job: the import was cancelled";
            job.errorCode = "job";
            job.message = "";
            return;
        }
        if (job.jobId) await this.sendCancel(job.jobId);
    }

    /** Remove a finished import from the list. */
    dismiss(key: string) {
        const i = this.state.jobs.findIndex((j) => j.key === key);
        if (i >= 0 && isFinished(this.state.jobs[i].stage)) this.state.jobs.splice(i, 1);
    }

    /** Remove every finished import from the list. */
    dismissFinished() {
        this.state.jobs = this.state.jobs.filter((j) => !isFinished(j.stage));
    }

    private jobs(key: string): VcImportJob | undefined {
        return this.state.jobs.find((j) => j.key === key);
    }

    private newJob(kind: ImportJobKind, source: string, index: string | null): VcImportJob {
        jobCounter += 1;
        return {
            key: `vcjob-${Date.now().toString(36)}-${jobCounter}`,
            kind,
            jobId: null,
            source,
            fileName: kind === "default_voice" ? "" : fileNameOf(source),
            index,
            stage: "queued",
            fraction: 0,
            message: "",
            error: null,
            errorCode: null,
            slot: null,
            cached: false,
            warnings: [],
            cancelRequested: false,
        };
    }

    /** Queue `job`; returns the queued (proxied) job, which is the one that gets updated. */
    private enqueue(job: VcImportJob): VcImportJob {
        this.state.jobs.push(job);
        this.trimFinished();
        if (!this.runner) this.runner = this.runQueue();
        return this.jobs(job.key) ?? job;
    }

    private trimFinished() {
        const finished = this.state.jobs.filter((j) => isFinished(j.stage));
        let excess = finished.length - MAX_FINISHED_JOBS;
        if (excess <= 0) return;
        // Oldest successes go first; failures stay until dismissed or the list is long.
        for (const stage of ["done", "failed"] as const) {
            for (const j of finished) {
                if (excess <= 0) return;
                if (j.stage !== stage) continue;
                const i = this.state.jobs.indexOf(j);
                if (i >= 0) {
                    this.state.jobs.splice(i, 1);
                    excess--;
                }
            }
        }
    }

    private async runQueue() {
        try {
            for (;;) {
                // The proxied job (the one the UI renders): updates go through it.
                const job = this.state.jobs.find((j) => j.stage === "queued" && !j.cancelRequested);
                if (!job) break;
                this.running = job;
                try {
                    await this.runJob(job);
                } finally {
                    this.running = null;
                }
            }
        } finally {
            this.runner = null;
        }
    }

    private async runJob(job: VcImportJob) {
        job.stage = "reading";
        job.message = "";
        const autoSelect = !this.hasSelectedVoice();
        try {
            let raw: unknown;
            if (job.kind === "default_voice") {
                raw = await invoke<unknown>(VC_COMMANDS.importDefaultVoice);
            } else {
                const request: VcImportRequest = { source: job.source, activate: false };
                if (job.index) request.index = job.index;
                raw = await invoke<unknown>(VC_COMMANDS.importVoice, { request });
            }
            const result = normalizeImportResult(raw);
            if (!result) throw new Error("io: the importer returned no voice");
            job.jobId = job.jobId ?? result.job_id;
            job.slot = result.slot;
            job.cached = result.cached;
            job.warnings = result.warnings;
            job.stage = "done";
            job.fraction = 1;
            job.error = null;
            job.errorCode = null;
            await this.refresh();
            // Nothing selected yet (first voice, or the default voice on first run): use it. The
            // import itself succeeded either way; a failed selection was shown to the user already.
            if (autoSelect && !result.slot.problem) {
                await this.host.selectVoice({ id: result.slot.id, path: result.slot.path }).catch((e: unknown) => {
                    devLog("[VoiceChanger] Could not select the imported voice:", e);
                    return false;
                });
            }
        } catch (e) {
            const text = errorText(e);
            job.stage = "failed";
            job.fraction = 1;
            job.error = text;
            job.errorCode = importErrorCode(text);
            devLog("[VoiceChanger] Import failed:", text);
        }
    }

    /** A voice is selected (a listed one, or a folder chosen directly). */
    private hasSelectedVoice(): boolean {
        return this.host.selectedVoiceDir().trim() !== "";
    }

    private async sendCancel(jobId: string) {
        try {
            await invoke(VC_COMMANDS.cancelImport, { jobId });
        } catch (e) {
            devLog("[VoiceChanger] Could not cancel the import:", e);
        }
    }

    private onProgress(p: VcImportProgress) {
        if (!p || typeof p !== "object" || typeof p.job_id !== "string") return;
        const job = this.running;
        if (!job) return;
        if (job.jobId === null) {
            job.jobId = p.job_id;
            if (job.cancelRequested) void this.sendCancel(p.job_id);
        } else if (job.jobId !== p.job_id) {
            return;
        }
        // The terminal stages come with the command's reply (and its error text).
        if (p.stage === "done" || p.stage === "failed") return;
        job.stage = p.stage;
        if (typeof p.fraction === "number" && Number.isFinite(p.fraction)) job.fraction = Math.min(1, Math.max(0, p.fraction));
        job.message = typeof p.message === "string" ? p.message : "";
    }

    // ---- slot operations ------------------------------------------------------------------------------

    private async withBusy<T>(id: string, op: SlotOperation, run: () => Promise<T>): Promise<T> {
        this.state.busy[id] = op;
        try {
            return await run();
        } finally {
            delete this.state.busy[id];
        }
    }

    /** Rename a voice (its display name; the folder and selection stay). */
    async rename(id: string, name: string): Promise<VcSlotInfo | null> {
        return this.withBusy(id, "rename", async () => {
            const info = normalizeSlotInfo(await invoke<unknown>(VC_COMMANDS.renameSlot, { id, name }));
            await this.refresh();
            return info;
        });
    }

    /** Copy a voice into a new slot (default name: "<name> (copy)"). */
    async duplicate(id: string, name?: string): Promise<VcSlotInfo | null> {
        return this.withBusy(id, "duplicate", async () => {
            const args: { id: string; name?: string } = { id };
            if (name && name.trim()) args.name = name.trim();
            const info = normalizeSlotInfo(await invoke<unknown>(VC_COMMANDS.duplicateSlot, args));
            await this.refresh();
            return info;
        });
    }

    /**
     * Delete a voice. Rust refuses the voice the voice changer is running on; deleting the selected
     * (stopped) voice clears the selection (through the params sync, after Rust's own clear).
     */
    async remove(id: string): Promise<void> {
        const slot = this.slotById(id);
        const wasSelected = !!slot && sameSlotPath(slot.path, this.host.selectedVoiceDir());
        await this.withBusy(id, "delete", async () => {
            await invoke(VC_COMMANDS.deleteSlot, { id });
            if (wasSelected) await this.host.clearVoice();
            await this.refresh();
        });
    }

    /** Weighted merge of two voices of one template into a new voice. */
    async merge(request: VcMergeRequest): Promise<VcSlotInfo | null> {
        return this.withBusy(request.a, "merge", async () => {
            const info = normalizeSlotInfo(await invoke<unknown>(VC_COMMANDS.mergeSlots, { request }));
            await this.refresh();
            return info;
        });
    }
}

export function isFinished(stage: VcImportStage): boolean {
    return stage === "done" || stage === "failed";
}

/** An import reply (`VcImportResult`), or `null` when it has no voice. */
export function normalizeImportResult(raw: unknown): VcImportResult | null {
    if (!raw || typeof raw !== "object") return null;
    const r = raw as Record<string, unknown>;
    const slot = normalizeSlotInfo(r.slot);
    if (!slot) return null;
    return {
        job_id: typeof r.job_id === "string" ? r.job_id : "",
        slot,
        cached: r.cached === true,
        warnings: Array.isArray(r.warnings) ? r.warnings.filter((w): w is string => typeof w === "string") : [],
        elapsed_ms: typeof r.elapsed_ms === "number" && Number.isFinite(r.elapsed_ms) ? r.elapsed_ms : 0,
    };
}
