import type { SttStopOptions } from "./types";

/** Shared start/stop bookkeeping for Rust-backed STT engines (Whisper, Vosk, Moonshine, local HTTP STT). */
export class SttStartLifecycle {
    starting = false;
    startAborted = false;
    private stopNotified = false;

    constructor(private readonly onStop: (error?: string, options?: SttStopOptions) => void) {}

    resetForStart(): void {
        this.startAborted = false;
        this.stopNotified = false;
    }

    beginStart(): void {
        this.starting = true;
        this.resetForStart();
    }

    finishStart(): void {
        this.starting = false;
    }

    abort(): void {
        this.startAborted = true;
    }

    get aborted(): boolean {
        return this.startAborted;
    }

    notifyStopOnce(error?: string, options?: SttStopOptions): void {
        if (this.stopNotified) return;
        this.stopNotified = true;
        if (options) this.onStop(error, options);
        else this.onStop(error);
    }

    /** When true, caller should tear down and return (optionally notifying stop). */
    checkAborted(notify = true): boolean {
        if (!this.startAborted) return false;
        if (notify) this.notifyStopOnce();
        return true;
    }
}
