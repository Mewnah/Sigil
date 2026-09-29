import { raceWithTimeoutAndAbort } from "./abort";

/** Loaded model from `vosk-browser` `createModel()` — `KaldiRecognizer` is a getter, not a module export. */
export type VoskLoadedModel = {
    KaldiRecognizer: new (sampleRate: number, grammar?: string) => VoskRecognizer;
    terminate(): void;
};

export type VoskRecognizer = {
    remove(): void;
    acceptWaveformFloat(buffer: Float32Array, sampleRate: number): void;
    on(event: string, cb: (msg: unknown) => void): void;
};

const VOSK_CREATE_MODEL_TIMEOUT_MS = 120_000;

export async function loadVoskModel(
    modelUrl: string,
    signal: AbortSignal | null,
): Promise<VoskLoadedModel> {
    let createModel: (url: string) => Promise<unknown>;
    try {
        ({ createModel } = await import("vosk-browser"));
    } catch (importErr) {
        throw new Error(
            `Failed to load Vosk WASM module (release build). ${importErr instanceof Error ? importErr.message : String(importErr)}`,
        );
    }

    return (await raceWithTimeoutAndAbort(
        createModel(modelUrl) as Promise<VoskLoadedModel>,
        VOSK_CREATE_MODEL_TIMEOUT_MS,
        "Vosk model load timed out",
        signal,
    )) as VoskLoadedModel;
}
