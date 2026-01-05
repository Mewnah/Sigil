// Type declarations for vosk-browser
declare module "vosk-browser" {
    export function createModel(modelUrl: string): Promise<Model>;

    export interface Model {
        terminate(): void;
    }

    export class KaldiRecognizer {
        constructor(model: Model, sampleRate: number);
        on(event: "result", callback: (message: { result?: { text: string } }) => void): void;
        on(event: "partialresult", callback: (message: { result?: { partial: string } }) => void): void;
        acceptWaveform(data: Int16Array | Float32Array): void;
        remove(): void;
    }
}
