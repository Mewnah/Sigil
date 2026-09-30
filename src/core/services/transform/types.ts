import { Transform_State } from "./schema";

export type ChatTurn = { role: "user" | "assistant"; content: string };

export interface TransformRequest {
    /** Source sentence, non-empty. */
    text: string;
    /** Earlier (source, result) exchanges, oldest first; empty when context history is off. */
    history: ChatTurn[];
    /**
     * Streaming progress: the sanitized reply so far (reasoning hidden). Engines that stream call it in order and
     * never after `signal` aborted; the resolved value is the final text.
     */
    onPartial?: (text: string) => void;
}

/**
 * One transform backend. Engines never publish: the service orders their results (see ordering.ts).
 *
 * - `start` validates settings; it rejects with a user-facing message (`TransformNotConfiguredError`
 *   when a required field is simply empty, which an automatic start at launch reports silently).
 * - `transform` resolves with the rewritten text or throws (`TransformRequestError` with a kind the
 *   service turns into the right notice). It must honour `signal`, which is aborted on timeout and stop.
 * - Settings are read through `getData()` on every request, so inspector edits apply to the next line.
 */
export interface ITransformEngine {
    start(getData: () => Transform_State, signal: AbortSignal): Promise<void>;
    transform(request: TransformRequest, signal: AbortSignal): Promise<string>;
    stop(): void;
}

export interface ITransformEngineConstructor {
    new(): ITransformEngine;
}

/** A required setting (key, URL, model) is empty. */
export class TransformNotConfiguredError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "TransformNotConfiguredError";
    }
}

/** `unauthorized`: the provider rejected the key (401/403); `network`: the server could not be reached. */
export type TransformErrorKind = "rate_limited" | "truncated" | "empty" | "unauthorized" | "network" | "request";

export class TransformRequestError extends Error {
    constructor(
        readonly kind: TransformErrorKind,
        message: string,
        readonly status?: number,
    ) {
        super(message);
        this.name = "TransformRequestError";
    }
}
