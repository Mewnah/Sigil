import { ErrorThrottle } from "@/core/services/tts/errorThrottle";
import { PROCESS_ID_BASE } from "@/utils/processNonce";
import {
    IServiceInterface,
    ServiceNetworkState,
    TextEvent,
    TextEventSource,
    TextEventType,
} from "@/types";
import i18n from "i18next";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import { i18nReady } from "../stt/statusMessages";
import { pushSystemLog } from "../systemLog";
import { raceWithSignal } from "../translation/abort";
import { isCancelMarker } from "../translation/sequencer";
import { OrderedQueue } from "./ordering";
import {
    openRouterModelFor,
    Transform_Backends,
    TRANSFORM_TIMEOUT_MS_DEFAULT,
    TRANSFORM_TIMEOUT_MS_MAX,
    TRANSFORM_TIMEOUT_MS_MIN,
} from "./schema";
import { Transform_DummyService } from "./services/dummy";
import { Transform_LLMService } from "./services/llm";
import { PartialThrottle } from "./streaming";
import {
    ChatTurn,
    ITransformEngine,
    ITransformEngineConstructor,
    TransformNotConfiguredError,
    TransformRequestError,
} from "./types";

export type TransformEngineMap = { [k in Transform_Backends]: ITransformEngineConstructor };

export const TRANSFORM_ENGINES: TransformEngineMap = {
    [Transform_Backends.dummy]: Transform_DummyService,
    [Transform_Backends.openai]: Transform_LLMService,
};

/** Default per-line deadline (setting `timeoutMs`); the request is aborted (not just abandoned) when it expires. */
export const TRANSFORM_TIMEOUT_MS = TRANSFORM_TIMEOUT_MS_DEFAULT;
/**
 * At most this often a streamed partial reply is published to text.transform_source (and only for the newest line,
 * until the next sentence starts).
 */
export const TRANSFORM_PARTIAL_INTERVAL_MS = 100;
/** Context sent with a line when context history is on (3 exchanges). */
const HISTORY_TURNS = 6;
const HISTORY_CAP = 20;

export class TransformTimeoutError extends Error {
    constructor(readonly timeoutMs: number) {
        super(`timed out after ${timeoutMs / 1000} s`);
        this.name = "TransformTimeoutError";
    }
}

/** Abort reason for requests cut off by stop/restart: their lines are passed through quietly. */
class TransformStopped extends Error {
    constructor() {
        super("stopped");
        this.name = "AbortError";
    }
}

/**
 * How one STT final ends up, published in sentence order:
 * - `ai`: the AI reply goes to text.transform and text.transform_source; the original to transform_raw.
 * - `fallback`: the AI failed while running; the original text goes everywhere, so consumers keep
 *   flowing (a notice explains why) instead of stalling on a "..." placeholder.
 * - `passthrough`: AI is off, was stopped mid-line, or the line is STT's cancel marker; only the
 *   synced topics (transform_source, transform_raw) get the original, to close their partials.
 */
type Outcome =
    | { kind: "ai"; event: TextEvent; value: string }
    | { kind: "fallback"; event: TextEvent }
    | { kind: "passthrough"; event: TextEvent };

function errorText(e: unknown): string {
    return e instanceof Error ? e.message : String(e);
}

/** The configured per-line deadline, clamped (a hand-edited settings file cannot disable it). */
function lineTimeoutMs(value: unknown): number {
    const n = typeof value === "number" ? value : Number(value);
    if (!Number.isFinite(n)) return TRANSFORM_TIMEOUT_MS;
    return Math.min(TRANSFORM_TIMEOUT_MS_MAX, Math.max(TRANSFORM_TIMEOUT_MS_MIN, Math.round(n)));
}

/**
 * A streamed partial reply for the synced captions. It pairs with its line by `id` only: the STT segment fields
 * (`segmentId`, `revision`, `stableLength`) describe the spoken text, and repeating the final's revision on every
 * partial could make a revision-aware caption drop the final reply as "not newer".
 */
function partialEvent(event: TextEvent, value: string): TextEvent {
    const partial: TextEvent = { ...event, type: TextEventType.interim, value };
    delete partial.segmentId;
    delete partial.revision;
    delete partial.stableLength;
    return partial;
}

/** Never empty, so a failed start that is not silent is always explained. */
function startErrorText(e: unknown): string {
    const detail = errorText(e).trim() || i18n.t("common.error_no_details") || "no details";
    return i18n.t("transform.error_start_failed", { error: detail }) || `AI Transform could not start: ${detail}`;
}

class Service_Transform implements IServiceInterface {
    serviceState = proxy({
        status: ServiceNetworkState.disconnected,
        error: "",
    });
    private eventDisposers: (() => void)[] = [];

    #initialized = false;
    #engine: ITransformEngine | null = null;
    /** Bumped on every start and stop; async work of an older session must not touch state. */
    #epoch = 0;
    #startController: AbortController | null = null;
    #inFlight = new Set<AbortController>();
    /**
     * Id of the sentence STT is currently speaking; attached to every transform_source/raw event. Starts at a random
     * per-process base so a restarted or linked host's line ids never pair with this host's lines.
     */
    #sentenceId = PROCESS_ID_BASE;
    /**
     * Id of the only line whose streamed partial replies may still go to text.transform_source: the newest line,
     * until STT starts the next sentence. Partials share that topic with the next sentence's STT interims, and
     * consumers that ignore `id` (OBS caption feed, OBS stream captions) show whichever interim came last, so both
     * streams at once would make their caption flip between two sentences. null = none.
     */
    #streamingLine: number | null = null;
    /** Context for the AI: (source, reply) pairs in sentence order. */
    #history: ChatTurn[] = [];
    #errors = new ErrorThrottle();
    #queue = new OrderedQueue<Outcome>((o) => this.#deliver(o));

    constructor(private readonly engines: TransformEngineMap = TRANSFORM_ENGINES) {}

    get data() {
        return window.ApiServer.state.services.transform.data;
    }

    /** Lines waiting for an AI reply or for an earlier line (diagnostics and tests). */
    get pendingLines(): number {
        return this.#queue.pending;
    }

    init(): void {
        if (this.#initialized) return;
        this.#initialized = true;

        const subId = window.ApiShared.pubsub.subscribeText(
            TextEventSource.stt,
            (e) => e && this.#onStt(e)
        );
        this.eventDisposers.push(() => window.ApiShared.pubsub.unsubscribe(subId));

        const streamSub = window.ApiShared.pubsub.subscribe("stream.on_ended", () => {
            if (this.data.stopWithStream && this.serviceState.status === ServiceNetworkState.connected)
                this.stop();
        });
        this.eventDisposers.push(() => window.ApiShared.pubsub.unsubscribe(streamSub));

        this.#migrateSettings();

        if (this.data.autoStart) {
            // Host services init before i18next, whose t() returns nothing until then; a start error
            // worded that early would be empty. A start or stop by the user in the meantime wins.
            const epoch = this.#epoch;
            void i18nReady().then(() => {
                if (epoch === this.#epoch && this.#initialized && this.data.autoStart) this.start({ auto: true });
            });
        }
    }

    /**
     * One-time fixes of stored values. The whole state is saved with its defaults, so a changed
     * default never reaches existing installs: the retired OpenRouter default is rewritten here (the
     * request path maps it as well, see resolveProvider) so the inspector shows what is sent.
     */
    #migrateSettings() {
        const stored = this.data.openRouterModel;
        const current = openRouterModelFor(stored);
        if (current !== stored.trim()) {
            window.ApiServer.patchService("transform", (s) => {
                s.data.openRouterModel = current;
            });
        }
    }

    /**
     * @param options.auto true for the launch-time auto start, which stays silent when the provider is
     *   simply not set up yet; a start the user asked for explains what is missing.
     */
    start(options: { auto?: boolean } = {}) {
        this.#teardown();
        const epoch = ++this.#epoch;
        this.serviceState.error = "";
        this.#errors.reset();
        this.#history = [];

        const Engine = this.engines[this.data.backend];
        if (!Engine) {
            this.#setStatus(ServiceNetworkState.disconnected);
            return;
        }
        const engine = new Engine();
        const controller = new AbortController();
        this.#engine = engine;
        this.#startController = controller;
        this.#setStatus(ServiceNetworkState.connecting);

        let starting: Promise<void>;
        try {
            starting = Promise.resolve(engine.start(() => this.data, controller.signal));
        } catch (e) {
            starting = Promise.reject(e);
        }
        void (async () => {
            try {
                await raceWithSignal(starting, controller.signal);
            } catch (e) {
                if (epoch !== this.#epoch) return;
                const silent = options.auto === true && e instanceof TransformNotConfiguredError;
                this.#failStart(silent ? "" : startErrorText(e));
                return;
            }
            if (epoch !== this.#epoch) return;
            this.#startController = null;
            this.#setStatus(ServiceNetworkState.connected);
        })();
    }

    stop() {
        this.#epoch++;
        this.#teardown();
        this.#setStatus(ServiceNetworkState.disconnected);
    }

    dispose() {
        this.stop();
        this.eventDisposers.forEach(d => d());
        this.eventDisposers = [];
        this.#initialized = false;
    }

    /** Aborts a pending start and every in-flight line (they are passed through), stops the engine. */
    #teardown() {
        this.#startController?.abort();
        this.#startController = null;
        this.#streamingLine = null;
        const stopped = new TransformStopped();
        for (const c of this.#inFlight) c.abort(stopped);
        this.#inFlight.clear();
        const engine = this.#engine;
        this.#engine = null;
        try {
            engine?.stop();
        } catch (e) {
            console.error("[Transform] engine stop failed:", e);
        }
    }

    #failStart(message: string) {
        this.#teardown();
        this.#setStatus(ServiceNetworkState.disconnected);
        if (!message) return;
        this.serviceState.error = message;
        pushSystemLog("AI Transform", message, "error");
        toast(message, { type: "error", autoClose: false });
    }

    #setStatus(value: ServiceNetworkState) {
        this.serviceState.status = value;
    }

    #aiActive(): boolean {
        return this.#engine !== null && this.serviceState.status === ServiceNetworkState.connected;
    }

    #onStt(e: TextEvent) {
        if (e.type !== TextEventType.final) {
            // Live partials of the sentence being spoken. With AI off, "Sync timing" captions
            // (transform_raw) simply follow STT.
            const interim = { ...e, id: this.#sentenceId };
            // The next sentence has started: an earlier line's reply now arrives only as its final.
            this.#streamingLine = null;
            window.ApiShared.pubsub.publishText(TextEventSource.transform_source, interim);
            if (!this.#aiActive())
                window.ApiShared.pubsub.publishText(TextEventSource.transform_raw, interim);
            return;
        }

        const id = this.#sentenceId++;
        const event: TextEvent = { ...e, id };
        const seq = this.#queue.reserve();
        if (!this.#aiActive() || isCancelMarker(event.value) || !event.value.trim()) {
            this.#streamingLine = null;
            this.#queue.resolve(seq, { kind: "passthrough", event });
            return;
        }
        this.#streamingLine = id;

        // Show the original at once while the AI works; the reply replaces it (same id).
        window.ApiShared.pubsub.publishText(TextEventSource.transform_source, { ...event, type: TextEventType.interim });
        window.ApiShared.pubsub.publishText(TextEventSource.transform_raw, {
            ...event,
            type: TextEventType.interim,
            value: event.value + " ...",
        });
        const history = this.data.contextHistory ? this.#history.slice(-HISTORY_TURNS) : [];
        void this.#runLine(this.#engine!, seq, event, history);
    }

    async #runLine(engine: ITransformEngine, seq: number, event: TextEvent, history: ChatTurn[]) {
        const controller = new AbortController();
        this.#inFlight.add(controller);
        const timeoutMs = lineTimeoutMs(this.data.timeoutMs);
        const timer = setTimeout(() => controller.abort(new TransformTimeoutError(timeoutMs)), timeoutMs);
        // Streamed replies replace the original on the synced captions (same id) as they arrive, while this is the
        // newest line and the next sentence has not started (see #streamingLine); the final reply is still delivered
        // in sentence order below. TTS, chat and text.transform only ever get the final.
        const partials: PartialThrottle = new PartialThrottle((value) => {
            if (controller.signal.aborted) return;
            if (this.#streamingLine !== event.id) {
                partials.close();
                return;
            }
            window.ApiShared.pubsub.publishText(TextEventSource.transform_source, partialEvent(event, value));
        }, TRANSFORM_PARTIAL_INTERVAL_MS);
        let outcome: Outcome;
        try {
            const value = await raceWithSignal(
                engine.transform({ text: event.value, history, onPartial: (text) => partials.push(text) }, controller.signal),
                controller.signal,
            );
            outcome = { kind: "ai", event, value };
        } catch (error) {
            if (controller.signal.reason instanceof TransformStopped) {
                outcome = { kind: "passthrough", event };
            } else {
                this.#reportError(controller.signal.reason instanceof TransformTimeoutError ? controller.signal.reason : error);
                outcome = { kind: "fallback", event };
            }
        } finally {
            partials.close();
            clearTimeout(timer);
            this.#inFlight.delete(controller);
        }
        this.#queue.resolve(seq, outcome);
    }

    #deliver(o: Outcome) {
        const pubsub = window.ApiShared.pubsub;
        const original = o.event;
        if (o.kind !== "passthrough") {
            pubsub.publishText(TextEventSource.transform, {
                value: o.kind === "ai" ? o.value : original.value,
                type: TextEventType.final,
            });
        }
        pubsub.publishText(TextEventSource.transform_source, {
            ...original,
            value: o.kind === "ai" ? o.value : original.value,
            type: TextEventType.final,
        });
        // Original text, finalized exactly when the AI line appears ("Sync timing" captions).
        pubsub.publishText(TextEventSource.transform_raw, { ...original, type: TextEventType.final });

        if (o.kind === "ai") {
            this.#history.push({ role: "user", content: original.value }, { role: "assistant", content: o.value });
            if (this.#history.length > HISTORY_CAP) this.#history = this.#history.slice(-HISTORY_CAP);
        }
    }

    #reportError(error: unknown) {
        let message: string;
        if (error instanceof TransformTimeoutError) {
            message = i18n.t("transform.error_timeout", { seconds: error.timeoutMs / 1000 });
        } else if (error instanceof TransformRequestError) {
            switch (error.kind) {
                case "rate_limited":
                    message = i18n.t("transform.error_rate_limited");
                    break;
                case "truncated":
                    message = i18n.t("transform.error_truncated");
                    break;
                case "empty":
                    message = i18n.t("transform.error_empty");
                    break;
                case "unauthorized":
                    message = i18n.t("transform.error_unauthorized", { error: error.message });
                    break;
                case "network":
                    message = i18n.t("transform.error_network", { error: error.message });
                    break;
                default:
                    message = i18n.t("transform.error_request", {
                        status: error.status ?? "-",
                        error: error.message,
                    });
            }
        } else {
            // TransformNotConfiguredError (settings cleared while running) and anything unexpected.
            message = i18n.t("transform.error_request", { status: "-", error: errorText(error) });
        }
        console.warn("[Transform] line failed, original text sent instead:", error);
        if (this.#errors.shouldShow(message)) {
            pushSystemLog("AI Transform", message, "warning");
            toast(message, { type: "warning", autoClose: 8_000 });
        }
    }
}

export default Service_Transform;
