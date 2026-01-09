import {
    IServiceInterface,
    ServiceNetworkState,
    TextEvent,
    TextEventType,
    TextEventSource,
} from "@/types";
import { serviceSubscibeToInput } from "@/utils";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import { Transform_Backends } from "./schema";
import { Transform_LLMService } from "./services/llm";
import { Transform_DummyService } from "./services/dummy";
import {
    ITransformReceiver,
    ITransformService,
    ITransformServiceConstructor,
} from "./types";

const backends: {
    [k in Transform_Backends]: ITransformServiceConstructor;
} = {
    [Transform_Backends.dummy]: Transform_DummyService,
    [Transform_Backends.openai]: Transform_LLMService,
};

class Service_Transform implements IServiceInterface, ITransformReceiver {
    serviceState = proxy({
        status: ServiceNetworkState.disconnected,
        error: "",
    });
    private eventDisposers: (() => void)[] = [];
    #id = 0; // Internal ID for AI requests (used by LLM service)
    #currentSentenceId = 0; // Public ID for UI text tracking

    #serviceInstance!: ITransformService;
    get data() {
        return window.ApiServer.state.services.transform.data;
    }

    #initialized = false;
    // History buffer for Context Awareness (Last 5 exchanges)
    #history: { role: "user" | "assistant", content: string }[] = [];

    init(): void {
        if (this.#initialized) return;
        this.#initialized = true;

        if (this.data.autoStart)
            this.start();

        const subId = window.ApiShared.pubsub.subscribeText(
            TextEventSource.stt,
            (e) => {
                if (e) {
                    if (e.type === TextEventType.final) {
                        // Attach ID to final event
                        const eventWithId = { ...e, id: this.#currentSentenceId };

                        // ECHO AS INTERIM FIRST so UI sees original text immediately while AI processes
                        // This fixes the "Blank screen while waiting for AI" issue
                        window.ApiShared.pubsub.publishText(TextEventSource.transform_source, {
                            ...eventWithId,
                            type: TextEventType.interim // Masquerade as interim so UI accepts it (or final, but interim allows update later)
                        });

                        // Publish "THINKING" Indicator to Synced Raw
                        window.ApiShared.pubsub.publishText(TextEventSource.transform_raw, {
                            ...eventWithId,
                            type: TextEventType.interim,
                            value: eventWithId.value + " ..."
                        });

                        // Prepare Context History for LLM (if enabled)
                        const history = this.data.contextHistory ? this.#history.slice(-6) : [];

                        this.transform(eventWithId, history);

                        // Increment ID for NEXT sentence
                        this.#currentSentenceId++;
                    } else {
                        // Pass interim events directly to transform_source for instant feedback
                        // Attach current ID
                        window.ApiShared.pubsub.publishText(TextEventSource.transform_source, { ...e, id: this.#currentSentenceId });
                    }
                }
            }
        );
        this.eventDisposers.push(() => window.ApiShared.pubsub.unsubscribe(subId));
    }

    #setStatus(value: ServiceNetworkState) {
        this.serviceState.status = value;
    }

    // #region ITransformReceiver
    onStart(): void {
        this.#setStatus(ServiceNetworkState.connected);
    }
    onStop(error: string): void {
        if (error) {
            toast(error, { type: "error", autoClose: false });
            this.serviceState.error = error;
        }
        this.#setStatus(ServiceNetworkState.disconnected);
    }

    onTransform(id: number, e: TextEvent, value: string): void {
        window.ApiShared.pubsub.publishText(TextEventSource.transform, {
            value,
            type: e.type,
        });
        // Publish the AI RESULT to transform_source (replacing the original)
        // Use the ID from the original event (e.id) which we attached in init()
        window.ApiShared.pubsub.publishText(TextEventSource.transform_source, {
            ...e,
            value: value,
            id: e.id // Ensure we target the same sentence
        });

        // Publish ORIGINAL RAW text to transform_raw (Synced Timing)
        // This allows a secondary element to show the raw text EXACTLY when the AI text updates
        window.ApiShared.pubsub.publishText(TextEventSource.transform_raw, {
            ...e,
            type: TextEventType.final, // Force final as this is the committed raw text
            value: e.value, // Original Raw Value
            id: e.id
        });

        // Update History Buffer
        this.#history.push({ role: "user", content: e.value });
        this.#history.push({ role: "assistant", content: value });
        // Cap history at 20 items to prevent memory bloat
        if (this.#history.length > 20) {
            this.#history = this.#history.slice(-20);
        }
    }
    // #endregion

    stop() {
        if (!this.#serviceInstance) return;
        this.#serviceInstance.stop();
    }
    start() {
        this.stop();
        this.serviceState.error = "";

        let backend = this.data.backend;
        if (!(backend in backends)) {
            return;
        }
        this.#serviceInstance = new backends[backend](this);

        if (!this.#serviceInstance) return;
        this.#setStatus(ServiceNetworkState.connecting);
        this.#serviceInstance.start(this.data);
    }

    transform(text: TextEvent, history: any[] = []) {
        if (
            !this.#serviceInstance ||
            this.serviceState.status !== ServiceNetworkState.connected
        ) {
            // Passthrough for Sync Timing when service is off
            // text already has ID from init()
            window.ApiShared.pubsub.publishText(TextEventSource.transform_source, text);
            return;
        }
        this.#serviceInstance.transform(this.#id, text, history);
        this.#id++;
    }

    dispose() {
        this.stop();
        this.eventDisposers.forEach(d => d());
        this.eventDisposers = [];
    }
}

export default Service_Transform;
