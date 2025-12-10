import {
    IServiceInterface,
    ServiceNetworkState,
    TextEvent,
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
    #id = 0;

    #serviceInstance!: ITransformService;
    get data() {
        return window.ApiServer.state.services.transform.data;
    }

    init(): void {
        if (this.data.autoStart)
            this.start();

        const subId = window.ApiShared.pubsub.subscribeText(
            TextEventSource.stt,
            (e) => e && this.transform(e)
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
        // ignore late results if applicable
        if (this.#id - 1 !== id) return;
        window.ApiShared.pubsub.publishText(TextEventSource.transform, {
            value,
            type: e.type,
        });
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

    transform(text: TextEvent) {
        if (
            !this.#serviceInstance ||
            this.serviceState.status !== ServiceNetworkState.connected
        )
            return;
        this.#serviceInstance.transform(this.#id, text);
        this.#id++;
    }

    dispose() {
        this.stop();
        this.eventDisposers.forEach(d => d());
        this.eventDisposers = [];
    }
}

export default Service_Transform;
