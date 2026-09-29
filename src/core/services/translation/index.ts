import { ErrorThrottle } from "@/core/services/tts/errorThrottle";
import {
  IServiceInterface,
  ServiceNetworkState,
  TextEvent,
  TextEventSource,
  TextEventType,
} from "@/types";
import { serviceSubscibeToInput } from "@/utils";
import { fetchWithTimeout } from "@/utils/fetchWithTimeout";
import i18n from "i18next";
import { toast } from "react-toastify";
import { proxy } from "valtio";
import { i18nReady } from "../stt/statusMessages";
import { pushSystemLog } from "../systemLog";
import { raceWithSignal } from "./abort";
import { withOneRetry } from "./retry";
import { Translation_Backends } from "./schema";
import {
  CANCEL_MARKER,
  isCancelMarker,
  SequencerJob,
  SequencerTimeoutError,
  TranslationSequencer,
} from "./sequencer";
import { Translation_AzureService } from "./services/azure";
import { Translation_LibreTranslateService } from "./services/libretranslate";
import { Translation_OnDeviceService } from "./services/onDevice";
import { ITranslationEngine, ITranslationEngineConstructor } from "./types";

export type TranslationEngineMap = { [k in Translation_Backends]: ITranslationEngineConstructor };

// Translation backend registry: the mapped type forces every enum value to be registered.
export const TRANSLATION_ENGINES: TranslationEngineMap = {
  [Translation_Backends.azure]: Translation_AzureService,
  [Translation_Backends.libretranslate]: Translation_LibreTranslateService,
  [Translation_Backends.onDevice]: Translation_OnDeviceService,
};

/** Input streams, sequenced independently: speech and typed text are separate utterance streams. */
type TranslationStream = "stt" | "textfield";

/** Message of `e`; "" when it has none (an engine threw `new Error(i18n.t(...))` before i18n was ready). */
function errorText(e: unknown): string {
  if (e instanceof SequencerTimeoutError) {
    return i18n.t("transl.error_timeout", { seconds: Math.round(e.timeoutMs / 1000) }) ?? "";
  }
  const text = e instanceof Error ? e.message : e === undefined || e === null ? "" : String(e);
  return text.trim();
}

/** Never empty: a failure the user cannot see explained is a failure they cannot fix. */
function startErrorText(e: unknown): string {
  return errorText(e) || i18n.t("transl.error_start_no_details") || "Translation could not start. Check the provider settings.";
}

function lineErrorText(e: unknown): string {
  return errorText(e) || i18n.t("common.error_no_details") || "no details";
}

class Service_Translation implements IServiceInterface {
  lib = proxy<{ azure: { label: string; value: string }[] }>({
    azure: [],
  });

  serviceState = proxy({
    status: ServiceNetworkState.disconnected,
    error: "",
  });
  private eventDisposers: (() => void)[] = [];

  #engine: ITranslationEngine | null = null;
  #backend: Translation_Backends | null = null;
  /** Bumped on every start and stop; async work of an older session must not touch state. */
  #epoch = 0;
  #startController: AbortController | null = null;
  #errors = new ErrorThrottle();
  #sequencer = new TranslationSequencer<TextEvent>({
    run: (job) => this.#run(job),
    publish: (job, value) => this.#publish(job, value),
    onError: (job, error) => this.#reportError(job, error),
  });

  constructor(private readonly engines: TranslationEngineMap = TRANSLATION_ENGINES) {}

  get data() {
    return window.ApiServer.state.services.translation.data;
  }

  async loadAzure() {
    try {
      const azureResp = await fetchWithTimeout(
        "https://api.cognitive.microsofttranslator.com/languages?api-version=3.0&scope=translation",
        { timeoutMs: 20_000 }
      );
      const json: {
        translation: {
          [key: string]: {
            name: string;
            nativeName: string;
            dir: string;
          };
        };
      } = await azureResp.json();
      if (!json || !("translation" in json)) return;
      this.lib.azure = Object.keys(json.translation).map((k) => ({
        label: json.translation[k].nativeName,
        value: k,
      }));
    } catch {
      // Offline: the inspector keeps the stored codes selectable.
    }
  }

  init(): void {
    void this.loadAzure();

    const subId = window.ApiShared.pubsub.subscribeText(
      TextEventSource.stt,
      (e) => e && this.#onText("stt", e)
    );
    this.eventDisposers.push(() => window.ApiShared.pubsub.unsubscribe(subId));

    this.eventDisposers.push(serviceSubscibeToInput(
      window.ApiServer.state.services.translation.data,
      "inputField",
      (e) => e && this.#onText("textfield", e)
    ));

    if (this.data.autoStart) {
      // Host services init before i18next (ApiServer.init awaits initI18n last). Engines word their
      // start errors with i18n.t, which returns nothing until then, so an auto start that failed right
      // away would stay unexplained. A start or stop by the user in the meantime wins.
      const epoch = this.#epoch;
      void i18nReady().then(() => {
        if (epoch === this.#epoch && this.data.autoStart) this.start();
      });
    }
  }

  start() {
    this.#teardown();
    const epoch = ++this.#epoch;
    this.serviceState.error = "";
    this.#errors.reset();

    const backend = this.data.backend;
    const Engine = this.engines[backend];
    if (!Engine) {
      this.#setStatus(ServiceNetworkState.disconnected);
      return;
    }
    const engine = new Engine();
    const controller = new AbortController();
    this.#engine = engine;
    this.#backend = backend;
    this.#startController = controller;
    this.#setStatus(ServiceNetworkState.connecting);

    let starting: Promise<void>;
    try {
      starting = Promise.resolve(engine.start(this.data, controller.signal));
    } catch (e) {
      starting = Promise.reject(e);
    }
    void (async () => {
      try {
        await raceWithSignal(starting, controller.signal);
      } catch (e) {
        // stop() or a newer start() owns the state now.
        if (epoch !== this.#epoch) return;
        this.#failStart(startErrorText(e));
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
  }

  /** Aborts a pending start, drops in-flight translations and stops the engine. */
  #teardown() {
    this.#startController?.abort();
    this.#startController = null;
    this.#sequencer.reset();
    const engine = this.#engine;
    this.#engine = null;
    this.#backend = null;
    try {
      engine?.stop();
    } catch (e) {
      console.error("[Translation] engine stop failed:", e);
    }
  }

  #failStart(message: string) {
    this.#teardown();
    this.#setStatus(ServiceNetworkState.disconnected);
    this.serviceState.error = message;
    pushSystemLog("Translation", message, "error");
    toast(message, { type: "error", autoClose: false });
  }

  #setStatus(value: ServiceNetworkState) {
    this.serviceState.status = value;
  }

  #interimEnabled(): boolean {
    const backend = this.#backend;
    return backend !== null && this.data[backend]?.interim !== false;
  }

  #onText(stream: TranslationStream, e: TextEvent) {
    if (!this.#engine || this.serviceState.status !== ServiceNetworkState.connected) return;
    if (!e.value?.trim()) return;
    if (e.type === TextEventType.final) {
      // STT's "sentence cancelled" marker is not text to translate: it only closes the partial.
      if (isCancelMarker(e.value)) this.#sequencer.cancel(stream, e, CANCEL_MARKER);
      else this.#sequencer.pushFinal(stream, e);
    }
    else if (this.#interimEnabled()) {
      this.#sequencer.pushInterim(stream, e);
    }
  }

  #run(job: SequencerJob<TextEvent>): Promise<string> {
    const engine = this.#engine;
    if (!engine) return Promise.reject(new DOMException("Aborted", "AbortError"));
    const request = { text: job.event.value, kind: job.kind };
    if (job.kind === "interim") return engine.translate(request, job.signal);
    return withOneRetry(() => engine.translate(request, job.signal), job.signal);
  }

  #publish(job: SequencerJob<TextEvent>, value: string) {
    if (!value.trim()) return;
    window.ApiShared.pubsub.publishText(TextEventSource.translation, {
      value,
      type: job.kind === "final" ? TextEventType.final : TextEventType.interim,
    });
  }

  #reportError(job: SequencerJob<TextEvent>, error: unknown) {
    const message = lineErrorText(error);
    const line = i18n.t("transl.error_line_failed", { error: message }) || message;
    console.warn(`[Translation] ${job.kind} translation failed:`, error);
    // A failed final loses a line, so it is always logged; interim failures (the next partial replaces
    // them) are logged only together with a toast. Toasts are throttled: a dead server fails every line.
    const show = this.#errors.shouldShow(message);
    if (job.kind === "final" || show) pushSystemLog("Translation", line, "warning");
    if (show) toast(line, { type: "warning", autoClose: 8_000 });
  }
}

export default Service_Translation;
