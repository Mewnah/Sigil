import { IServiceInterface, TextEventSource } from "@/types";
import { serviceSubscibeToInput } from "@/utils";
import { fetchWithTimeout } from "@/utils/fetchWithTimeout";
import { proxy } from "valtio";
import { llmServerUi } from "../llm_server";
import { i18nReady } from "../stt/statusMessages";
import { bundledTranslatorOnCpu } from "./interimPolicy";
import { createTranslationPipeline, TranslationEngineMap, TranslationPipeline, TranslationServiceState } from "./pipeline";
import { Translation_RouterService } from "./router";
import { Translation_Backends } from "./schema";
import { Translation_AzureService } from "./services/azure";
import { Translation_BergamotService } from "./services/bergamot";
import { Translation_DeeplService } from "./services/deepl";
import { isLiveTranslatedStt } from "./services/geminiLive";
import { Translation_LibreTranslateService } from "./services/libretranslate";
import { Translation_LlmService } from "./services/llm";
import { Translation_NllbService } from "./services/nllb";
import { Translation_OnDeviceService } from "./services/onDevice";

export { createTranslationPipeline } from "./pipeline";
export { bundledTranslatorOnCpu, interimHeldOffByCpu, interimTranslationsOn } from "./interimPolicy";
export type {
  TranslationEngineMap,
  TranslationOutputEvent,
  TranslationPipeline,
  TranslationPipelineOptions,
  TranslationServiceState,
} from "./pipeline";

// Translation backend registry: the mapped type forces every enum value to be registered. The router (`auto`) builds
// its chain from the other entries (the pipeline hands them over through `ITranslationEngine.attach`).
export const TRANSLATION_ENGINES: TranslationEngineMap = {
  [Translation_Backends.azure]: Translation_AzureService,
  [Translation_Backends.libretranslate]: Translation_LibreTranslateService,
  [Translation_Backends.onDevice]: Translation_OnDeviceService,
  [Translation_Backends.llm]: Translation_LlmService,
  [Translation_Backends.deepl]: Translation_DeeplService,
  [Translation_Backends.bergamot]: Translation_BergamotService,
  [Translation_Backends.nllb]: Translation_NllbService,
  [Translation_Backends.auto]: Translation_RouterService,
};

/**
 * `TranslationPipelineOptions.translatorOnCpu` for the live settings: the bundled llama-server translator runs on
 * the processor (its status once it is up, else the `llm` slice), which turns `interim: "auto"` off.
 */
export function translatorOnCpuNow(): boolean {
  return bundledTranslatorOnCpu(window.ApiServer.state.services.translation.data.llm, llmServerUi.status);
}

/** Input streams, sequenced independently: speech and typed text are separate utterance streams. */
type TranslationStream = "stt" | "textfield";

/**
 * The translation service: one pipeline (`pipeline.ts`) from `text.stt` and typed text (`inputField`) to
 * `text.translation`, configured by `state.services.translation.data`.
 */
class Service_Translation implements IServiceInterface {
  lib = proxy<{ azure: { label: string; value: string }[] }>({
    azure: [],
  });

  private eventDisposers: (() => void)[] = [];
  readonly #pipeline: TranslationPipeline;
  /** Bumped by every start and stop request; a deferred auto start yields to a user action. */
  #requests = 0;

  constructor(engines: TranslationEngineMap = TRANSLATION_ENGINES) {
    this.#pipeline = createTranslationPipeline({
      name: "tr",
      settings: () => this.data,
      engines,
      publish: (event) => window.ApiShared.pubsub.publishText(TextEventSource.translation, event),
      translatorOnCpu: translatorOnCpuNow,
    });
  }

  /** Pipeline state (status, start error, preparation status, active engine) for the inspector. */
  get serviceState(): TranslationServiceState {
    return this.#pipeline.state;
  }

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
      const request = this.#requests;
      void i18nReady().then(() => {
        if (request === this.#requests && this.data.autoStart) this.start();
      });
    }
  }

  start() {
    this.#requests++;
    this.#pipeline.start();
  }

  stop() {
    this.#requests++;
    this.#pipeline.stop();
  }

  dispose() {
    this.stop();
    this.eventDisposers.forEach(d => d());
    this.eventDisposers = [];
  }

  #onText(stream: TranslationStream, e: Parameters<TranslationPipeline["push"]>[1]) {
    // Gemini Live Translate already publishes its own translation on text.translation (P6-K).
    if (stream === "stt" && isLiveTranslatedStt(e)) return;
    this.#pipeline.push(stream, e);
  }
}

export default Service_Translation;
