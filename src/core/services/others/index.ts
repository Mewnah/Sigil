import { IServiceInterface, OthersEventSource } from "@/types";
import i18n from "i18next";
import { toast } from "react-toastify";
import { subscribeKey } from "valtio/utils";
import { entryTitle } from "@/core/services/models/format";
import { i18nReady } from "@/core/services/stt/statusMessages";
import { pushSystemLog } from "@/core/services/systemLog";
import { createTranslationPipeline, modelInstalledNow, TRANSLATION_ENGINES, translatorOnCpuNow, type TranslationEngineMap, type TranslationPipeline } from "../translation";
import { OthersSpeechDriver, type OthersDriverDeps, type OthersDriverState } from "./driver";
import { OthersPublisher } from "./publish";
import type { Others_State } from "./schema";
import { othersTranslationPair } from "./translation";

/** System-log source of the Others service. */
const LOG_SOURCE = "Other players";

/**
 * Other players (P6-A): transcribes what other players say in VRChat (per-process loopback of `VRChat.exe`, falling
 * back to an output device's loopback) with a second speech pipeline (`loopback` key, `./driver.ts`) and publishes it
 * on `others.stt` (`OthersEventSource`), plus a reverse-direction translation pipeline instance
 * (`createTranslationPipeline` with the reversed pair, `./translation.ts`) on `others.translation`. Never `text.*`:
 * other players never reach the chatbox, TTS or history. Settings: `services.others` (`./schema.ts`); inspector
 * `src/core/ui/inspector/inspector_others.tsx`. `ApiServer.others`, its init task, `Services.others` and the panel
 * entry are frozen (P6-S0).
 *
 * Off by default: it starts only from its panel or with `autoStart`.
 */
class Service_Others implements IServiceInterface {
  readonly #driver: OthersSpeechDriver;
  readonly #translation: TranslationPipeline;
  readonly #publisher: OthersPublisher;
  /** Bumped by every start and stop request; a deferred auto start yields to a user action. */
  #requests = 0;
  #disposers: (() => void)[] = [];

  constructor(engines: TranslationEngineMap = TRANSLATION_ENGINES, deps: Partial<OthersDriverDeps> = {}) {
    this.#translation = createTranslationPipeline({
      name: "others",
      settings: () => window.ApiServer.state.services.translation.data,
      engines,
      publish: (event) => window.ApiShared.pubsub.publishText(OthersEventSource.translation, event),
      pair: () => this.translationPair(),
      logSource: LOG_SOURCE,
      translatorOnCpu: translatorOnCpuNow,
      modelInstalled: modelInstalledNow,
    });
    this.#publisher = new OthersPublisher({
      publish: (topic, event) => window.ApiShared.pubsub.publishText(topic, event),
      translate: (stream, event) => this.#translation.push(stream, event),
      interimEnabled: () => this.data.interim !== false,
    });
    this.#driver = new OthersSpeechDriver(
      { interim: (text, meta) => this.#publisher.interim(text, meta), final: (text, meta) => this.#publisher.final(text, meta) },
      (error) => this.#onDriverStopped(error),
      {
        title: (id) => {
          const entry = window.ApiServer?.models?.state.catalog.find((e) => e.id === id);
          return entry ? entryTitle(entry) : id;
        },
        log: (line, level) => pushSystemLog(LOG_SOURCE, line, level),
        ...deps,
      },
    );
  }

  /** Capture and engine state (status, phase, source plan, status line, error) for the panel. */
  get serviceState(): OthersDriverState {
    return this.#driver.state;
  }

  /** State of the reverse translation pipeline (status, error, preparation, missing models). */
  get translationState() {
    return this.#translation.state;
  }

  get data(): Others_State {
    return window.ApiServer.state.services.others.data;
  }

  /** The reverse language pair (their language → the user's), read per line. */
  translationPair(): { from: string; to: string } {
    const state = window.ApiServer.state.services;
    return othersTranslationPair({
      othersLanguage: this.data.language,
      translateTo: this.data.translateTo,
      mainSource: state.translation.data.sourceLanguage,
      sttLanguage: state.stt.data.local?.language,
      appLanguage: i18n.language,
    });
  }

  /** Off by default: starts only when `autoStart` is on. */
  init() {
    // Turning reverse translation on or off applies while running.
    this.#disposers.push(
      subscribeKey(this.data, "translate", (on) => {
        if (!this.#driver.running) return;
        if (on) this.#translation.start();
        else this.#translation.stop();
      }),
    );
    if (this.data.autoStart) {
      // Host services init before i18next: wait for it so status and error texts are worded.
      const request = this.#requests;
      void i18nReady().then(() => {
        if (request === this.#requests && this.data.autoStart && !this.#driver.running) this.#start();
      });
    }
  }

  start() {
    this.#requests++;
    this.#start();
  }

  stop() {
    this.#requests++;
    this.#translation.stop();
    void this.#driver.stop();
  }

  dispose() {
    this.stop();
    this.#disposers.forEach((d) => d());
    this.#disposers = [];
  }

  #start() {
    const settings = { ...this.data };
    this.#translation.stop();
    void this.#driver.start(settings).then(() => {
      // Running or waiting for the program: translate what comes.
      if (this.#driver.running && this.data.translate) this.#translation.start();
    });
  }

  #onDriverStopped(error?: string) {
    this.#translation.stop();
    if (!error) return;
    // The driver already wrote the failure to the system log (its `log`): only the toast here.
    try {
      toast.error(i18n.t("others.toast_stopped", { error }), { toastId: "others-stopped", autoClose: 10_000 });
    } catch {
      /* toasts are best effort */
    }
  }
}

export default Service_Others;
export type { OthersDriverState };
