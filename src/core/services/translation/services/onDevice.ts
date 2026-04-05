import { TextEvent, TextEventType } from "@/types";
import { devLog } from "@/utils/devLog";
import { Translation_State } from "../schema";
import {
  ITranslationReceiver,
  ITranslationService,
} from "../types";

type TranslatorInstance = {
  translate: (text: string) => Promise<string>;
};

type TranslatorCtor = {
  create: (opts: {
    sourceLanguage: string;
    targetLanguage: string;
  }) => Promise<TranslatorInstance>;
};

function getTranslatorCtor(): TranslatorCtor | undefined {
  const t = (globalThis as Record<string, unknown>)["Translator"];
  if (
    typeof t === "object" &&
    t !== null &&
    "create" in t &&
    typeof (t as TranslatorCtor).create === "function"
  ) {
    return t as TranslatorCtor;
  }
  return undefined;
}

type LanguageDetectorInstance = {
  detect?: (text: string) => unknown;
};

type LanguageDetectorCtor = {
  create: () => Promise<LanguageDetectorInstance>;
};

function getLanguageDetectorCtor(): LanguageDetectorCtor | undefined {
  const d = (globalThis as Record<string, unknown>)["LanguageDetector"];
  if (
    typeof d === "object" &&
    d !== null &&
    "create" in d &&
    typeof (d as LanguageDetectorCtor).create === "function"
  ) {
    return d as LanguageDetectorCtor;
  }
  return undefined;
}

function normalizeTag(code: string): string {
  return code.trim().toLowerCase().replace(/_/g, "-");
}

async function firstDetectedLanguage(
  text: string,
  fallback: string
): Promise<string> {
  const ctor = getLanguageDetectorCtor();
  if (!ctor) return fallback;
  try {
    const det = await ctor.create();
    if (typeof det.detect !== "function") return fallback;
    const raw = await det.detect(text);

    if (typeof raw === "string" && raw.length > 0) {
      return normalizeTag(raw);
    }

    if (Array.isArray(raw) && raw.length > 0) {
      const first = raw[0] as Record<string, unknown>;
      const lang =
        (typeof first?.detectedLanguage === "string" && first.detectedLanguage) ||
        (typeof first?.locale === "string" && first.locale);
      if (lang) return normalizeTag(lang);
    }

    if (raw && typeof (raw as AsyncIterable<{ detectedLanguage?: string }>)[Symbol.asyncIterator] === "function") {
      for await (const chunk of raw as AsyncIterable<{ detectedLanguage?: string }>) {
        if (chunk?.detectedLanguage) return normalizeTag(chunk.detectedLanguage);
      }
    }
  } catch {
    // fall through
  }
  return fallback;
}

export class Translation_OnDeviceService implements ITranslationService {
  #receiver: ITranslationReceiver;
  #running = false;
  #cfg: Translation_State["onDevice"] | null = null;
  /** Used when autoDetect is off */
  #fixedTranslator: TranslatorInstance | null = null;
  /** Recreated when source/target pair changes while autoDetect is on */
  #dynamicTranslator: TranslatorInstance | null = null;
  #lastPair = "";

  constructor(receiver: ITranslationReceiver) {
    this.#receiver = receiver;
  }

  dispose(): void {
    this.stop();
  }

  start(state: Translation_State): void {
    void this.#startAsync(state);
  }

  async #startAsync(state: Translation_State): Promise<void> {
    const ctor = getTranslatorCtor();
    if (!ctor) {
      this.#receiver.onStop(
        "On-device translation is not available in this WebView runtime. Try Microsoft Edge / Chrome with the Translator API enabled, or use LibreTranslate / Azure."
      );
      return;
    }

    devLog("[Translation] Built-in Translator constructor present (experimental Web API)");

    this.#cfg = state.onDevice;
    this.#fixedTranslator = null;
    this.#dynamicTranslator = null;
    this.#lastPair = "";

    if (!state.onDevice.autoDetect) {
      try {
        const source = normalizeTag(state.onDevice.languageFrom);
        const target = normalizeTag(state.onDevice.language);
        this.#fixedTranslator = await ctor.create({
          sourceLanguage: source,
          targetLanguage: target,
        });
        this.#lastPair = `${source}|${target}`;
        devLog("[Translation] On-device Translator ready:", this.#lastPair);
      } catch (e) {
        this.#receiver.onStop(
          `On-device translator failed to start (model download or unsupported language pair): ${e}`
        );
        return;
      }
    } else if (!getLanguageDetectorCtor()) {
      devLog("[Translation] LanguageDetector API missing; auto-detect will use configured source language");
    }

    this.#running = true;
    this.#receiver.onStart();
  }

  stop(): void {
    this.#running = false;
    this.#cfg = null;
    this.#fixedTranslator = null;
    this.#dynamicTranslator = null;
    this.#lastPair = "";
    this.#receiver.onStop("");
  }

  async translate(id: number, event: TextEvent): Promise<void> {
    if (!this.#running || !this.#cfg || !event.value) return;
    if (event.type === TextEventType.interim && !this.#cfg.interim) return;

    const ctor = getTranslatorCtor();
    if (!ctor) return;

    const target = normalizeTag(this.#cfg.language);
    let source = normalizeTag(this.#cfg.languageFrom);
    if (this.#cfg.autoDetect) {
      source = await firstDetectedLanguage(event.value, source);
    }

    const pair = `${source}|${target}`;
    try {
      let tr: TranslatorInstance;
      if (!this.#cfg.autoDetect) {
        const fixed = this.#fixedTranslator;
        if (!fixed) return;
        tr = fixed;
      } else {
        if (pair !== this.#lastPair || !this.#dynamicTranslator) {
          this.#dynamicTranslator = await ctor.create({
            sourceLanguage: source,
            targetLanguage: target,
          });
          this.#lastPair = pair;
        }
        tr = this.#dynamicTranslator!;
      }
      const translated = await tr.translate(event.value);
      this.#receiver.onTranslation(id, event, translated);
    } catch (e) {
      devLog("[Translation] On-device translate error:", e);
    }
  }
}
