import { devLog } from "@/utils/devLog";
import i18n from "i18next";
import { raceWithSignal } from "../abort";
import { Translation_State } from "../schema";
import { ITranslationEngine, TranslationError, TranslationRequest } from "../types";

type TranslatorInstance = {
  translate: (text: string, options?: { signal?: AbortSignal }) => Promise<string>;
  destroy?: () => void;
};

type TranslatorCtor = {
  create: (opts: {
    sourceLanguage: string;
    targetLanguage: string;
    signal?: AbortSignal;
  }) => Promise<TranslatorInstance>;
};

/** Built-in AI globals are interface objects (`typeof === "function"`), not plain objects. */
function isCtorLike(v: unknown): v is object {
  return (typeof v === "function" || typeof v === "object") && v !== null;
}

function getTranslatorCtor(): TranslatorCtor | undefined {
  const t = (globalThis as Record<string, unknown>)["Translator"];
  if (
    isCtorLike(t) &&
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
    isCtorLike(d) &&
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

let sharedDetector: Promise<LanguageDetectorInstance> | null = null;

async function firstDetectedLanguage(
  text: string,
  fallback: string
): Promise<string> {
  const ctor = getLanguageDetectorCtor();
  if (!ctor) return fallback;
  try {
    // One detector per session: creating it can trigger a model download.
    sharedDetector ??= ctor.create();
    const det = await sharedDetector;
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
    sharedDetector = null;
  }
  return fallback;
}

function destroyTranslator(t: TranslatorInstance | null | undefined): void {
  try {
    t?.destroy?.();
  } catch {
    // ignore
  }
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Translators kept alive with auto-detect (each holds a loaded model); the oldest is destroyed. */
const MAX_CACHED_TRANSLATORS = 3;

export class Translation_OnDeviceService implements ITranslationEngine {
  #cfg: Translation_State["onDevice"] | null = null;
  /**
   * One translator per `source|target` pair, created once and shared by concurrent calls (an interim
   * and a final can overlap). Insertion order is least-recently-used first.
   */
  #translators = new Map<string, Promise<TranslatorInstance>>();
  /** Aborted by stop(): cancels model downloads and translator creation. */
  #lifetime = new AbortController();

  async start(state: Translation_State, signal: AbortSignal): Promise<void> {
    const ctor = getTranslatorCtor();
    if (!ctor) throw new Error(i18n.t("transl.ondevice_error_unavailable"));

    devLog("[Translation] Built-in Translator constructor present (experimental Web API)");
    const cfg = { ...state.onDevice };
    const source = normalizeTag(cfg.languageFrom);
    const target = normalizeTag(cfg.language);
    if (!target) throw new Error(i18n.t("transl.error_target_missing"));

    // Same source and target needs no translator: translate() passes such lines through.
    if (!cfg.autoDetect && source !== target) {
      // Created up front so a model download or an unsupported pair fails Start, not the first line.
      try {
        await raceWithSignal(this.#translatorFor(ctor, source, target), signal);
      } catch (e) {
        if (signal.aborted) throw e;
        throw new Error(i18n.t("transl.ondevice_error_start", { error: errorText(e) }));
      }
      devLog("[Translation] On-device Translator ready:", `${source}|${target}`);
    } else if (cfg.autoDetect && !getLanguageDetectorCtor()) {
      devLog("[Translation] LanguageDetector API missing; auto-detect will use configured source language");
    }
    this.#cfg = cfg;
  }

  stop(): void {
    this.#cfg = null;
    this.#lifetime.abort();
    this.#lifetime = new AbortController();
    const pending = [...this.#translators.values()];
    this.#translators.clear();
    for (const p of pending) void p.then(destroyTranslator, () => {});
  }

  async translate(request: TranslationRequest, signal: AbortSignal): Promise<string> {
    const cfg = this.#cfg;
    const ctor = getTranslatorCtor();
    if (!cfg || !ctor) throw new TranslationError(i18n.t("transl.error_not_running"), { retryable: false });

    const target = normalizeTag(cfg.language);
    let source = normalizeTag(cfg.languageFrom);
    if (cfg.autoDetect) {
      source = await raceWithSignal(firstDetectedLanguage(request.text, source), signal);
    }
    // A line already in the target language (a bilingual speaker with auto-detect) is its own
    // translation; the Translator API has no same-language translator and would reject the pair.
    if (source === target) return request.text;

    let translator: TranslatorInstance;
    try {
      translator = await raceWithSignal(this.#translatorFor(ctor, source, target), signal);
    } catch (e) {
      if (signal.aborted) throw e;
      throw new TranslationError(i18n.t("transl.ondevice_error_pair", { pair: `${source} → ${target}`, error: errorText(e) }), {
        retryable: false,
      });
    }
    try {
      return await raceWithSignal(translator.translate(request.text, { signal }), signal);
    } catch (e) {
      if (signal.aborted) throw e;
      throw new TranslationError(errorText(e));
    }
  }

  #translatorFor(ctor: TranslatorCtor, source: string, target: string): Promise<TranslatorInstance> {
    const pair = `${source}|${target}`;
    const cached = this.#translators.get(pair);
    if (cached) {
      this.#translators.delete(pair);
      this.#translators.set(pair, cached);
      return cached;
    }
    const lifetime = this.#lifetime.signal;
    const created = ctor.create({ sourceLanguage: source, targetLanguage: target, signal: lifetime }).then((t) => {
      if (lifetime.aborted) {
        // stop() ran while the model was loading.
        destroyTranslator(t);
        throw new DOMException("Aborted", "AbortError");
      }
      return t;
    });
    // A failed creation is forgotten so the next line retries it.
    created.catch(() => {
      if (this.#translators.get(pair) === created) this.#translators.delete(pair);
    });
    this.#translators.set(pair, created);
    while (this.#translators.size > MAX_CACHED_TRANSLATORS) {
      const [oldestPair, oldest] = this.#translators.entries().next().value as [string, Promise<TranslatorInstance>];
      this.#translators.delete(oldestPair);
      void oldest.then(destroyTranslator, () => {});
    }
    return created;
  }
}
