import type {
  RustDeeplLanguage,
  RustDeeplPlan,
  RustDeeplTranslateRequest,
  RustDeeplTranslateResponse,
  RustDeeplUsage,
} from "@/types";
import { invoke } from "@tauri-apps/api/core";
import i18n from "i18next";
import { raceWithSignal } from "../abort";
import { canonicalLanguage, languageName, sameLanguage } from "../languages";
import { Translation_State } from "../schema";
import {
  isRetryableStatus,
  ITranslationEngine,
  TranslationContextPair,
  TranslationEngineHost,
  TranslationError,
  TranslationRequest,
} from "../types";

type DeeplConfig = Translation_State["deepl"];

/** DeepL's source languages when the list cannot be fetched (API v2, 2026). */
export const DEEPL_FALLBACK_SOURCES: readonly string[] = [
  "AR", "BG", "CS", "DA", "DE", "EL", "EN", "ES", "ET", "FI", "FR", "HE", "HU", "ID", "IT", "JA", "KO", "LT", "LV",
  "NB", "NL", "PL", "PT", "RO", "RU", "SK", "SL", "SV", "TH", "TR", "UK", "VI", "ZH",
];

/** DeepL's target languages when the list cannot be fetched. */
export const DEEPL_FALLBACK_TARGETS: readonly string[] = [
  "AR", "BG", "CS", "DA", "DE", "EL", "EN-GB", "EN-US", "ES", "ET", "FI", "FR", "HE", "HU", "ID", "IT", "JA", "KO",
  "LT", "LV", "NB", "NL", "PL", "PT-BR", "PT-PT", "RO", "RU", "SK", "SL", "SV", "TH", "TR", "UK", "VI", "ZH",
  "ZH-HANS", "ZH-HANT",
];

/** Interims are cheap to lose; finals get more time. The sequencer's own deadline stays above both. */
export const DEEPL_TIMEOUT_MS = { interim: 5_000, final: 10_000 } as const;
/** Earlier lines sent as `context` (not translated, not billed). */
export const DEEPL_CONTEXT_LINES = 3;
const DEEPL_CONTEXT_MAX_CHARS = 1_500;
/** Warn in the status line once this share of the monthly quota is used. */
const QUOTA_WARNING_RATIO = 0.9;

/** DeepL source code for a canonical language (`en` → `EN`, `zh-Hant` → `ZH`, `pt-PT` → `PT`); null = unknown. */
export function deeplSourceCode(code: string): string | null {
  const canonical = canonicalLanguage(code);
  if (!canonical) return null;
  const base = canonical.split("-")[0];
  if (base === "no") return "NB";
  return base.toUpperCase();
}

/**
 * DeepL target codes for a canonical language, best first: English defaults to American, Portuguese to Brazilian,
 * Chinese to the explicit script codes (older accounts only know `ZH`).
 */
export function deeplTargetCandidates(code: string): string[] {
  const canonical = canonicalLanguage(code);
  if (!canonical) return [];
  const [base, ...rest] = canonical.toLowerCase().split("-");
  switch (base) {
    case "en":
      return rest.includes("gb") ? ["EN-GB", "EN-US", "EN"] : ["EN-US", "EN-GB", "EN"];
    case "pt":
      return rest.includes("pt") ? ["PT-PT", "PT-BR", "PT"] : ["PT-BR", "PT-PT", "PT"];
    case "zh":
      return rest.includes("hant") ? ["ZH-HANT"] : ["ZH-HANS", "ZH"];
    case "no":
    case "nb":
      return ["NB"];
    default:
      return [base.toUpperCase()];
  }
}

/** The first candidate DeepL offers (`available` = its target list; null = unknown, take the first candidate). */
export function pickDeeplTarget(code: string, available: ReadonlySet<string> | null): string | null {
  const candidates = deeplTargetCandidates(code);
  if (!available) return candidates[0] ?? null;
  return candidates.find((c) => available.has(c)) ?? null;
}

/** DeepL `formality`: the strict values would fail for languages without formality, so the `prefer_` ones are sent. */
export function deeplFormality(value: DeeplConfig["formality"]): RustDeeplTranslateRequest["formality"] {
  switch (value) {
    case "more":
    case "prefer_more":
      return "prefer_more";
    case "less":
    case "prefer_less":
      return "prefer_less";
    default:
      return null;
  }
}

/** Earlier source lines for DeepL's `context` (the last few, newest last, at most ~1.5k characters); null = none. */
export function deeplContext(pairs: readonly TranslationContextPair[] | undefined): string | null {
  if (!pairs?.length) return null;
  const lines = pairs
    .slice(-DEEPL_CONTEXT_LINES)
    .map((p) => p.source.trim())
    .filter(Boolean);
  let text = lines.join("\n");
  if (text.length > DEEPL_CONTEXT_MAX_CHARS) text = text.slice(text.length - DEEPL_CONTEXT_MAX_CHARS);
  return text || null;
}

/** HTTP status inside a Rust error string (`"DeepL translation: HTTP 456: ..."`). */
export function deeplErrorStatus(message: string): number | undefined {
  const m = /\bHTTP (\d{3})\b/.exec(message);
  return m ? Number(m[1]) : undefined;
}

/**
 * A user-facing `TranslationError` for a failed `plugin:translate|deepl_*` call. 403 (wrong key or plan) and 456
 * (quota used up) never succeed on a retry; 429, 529 and 5xx do; errors the Rust side raised before sending
 * (`"DeepL: ..."`: missing key, invalid option) are permanent; transport failures (no connection, timeout) are
 * retryable.
 */
export function deeplError(e: unknown): TranslationError {
  const message = (e instanceof Error ? e.message : String(e ?? "")).trim();
  const status = deeplErrorStatus(message);
  if (status === 401 || status === 403) {
    return new TranslationError(i18n.t("transl.deepl_error_auth", { status }) || message, { status, retryable: false });
  }
  if (status === 456) {
    return new TranslationError(i18n.t("transl.deepl_error_quota") || message, { status, retryable: false });
  }
  if (status !== undefined) {
    return new TranslationError(i18n.t("transl.deepl_error_http", { status, error: message }) || message, {
      status,
      retryable: isRetryableStatus(status),
    });
  }
  if (message.startsWith("DeepL: ")) {
    return new TranslationError(message, { retryable: false });
  }
  return new TranslationError(i18n.t("transl.deepl_error_network", { error: message }) || message, { retryable: true });
}

/**
 * DeepL engine (`deepl` backend) over `plugin:translate|deepl_translate` (Rust `translate/deepl.rs`: API Free or Pro,
 * the key only in the `Authorization` header, a per-request timeout).
 *
 * - `start` needs the key and a target DeepL offers. It checks the key once (`deepl_usage`): a rejected key or a
 *   used-up quota fails the start, while an unreachable DeepL does not (lines retry). It loads DeepL's language
 *   lists for `supports` (built-in lists when they cannot be fetched) and warns at 90% of the monthly quota.
 * - `translate` maps canonical codes to DeepL's (`en` → `EN-US`, `zh-Hant` → `ZH-HANT`; an unknown source lets DeepL
 *   detect it), sends the last lines as `context` when `deepl.sendContext` (DeepL does not bill it), `prefer_*`
 *   formality and the model type. A line already in the target language comes back unchanged.
 *
 * Owned by P4-D. The static helpers serve the inspector (P4-F).
 */
export class Translation_DeeplService implements ITranslationEngine {
  #host: TranslationEngineHost | null = null;
  #key = "";
  #plan: RustDeeplPlan = "auto";
  #config: DeeplConfig | null = null;
  #source = "";
  #target = "";
  #sources: ReadonlySet<string> | null = null;
  #targets: ReadonlySet<string> | null = null;

  attach(host: TranslationEngineHost): void {
    this.#host = host;
  }

  async start(state: Translation_State, signal: AbortSignal): Promise<void> {
    const cfg = state.deepl;
    const key = cfg.apiKey.trim();
    if (!key) throw new TranslationError(i18n.t("transl.deepl_error_key_missing"), { retryable: false });
    const target = canonicalLanguage(state.targetLanguage);
    if (!target) throw new TranslationError(i18n.t("transl.error_target_missing"), { retryable: false });
    const plan = cfg.plan;

    // One key check. A rejected key or a used-up quota cannot work; an unreachable DeepL may recover.
    try {
      const usage = await raceWithSignal(invoke<RustDeeplUsage>("plugin:translate|deepl_usage", { apiKey: key, plan }), signal);
      this.#reportUsage(usage);
    } catch (e) {
      if (signal.aborted) throw e;
      const error = e instanceof TranslationError ? e : deeplError(e);
      if (!error.retryable || error.status === 456) throw error;
    }

    const [sources, targets] = await Promise.all([
      raceWithSignal(Translation_DeeplService.getLanguages(key, plan, "source"), signal),
      raceWithSignal(Translation_DeeplService.getLanguages(key, plan, "target"), signal),
    ]);
    this.#sources = new Set(sources.length ? sources.map((l) => l.code) : DEEPL_FALLBACK_SOURCES);
    this.#targets = new Set(targets.length ? targets.map((l) => l.code) : DEEPL_FALLBACK_TARGETS);
    if (!pickDeeplTarget(target, this.#targets)) {
      throw new TranslationError(i18n.t("transl.deepl_error_language", { language: languageName(state.targetLanguage) }), {
        retryable: false,
      });
    }

    this.#key = key;
    this.#plan = plan;
    this.#config = { ...cfg };
    this.#source = canonicalLanguage(state.sourceLanguage);
    this.#target = target;
  }

  async translate(request: TranslationRequest, signal: AbortSignal): Promise<string> {
    const cfg = this.#config;
    if (!cfg || !this.#key) throw new TranslationError(i18n.t("transl.error_not_running"), { retryable: false });
    const to = canonicalLanguage(request.to ?? this.#target) || this.#target;
    const from = canonicalLanguage(request.from ?? this.#source);
    if (from && sameLanguage(from, to)) return request.text;
    const target = pickDeeplTarget(to, this.#targets);
    if (!target) {
      throw new TranslationError(i18n.t("transl.deepl_error_language", { language: languageName(to) }), { retryable: false });
    }
    const source = from ? deeplSourceCode(from) : null;
    const payload: RustDeeplTranslateRequest = {
      text: request.text,
      // A source DeepL does not offer is left to its detection.
      source: source && (!this.#sources || this.#sources.has(source)) ? source : null,
      target,
      apiKey: this.#key,
      plan: this.#plan,
      formality: deeplFormality(cfg.formality),
      context: cfg.sendContext ? deeplContext(request.context) : null,
      modelType: cfg.modelType === "default" ? null : cfg.modelType,
      timeoutMs: DEEPL_TIMEOUT_MS[request.kind],
    };
    try {
      const response = await raceWithSignal(
        invoke<RustDeeplTranslateResponse>("plugin:translate|deepl_translate", { request: payload }),
        signal,
      );
      return response.text;
    } catch (e) {
      if (signal.aborted) throw signal.reason ?? e;
      throw deeplError(e);
    }
  }

  stop(): void {
    this.#key = "";
    this.#config = null;
  }

  supports(from: string, to: string): boolean {
    if (!pickDeeplTarget(to, this.#targets ?? new Set(DEEPL_FALLBACK_TARGETS))) return false;
    if (!from) return true;
    const source = deeplSourceCode(from);
    return source !== null && (this.#sources ?? new Set(DEEPL_FALLBACK_SOURCES)).has(source);
  }

  #reportUsage(usage: RustDeeplUsage): void {
    const { characterCount: used, characterLimit: limit } = usage;
    if (!(limit > 0)) return;
    if (used >= limit) throw deeplError("DeepL usage: HTTP 456: quota exceeded");
    if (used >= limit * QUOTA_WARNING_RATIO) {
      this.#host?.status({
        text: i18n.t("transl.deepl_status_quota", { percent: Math.floor((used / limit) * 100) }) || "",
        level: "warning",
      });
    }
  }

  /** Characters used / allowed this billing period; null when the key is empty or DeepL cannot be reached. */
  static async getUsage(apiKey: string, plan: RustDeeplPlan = "auto"): Promise<RustDeeplUsage | null> {
    const key = apiKey.trim();
    if (!key) return null;
    try {
      return await invoke<RustDeeplUsage>("plugin:translate|deepl_usage", { apiKey: key, plan });
    } catch {
      return null;
    }
  }

  /** DeepL's source or target languages; [] when unavailable. */
  static async getLanguages(apiKey: string, plan: RustDeeplPlan, kind: "source" | "target"): Promise<RustDeeplLanguage[]> {
    const key = apiKey.trim();
    if (!key) return [];
    try {
      const rows = await invoke<RustDeeplLanguage[]>("plugin:translate|deepl_languages", { apiKey: key, plan, kind });
      return Array.isArray(rows) ? rows : [];
    } catch {
      return [];
    }
  }

  /** The DeepL target code a canonical language would be sent as (the inspector shows it); null = unsupported. */
  static targetCode(code: string, available: readonly RustDeeplLanguage[] = []): string | null {
    return pickDeeplTarget(code, available.length ? new Set(available.map((l) => l.code.toUpperCase())) : null);
  }
}
