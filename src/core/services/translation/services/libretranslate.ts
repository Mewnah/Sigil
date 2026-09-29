import { invoke } from "@tauri-apps/api/core";
import i18n from "i18next";
import { raceWithSignal } from "../abort";
import { Translation_State } from "../schema";
import { isRetryableStatus, ITranslationEngine, TranslationError, TranslationRequest } from "../types";

export interface LibreTranslateLanguage {
  code: string;
  name: string;
}

/**
 * Endpoint and key for one `plugin:translate` call (per-call contract, see translate.rs). The key is
 * always passed explicitly, so an empty key means "no key" and never falls back to a key another
 * caller stored with `set_api_key`; the global `set_endpoint` / `set_api_key` are not used at all.
 */
export interface LibreConnection {
  endpoint: string;
  apiKey: string;
}

export function libreConnection(endpoint: string, apiKey: string | undefined): LibreConnection {
  return { endpoint: endpoint.trim(), apiKey: (apiKey ?? "").trim() };
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** HTTP status inside a Rust error string ("LibreTranslate translation: HTTP 403 Forbidden: ..."). */
export function libreErrorStatus(message: string): number | undefined {
  const m = /\bHTTP (\d{3})\b/.exec(message);
  return m ? Number(m[1]) : undefined;
}

/** Rust errors that repeat on every attempt: a malformed endpoint, or a body that is not the API's. */
const PERMANENT_ERROR = /^Invalid endpoint URL|^The endpoint URL is empty|: unexpected response \(/;

/**
 * Wraps a failed `plugin:translate|translate` call. Only statuses worth repeating (408, 429, 5xx) and
 * status-less transport failures (refused or reset connection, timeout) are retryable; a 400
 * (unsupported pair), 403 (bad key) or invalid URL fails the same way every time.
 */
export function libreTranslationError(e: unknown): TranslationError {
  const message = errorText(e);
  const status = libreErrorStatus(message);
  if (status !== undefined) return new TranslationError(message, { status, retryable: isRetryableStatus(status) });
  return new TranslationError(message, { retryable: !PERMANENT_ERROR.test(message) });
}

/** Language codes compared the way the server does (case-insensitive, `_` = `-`). */
function sameLanguage(a: string, b: string): boolean {
  const norm = (code: string) => code.trim().toLowerCase().replace(/_/g, "-");
  return norm(a) !== "" && norm(a) === norm(b);
}

export class Translation_LibreTranslateService implements ITranslationEngine {
  #conn: LibreConnection | null = null;
  #source = "en";
  #targetLanguage = "es";
  #autoDetect = false;

  async start(state: Translation_State, signal: AbortSignal): Promise<void> {
    const cfg = state.libretranslate;
    const conn = libreConnection(cfg.endpoint, cfg.apiKey);
    if (!conn.endpoint) throw new Error(i18n.t("transl.libretranslate_error_endpoint_missing"));
    if (!cfg.language.trim()) throw new Error(i18n.t("transl.error_target_missing"));

    let available: boolean;
    try {
      available = await raceWithSignal(invoke<boolean>("plugin:translate|check_availability", { ...conn }), signal);
    } catch (e) {
      if (signal.aborted) throw e;
      throw new Error(i18n.t("transl.libretranslate_error_connect", { error: errorText(e) }));
    }
    if (!available) throw new Error(i18n.t("transl.libretranslate_error_unavailable", { endpoint: conn.endpoint }));

    this.#conn = conn;
    this.#source = cfg.languageFrom.trim();
    this.#targetLanguage = cfg.language.trim();
    this.#autoDetect = cfg.autoDetect;
  }

  async translate(request: TranslationRequest, signal: AbortSignal): Promise<string> {
    const conn = this.#conn;
    if (!conn) throw new TranslationError(i18n.t("transl.error_not_running"), { retryable: false });

    let source = this.#source;
    if (this.#autoDetect) {
      try {
        source = await raceWithSignal(
          invoke<string>("plugin:translate|detect_language", { text: request.text, ...conn }),
          signal,
        );
      } catch (e) {
        if (signal.aborted) throw e;
        // Detection is best effort: fall back to the configured source language.
        source = this.#source;
      }
    }
    // A line already in the target language (e.g. a bilingual speaker with auto-detect) is its own
    // translation; the server would reject the pair.
    if (sameLanguage(source, this.#targetLanguage)) return request.text;

    try {
      return await raceWithSignal(
        invoke<string>("plugin:translate|translate", {
          text: request.text,
          source,
          target: this.#targetLanguage,
          ...conn,
        }),
        signal,
      );
    } catch (e) {
      if (signal.aborted) throw e;
      // The Rust side already words the error ("LibreTranslate translation: HTTP 429 ...").
      throw libreTranslationError(e);
    }
  }

  stop(): void {
    this.#conn = null;
  }

  /** Languages offered by the server at `endpoint`; [] when it cannot be reached. */
  static async getLanguages(endpoint: string, apiKey = ""): Promise<LibreTranslateLanguage[]> {
    const conn = libreConnection(endpoint, apiKey);
    if (!conn.endpoint) return [];
    try {
      return await invoke<LibreTranslateLanguage[]>("plugin:translate|get_languages", { ...conn });
    } catch {
      return [];
    }
  }
}
