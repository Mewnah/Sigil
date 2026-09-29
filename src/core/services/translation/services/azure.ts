import { fetchWithTimeout } from "@/utils/fetchWithTimeout";
import i18n from "i18next";
import { Translation_State } from "../schema";
import {
  isRetryableStatus,
  ITranslationEngine,
  TranslationError,
  TranslationRequest,
} from "../types";

type AzureConfig = Translation_State["azure"];

export const AZURE_TRANSLATE_ENDPOINT = "https://api.cognitive.microsofttranslator.com/translate";
/** Per request; the sequencer's own deadline (30 s for finals) stays above it. */
export const AZURE_TIMEOUT_MS = 20_000;

/** Translate URL. `from` is left out when blank, which makes Azure auto-detect the source. */
export function azureTranslateUrl(cfg: Pick<AzureConfig, "languageFrom" | "language" | "profanity">): string {
  const params = new URLSearchParams({ "api-version": "3.0" });
  const from = cfg.languageFrom.trim();
  if (from) params.set("from", from);
  params.set("to", cfg.language.trim());
  params.set("profanityAction", cfg.profanity);
  return `${AZURE_TRANSLATE_ENDPOINT}?${params.toString()}`;
}

/**
 * Request headers. The region header is required only for regional and multi-service resources; a
 * global Translator resource must work without it, so a blank location is not sent at all.
 */
export function azureHeaders(cfg: Pick<AzureConfig, "key" | "location">): Record<string, string> {
  const headers: Record<string, string> = {
    "Ocp-Apim-Subscription-Key": cfg.key.trim(),
    "Content-Type": "application/json",
  };
  const region = cfg.location.trim();
  if (region) headers["Ocp-Apim-Subscription-Region"] = region;
  return headers;
}

/** Reads Azure's `{ error: { code, message } }` body; falls back to the status text. */
async function azureErrorDetail(resp: Response): Promise<string> {
  try {
    const body: unknown = await resp.json();
    const message = (body as { error?: { message?: unknown } } | null)?.error?.message;
    if (typeof message === "string" && message.trim()) return message.trim();
  } catch {
    // not JSON
  }
  return resp.statusText || `HTTP ${resp.status}`;
}

export function azureHttpError(status: number, detail: string): TranslationError {
  const message =
    status === 401 || status === 403
      ? i18n.t("transl.azure_error_auth", { status, error: detail })
      : i18n.t("transl.azure_error_http", { status, error: detail });
  return new TranslationError(message, { status, retryable: isRetryableStatus(status) });
}

/** Extracts the first translation from Azure's response body. */
export function parseAzureTranslation(body: unknown): string {
  const text = (body as { translations?: { text?: unknown }[] }[] | null)?.[0]?.translations?.[0]?.text;
  if (typeof text !== "string") {
    throw new TranslationError(i18n.t("transl.azure_error_response"), { retryable: false });
  }
  return text;
}

export class Translation_AzureService implements ITranslationEngine {
  #cfg: AzureConfig | null = null;

  // Nothing to await: the key is checked by the first request (a probe would cost characters).
  async start(state: Translation_State, _signal?: AbortSignal): Promise<void> {
    // Copied: the inspector locks these fields while the service runs.
    const cfg = { ...state.azure };
    if (!cfg.key.trim()) throw new Error(i18n.t("transl.azure_error_key_missing"));
    if (!cfg.language.trim()) throw new Error(i18n.t("transl.error_target_missing"));
    this.#cfg = cfg;
  }

  async translate(request: TranslationRequest, signal: AbortSignal): Promise<string> {
    const cfg = this.#cfg;
    if (!cfg) throw new TranslationError(i18n.t("transl.error_not_running"), { retryable: false });

    let resp: Response;
    try {
      resp = await fetchWithTimeout(azureTranslateUrl(cfg), {
        method: "POST",
        timeoutMs: AZURE_TIMEOUT_MS,
        signal,
        body: JSON.stringify([{ text: request.text }]),
        headers: azureHeaders(cfg),
      });
    } catch (e) {
      if (signal.aborted) throw signal.reason ?? e;
      if (e instanceof DOMException && e.name === "AbortError") {
        throw new TranslationError(i18n.t("transl.error_timeout", { seconds: AZURE_TIMEOUT_MS / 1000 }), {
          retryable: false,
        });
      }
      const detail = e instanceof Error ? e.message : String(e);
      throw new TranslationError(i18n.t("transl.azure_error_network", { error: detail }));
    }

    if (!resp.ok) throw azureHttpError(resp.status, await azureErrorDetail(resp));

    let body: unknown;
    try {
      body = await resp.json();
    } catch {
      throw new TranslationError(i18n.t("transl.azure_error_response"), { retryable: false });
    }
    return parseAzureTranslation(body);
  }

  stop(): void {
    this.#cfg = null;
  }
}
