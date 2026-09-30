import i18n from "i18next";
import type { RustSpeechEngineSpec } from "@/types";
import { STT_State } from "../schema";
import { normalizeEndpointUrl, RustUtteranceService, shortDetail, type PipelineEngineSetup } from "../rustCapturePolling";

/** Moonshine server default (Rust `speech::engines::http::MOONSHINE_DEFAULT_URL`). */
export const MOONSHINE_DEFAULT_ENDPOINT = "http://localhost:8090";

/**
 * `moonshine` settings → the Rust HTTP engine's Moonshine preset (`POST {endpoint}/transcribe` with a base64 WAV,
 * after a `GET {endpoint}/health` when the session starts), or the message to stop with. Pure.
 */
export function moonshineSetup(settings: Partial<STT_State["moonshine"]>): PipelineEngineSetup | string {
  const raw = String(settings.endpoint ?? "").trim() || MOONSHINE_DEFAULT_ENDPOINT;
  const baseUrl = normalizeEndpointUrl(raw);
  if (!baseUrl) return i18n.t("stt.error_http_base_url_invalid", { url: raw });
  const engine: RustSpeechEngineSpec = { engine: "http", model: "", options: { preset: "moonshine", baseUrl } };
  return { engine, language: String(settings.language ?? "").trim() || "auto" };
}

/**
 * Legacy: Useful Sensors Moonshine behind a server the user runs (Docker or another process). The `local` backend
 * runs Moonshine v2 in-process with nothing to install; this backend stays for existing setups. Requests go through
 * Sigil's Rust speech pipeline (one per VAD utterance); see {@link RustUtteranceService}.
 */
export class STT_MoonshineService extends RustUtteranceService {
  protected readonly logTag = "[Moonshine]";
  #endpoint = MOONSHINE_DEFAULT_ENDPOINT;

  protected engineLabel(): string {
    return i18n.t("stt.provider_moonshine");
  }

  protected inputDevice(params: STT_State): string {
    return params.moonshine.device;
  }

  protected prepare(params: STT_State): PipelineEngineSetup | string {
    const setup = moonshineSetup(params.moonshine);
    if (typeof setup !== "string") this.#endpoint = String(setup.engine.options?.baseUrl ?? MOONSHINE_DEFAULT_ENDPOINT);
    return setup;
  }

  protected connectingStatus(): string {
    return i18n.t("stt.status_moonshine_connecting");
  }

  /** The start fails mostly because the server does not answer its health check. */
  protected startFailedMessage(detail: string): string {
    return i18n.t("stt.error_moonshine_start", { endpoint: this.#endpoint, detail: shortDetail(detail) });
  }
}
