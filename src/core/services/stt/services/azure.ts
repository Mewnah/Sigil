import type {
  SpeechRecognitionCanceledEventArgs,
  SpeechRecognitionEventArgs,
  SpeechRecognitionResult,
  SpeechRecognizer,
} from "microsoft-cognitiveservices-speech-sdk";
import i18n from "i18next";
import { isEmptyValue } from "../../../../utils";
import { isAbortError, raceWithAbort } from "../abort";
import { networkLooksUp } from "../restartPolicy";
import { baseLanguageTag, errorText, shortDetail } from "../rustCapturePolling";
import { STT_State } from "../schema";
import { azureSegmentationSilenceMs } from "../sttOptions";
import { ISTTReceiver, ISTTService, SttStopOptions, SttTextMeta } from "../types";

type AzureSdk = typeof import("microsoft-cognitiveservices-speech-sdk");

/**
 * The Speech SDK is loaded on first start: it is large, and its CommonJS bundle has a circular init that must stay out
 * of the app's startup graph (a static import here broke the production app when module order shifted, P2-S0).
 */
let sdkPromise: Promise<AzureSdk> | null = null;
function loadSdk(): Promise<AzureSdk> {
  if (!sdkPromise) {
    sdkPromise = import("microsoft-cognitiveservices-speech-sdk").catch((error: unknown) => {
      sdkPromise = null;
      throw error;
    });
  }
  return sdkPromise;
}

/**
 * Stop message for an Azure cancellation (`CancellationErrorCode` name + the SDK's `errorDetails`). Pure.
 * `null` = no error (the session simply ended).
 */
export function azureCancelMessage(code: string | undefined, details: unknown): string | null {
  const detail = shortDetail(details);
  switch (code) {
    case undefined:
    case "":
    case "NoError":
      return null;
    case "AuthenticationFailure":
    case "Forbidden":
      return i18n.t("stt.error_azure_auth");
    case "ConnectionFailure":
    case "ServiceTimeout":
      return i18n.t("stt.error_azure_connection");
    case "TooManyRequests":
      return i18n.t("stt.error_azure_quota");
    case "BadRequestParameters":
      return i18n.t("stt.error_azure_bad_request", { detail: detail || code });
    default:
      return i18n.t("stt.error_azure_generic", { detail: detail || code });
  }
}

/**
 * Whether an Azure cancellation (`CancellationErrorCode` name) is a failure that may pass by itself: the connection
 * failed or timed out, or the service had an error. A rejected key, bad parameters or a used-up quota need the user.
 */
export function azureCancelRetryable(code: string | undefined): boolean {
  switch (code) {
    case "ConnectionFailure":
    case "ServiceTimeout":
    case "ServiceError":
    case "ServiceUnavailable":
      return true;
    default:
      return false;
  }
}

/**
 * How an Azure cancellation is reported: `retry` when it may pass (`azureCancelRetryable`), nothing when it needs the
 * user. A connection that failed or timed out before the session ever started (`started`) may be a wrong region
 * rather than an outage, and a browser cannot tell: it is `unproven` (Service_STT retries it on the fast ladder and
 * then stops with the message) unless the webview knows the network to be down. Pure.
 */
export function azureStopOptions(code: string | undefined, started: boolean, online: boolean): SttStopOptions | undefined {
  if (!azureCancelRetryable(code)) return undefined;
  const refusable = code === "ConnectionFailure" || code === "ServiceTimeout";
  return refusable && !started && online ? { retry: true, unproven: true } : { retry: true };
}

/**
 * Azure Speech in the webview (continuous recognition, optional second language through continuous language ID).
 * Interims and finals carry the detected language as `lang`.
 */
export class STT_AzureService implements ISTTService {
  startAbortSignal: AbortSignal | null = null;
  #instance?: SpeechRecognizer;
  /** Bumped by `start()` and `stop()`: an SDK that finishes loading for an older start creates nothing. */
  #token = 0;
  /** A partial is on screen (a `NoMatch` result then abandons it). */
  #interimOpen = false;
  /** The session reported its end (Azure follows a cancellation with a session stop). */
  #ended = false;

  constructor(private readonly bindings: ISTTReceiver) {}

  dispose(): void {}

  async start(state: STT_State): Promise<void> {
    const token = ++this.#token;
    const az = state.azure;
    if (isEmptyValue(az.key) || isEmptyValue(az.location) || isEmptyValue(az.language))
      return this.bindings.onStop(i18n.t("stt.error_azure_options"));

    this.bindings.onStatus(i18n.t("stt.status_connecting_engine", { engine: "Azure" }));
    let sdk: AzureSdk;
    try {
      sdk = await raceWithAbort(loadSdk(), this.startAbortSignal);
    } catch (error) {
      if (isAbortError(error) || token !== this.#token) return;
      this.bindings.onStatus("");
      return this.bindings.onStop(i18n.t("stt.error_azure_sdk", { detail: shortDetail(errorText(error)) }));
    }
    if (token !== this.#token) return;

    const speechConfig = sdk.SpeechConfig.fromSubscription(az.key, az.location);
    speechConfig.speechRecognitionLanguage = az.language;
    speechConfig.setProperty(sdk.PropertyId.SpeechServiceResponse_ProfanityOption, az.profanity);
    speechConfig.setProperty(sdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "60000");
    speechConfig.setProperty(
      sdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
      ((parseInt(az.silenceTimeout) || 20) * 1000).toString(),
    );
    const segmentationMs = azureSegmentationSilenceMs(az.segmentationSilenceMs);
    if (segmentationMs !== null) {
      // Main knob for how fast a phrase is finalized (EndSilenceTimeoutMs is deprecated).
      speechConfig.setProperty(sdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, segmentationMs.toString());
    }
    speechConfig.enableDictation();

    const langConfig = sdk.AutoDetectSourceLanguageConfig.fromLanguages(
      az.secondary_language && az.use_secondary_language ? [az.language, az.secondary_language] : [az.language],
    );
    langConfig.mode = sdk.LanguageIdMode.Continuous;
    const deviceId = az.device?.trim();
    const audioConfig =
      deviceId && deviceId !== "default"
        ? sdk.AudioConfig.fromMicrophoneInput(deviceId)
        : sdk.AudioConfig.fromDefaultMicrophoneInput();

    let recognizer: SpeechRecognizer;
    try {
      recognizer = sdk.SpeechRecognizer.FromConfig(speechConfig, langConfig, audioConfig);
    } catch (error) {
      this.bindings.onStatus("");
      return this.bindings.onStop(i18n.t("stt.error_azure_generic", { detail: shortDetail(errorText(error)) }));
    }
    this.#instance = recognizer;
    this.#interimOpen = false;
    this.#ended = false;
    const current = () => token === this.#token;
    /** The session reached the service (`sessionStarted`): a failure after that is an outage, not a refused setup. */
    let started = false;
    /** Reports the end of the session once; `options.retry`: a failure that may pass (Service_STT reconnects). */
    const end = (message: string, options?: SttStopOptions) => {
      if (this.#ended) return;
      this.#ended = true;
      this.bindings.onStatus("");
      if (options) this.bindings.onStop(message, options);
      else this.bindings.onStop(message);
    };
    const meta = (result: SpeechRecognitionResult): SttTextMeta | undefined => {
      try {
        const lang = baseLanguageTag(sdk.AutoDetectSourceLanguageResult.fromResult(result).language);
        return lang ? { lang } : undefined;
      } catch {
        return undefined;
      }
    };

    recognizer.sessionStarted = () => {
      if (!current()) return;
      started = true;
      this.bindings.onStart();
    };
    recognizer.sessionStopped = () => {
      // A stop of ours makes the session stale first, so this is the service or the connection ending it.
      if (current()) end(i18n.t("stt.error_azure_closed"), { retry: true });
    };
    recognizer.canceled = (_r, e: SpeechRecognitionCanceledEventArgs) => {
      if (!current()) return;
      const code = sdk.CancellationErrorCode[e.errorCode];
      const message = azureCancelMessage(code, e.errorDetails);
      // A cancellation without an error is the stream ending under a running session: it reconnects too.
      if (message === null) end(i18n.t("stt.error_azure_closed"), { retry: true });
      else end(message, azureStopOptions(code, started, networkLooksUp()));
    };
    recognizer.recognizing = (_r, e: SpeechRecognitionEventArgs) => {
      if (!current() || !az.interim || !e.result.text) return;
      this.#interimOpen = true;
      this.bindings.onInterim(e.result.text, meta(e.result));
    };
    recognizer.recognized = (_r, e: SpeechRecognitionEventArgs) => {
      if (!current()) return;
      const text = e.result.reason === sdk.ResultReason.RecognizedSpeech ? e.result.text?.trim() ?? "" : "";
      const open = this.#interimOpen;
      this.#interimOpen = false;
      if (text) this.bindings.onFinal(e.result.text, meta(e.result));
      else if (open) this.bindings.onInterim(""); // NoMatch: the open partial was not speech after all
    };
    recognizer.startContinuousRecognitionAsync(undefined, (error: string) => {
      if (!current()) return;
      this.bindings.onStatus("");
      this.bindings.onStop(i18n.t("stt.error_azure_generic", { detail: shortDetail(error) }));
    });
  }

  stop(): void {
    this.#token++;
    const recognizer = this.#instance;
    this.#instance = undefined;
    if (!recognizer) return;
    const close = () => {
      try {
        recognizer.close();
      } catch {
        /* already closed */
      }
      this.bindings.onStop();
    };
    recognizer.stopContinuousRecognitionAsync(close, close);
  }
}
