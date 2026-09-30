import type {
  SpeechRecognitionCanceledEventArgs,
  SpeechRecognitionEventArgs,
  SpeechRecognitionResult,
  SpeechRecognizer,
} from "microsoft-cognitiveservices-speech-sdk";
import i18n from "i18next";
import { isEmptyValue } from "../../../../utils";
import { isAbortError, raceWithAbort } from "../abort";
import { baseLanguageTag, errorText, shortDetail } from "../rustCapturePolling";
import { STT_State } from "../schema";
import { azureSegmentationSilenceMs } from "../sttOptions";
import { ISTTReceiver, ISTTService, SttTextMeta } from "../types";

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
    const current = () => token === this.#token;
    const meta = (result: SpeechRecognitionResult): SttTextMeta | undefined => {
      try {
        const lang = baseLanguageTag(sdk.AutoDetectSourceLanguageResult.fromResult(result).language);
        return lang ? { lang } : undefined;
      } catch {
        return undefined;
      }
    };

    recognizer.sessionStarted = () => {
      if (current()) this.bindings.onStart();
    };
    recognizer.sessionStopped = () => {
      if (current()) this.bindings.onStop();
    };
    recognizer.canceled = (_r, e: SpeechRecognitionCanceledEventArgs) => {
      if (!current()) return;
      const message = azureCancelMessage(sdk.CancellationErrorCode[e.errorCode], e.errorDetails);
      this.bindings.onStatus("");
      this.bindings.onStop(message ?? undefined);
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
