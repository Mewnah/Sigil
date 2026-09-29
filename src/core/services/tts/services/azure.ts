import i18n from "i18next";
import {
  CancellationDetails,
  CancellationErrorCode,
  Connection,
  ResultReason,
  SpeechConfig,
  SpeechSynthesisOutputFormat,
  SpeechSynthesisResult,
  SpeechSynthesizer
} from "microsoft-cognitiveservices-speech-sdk";
import { isEmptyValue } from "../../../../utils";
import { clampedOr, positiveOr } from "../num";
import { TTS_State } from "../schema";
import { azureVoiceLocale, buildAzureSsml, normalizeAzureVoiceName } from "../ssml";
import { ITTSReceiver, ITTSService, TTS_OPTIONS_MISSING, errorMessage } from "../types";

/**
 * Cancellations that will fail the same way for every line: stop instead of failing each one.
 * The Speech SDK (1.48) raises these only on its recognition path; for synthesis a rejected key or
 * an unknown region shows up as ConnectionFailure ("StatusCode: 1006"), which is why start()
 * opens the connection before reporting onStart. Kept in case a later SDK reports them here too.
 */
const FATAL_CANCELLATIONS = new Set<CancellationErrorCode>([
  CancellationErrorCode.AuthenticationFailure,
  CancellationErrorCode.Forbidden,
]);

/** Upper bound for opening the synthesis connection in start(). */
export const AZURE_CONNECT_TIMEOUT_MS = 15_000;

type ConnectOutcome = { ok: true } | { ok: false; timedOut: boolean; error: string };

/**
 * Open the synthesizer's WebSocket once, so a wrong key or region fails Start instead of every
 * line. In the webview the handshake rejection carries no HTTP status (the SDK reports close code
 * 1006), so any failure here means "could not connect with this key and region".
 */
function openSynthesisConnection(synth: SpeechSynthesizer, signal: AbortSignal): Promise<ConnectOutcome | null> {
  return new Promise((resolve) => {
    let connection: Connection | undefined;
    let settled = false;
    const finish = (outcome: ConnectOutcome | null) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
      try {
        connection?.close();
      } catch {
        // nothing to release
      }
      resolve(outcome);
    };
    const onAbort = () => finish(null);
    const timer = setTimeout(() => finish({ ok: false, timedOut: true, error: "" }), AZURE_CONNECT_TIMEOUT_MS);
    if (signal.aborted) return finish(null);
    signal.addEventListener("abort", onAbort, { once: true });
    try {
      connection = Connection.fromSynthesizer(synth);
      connection.openConnection(
        () => finish({ ok: true }),
        (error) => finish({ ok: false, timedOut: false, error: errorMessage(error) }),
      );
    } catch (error) {
      finish({ ok: false, timedOut: false, error: errorMessage(error) });
    }
  });
}

/** The SDK's connect error spans several indented lines; keep it on one. */
function oneLine(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

/** Azure neural voices via the Speech SDK (SSML in, MP3 out); clips play through Service_Sound. */
export class TTS_AzureService implements ITTSService {
  #instance?: SpeechSynthesizer;
  /** Aborts the connection check of a start() still in progress. */
  #connecting?: AbortController;

  constructor(private bindings: ITTSReceiver) {}

  dispose(): void {
    this.#close();
  }

  async start(state: TTS_State): Promise<void> {
    const { key, location } = state.azure;
    const voice = normalizeAzureVoiceName(state.azure.voice);
    // Style, role, prosody and device are optional (empty device = the app's output device).
    if ([key, location, voice].some(isEmptyValue)) return this.bindings.onStop(TTS_OPTIONS_MISSING);
    const region = location.trim();

    let instance: SpeechSynthesizer;
    try {
      const speechConfig = SpeechConfig.fromSubscription(key.trim(), region);
      speechConfig.speechSynthesisVoiceName = voice;
      // `language` stores the display group ("English (United States)"), not a locale.
      const locale = azureVoiceLocale(voice);
      if (locale) speechConfig.speechSynthesisLanguage = locale;
      speechConfig.speechSynthesisOutputFormat = SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3;
      // No audio config: the SDK must not play by itself; clips go to the selected device instead.
      instance = new SpeechSynthesizer(speechConfig, null);
      this.#instance = instance;
    } catch (error) {
      this.#close();
      return this.bindings.onStop(errorMessage(error));
    }

    const connecting = new AbortController();
    this.#connecting = connecting;
    const outcome = await openSynthesisConnection(instance, connecting.signal);
    if (this.#connecting === connecting) this.#connecting = undefined;
    if (!outcome || this.#instance !== instance) return; // stopped meanwhile
    if (outcome.ok) return this.bindings.onStart();

    this.#close();
    const message = outcome.timedOut
      ? i18n.t("tts.azure_connect_timeout", { region })
      : i18n.t("tts.azure_connect_failed", { region, error: oneLine(outcome.error) });
    this.bindings.onStop(`Azure: ${message}`);
  }

  get state() {
    return window.ApiServer.state.services.tts.data.azure;
  }

  play(value: string): void {
    const instance = this.#instance;
    if (!instance) return;
    const ssml = buildAzureSsml(value, {
      voice: this.state.voice,
      style: this.state.voiceStyle,
      role: this.state.voiceRole,
      volume: this.state.voiceVolume,
      rate: this.state.voiceRate,
      pitch: this.state.voicePitch,
      range: this.state.voiceRange,
    });
    if (!ssml) return;

    instance.speakSsmlAsync(
      ssml,
      (result) => {
        if (this.#instance !== instance) return; // stopped meanwhile
        this.#handleResult(result);
      },
      (error) => {
        if (this.#instance !== instance) return;
        // Service errors arrive as Canceled results above; this callback means the synthesizer threw
        // (and the SDK disposed it), so it cannot speak again.
        this.#close();
        this.bindings.onStop(`Azure: ${errorMessage(error)}`);
      },
    );
  }

  #handleResult(result: SpeechSynthesisResult) {
    if (result.reason === ResultReason.SynthesizingAudioCompleted) {
      if (!result.audioData?.byteLength) return;
      window.ApiServer.sound
        .enqueueVoiceClip(result.audioData, {
          volume: clampedOr(this.state.volume, 1, 0, 1),
          rate: positiveOr(this.state.rate, 1),
          device_name: this.state.device,
        })
        .catch((error) => {
          if (this.#instance) this.bindings.onError(errorMessage(error));
        });
      return;
    }
    if (result.reason === ResultReason.Canceled) {
      const details = CancellationDetails.fromResult(result);
      const message = `Azure: ${details.errorDetails || CancellationErrorCode[details.ErrorCode]}`;
      if (FATAL_CANCELLATIONS.has(details.ErrorCode)) {
        this.#close();
        this.bindings.onStop(message);
      } else {
        this.bindings.onError(message);
      }
    }
  }

  stop(): void {
    this.#close();
    // Clips already synthesized but not yet playing are dropped; the one playing finishes.
    window.ApiServer.sound.clearVoiceClips();
    this.bindings.onStop();
  }

  #close() {
    this.#connecting?.abort();
    this.#connecting = undefined;
    const instance = this.#instance;
    this.#instance = undefined;
    try {
      instance?.close();
    } catch {
      // already disposed
    }
  }
}
