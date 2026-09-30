import i18n from "i18next";
import type {
  Connection,
  SpeechSynthesisResult,
  SpeechSynthesizer,
} from "microsoft-cognitiveservices-speech-sdk";
import { isEmptyValue } from "../../../../utils";
import { pcm16leToFloat32 } from "../../sound/pcm";
import { clampedOr, positiveOr } from "../num";
import { TTS_State } from "../schema";
import { azureVoiceLocale, buildAzureSsml, normalizeAzureVoiceName } from "../ssml";
import { ITTSReceiver, ITTSService, PcmChunk, TTS_OPTIONS_MISSING, errorMessage } from "../types";

type AzureSdk = typeof import("microsoft-cognitiveservices-speech-sdk");

/**
 * The Speech SDK is loaded on first start (it is large, and its CommonJS bundle has a circular
 * init that must not be pulled into the app's startup graph).
 */
let sdkPromise: Promise<AzureSdk> | null = null;
function loadSdk(): Promise<AzureSdk> {
  if (!sdkPromise) {
    sdkPromise = import("microsoft-cognitiveservices-speech-sdk").catch((error) => {
      sdkPromise = null;
      throw error;
    });
  }
  return sdkPromise;
}

/** Upper bound for opening the synthesis connection in start(). */
export const AZURE_CONNECT_TIMEOUT_MS = 15_000;

/** Output: raw 16-bit mono PCM at this rate, streamed to the sink as Azure synthesizes it. */
export const AZURE_PCM_RATE = 48_000;

type ConnectOutcome = { ok: true } | { ok: false; timedOut: boolean; error: string };

/**
 * Open the synthesizer's WebSocket once, so a wrong key or region fails Start instead of every
 * line. In the webview the handshake rejection carries no HTTP status (the SDK reports close code
 * 1006), so any failure here means "could not connect with this key and region".
 */
function openSynthesisConnection(sdk: AzureSdk, synth: SpeechSynthesizer, signal: AbortSignal): Promise<ConnectOutcome | null> {
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
      connection = sdk.Connection.fromSynthesizer(synth);
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

/** Audio chunks of one synthesis request, as an async iterable (filled by SDK callbacks). */
class ChunkQueue implements AsyncIterable<ArrayBuffer> {
  #items: ArrayBuffer[] = [];
  #closed = false;
  #error: Error | null = null;
  #wake: (() => void) | null = null;
  /** Whether any chunk was streamed (else the final result's audio is used). */
  streamed = false;

  push(bytes: ArrayBuffer) {
    if (this.#closed || !bytes?.byteLength) return;
    this.streamed = true;
    this.#items.push(bytes);
    this.#notify();
  }

  close() {
    this.#closed = true;
    this.#notify();
  }

  fail(error: Error) {
    if (this.#closed) return;
    this.#error = error;
    this.close();
  }

  #notify() {
    const wake = this.#wake;
    this.#wake = null;
    wake?.();
  }

  async *[Symbol.asyncIterator](): AsyncIterator<ArrayBuffer> {
    for (;;) {
      const next = this.#items.shift();
      if (next) {
        yield next;
        continue;
      }
      if (this.#error) throw this.#error;
      if (this.#closed) return;
      await new Promise<void>((resolve) => {
        this.#wake = resolve;
      });
    }
  }
}

/**
 * Azure neural voices via the Speech SDK (SSML in, raw 48 kHz PCM out).
 *
 * - PCM contract (ITTSService v2): {@link synthesizePcm} streams each sentence's audio as Azure synthesizes it
 *   (the SDK's `synthesizing` events), so playback starts before the sentence is complete.
 * - Legacy {@link play}: whole lines through Service_Sound.
 * - Requests run one at a time on the one synthesizer (the SDK queues them anyway), which is what routes each
 *   streamed chunk to its sentence.
 */
export class TTS_AzureService implements ITTSService {
  #sdk?: AzureSdk;
  #instance?: SpeechSynthesizer;
  /** Aborts the connection check of a start() still in progress. */
  #connecting?: AbortController;
  /** One request at a time. */
  #chain: Promise<void> = Promise.resolve();
  /** The request whose streamed chunks arrive now (null for legacy whole-clip requests). */
  #active: ChunkQueue | null = null;

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

    const connecting = new AbortController();
    this.#connecting = connecting;
    let sdk: AzureSdk;
    try {
      sdk = await loadSdk();
    } catch (error) {
      if (connecting.signal.aborted) return;
      this.#connecting = undefined;
      return this.bindings.onStop(`Azure: ${errorMessage(error)}`);
    }
    if (connecting.signal.aborted) return; // stopped while loading

    let instance: SpeechSynthesizer;
    try {
      const speechConfig = sdk.SpeechConfig.fromSubscription(key.trim(), region);
      speechConfig.speechSynthesisVoiceName = voice;
      // `language` stores the display group ("English (United States)"), not a locale.
      const locale = azureVoiceLocale(voice);
      if (locale) speechConfig.speechSynthesisLanguage = locale;
      speechConfig.speechSynthesisOutputFormat = sdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm;
      // No audio config: the SDK must not play by itself; audio goes to the TTS sink instead.
      instance = new sdk.SpeechSynthesizer(speechConfig, null);
      instance.synthesizing = (_sender, event) => {
        if (this.#instance === instance) this.#active?.push(event.result.audioData);
      };
      this.#sdk = sdk;
      this.#instance = instance;
    } catch (error) {
      this.#close();
      return this.bindings.onStop(errorMessage(error));
    }

    const outcome = await openSynthesisConnection(sdk, instance, connecting.signal);
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

  #ssml(value: string): string {
    return buildAzureSsml(value, {
      voice: this.state.voice,
      style: this.state.voiceStyle,
      role: this.state.voiceRole,
      volume: this.state.voiceVolume,
      rate: this.state.voiceRate,
      pitch: this.state.voicePitch,
      range: this.state.voiceRange,
    });
  }

  /**
   * Run one synthesis request after the previous one. `queue` receives streamed chunks (null: legacy, whole
   * result only). Resolves with the result's audio, or `null` when stopped / cancelled; rejects with a readable
   * message when Azure refused this line. A fatal failure also stops the backend.
   */
  #request(ssml: string, queue: ChunkQueue | null): Promise<ArrayBuffer | null> {
    const instance = this.#instance;
    const sdk = this.#sdk;
    const run = this.#chain.then(
      () =>
        new Promise<ArrayBuffer | null>((resolve, reject) => {
          if (!instance || !sdk || this.#instance !== instance) return resolve(null);
          this.#active = queue;
          instance.speakSsmlAsync(
            ssml,
            (result) => {
              if (this.#active === queue) this.#active = null;
              if (this.#instance !== instance) return resolve(null); // stopped meanwhile
              const failure = this.#failure(sdk, result);
              if (failure) return reject(new Error(failure));
              resolve(result.audioData?.byteLength ? result.audioData : null);
            },
            (error) => {
              if (this.#active === queue) this.#active = null;
              if (this.#instance !== instance) return resolve(null);
              // Service errors arrive as Canceled results above; this callback means the synthesizer threw
              // (and the SDK disposed it), so it cannot speak again.
              this.#close();
              this.bindings.onStop(`Azure: ${errorMessage(error)}`);
              resolve(null);
            },
          );
        }),
    );
    this.#chain = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  /** The error of a cancelled result (stopping the backend when it is fatal), or null when it succeeded. */
  #failure(sdk: AzureSdk, result: SpeechSynthesisResult): string | null {
    if (result.reason === sdk.ResultReason.SynthesizingAudioCompleted) return null;
    if (result.reason !== sdk.ResultReason.Canceled) return null;
    const details = sdk.CancellationDetails.fromResult(result);
    const message = `Azure: ${details.errorDetails || sdk.CancellationErrorCode[details.ErrorCode]}`;
    // The SDK (1.48) raises these only on its recognition path; for synthesis a rejected key or an
    // unknown region shows up as ConnectionFailure ("StatusCode: 1006"), which is why start() opens
    // the connection before reporting onStart. Kept in case a later SDK reports them here too.
    const fatal =
      details.ErrorCode === sdk.CancellationErrorCode.AuthenticationFailure ||
      details.ErrorCode === sdk.CancellationErrorCode.Forbidden;
    if (fatal) {
      this.#close();
      this.bindings.onStop(message);
    }
    return message;
  }

  /** v2: stream one sentence as PCM while Azure synthesizes it. */
  async *synthesizePcm(text: string, signal: AbortSignal): AsyncIterable<PcmChunk> {
    if (!this.#instance || signal.aborted) return;
    const ssml = this.#ssml(text);
    if (!ssml) return;
    const queue = new ChunkQueue();
    const done = this.#request(ssml, queue).then(
      (audio) => {
        // No streamed chunks (older SDK behaviour): use the whole result.
        if (!queue.streamed && audio) queue.push(audio);
        queue.close();
      },
      (error: Error) => queue.fail(error),
    );
    const onAbort = () => queue.close();
    signal.addEventListener("abort", onAbort, { once: true });
    let carry: number | null = null;
    try {
      for await (const bytes of queue) {
        if (signal.aborted) return;
        const converted = pcm16leToFloat32(bytes, carry);
        carry = converted.carry;
        if (converted.samples.length) yield { samples: converted.samples, sampleRate: AZURE_PCM_RATE, channels: 1 };
      }
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
    // Cancelled: settle now. The SDK's callback may come late (or never); `done` handles it either way.
    if (signal.aborted) return;
    await done;
  }

  play(value: string): void {
    if (!this.#instance) return;
    const ssml = this.#ssml(value);
    if (!ssml) return;
    this.#request(ssml, null).then(
      (audio) => {
        if (!audio || !this.#instance) return;
        const { samples } = pcm16leToFloat32(audio);
        window.ApiServer.sound
          .enqueuePcm(
            { samples, sampleRate: AZURE_PCM_RATE, channels: 1 },
            {
              volume: clampedOr(this.state.volume, 1, 0, 1),
              rate: positiveOr(this.state.rate, 1),
              device_name: this.state.device,
            },
          )
          .catch((error) => {
            if (this.#instance) this.bindings.onError(errorMessage(error));
          });
      },
      (error) => {
        if (this.#instance) this.bindings.onError(errorMessage(error));
      },
    );
  }

  stop(): void {
    this.#close();
    // Clips queued or playing through Service_Sound stop too.
    window.ApiServer.sound.clearVoiceClips();
    this.bindings.onStop();
  }

  #close() {
    this.#connecting?.abort();
    this.#connecting = undefined;
    this.#active?.close();
    this.#active = null;
    const instance = this.#instance;
    this.#instance = undefined;
    try {
      instance?.close();
    } catch {
      // already disposed
    }
  }
}
