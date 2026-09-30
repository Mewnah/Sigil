import { fetchWithTimeout } from "@/utils/fetchWithTimeout";
import { devLog } from "@/utils/devLog";
import i18n from "i18next";
import { decodeWav } from "../../sound/pcm";
import { TTS_State } from "../schema";
import { ITTSReceiver, ITTSService, PcmChunk, errorMessage, isAbortError } from "../types";
import { normalizeVoicevoxHost, voicevoxHostProblem, voicevoxScales } from "../voicevoxPresets";

interface VoicevoxSpeaker {
  name: string;
  speaker_uuid: string;
  styles: { name: string; id: number }[];
}

interface AudioQuery {
  accent_phrases: unknown[];
  speedScale: number;
  pitchScale: number;
  intonationScale: number;
  volumeScale: number;
  prePhonemeLength: number;
  postPhonemeLength: number;
  outputSamplingRate: number;
  outputStereo: boolean;
  kana: string;
}

const CONNECT_TIMEOUT_MS = 15_000;
const SYNTHESIS_TIMEOUT_MS = 120_000;

/** Localized message for a host the webview cannot reach, or `null` when the host is usable. */
export function voicevoxHostError(host: string): string | null {
  const problem = voicevoxHostProblem(host);
  if (!problem) return null;
  return i18n.t(`tts.voicevox_host_error_${problem}`, { host });
}

/**
 * VOICEVOX-compatible engines (VOICEVOX, AivisSpeech, or a custom host) over their HTTP API:
 * `POST /audio_query`, then `POST /synthesis` (WAV).
 *
 * - PCM contract (ITTSService v2): {@link synthesizePcm} synthesizes one sentence and yields it as
 *   PCM; the sentence pipeline plays it through the TTS sink (device / voice changer), so a long text
 *   starts speaking after its first sentence.
 * - Legacy {@link play}: whole lines, one at a time in arrival order while the previous clip plays,
 *   played through Service_Sound (itself on the sink).
 * - Stop aborts every request in flight and drops clips that have not started playing.
 */
export class TTS_VoicevoxService implements ITTSService {
  #controllers = new Set<AbortController>();
  #chain: Promise<void> = Promise.resolve();
  #host = "";
  #running = false;
  #stopped = false;
  /** Bumped by stop(): lines queued before it are dropped. */
  #generation = 0;

  constructor(private bindings: ITTSReceiver) { }

  dispose(): void {
    this.#halt();
  }

  get state() {
    return window.ApiServer.state.services.tts.data.voicevox;
  }

  async start(state: TTS_State): Promise<void> {
    const host = normalizeVoicevoxHost(state.voicevox.host);
    const hostError = voicevoxHostError(host);
    if (hostError) return this.bindings.onStop(hostError);

    const controller = this.#track(new AbortController());
    try {
      const response = await fetchWithTimeout(`${host}/version`, {
        method: "GET",
        signal: controller.signal,
        timeoutMs: CONNECT_TIMEOUT_MS,
      });
      if (this.#stopped) return;
      if (!response.ok) {
        throw new Error(i18n.t("tts.voicevox_not_responding", { host, status: response.status }));
      }

      const version = await response.text();
      if (this.#stopped) return;
      devLog(`[VoiceVox] Connected to ${host}, engine version: ${version}`);

      this.#host = host;
      this.#running = true;
      this.bindings.onStart();
    } catch (error: unknown) {
      if (this.#stopped) return;
      if (isAbortError(error)) {
        // Not our Stop, so the connect timeout fired.
        this.bindings.onStop(i18n.t("tts.voicevox_timeout", { host }));
      } else if (error instanceof TypeError) {
        // fetch() rejects with TypeError when nothing listens on the port (or the CSP blocks it).
        this.bindings.onStop(i18n.t("tts.voicevox_not_running", { host }));
      } else {
        this.bindings.onStop(errorMessage(error) || i18n.t("tts.voicevox_not_running", { host }));
      }
    } finally {
      this.#controllers.delete(controller);
    }
  }

  /** v2: one sentence as PCM (the engine's WAV, decoded). Aborting `signal` cancels the requests. */
  async *synthesizePcm(text: string, signal: AbortSignal): AsyncIterable<PcmChunk> {
    if (!this.#running || !text.trim() || signal.aborted) return;
    let wav: ArrayBuffer | null;
    try {
      wav = await this.#synthesizeWav(text, signal);
    } catch (error) {
      if (signal.aborted || !this.#running) return;
      throw new Error(`VOICEVOX: ${this.#describe(error)}`);
    }
    if (!wav || signal.aborted) return;
    let pcm;
    try {
      pcm = decodeWav(wav);
    } catch (error) {
      throw new Error(`VOICEVOX: ${errorMessage(error)}`);
    }
    yield { samples: pcm.samples, sampleRate: pcm.sampleRate, channels: pcm.channels };
  }

  play(text: string): void {
    if (!this.#running || !text.trim()) return;
    const generation = this.#generation;
    this.#chain = this.#chain
      .then(() => this.#speak(text, generation))
      .catch((error) => {
        // Stop bumps the generation before aborting, so an AbortError from the current generation
        // is fetchWithTimeout's per-request limit firing, not a Stop.
        if (generation !== this.#generation) return;
        console.error("[VoiceVox] Synthesis error:", error);
        this.bindings.onError(`VOICEVOX: ${this.#describe(error)}`);
      });
  }

  /** Readable message for a failed request (an abort here is the per-request time limit). */
  #describe(error: unknown): string {
    return isAbortError(error) ? i18n.t("tts.voicevox_timeout", { host: this.#host }) : errorMessage(error);
  }

  async #speak(text: string, generation: number): Promise<void> {
    if (generation !== this.#generation) return;
    const { device } = this.state;
    const audioData = await this.#synthesizeWav(text);
    if (!audioData || generation !== this.#generation) return;

    // Playback runs on its own queue so the next line can synthesize meanwhile.
    window.ApiServer.sound
      .enqueueVoiceClip(audioData, {
        volume: 1, // volumeScale is already applied by the engine
        rate: 1, // speedScale is already applied by the engine
        device_name: device,
      })
      .catch((error) => {
        if (generation === this.#generation) this.bindings.onError(errorMessage(error));
      });
  }

  /**
   * `/audio_query` then `/synthesis` for `text`: the WAV, or `null` when stopped meanwhile. Rejects with an
   * AbortError when the per-request limit fired (or `outer` aborted), else a readable error.
   */
  async #synthesizeWav(text: string, outer?: AbortSignal): Promise<ArrayBuffer | null> {
    const generation = this.#generation;
    const { speaker } = this.state;
    const host = this.#host;
    const speakerId = encodeURIComponent(speaker.trim());
    const controller = this.#track(new AbortController());
    const onOuterAbort = () => controller.abort();
    if (outer?.aborted) controller.abort();
    outer?.addEventListener("abort", onOuterAbort, { once: true });
    try {
      // Step 1: audio query (accent phrases and default prosody for this speaker)
      const queryResponse = await fetchWithTimeout(
        `${host}/audio_query?text=${encodeURIComponent(text)}&speaker=${speakerId}`,
        { method: "POST", signal: controller.signal, timeoutMs: SYNTHESIS_TIMEOUT_MS },
      );
      if (!queryResponse.ok) {
        throw new Error(i18n.t("tts.voicevox_query_failed", { status: queryResponse.status }));
      }
      const audioQuery: AudioQuery = await queryResponse.json();
      if (generation !== this.#generation) return null;

      // User settings; 0 is a valid pitch/intonation/volume and survives.
      Object.assign(audioQuery, voicevoxScales(this.state));

      // Step 2: synthesize WAV
      const synthesisResponse = await fetchWithTimeout(`${host}/synthesis?speaker=${speakerId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(audioQuery),
        signal: controller.signal,
        timeoutMs: SYNTHESIS_TIMEOUT_MS,
      });
      if (!synthesisResponse.ok) {
        throw new Error(i18n.t("tts.voicevox_synthesis_failed", { status: synthesisResponse.status }));
      }
      const audioData = await synthesisResponse.arrayBuffer();
      if (generation !== this.#generation) return null;
      return audioData;
    } finally {
      this.#controllers.delete(controller);
      outer?.removeEventListener("abort", onOuterAbort);
    }
  }

  stop(): void {
    this.#halt();
    // Clips queued or playing through Service_Sound stop too.
    window.ApiServer.sound.clearVoiceClips();
    this.bindings.onStop();
  }

  #track(controller: AbortController): AbortController {
    this.#controllers.add(controller);
    return controller;
  }

  #halt() {
    this.#stopped = true;
    this.#running = false;
    this.#generation++;
    for (const controller of this.#controllers) controller.abort();
    this.#controllers.clear();
  }
}

/** Speakers (and their styles) from the engine at `host`. Rejects when the engine cannot be reached. */
export async function fetchVoicevoxSpeakers(host: string): Promise<VoicevoxSpeaker[]> {
  const base = normalizeVoicevoxHost(host);
  const hostError = voicevoxHostError(base);
  if (hostError) throw new Error(hostError);
  let response: Response;
  try {
    response = await fetchWithTimeout(`${base}/speakers`, { timeoutMs: CONNECT_TIMEOUT_MS });
  } catch (error) {
    if (isAbortError(error)) throw new Error(i18n.t("tts.voicevox_timeout", { host: base }));
    if (error instanceof TypeError) throw new Error(i18n.t("tts.voicevox_not_running", { host: base }));
    throw error;
  }
  if (!response.ok) {
    throw new Error(i18n.t("tts.voicevox_speakers_failed", { status: response.status }));
  }
  const speakers: unknown = await response.json();
  return Array.isArray(speakers) ? (speakers as VoicevoxSpeaker[]) : [];
}
