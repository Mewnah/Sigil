import { fetchWithTimeout } from "@/utils/fetchWithTimeout";
import { devLog } from "@/utils/devLog";
import i18n from "i18next";
import { TTS_State } from "../schema";
import { ITTSReceiver, ITTSService, errorMessage, isAbortError } from "../types";
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
 * `POST /audio_query`, then `POST /synthesis` (WAV), played through Service_Sound on the chosen device.
 *
 * Lines are synthesized one at a time in arrival order, while the previous clip plays, so they
 * cannot be enqueued out of order. Stop aborts every request in flight and drops clips that have
 * not started playing.
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
        const message = isAbortError(error)
          ? i18n.t("tts.voicevox_timeout", { host: this.#host })
          : errorMessage(error);
        this.bindings.onError(`VOICEVOX: ${message}`);
      });
  }

  async #speak(text: string, generation: number): Promise<void> {
    if (generation !== this.#generation) return;
    const { speaker, device } = this.state;
    const host = this.#host;
    const speakerId = encodeURIComponent(speaker.trim());
    const controller = this.#track(new AbortController());
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
      if (generation !== this.#generation) return;

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
      if (generation !== this.#generation) return;

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
    } finally {
      this.#controllers.delete(controller);
    }
  }

  stop(): void {
    this.#halt();
    // Clips already synthesized but not yet playing are dropped; the one playing finishes.
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
