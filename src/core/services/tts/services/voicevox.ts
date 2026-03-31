import { fetchWithTimeout } from "@/utils/fetchWithTimeout";
import { devLog } from "@/utils/devLog";
import { TTS_State } from "../schema";
import { ITTSReceiver, ITTSService } from "../types";

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

export class TTS_VoicevoxService implements ITTSService {
  #abortController?: AbortController;

  constructor(private bindings: ITTSReceiver) { }

  dispose(): void {
    this.#abortController?.abort();
  }

  get state() {
    return window.ApiServer.state.services.tts.data.voicevox;
  }

  async start(state: TTS_State): Promise<void> {
    const { host } = state.voicevox;

    try {
      // Verify VoiceVox server is running
      const response = await fetchWithTimeout(`${host}/version`, {
        method: "GET",
        timeoutMs: 15_000,
      });

      if (!response.ok) {
        throw new Error(`VoiceVox server not responding (status ${response.status})`);
      }

      const version = await response.text();
      devLog(`[VoiceVox] Connected to server version: ${version}`);

      this.bindings.onStart();
    } catch (error: any) {
      if (error.name === "TypeError" && error.message.includes("fetch")) {
        this.bindings.onStop("VoiceVox server not running. Please start VoiceVox engine.");
      } else {
        this.bindings.onStop(error?.message || "Failed to connect to VoiceVox");
      }
    }
  }

  async play(text: string): Promise<void> {
    if (!text.trim()) return;

    const { host, speaker, speedScale, pitchScale, intonationScale, volumeScale, device } = this.state;

    try {
      this.#abortController = new AbortController();
      const signal = this.#abortController.signal;

      // Step 1: Create audio query
      const queryResponse = await fetchWithTimeout(
        `${host}/audio_query?text=${encodeURIComponent(text)}&speaker=${speaker}`,
        {
          method: "POST",
          signal,
          timeoutMs: 120_000,
        }
      );

      if (!queryResponse.ok) {
        throw new Error(`Audio query failed: ${queryResponse.status}`);
      }

      const audioQuery: AudioQuery = await queryResponse.json();

      // Apply user settings to the query
      audioQuery.speedScale = parseFloat(speedScale) || 1.0;
      audioQuery.pitchScale = parseFloat(pitchScale) || 0.0;
      audioQuery.intonationScale = parseFloat(intonationScale) || 1.0;
      audioQuery.volumeScale = parseFloat(volumeScale) || 1.0;

      // Step 2: Synthesize audio
      const synthesisResponse = await fetchWithTimeout(
        `${host}/synthesis?speaker=${speaker}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(audioQuery),
          signal,
          timeoutMs: 120_000,
        }
      );

      if (!synthesisResponse.ok) {
        throw new Error(`Synthesis failed: ${synthesisResponse.status}`);
      }

      // Get WAV audio data
      const audioData = await synthesisResponse.arrayBuffer();

      // Enqueue for playback
      window.ApiServer.sound.enqueueVoiceClip(audioData, {
        volume: 1, // Volume is already applied in VoiceVox
        rate: 1,   // Speed is already applied in VoiceVox
        device_name: device,
      });

    } catch (error: any) {
      if (error.name === "AbortError") {
        // Playback was cancelled, ignore
        return;
      }
      console.error("[VoiceVox] Synthesis error:", error);
    }
  }

  stop(): void {
    this.#abortController?.abort();
    this.bindings.onStop();
  }
}

// Utility function to fetch speakers from VoiceVox
export async function fetchVoicevoxSpeakers(host: string): Promise<VoicevoxSpeaker[]> {
  try {
    const response = await fetchWithTimeout(`${host}/speakers`, { timeoutMs: 15_000 });
    if (!response.ok) {
      throw new Error(`Failed to fetch speakers: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("[VoiceVox] Failed to fetch speakers:", error);
    return [];
  }
}
