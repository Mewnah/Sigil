import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { floatSamplesToWavBlob } from "../encodeWav";
import { ISTTReceiver, ISpeechRecognitionService } from "../types";
import { STT_State } from "../schema";

interface AudioChunkPayload {
  samples: number[];
  sample_rate: number;
  channels: number;
}

const POLL_MS = 500;
const MIN_SAMPLES = 4000;

function transcriptionsUrl(baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/+$/, "");
  return `${trimmed}/audio/transcriptions`;
}

function truncate(s: string, max: number): string {
  const t = s.replace(/\s+/g, " ").trim();
  return t.length <= max ? t : `${t.slice(0, max)}…`;
}

export class STT_OpenAI_AudioService implements ISpeechRecognitionService {
  private isRunning = false;
  private unlistenAudio?: UnlistenFn;
  private audioBuffer: Float32Array[] = [];
  private processingInterval?: ReturnType<typeof setInterval>;

  constructor(private readonly receiver: ISTTReceiver) {}

  private async teardownPipeline(): Promise<void> {
    this.isRunning = false;
    if (this.processingInterval) {
      clearInterval(this.processingInterval);
      this.processingInterval = undefined;
    }
    try {
      await invoke("plugin:audio|stop_audio_capture");
    } catch {
      /* ignore */
    }
    if (this.unlistenAudio) {
      this.unlistenAudio();
      this.unlistenAudio = undefined;
    }
    this.audioBuffer = [];
  }

  async start(params: STT_State) {
    const config = params.openai_audio;
    const base = config.baseUrl.trim();
    const model = config.model.trim();

    if (!base) {
      this.receiver.onStop("Local transcription API: set a base URL (e.g. http://127.0.0.1:8000/v1).");
      return;
    }
    if (!model) {
      this.receiver.onStop("Local transcription API: set the model id your local server expects (see server docs).");
      return;
    }

    try {
      this.receiver.onInterim("Connecting to local transcription server...");

      this.unlistenAudio = await listen<AudioChunkPayload>("audio:chunk", (event) => {
        if (!this.isRunning) return;
        const { samples } = event.payload;
        this.audioBuffer.push(new Float32Array(samples));
      });

      this.isRunning = true;

      const deviceName = config.device || window.ApiServer.state.audioInputDevice || "";
      await invoke("plugin:audio|start_audio_capture", {
        deviceName: deviceName || null,
        sampleRate: 16000,
      });

      this.processingInterval = setInterval(() => {
        void this.processAudioBuffer();
      }, POLL_MS);

      this.receiver.onStart();
    } catch (error) {
      console.error("[Local HTTP STT] Error starting:", error);
      await this.teardownPipeline();
      this.receiver.onStop(String(error));
    }
  }

  private async processAudioBuffer() {
    if (!this.isRunning || this.audioBuffer.length === 0) return;

    const config = window.ApiServer.state.services.stt.data.openai_audio;
    const base = config.baseUrl.trim();
    const model = config.model.trim();
    if (!base || !model) return;

    const totalLength = this.audioBuffer.reduce((acc, chunk) => acc + chunk.length, 0);
    if (totalLength < MIN_SAMPLES) return;

    const combined = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.audioBuffer) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }
    this.audioBuffer = [];

    this.receiver.onInterim("...");

    const url = transcriptionsUrl(base);
    const blob = floatSamplesToWavBlob(combined, 16000);
    const form = new FormData();
    form.append("file", blob, "audio.wav");
    form.append("model", model);
    const lang = config.language?.trim();
    if (lang) {
      form.append("language", lang);
    }

    const headers: HeadersInit = {};
    if (config.apiKey.trim()) {
      headers.Authorization = `Bearer ${config.apiKey.trim()}`;
    }

    try {
      const res = await fetch(url, { method: "POST", body: form, headers });
      const raw = await res.text();
      if (!res.ok) {
        const hint = truncate(raw, 120);
        this.receiver.onInterim(
          hint ? `Local STT HTTP ${res.status}: ${hint}` : `Local STT HTTP ${res.status}`
        );
        return;
      }
      let text = "";
      try {
        const json = JSON.parse(raw) as { text?: string; error?: { message?: string } };
        if (json.error?.message) {
          this.receiver.onInterim(`Local STT: ${truncate(json.error.message, 200)}`);
          return;
        }
        text = (json.text ?? "").trim();
      } catch {
        this.receiver.onInterim("Local STT: server returned non-JSON response.");
        return;
      }
      if (text) {
        this.receiver.onFinal(text);
      }
    } catch (error) {
      console.error("[Local HTTP STT] Request failed:", error);
      this.receiver.onInterim(`Local STT: ${truncate(String(error), 150)}`);
    }
  }

  async stop() {
    if (!this.isRunning && !this.unlistenAudio && !this.processingInterval) {
      return;
    }
    await this.teardownPipeline();
    this.receiver.onStop();
  }

  dispose() {
    void this.stop();
  }
}
