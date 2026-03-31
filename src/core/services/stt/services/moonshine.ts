import { invoke } from "@tauri-apps/api/core";
import { listen, UnlistenFn } from "@tauri-apps/api/event";
import { floatSamplesToWavBase64 } from "../encodeWav";
import { ISTTReceiver, ISpeechRecognitionService } from "../types";
import { STT_State } from "../schema";
import { devLog } from "@/utils/devLog";

interface AudioChunkPayload {
  samples: number[];
  sample_rate: number;
  channels: number;
}

export class STT_MoonshineService implements ISpeechRecognitionService {
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
    try {
      devLog("[Moonshine] Starting service...");
      this.receiver.onInterim("Connecting to Moonshine...");

      const config = params.moonshine;

      await invoke("plugin:moonshine-stt|set_moonshine_endpoint", {
        endpoint: config.endpoint,
      });

      const isAvailable = await invoke<boolean>("plugin:moonshine-stt|check_moonshine_availability");
      if (!isAvailable) {
        const setupInstructions = [
          `Moonshine server not found at ${config.endpoint}`,
          "",
          "To start Moonshine, run:",
          "docker run -p 8090:8090 useful-sensors/moonshine-onnx-server",
          "",
          "Or use the tiny model for faster startup:",
          "docker run -p 8090:8090 useful-sensors/moonshine-onnx-server:tiny",
        ].join("\n");

        this.receiver.onStop(setupInstructions);
        return;
      }

      devLog("[Moonshine] Server connected");

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
      }, 500);

      this.receiver.onStart();
      devLog("[Moonshine] Recording started");
    } catch (error) {
      console.error("[Moonshine] Error starting:", error);
      await this.teardownPipeline();
      this.receiver.onStop(String(error));
    }
  }

  private async processAudioBuffer() {
    if (!this.isRunning || this.audioBuffer.length === 0) return;

    const language = window.ApiServer.state.services.stt.data.moonshine.language;

    const totalLength = this.audioBuffer.reduce((acc, chunk) => acc + chunk.length, 0);
    if (totalLength < 4000) return;

    const combined = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.audioBuffer) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }
    this.audioBuffer = [];

    this.receiver.onInterim("...");

    const audioBase64 = floatSamplesToWavBase64(combined, 16000);

    try {
      const text = await invoke<string>("plugin:moonshine-stt|moonshine_transcribe", {
        audioBase64,
        language: language || null,
      });

      if (text.trim()) {
        devLog("[Moonshine] Result:", text);
        this.receiver.onFinal(text);
      }
    } catch (error) {
      console.error("[Moonshine] Transcription error:", error);
      const msg = String(error);
      this.receiver.onInterim(
        msg.length > 180 ? `Moonshine: ${msg.slice(0, 180)}…` : `Moonshine: ${msg}`
      );
    }
  }

  async stop() {
    if (!this.isRunning && !this.unlistenAudio && !this.processingInterval) return;

    devLog("[Moonshine] Stopping...");
    await this.teardownPipeline();
    this.receiver.onStop();
    devLog("[Moonshine] Stopped");
  }

  dispose() {
    void this.stop();
  }
}
