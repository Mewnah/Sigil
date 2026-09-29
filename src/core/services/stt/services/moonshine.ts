import { invoke } from "@tauri-apps/api/core";
import { floatSamplesToWavBase64 } from "../encodeWav";
import { STT_State } from "../schema";
import { devLog } from "@/utils/devLog";
import { raceWithAbort } from "../abort";
import { RUST_CAPTURE_SAMPLE_RATE, RustCapturePollingService } from "../rustCapturePolling";

export class STT_MoonshineService extends RustCapturePollingService {
  protected readonly logTag = "[Moonshine]";
  protected readonly errorLabel = "Moonshine";

  protected inputDevice(params: STT_State): string {
    return params.moonshine.device;
  }

  protected async prepare(params: STT_State): Promise<string | null> {
    devLog("[Moonshine] Starting service...");
    this.receiver.onInterim("Connecting to Moonshine...");

    const { endpoint } = params.moonshine;
    await raceWithAbort(
      invoke("plugin:moonshine-stt|set_moonshine_endpoint", { endpoint }),
      this.startAbortSignal,
    );
    if (this.lifecycle.aborted) return null;

    const isAvailable = await raceWithAbort(
      invoke<boolean>("plugin:moonshine-stt|check_moonshine_availability"),
      this.startAbortSignal,
    );
    if (!isAvailable) {
      return [
        `Moonshine server not found at ${endpoint}`,
        "",
        "To start Moonshine, run:",
        "docker run -p 8090:8090 useful-sensors/moonshine-onnx-server",
        "",
        "Or use the tiny model for faster startup:",
        "docker run -p 8090:8090 useful-sensors/moonshine-onnx-server:tiny",
      ].join("\n");
    }

    devLog("[Moonshine] Server connected");
    return null;
  }

  protected async transcribe(samples: Float32Array, signal: AbortSignal): Promise<string> {
    const language = window.ApiServer.state.services.stt.data.moonshine.language;
    // The IPC call cannot be cancelled; racing it lets teardown discard a late result immediately.
    return raceWithAbort(
      invoke<string>("plugin:moonshine-stt|moonshine_transcribe", {
        audioBase64: floatSamplesToWavBase64(samples, RUST_CAPTURE_SAMPLE_RATE),
        language: language || null,
      }),
      signal,
    );
  }
}
