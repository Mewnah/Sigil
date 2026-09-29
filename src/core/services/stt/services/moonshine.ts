import { invoke } from "@tauri-apps/api/core";
import i18n from "i18next";
import { floatSamplesToWavBase64 } from "../encodeWav";
import { STT_State } from "../schema";
import { devLog } from "@/utils/devLog";
import { raceWithAbort } from "../abort";
import { RUST_CAPTURE_SAMPLE_RATE, RustUtteranceService } from "../rustCapturePolling";

export class STT_MoonshineService extends RustUtteranceService {
  protected readonly logTag = "[Moonshine]";
  #language = "";

  protected engineLabel(): string {
    return i18n.t("stt.provider_moonshine");
  }

  protected inputDevice(params: STT_State): string {
    return params.moonshine.device;
  }

  protected async prepare(params: STT_State): Promise<string | null> {
    devLog("[Moonshine] Starting service...");
    this.receiver.onStatus(i18n.t("stt.status_moonshine_connecting"));

    const { endpoint, language } = params.moonshine;
    this.#language = language?.trim() ?? "";
    await raceWithAbort(
      invoke("plugin:moonshine-stt|set_moonshine_endpoint", { endpoint }),
      this.startAbortSignal,
    );
    if (this.lifecycle.aborted) return null;

    const isAvailable = await raceWithAbort(
      invoke<boolean>("plugin:moonshine-stt|check_moonshine_availability"),
      this.startAbortSignal,
    );
    if (!isAvailable) return i18n.t("stt.error_moonshine_not_found", { endpoint });

    devLog("[Moonshine] Server connected");
    return null;
  }

  protected async transcribe(samples: Float32Array, signal: AbortSignal): Promise<string> {
    // The IPC call cannot be cancelled; racing it lets teardown discard a late result immediately. Rust errors are
    // connection / HTTP failures and count as retryable.
    return raceWithAbort(
      invoke<string>("plugin:moonshine-stt|moonshine_transcribe", {
        audioBase64: floatSamplesToWavBase64(samples, RUST_CAPTURE_SAMPLE_RATE),
        language: this.#language || null,
      }),
      signal,
    );
  }
}
