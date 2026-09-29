import { invoke } from "@tauri-apps/api/core";
import i18n from "i18next";
import { proxy } from "valtio";
import { IServiceInterface } from "@/types";

export type VoiceClipOptions = {
  /** Output device name (cpal); empty = the app's output device, else the system default. */
  device_name: string;
  /** 1 = unchanged. */
  volume: number;
  /** Playback speed; 1 = unchanged. */
  rate: number;
};

type QueuedClip = {
  data: ArrayBuffer;
  options: VoiceClipOptions;
  resolve: () => void;
  reject: (error: Error) => void;
};

/** Bounds for the Rust sink: NaN would fail to deserialize, and a speed of 0 would never finish. */
const MAX_VOLUME = 4;
const MIN_RATE = 0.1;
const MAX_RATE = 10;

function finite(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/** Clamp clip options to values `plugin:audio|play_async` accepts. */
export function sanitizeVoiceClipOptions(options: VoiceClipOptions, defaultDevice: string): VoiceClipOptions {
  const rate = finite(options.rate, 1);
  return {
    device_name: (options.device_name || defaultDevice || "").trim(),
    volume: Math.min(MAX_VOLUME, Math.max(0, finite(options.volume, 1))),
    rate: rate > 0 ? Math.min(MAX_RATE, Math.max(MIN_RATE, rate)) : 1,
  };
}

function describePlaybackError(error: unknown, device: string): string {
  const detail =
    typeof error === "string" ? error : error instanceof Error ? error.message : String(error ?? "");
  return device
    ? i18n.t("toasts.sound_playback_failed_device", { device, error: detail })
    : i18n.t("toasts.sound_playback_failed", { error: detail });
}

/**
 * Serial voice-clip player (Azure and VOICEVOX TTS). Each clip goes to the Rust `audio` plugin, which
 * decodes it and plays it on the chosen output device. Clips play one at a time in enqueue order.
 */
class Service_Sound implements IServiceInterface {
  public serviceState = proxy({
    muted: false,
  });

  #queue: QueuedClip[] = [];
  #playing = false;

  async init() {
    // Voice clips use native playback via Tauri; no Web Audio setup required here.
  }

  /**
   * Queue a clip. The promise resolves when the clip has finished playing (or was dropped by
   * `clearVoiceClips`), and rejects with a readable, localized message when it could not be played:
   * missing output device, undecodable audio, and so on. Earlier versions swallowed these errors.
   */
  enqueueVoiceClip(buffer: ArrayBuffer, options: VoiceClipOptions): Promise<void> {
    if (!buffer || buffer.byteLength === 0) return Promise.resolve();
    const defaultDevice = window.ApiServer?.state?.audioOutputDevice ?? "";
    return new Promise<void>((resolve, reject) => {
      this.#queue.push({
        data: buffer,
        options: sanitizeVoiceClipOptions(options, defaultDevice),
        resolve,
        reject,
      });
      void this.#pump();
    });
  }

  /**
   * Drop clips that have not started. The clip already playing finishes, because `play_async`
   * cannot be interrupted yet (the P2 TTS sink adds cancel).
   */
  clearVoiceClips(): void {
    const dropped = this.#queue.splice(0);
    for (const clip of dropped) clip.resolve();
  }

  /** Clips waiting to play (excluding the one playing). */
  get pendingVoiceClips(): number {
    return this.#queue.length;
  }

  async #pump(): Promise<void> {
    if (this.#playing) return;
    const clip = this.#queue.shift();
    if (!clip) return;

    this.#playing = true;
    try {
      await invoke("plugin:audio|play_async", {
        data: {
          data: Array.from(new Uint8Array(clip.data)),
          device_name: clip.options.device_name,
          volume: clip.options.volume,
          rate: clip.options.rate,
        },
      });
      clip.resolve();
    } catch (error) {
      clip.reject(new Error(describePlaybackError(error, clip.options.device_name)));
    } finally {
      this.#playing = false;
      void this.#pump();
    }
  }
}

export default Service_Sound;
