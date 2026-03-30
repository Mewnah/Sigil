import { invoke } from "@tauri-apps/api/core";
import { proxy } from "valtio";
import { IServiceInterface } from "@/types";

type VoiceClipOptions = {
  device_name: string;
  volume: number; // 1 - base
  rate: number; // 1 - base
};

class Service_Sound implements IServiceInterface {
  constructor() {}

  public serviceState = proxy({
    muted: false,
  });

  async init() {
    // Voice clips use native playback via Tauri; no Web Audio setup required here.
  }

  #voiceClipQueue: { data: ArrayBuffer; options: VoiceClipOptions }[] = [];

  #isVoiceClipPlaying = false;

  async #tryDequeueVoiceClip() {
    if (this.#isVoiceClipPlaying) return;

    const clip = this.#voiceClipQueue.shift();
    if (!clip) return;

    this.#isVoiceClipPlaying = true;

    try {
      await invoke<any>("plugin:audio|play_async", {
        data: {
          data: Array.from(new Uint8Array(clip.data)),
          ...clip.options,
          speed: 1,
        },
      });
    } catch {
      //
    } finally {
      this.#isVoiceClipPlaying = false;
      this.#tryDequeueVoiceClip();
    }
  }

  enqueueVoiceClip(buffer: ArrayBuffer, options: VoiceClipOptions) {
    if (!buffer) return;
    const deviceName = options.device_name || window.ApiServer.state.audioOutputDevice || "";
    this.#voiceClipQueue.push({
      data: buffer,
      options: { ...options, device_name: deviceName },
    });
    this.#tryDequeueVoiceClip();
  }
}

export default Service_Sound;
