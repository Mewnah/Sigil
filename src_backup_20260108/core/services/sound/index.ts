import { invoke } from "@tauri-apps/api/tauri";
import { proxy } from "valtio";
import { IServiceInterface } from "@/types";

type SoundEffects = {
  volume?: number;
  playbackMin?: number;
  playbackMax?: number;
  detuneMin?: number;
  detuneMax?: number;
};

type VoiceClipOptions = {
  device_name: string;
  volume: number; // 1 - base
  rate: number; // 1 - base
};

class Service_Sound implements IServiceInterface {
  constructor() {
  }

  private audioContext!: AudioContext;
  private mediaStream: MediaStream | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;

  // Public analyser for audio visualization
  public analyser: AnalyserNode | null = null;

  async init() {
    this.audioContext = new AudioContext();
  }

  public serviceState = proxy({
    muted: false,
    inputActive: false,
  });

  /**
   * Start microphone input for audio visualization
   */
  async startInput(): Promise<void> {
    if (this.mediaStream) return; // Already started

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

      if (!this.audioContext || this.audioContext.state === "closed") {
        this.audioContext = new AudioContext();
      }

      if (this.audioContext.state === "suspended") {
        await this.audioContext.resume();
      }

      this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.8;

      this.sourceNode.connect(this.analyser);
      // Don't connect to destination - we don't want to hear ourselves

      this.serviceState.inputActive = true;
    } catch (error) {
      console.error("Failed to start audio input:", error);
      throw error;
    }
  }

  /**
   * Stop microphone input
   */
  stopInput(): void {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
    this.analyser = null;
    this.serviceState.inputActive = false;
  }


  #voiceClipQueue: { data: ArrayBuffer; options: VoiceClipOptions }[] = [];

  #isVoiceClipPlaying = false;

  async #tryDequeueVoiceClip() {
    if (this.#isVoiceClipPlaying) return;

    const clip = this.#voiceClipQueue.shift();
    // empty queue
    if (!clip) return;

    this.#isVoiceClipPlaying = true;

    try {
      await invoke<any>("plugin:audio|play_async", {
        data: {
          data: Array.from(new Uint8Array(clip.data)),
          ...clip.options,
          speed: 1
        },
      });
    } catch (error) {
      // 
    } finally {
      this.#isVoiceClipPlaying = false;
      this.#tryDequeueVoiceClip();
    }
  }

  enqueueVoiceClip(buffer: ArrayBuffer, options: VoiceClipOptions) {
    if (!buffer)
      return;
    // Use global output device as fallback if service doesn't specify one
    const deviceName = options.device_name || window.ApiServer.state.audioOutputDevice || "";
    this.#voiceClipQueue.push({
      data: buffer,
      options: { ...options, device_name: deviceName }
    });
    this.#tryDequeueVoiceClip();
  }

  private random = (min: number, max: number) =>
    Math.random() * (max - min) + min;
}

export default Service_Sound;
