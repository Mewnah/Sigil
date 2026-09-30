import { invoke } from "@tauri-apps/api/core";
import { isEmptyValue } from "../../../../utils";
import { clampedOr } from "../num";
import { TTS_State } from "../schema";
import { ITTSReceiver, ITTSService, TTS_OPTIONS_MISSING, errorMessage } from "../types";

/**
 * Windows SAPI voices (`windows-tts` plugin), spoken directly to the chosen SAPI output device. Self-playing
 * (ITTSService v2) until SAPI renders into the TTS sink (P6-J): whole finals, no voice-changer routing.
 */
export class TTS_WindowsService implements ITTSService {
  get selfPlaying(): boolean {
    return true;
  }

  #running = false;

  constructor(private bindings: ITTSReceiver) {}

  dispose(): void {
    this.#running = false;
  }

  get state() {
    return window.ApiServer.state.services.tts.data.windows;
  }

  start(_state: TTS_State): void {
    if (isEmptyValue(this.state.voice) || isEmptyValue(this.state.device))
      return this.bindings.onStop(TTS_OPTIONS_MISSING);
    this.#running = true;
    this.bindings.onStart();
  }

  play(value: string): void {
    if (!this.#running || !value.trim()) return;
    invoke("plugin:windows-tts|speak", {
      data: {
        voice: this.state.voice,
        device: this.state.device,
        volume: clampedOr(this.state.volume, 1, 0, 1),
        rate: clampedOr(this.state.rate, 1, 0.1, 5),
        value,
      },
    }).catch((error) => {
      if (this.#running) this.bindings.onError(`Windows speech: ${errorMessage(error)}`);
    });
  }

  stop(): void {
    this.#running = false;
    this.bindings.onStop();
  }
}
