import { invoke } from "@tauri-apps/api/core";
import { isEmptyValue } from "../../../../utils";
import { clampedOr } from "../num";
import { TTS_State } from "../schema";
import { ITTSReceiver, ITTSService, TTS_OPTIONS_MISSING, errorMessage } from "../types";

/**
 * Uberduck v1 API (`uberduck-tts` plugin). Rust synthesizes, downloads and plays each clip on the
 * chosen device and resolves when playback ends, so lines are spoken one after another in order.
 */
export class TTS_UberduckService implements ITTSService {
  #running = false;
  /** Lines wait here so two clips never play over each other. */
  #chain: Promise<void> = Promise.resolve();
  /** Bumped by stop(): lines queued before it are dropped. */
  #generation = 0;

  constructor(private bindings: ITTSReceiver) {}

  dispose(): void {
    this.#halt();
  }

  get state() {
    return window.ApiServer.state.services.tts.data.uberduck;
  }

  start(_state: TTS_State): void {
    // The v1 API only needs the API key and a voice; secret_key and device are optional.
    if (isEmptyValue(this.state.api_key) || isEmptyValue(this.state.voice))
      return this.bindings.onStop(TTS_OPTIONS_MISSING);
    this.#running = true;
    this.bindings.onStart();
  }

  play(value: string): void {
    if (!this.#running || !value.trim()) return;
    const generation = this.#generation;
    this.#chain = this.#chain.then(async () => {
      if (generation !== this.#generation) return;
      try {
        await invoke("plugin:uberduck-tts|speak", {
          data: {
            auth: {
              api_key: this.state.api_key,
              secret_key: this.state.secret_key,
            },
            text: value,
            // Same fallback as the other native players: the app's output device, else system default.
            device_name: this.state.device || window.ApiServer.state.audioOutputDevice || "",
            volume: clampedOr(this.state.volume, 1, 0, 1),
            voicemodel_uuid: this.state.voice,
          },
        });
      } catch (error) {
        if (generation === this.#generation) this.bindings.onError(`Uberduck: ${errorMessage(error)}`);
      }
    });
  }

  stop(): void {
    this.#halt();
    this.bindings.onStop();
  }

  #halt() {
    this.#running = false;
    this.#generation++;
  }
}
