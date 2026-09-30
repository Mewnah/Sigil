import { invoke } from "@tauri-apps/api/core";
import type { RustTtsSpeakReport } from "@/types";
import { isEmptyValue } from "../../../../utils";
import { isSinkCancelled } from "../../sound/sink";
import { clampedOr } from "../num";
import { TTS_State } from "../schema";
import { ITTSReceiver, ITTSService, TTS_OPTIONS_MISSING, TtsSinkContext, errorMessage } from "../types";

/**
 * Uberduck v1 API (`uberduck-tts` plugin).
 *
 * - PCM contract (ITTSService v2): {@link synthesizeToSink} has Rust synthesize, download and decode one sentence
 *   and push it straight into the begun TTS-sink utterance (no audio crosses IPC); the sentence pipeline routes it
 *   (device / voice changer).
 * - Legacy {@link play}: Rust speaks whole lines on the chosen device (through the sink), one after another.
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

  #auth() {
    return { api_key: this.state.api_key, secret_key: this.state.secret_key };
  }

  /** v2: Rust fetches one sentence and pushes it into the sink utterance `sink.utteranceId`. */
  async synthesizeToSink(text: string, sink: TtsSinkContext, signal: AbortSignal): Promise<void> {
    if (!this.#running || signal.aborted || !text.trim()) return;
    try {
      await invoke<RustTtsSpeakReport>("plugin:uberduck-tts|speak_to_sink", {
        data: { auth: this.#auth(), text, voicemodel_uuid: this.state.voice },
        utteranceId: sink.utteranceId,
      });
    } catch (error) {
      // Stop / skip cancel the utterance; Rust then stops with "cancelled".
      if (signal.aborted || isSinkCancelled(error)) return;
      throw new Error(`Uberduck: ${errorMessage(error)}`);
    }
  }

  play(value: string): void {
    if (!this.#running || !value.trim()) return;
    const generation = this.#generation;
    this.#chain = this.#chain.then(async () => {
      if (generation !== this.#generation) return;
      try {
        await invoke("plugin:uberduck-tts|speak", {
          data: {
            auth: this.#auth(),
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
