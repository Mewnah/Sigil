import { invoke } from "@tauri-apps/api/core";
import type { RustTtsSpeakReport } from "@/types";
import { isEmptyValue } from "../../../../utils";
import { isSinkCancelled } from "../../sound/sink";
import { clampedOr } from "../num";
import { TTS_State } from "../schema";
import { ITTSReceiver, ITTSService, TTS_OPTIONS_MISSING, TtsSinkContext, errorMessage } from "../types";

/**
 * `speak_to_sink` request of the Rust `windows-tts` plugin (P6-J): one sentence. `voice` is a SAPI voice token id
 * (`""` = the Windows default voice); `rate` is Sigil's multiplier 0.1-1-5 (Rust maps it onto SAPI's -10..10).
 */
export interface WindowsSinkSpeakRequest {
  voice: string;
  text: string;
  rate: number;
}

/** The `speak_to_sink` request for one sentence with the `windows` settings. */
export function windowsSinkRequest(c: TTS_State["windows"], text: string): WindowsSinkSpeakRequest {
  return {
    voice: String(c.voice ?? "").trim(),
    text,
    rate: clampedOr(c.rate, 1, 0.1, 5),
  };
}

/** Fire-and-forget IPC: never throws, synchronously or later. */
function send(command: string, args?: Record<string, unknown>): void {
  try {
    void Promise.resolve(invoke(command, args)).catch(() => {});
  } catch {
    // Without Tauri (web build, tests) there is nothing to tell.
  }
}

/**
 * Windows SAPI voices (`windows-tts` plugin) through the TTS sink (ITTSService v2 `synthesizeToSink`, P6-J).
 *
 * Rust renders each sentence on its own SAPI thread with the voice's native format and streams it into the sink
 * utterance the sentence pipeline began, so SAPI speaks sentence by sentence, can be skipped and stopped, and follows
 * the top-level route: the output device, the voice changer (speech comes out in the converted voice) or both. The
 * sink applies the device and volume of the `windows` settings; a device saved from the old SAPI output list (a SAPI
 * token id) still works (the sink resolves its endpoint), and an empty device is the app's output device.
 *
 * - `start` only needs a voice.
 * - Stop / skip abort the sentence's signal, which cancels its render at once (the sink cancel stops it too).
 * - {@link play} is the legacy whole-line path that speaks straight to the SAPI output device with SAPI's own
 *   playback (no sink, no voice changer). The service never uses it for this backend (the pipeline speaks through
 *   the sink); it stays for the v1 contract.
 */
export class TTS_WindowsService implements ITTSService {
  #running = false;
  #stopped = false;

  constructor(private bindings: ITTSReceiver) {}

  get state() {
    return window.ApiServer.state.services.tts.data.windows;
  }

  start(_state: TTS_State): void {
    if (isEmptyValue(this.state.voice))
      return this.bindings.onStop(TTS_OPTIONS_MISSING);
    this.#running = true;
    this.bindings.onStart();
  }

  /** v2: render one sentence into the begun sink utterance `sink.utteranceId`. */
  async synthesizeToSink(text: string, sink: TtsSinkContext, signal: AbortSignal): Promise<void> {
    if (!this.#running || signal.aborted || !text.trim()) return;
    const request = windowsSinkRequest(this.state, text);
    const onAbort = () => send("plugin:windows-tts|cancel_speak_to_sink", { utteranceId: sink.utteranceId });
    signal.addEventListener("abort", onAbort, { once: true });
    try {
      await invoke<RustTtsSpeakReport>("plugin:windows-tts|speak_to_sink", { request, utteranceId: sink.utteranceId });
    } catch (error) {
      // Stop / skip cancel the utterance; Rust then answers "cancelled".
      if (signal.aborted || isSinkCancelled(error)) return;
      throw new Error(`Windows speech: ${errorMessage(error)}`);
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
  }

  /** Legacy (v1): speak a whole line straight to the SAPI output device. Unused by the sentence pipeline. */
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
    this.#halt();
    this.bindings.onStop();
  }

  dispose(): void {
    this.#halt();
  }

  #halt() {
    const wasActive = !this.#stopped;
    this.#stopped = true;
    this.#running = false;
    // Cancel whatever the SAPI thread still has queued or is rendering for this backend.
    if (wasActive) send("plugin:windows-tts|cancel_speak_to_sink", { utteranceId: null });
  }
}
