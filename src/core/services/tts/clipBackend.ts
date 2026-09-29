import { TTS_State } from "./schema";
import { ITTSReceiver, ITTSService, errorMessage } from "./types";
import { ClipBytes, WebAudioClipQueue } from "./webAudioQueue";

/**
 * Shared lifecycle for backends that synthesize whole clips in Rust and play them with Web Audio
 * (Kokoro, MeloTTS, Chatterbox, Fish Speech, ONNX (local), HTTP speech server):
 * - `start` runs `connect()` and reports `onStart`, or `onStop(error)` when it fails. A Stop during
 *   `connect()` wins: the late result is dropped.
 * - `play` queues synthesis on a {@link WebAudioClipQueue}, so clips play in order without
 *   overlapping. A failed utterance goes to `onError`, and the backend keeps running.
 * - `stop` drops queued and in-flight clips, silences playback and reports `onStop()`.
 */
export abstract class ClipTtsBackend implements ITTSService {
  protected readonly clips: WebAudioClipQueue;
  #running = false;
  #stopped = false;

  constructor(
    protected readonly bindings: ITTSReceiver,
    /** Prefix for error messages, e.g. "Kokoro". */
    protected readonly label: string,
    clips: WebAudioClipQueue = new WebAudioClipQueue(),
  ) {
    this.clips = clips;
  }

  /** Configure Rust and check that the engine answers. Resolve with an error message to refuse. */
  protected abstract connect(state: TTS_State): Promise<string | null>;

  /** Produce one clip, or nothing when the audio was routed elsewhere (voice changer). */
  protected abstract synthesize(text: string): Promise<ClipBytes | null | undefined | void>;

  get running(): boolean {
    return this.#running;
  }

  async start(state: TTS_State): Promise<void> {
    try {
      const refusal = await this.connect(state);
      if (this.#stopped) return;
      if (refusal) {
        this.bindings.onStop(refusal);
        return;
      }
      this.clips.warmUp();
      this.#running = true;
      this.bindings.onStart();
    } catch (error) {
      if (this.#stopped) return;
      console.error(`[${this.label}] start failed:`, error);
      this.bindings.onStop(errorMessage(error));
    }
  }

  play(text: string): void {
    if (!this.#running || !text.trim()) return;
    this.clips
      .enqueue(() => this.synthesize(text))
      .catch((error) => {
        if (!this.#running) return;
        console.error(`[${this.label}] speech failed:`, error);
        this.bindings.onError(`${this.label}: ${errorMessage(error)}`);
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
    this.#stopped = true;
    this.#running = false;
    this.clips.stop();
  }
}
