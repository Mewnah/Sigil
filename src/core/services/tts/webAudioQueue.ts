/**
 * Ordered, non-overlapping Web Audio playback for TTS backends that receive whole clips (WAV/MP3
 * bytes) from Rust: Kokoro, MeloTTS, Chatterbox, Fish Speech, ONNX (local) and the HTTP speech server.
 *
 * The six copies of `playAudio` this replaces had three problems:
 * - Every clip started immediately, so a second final overlapped the first.
 * - Clips could play out of order when a later synthesis finished first.
 * - Stop did not cancel synthesis already in flight, so audio could still play after Stop.
 *
 * Output is the webview's default device, as before (see `tts.neural_web_audio_output_note`). The
 * unified Rust TTS sink (P2) replaces this with per-device routing.
 */

export type ClipBytes = ArrayBuffer | Uint8Array | number[];

/** Produces one clip, or nothing when the audio went elsewhere (e.g. into the voice changer). */
export type ClipProducer = () => Promise<ClipBytes | null | undefined | void>;

function toArrayBuffer(bytes: ClipBytes): ArrayBuffer {
  // decodeAudioData detaches its argument, so always hand it a private copy.
  if (bytes instanceof ArrayBuffer) return bytes.slice(0);
  if (bytes instanceof Uint8Array) return new Uint8Array(bytes).buffer;
  return Uint8Array.from(bytes).buffer;
}

function byteLength(bytes: ClipBytes): number {
  return Array.isArray(bytes) ? bytes.length : bytes.byteLength;
}

export class WebAudioClipQueue {
  #ctx: AudioContext | null = null;
  /** Context time at which the last scheduled clip ends. */
  #endAt = 0;
  #sources = new Set<AudioBufferSourceNode>();
  #tail: Promise<void> = Promise.resolve();
  /** Bumped by stop(); work queued under an older generation is dropped. */
  #generation = 0;

  constructor(private readonly createContext: () => AudioContext = () => new AudioContext()) {}

  /**
   * Create the audio context now. Call this from `start()`, which usually runs after a click, so the
   * context is not created later from a background event in a suspended (autoplay-blocked) state.
   */
  warmUp(): void {
    this.#context();
  }

  /**
   * Queue one clip. `produce` runs only after every earlier clip was produced and scheduled, so clips
   * play in call order, back to back, without overlapping. Synthesis of the next clip overlaps
   * playback of the current one.
   *
   * Resolves once the clip is scheduled (or dropped by `stop()`), and rejects when `produce`,
   * decoding or scheduling fails. A failure does not block later clips.
   */
  enqueue(produce: ClipProducer): Promise<void> {
    const generation = this.#generation;
    const task = this.#tail.then(async () => {
      if (generation !== this.#generation) return;
      const bytes = await produce();
      if (generation !== this.#generation || !bytes || byteLength(bytes) === 0) return;
      await this.#schedule(toArrayBuffer(bytes), generation);
    });
    this.#tail = task.catch(() => {});
    return task;
  }

  /** Silence everything now, drop queued and in-flight clips, and release the audio context. */
  stop(): void {
    this.#generation++;
    for (const source of this.#sources) {
      try {
        source.stop();
      } catch {
        // already stopped
      }
      source.disconnect();
    }
    this.#sources.clear();
    this.#endAt = 0;
    const ctx = this.#ctx;
    this.#ctx = null;
    if (ctx && ctx.state !== "closed") void ctx.close().catch(() => {});
  }

  /** Clips scheduled or playing right now (for tests and diagnostics). */
  get activeClips(): number {
    return this.#sources.size;
  }

  #context(): AudioContext {
    if (!this.#ctx || this.#ctx.state === "closed") {
      this.#ctx = this.createContext();
      this.#endAt = 0;
    }
    return this.#ctx;
  }

  async #schedule(data: ArrayBuffer, generation: number): Promise<void> {
    const ctx = this.#context();
    if (ctx.state === "suspended") await ctx.resume().catch(() => {});
    const buffer = await ctx.decodeAudioData(data);
    if (generation !== this.#generation || ctx !== this.#ctx) return;

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    const startAt = Math.max(ctx.currentTime, this.#endAt);
    this.#endAt = startAt + buffer.duration;
    this.#sources.add(source);
    source.onended = () => {
      this.#sources.delete(source);
      source.disconnect();
    };
    source.start(startAt);
  }
}
