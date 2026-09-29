import type { RequestLimiter } from "./rateLimiter";

/**
 * Per-utterance transcription pipeline for HTTP STT engines (local OpenAI-compatible servers, Groq, Moonshine).
 *
 * Rust's Silero segmenter closes utterances (`audio:utterance`); each one is queued here and transcribed with one
 * request, strictly in order, one request at a time. When requests cannot keep up (a rate limit, a slow server, a
 * retry), the utterances waiting in the queue are merged into the next request (up to `maxBatchSeconds`, separated
 * by a short silence), so the queue drains faster than speech arrives and no final is ever dropped.
 *
 * Billing minimum (Groq bills every request as at least 10 s of audio and caps requests per day): while the limiter
 * asks for it (`RequestLimiter.minBatchSeconds`, e.g. once requests run ahead of the daily pace), a batch shorter
 * than the minimum is held back briefly so the next utterances can join it. It is sent once it reaches the minimum,
 * or once its oldest utterance has waited `maxHoldMs` (`maxHoldWhileSpeakingMs` while the VAD reports the next
 * utterance in progress, see {@link UtterancePipeline.speechStarted}), whichever comes first.
 *
 * Pure (no Tauri, no `window`); timers come from the global `setTimeout`, so tests drive it with fake timers.
 */

export interface PendingUtterance {
  seq: number;
  /** Mono f32 PCM at the pipeline's sample rate. */
  samples: Float32Array;
  /** The VAD force-split a long utterance here and the speech goes on (Rust `forced`). */
  continues?: boolean;
}

export interface UtteranceBatch {
  /** Utterance seqs in the batch, oldest first. */
  seqs: number[];
  /** Concatenated PCM, utterances separated by `gapSamples` of silence. */
  samples: Float32Array;
}

export interface BatchLimits {
  /** A batch grows only while it stays at or below this many samples (its first utterance is always taken). */
  maxBatchSamples: number;
  /** Silence inserted between merged utterances. */
  gapSamples: number;
}

/**
 * `plugin:audio|take_utterance` result → mono f32 PCM (little-endian, like every Tauri target). The IPC normally
 * delivers an `ArrayBuffer`; typed-array views and plain byte arrays (IPC fallback transports) are accepted too.
 * A trailing partial sample is ignored.
 */
export function pcmFromIpc(raw: unknown): Float32Array {
  let bytes: Uint8Array;
  // `instanceof` fails across realms (e.g. a buffer made in another frame); the tag check does not.
  if (raw instanceof ArrayBuffer || Object.prototype.toString.call(raw) === "[object ArrayBuffer]")
    bytes = new Uint8Array(raw as ArrayBuffer);
  else if (ArrayBuffer.isView(raw)) bytes = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);
  else if (Array.isArray(raw)) bytes = Uint8Array.from(raw as number[]);
  else throw new TypeError("take_utterance returned no PCM");
  const whole = bytes.byteLength - (bytes.byteLength % 4);
  // Float32Array views need a 4-byte aligned offset; copy otherwise.
  if (bytes.byteOffset % 4 === 0) return new Float32Array(bytes.buffer, bytes.byteOffset, whole / 4);
  return new Float32Array(bytes.slice(0, whole).buffer);
}

/** What `takeBatch` would return now: how many utterances and the merged length in samples (gaps included). */
export interface BatchPlan {
  count: number;
  samples: number;
}

/** FIFO of closed utterances with merge-on-dequeue. */
export class UtteranceQueue {
  #items: PendingUtterance[] = [];
  /** Enqueue time (ms) of each item in `#items`. */
  #queuedAt: number[] = [];
  #samples = 0;

  get length(): number {
    return this.#items.length;
  }

  /** Speech samples waiting (gaps not counted). */
  get queuedSamples(): number {
    return this.#samples;
  }

  /** Enqueue time of the oldest waiting utterance; `null` when empty. */
  get oldestQueuedAt(): number | null {
    return this.#queuedAt.length > 0 ? this.#queuedAt[0] : null;
  }

  push(utterance: PendingUtterance, queuedAt = 0): void {
    this.#items.push(utterance);
    this.#queuedAt.push(queuedAt);
    this.#samples += utterance.samples.length;
  }

  /** The batch `takeBatch` would return now; `null` when empty. */
  peekBatch(limits: BatchLimits): BatchPlan | null {
    if (this.#items.length === 0) return null;
    let total = this.#items[0].samples.length;
    let count = 1;
    while (count < this.#items.length) {
      const next = total + limits.gapSamples + this.#items[count].samples.length;
      if (next > limits.maxBatchSamples) break;
      total = next;
      count++;
    }
    return { count, samples: total };
  }

  /** Length in samples of the batch `takeBatch` would return now (0 when empty). */
  peekBatchSamples(limits: BatchLimits): number {
    return this.peekBatch(limits)?.samples ?? 0;
  }

  /** Removes the oldest utterances that fit in one batch and merges them; `null` when empty. */
  takeBatch(limits: BatchLimits): UtteranceBatch | null {
    const plan = this.peekBatch(limits);
    if (!plan) return null;
    const taken = this.#items.splice(0, plan.count);
    this.#queuedAt.splice(0, plan.count);
    for (const u of taken) this.#samples -= u.samples.length;
    if (taken.length === 1) return { seqs: [taken[0].seq], samples: taken[0].samples };
    const samples = new Float32Array(plan.samples);
    let offset = 0;
    taken.forEach((u, i) => {
      if (i > 0) offset += limits.gapSamples;
      samples.set(u.samples, offset);
      offset += u.samples.length;
    });
    return { seqs: taken.map((u) => u.seq), samples };
  }

  clear(): void {
    this.#items = [];
    this.#queuedAt = [];
    this.#samples = 0;
  }
}

/**
 * How a failed transcription is handled:
 * - `retryable`: network error, timeout, 5xx; the same batch is retried with backoff, then the session stops.
 * - `rate_limited`: HTTP 429; the batch waits (`retryAfterMs` or a fallback) and is retried without counting as a failure.
 * - `fatal`: retrying cannot help (bad key, unknown model, bad request); the session stops at once.
 */
export type TranscriptionFailureKind = "retryable" | "rate_limited" | "fatal";

export class TranscriptionError extends Error {
  constructor(
    message: string,
    readonly kind: TranscriptionFailureKind,
    readonly retryAfterMs: number | null = null,
  ) {
    super(message);
    this.name = "TranscriptionError";
  }
}

/**
 * Classifies a failed HTTP transcription response: 429 waits (honouring `Retry-After`), 408 / 5xx are retried, and
 * other 4xx (bad key, unknown model or URL, rejected audio) stop the session because retrying cannot help.
 */
export function httpTranscriptionError(status: number, detail: string, retryAfterMs: number | null = null): TranscriptionError {
  const message = detail ? `HTTP ${status}: ${detail}` : `HTTP ${status}`;
  if (status === 429) return new TranscriptionError(message, "rate_limited", retryAfterMs);
  if (status === 408 || status === 425 || status >= 500) return new TranscriptionError(message, "retryable");
  return new TranscriptionError(message, "fatal");
}

/** Transient pipeline state worth showing to the user; `null` clears it. */
export type PipelineStatus =
  | { kind: "rate_limited"; waitMs: number; queued: number }
  | { kind: "retrying"; attempt: number; maxAttempts: number; waitMs: number; detail: string };

/** Why the pipeline gave up; the engine stops the session with a message built from this. */
export type PipelineFatal =
  | { reason: "fatal"; detail: string }
  | { reason: "failed"; attempts: number; detail: string }
  | { reason: "wait_too_long"; waitMs: number }
  | { reason: "backlog"; queuedSeconds: number };

export interface UtterancePipelineOptions {
  sampleRate: number;
  /** One request. Throw a {@link TranscriptionError} to classify the failure (other errors count as retryable). */
  transcribe(samples: Float32Array, signal: AbortSignal): Promise<string>;
  /** Non-empty transcript of one batch, in utterance order. */
  onText(text: string, seqs: number[]): void;
  onStatus(status: PipelineStatus | null): void;
  /** Called once; the pipeline is disposed afterwards. */
  onFatal(fatal: PipelineFatal): void;
  /** Request pacing (Groq); `null` = send as soon as the previous request finished. */
  limiter?: RequestLimiter | null;
  /**
   * Hold a batch shorter than this (gaps included) so later utterances can join it. Default: asked from the
   * limiter before each request (`RequestLimiter.minBatchSeconds`), else 0 = never hold.
   */
  minBatchSeconds?: number;
  /** Longest hold of a short batch, from its oldest utterance's arrival (default 3.5 s). */
  maxHoldMs?: number;
  /** Longest hold while the VAD reports the next utterance in progress (default 8 s). */
  maxHoldWhileSpeakingMs?: number;
  /** Longest merged request (default 28 s, inside Whisper's 30 s window). */
  maxBatchSeconds?: number;
  /** Silence between merged utterances (default 0.2 s). */
  gapSeconds?: number;
  /** Queued speech that ends the session instead of growing without bound (default 300 s). */
  maxQueuedSeconds?: number;
  /** A limiter or server wait above this ends the session (default 10 min; e.g. a daily cap). */
  maxWaitMs?: number;
  /** Attempts per batch before the session stops (default 4). */
  maxAttempts?: number;
  /** Backoff after a failed attempt: base * 2^(attempt-1), capped (defaults 1 s, 15 s). */
  retryBaseMs?: number;
  retryMaxMs?: number;
  /** Wait after a 429 without a readable Retry-After (default 5 s). */
  rateLimitFallbackMs?: number;
  /** Rate-limit waits shorter than this are not reported (default 1.5 s). */
  statusDelayMs?: number;
  /** Clock (epoch ms). */
  now?: () => number;
}

export class UtterancePipeline {
  readonly #o: Required<Omit<UtterancePipelineOptions, "limiter" | "minBatchSeconds">> & { limiter: RequestLimiter | null };
  readonly #limits: BatchLimits;
  /** Fixed hold target in seconds (`minBatchSeconds` option); `null` = ask the limiter. */
  readonly #fixedMinBatchSeconds: number | null;
  readonly #queue = new UtteranceQueue();
  /** A batch whose request failed and is retried before anything newer. */
  #retry: UtteranceBatch | null = null;
  #attempts = 0;
  #inFlight: AbortController | null = null;
  #timer: ReturnType<typeof setTimeout> | null = null;
  /** The pending timer only holds a short batch (a new utterance or speech start re-plans at once). */
  #holding = false;
  /** Seq of the utterance the VAD reports in progress (`null` = none). */
  #speakingSeq: number | null = null;
  /** Nothing is sent before this (429 / retry backoff). */
  #blockedUntil = 0;
  #disposed = false;
  #statusShown = false;

  constructor(options: UtterancePipelineOptions) {
    // An explicit `undefined` must not replace a default.
    const given = Object.fromEntries(
      Object.entries(options).filter(([, v]) => v !== undefined),
    ) as UtterancePipelineOptions;
    const { minBatchSeconds, ...rest } = given;
    this.#fixedMinBatchSeconds = minBatchSeconds ?? null;
    this.#o = {
      maxHoldMs: 3_500,
      maxHoldWhileSpeakingMs: 8_000,
      maxBatchSeconds: 28,
      gapSeconds: 0.2,
      maxQueuedSeconds: 300,
      maxWaitMs: 600_000,
      maxAttempts: 4,
      retryBaseMs: 1_000,
      retryMaxMs: 15_000,
      rateLimitFallbackMs: 5_000,
      statusDelayMs: 1_500,
      now: () => Date.now(),
      ...rest,
      limiter: options.limiter ?? null,
    };
    this.#limits = {
      maxBatchSamples: Math.max(1, Math.round(this.#o.maxBatchSeconds * this.#o.sampleRate)),
      gapSamples: Math.max(0, Math.round(this.#o.gapSeconds * this.#o.sampleRate)),
    };
  }

  /** Samples a batch should reach before it is sent (0 = send at once). */
  #minBatchSamples(now: number): number {
    const seconds = this.#fixedMinBatchSeconds ?? this.#o.limiter?.minBatchSeconds?.(now) ?? 0;
    if (!Number.isFinite(seconds) || seconds <= 0) return 0;
    // A target above the batch limit could never be reached.
    return Math.min(Math.round(seconds * this.#o.sampleRate), this.#limits.maxBatchSamples);
  }

  get disposed(): boolean {
    return this.#disposed;
  }

  /** Utterances waiting (a batch being retried counts as one). */
  get queued(): number {
    return this.#queue.length + (this.#retry ? 1 : 0);
  }

  get busy(): boolean {
    return this.#inFlight !== null;
  }

  push(utterance: PendingUtterance): void {
    if (this.#disposed) return;
    this.#utteranceClosed(utterance.seq, utterance.continues === true);
    if (utterance.samples.length === 0) {
      this.#replanHold();
      return;
    }
    this.#queue.push(utterance, this.#o.now());
    const queuedSeconds = this.#queue.queuedSamples / this.#o.sampleRate;
    if (queuedSeconds > this.#o.maxQueuedSeconds) {
      this.#fail({ reason: "backlog", queuedSeconds });
      return;
    }
    // A held batch may be complete now; a pacing or backoff wait stays as it is (the utterance joins that batch).
    if (this.#holding) this.#clearTimer();
    this.#pump();
  }

  /**
   * The VAD confirmed speech for utterance `seq` (Rust `speech_start`): another utterance is on its way, so a short
   * batch may wait up to `maxHoldWhileSpeakingMs` for it. Cleared when that utterance (or a later one) is pushed or
   * reported gone with {@link speechEnded}.
   */
  speechStarted(seq: number): void {
    if (this.#disposed) return;
    if (this.#speakingSeq === null || seq >= this.#speakingSeq) this.#speakingSeq = seq;
    this.#replanHold();
  }

  /** Utterance `seq` will never be pushed (dropped by Rust, skipped while muted, unavailable). */
  speechEnded(seq: number): void {
    if (this.#disposed) return;
    this.#utteranceClosed(seq, false);
    this.#replanHold();
  }

  /** Aborts the request in flight and forgets everything queued. Idempotent. */
  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#clearTimer();
    this.#inFlight?.abort();
    this.#inFlight = null;
    this.#queue.clear();
    this.#retry = null;
    this.#speakingSeq = null;
  }

  #utteranceClosed(seq: number, continues: boolean): void {
    if (continues) {
      // A forced split: the rest of the same speech follows as the next seq.
      if (this.#speakingSeq === null || seq + 1 > this.#speakingSeq) this.#speakingSeq = seq + 1;
      return;
    }
    if (this.#speakingSeq !== null && seq >= this.#speakingSeq) this.#speakingSeq = null;
  }

  /** A hold's deadline depends on the queue and on speech in progress: plan it again (sends now if it is over). */
  #replanHold(): void {
    if (!this.#holding) return;
    this.#clearTimer();
    this.#pump();
  }

  #clearTimer(): void {
    this.#holding = false;
    if (this.#timer !== null) {
      clearTimeout(this.#timer);
      this.#timer = null;
    }
  }

  #wake(ms: number, holding = false): void {
    this.#clearTimer();
    this.#holding = holding;
    this.#timer = setTimeout(() => {
      this.#timer = null;
      this.#holding = false;
      this.#pump();
    }, Math.max(0, Math.ceil(ms)));
  }

  /**
   * How long the next batch should still wait for more utterances to join it (0 = send it). Only a batch below the
   * billing minimum is held, never a retry, and never one that cannot grow (the next queued utterance does not fit).
   */
  #holdMs(now: number, plan: BatchPlan): number {
    if (this.#retry !== null || plan.count < this.#queue.length) return 0;
    const minSamples = this.#minBatchSamples(now);
    if (minSamples <= 0 || plan.samples >= minSamples) return 0;
    const oldest = this.#queue.oldestQueuedAt;
    if (oldest === null) return 0;
    const limit = this.#speakingSeq !== null ? this.#o.maxHoldWhileSpeakingMs : this.#o.maxHoldMs;
    return Math.max(0, oldest + limit - now);
  }

  #setStatus(status: PipelineStatus | null): void {
    if (status === null) {
      if (!this.#statusShown) return;
      this.#statusShown = false;
    } else {
      this.#statusShown = true;
    }
    this.#o.onStatus(status);
  }

  #fail(fatal: PipelineFatal): void {
    if (this.#disposed) return;
    this.dispose();
    this.#o.onFatal(fatal);
  }

  #pump(): void {
    if (this.#disposed || this.#inFlight || this.#timer !== null) return;
    const retrying = this.#retry !== null;
    const plan: BatchPlan | null = this.#retry
      ? { count: this.#retry.seqs.length, samples: this.#retry.samples.length }
      : this.#queue.peekBatch(this.#limits);
    const batchSamples = plan?.samples ?? 0;
    if (!plan || batchSamples === 0) {
      this.#setStatus(null);
      return;
    }
    const now = this.#o.now();
    const audioSeconds = batchSamples / this.#o.sampleRate;
    const blockedMs = Math.max(0, this.#blockedUntil - now);
    const limiterMs = this.#o.limiter ? this.#o.limiter.waitMs(now, audioSeconds) : 0;
    const waitMs = Math.max(blockedMs, limiterMs);
    if (waitMs > this.#o.maxWaitMs) {
      this.#fail({ reason: "wait_too_long", waitMs });
      return;
    }
    // A short batch waits for more utterances first (billing minimum); the pacing wait below re-plans afterwards.
    const holdMs = this.#holdMs(now, plan);
    if (holdMs > waitMs) {
      this.#wake(holdMs, true);
      return;
    }
    if (waitMs > 0) {
      // A retry backoff already reported its own status; report plain pacing waits once they are noticeable.
      if (!retrying && waitMs >= this.#o.statusDelayMs)
        this.#setStatus({ kind: "rate_limited", waitMs, queued: this.queued });
      this.#wake(waitMs);
      return;
    }

    const batch = this.#retry ?? this.#queue.takeBatch(this.#limits);
    if (!batch) return;
    this.#retry = null;
    this.#o.limiter?.record(now, audioSeconds);
    if (!retrying) this.#setStatus(null);

    const controller = new AbortController();
    this.#inFlight = controller;
    let request: Promise<string>;
    try {
      request = Promise.resolve(this.#o.transcribe(batch.samples, controller.signal));
    } catch (e) {
      request = Promise.reject(e);
    }
    request.then(
      (text) => {
        if (controller.signal.aborted || this.#disposed) return;
        this.#inFlight = null;
        this.#attempts = 0;
        if (retrying) this.#setStatus(null);
        const clean = String(text ?? "").trim();
        if (clean) {
          try {
            this.#o.onText(clean, batch.seqs);
          } catch (e) {
            console.error("[STT] transcript handler failed:", e);
          }
        }
        this.#pump();
      },
      (error: unknown) => {
        if (controller.signal.aborted || this.#disposed) return;
        this.#inFlight = null;
        this.#onFailure(batch, error);
        this.#pump();
      },
    );
  }

  #onFailure(batch: UtteranceBatch, error: unknown): void {
    const now = this.#o.now();
    const detail = error instanceof Error ? error.message : String(error);
    const kind: TranscriptionFailureKind = error instanceof TranscriptionError ? error.kind : "retryable";
    if (kind === "fatal") {
      this.#fail({ reason: "fatal", detail });
      return;
    }
    this.#retry = batch;
    if (kind === "rate_limited") {
      const retryAfter = error instanceof TranscriptionError ? error.retryAfterMs : null;
      const waitMs = Math.max(retryAfter ?? this.#o.rateLimitFallbackMs, 250);
      this.#blockedUntil = Math.max(this.#blockedUntil, now + waitMs);
      this.#setStatus({ kind: "rate_limited", waitMs, queued: this.queued });
      return;
    }
    this.#attempts++;
    if (this.#attempts >= this.#o.maxAttempts) {
      this.#fail({ reason: "failed", attempts: this.#attempts, detail });
      return;
    }
    const waitMs = Math.min(this.#o.retryMaxMs, this.#o.retryBaseMs * 2 ** (this.#attempts - 1));
    this.#blockedUntil = Math.max(this.#blockedUntil, now + waitMs);
    this.#setStatus({ kind: "retrying", attempt: this.#attempts, maxAttempts: this.#o.maxAttempts, waitMs, detail });
  }
}
