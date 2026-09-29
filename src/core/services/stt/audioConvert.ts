/**
 * PCM helpers for Rust `audio:chunk` payloads. Rust captures at the DEVICE rate and channel count (the rate
 * requested by `start_audio_capture` is ignored), so engines that need a fixed format (Moonshine and local HTTP
 * STT expect mono 16 kHz) must downmix and resample here.
 */

/** Downmix interleaved PCM to mono by averaging the channels (a trailing partial frame is dropped). */
export function toMonoFloat32(samples: ArrayLike<number>, channels: number): Float32Array {
  if (!(channels > 1)) {
    return new Float32Array(samples);
  }
  const n = Math.floor(samples.length / channels);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) {
      sum += samples[i * channels + c]!;
    }
    out[i] = sum / channels;
  }
  return out;
}

/** Zero crossings of the windowed-sinc kernel on each side (quality vs. cost). */
const ZERO_CROSSINGS = 16;
/** Cutoff as a fraction of the lower Nyquist rate; leaves room for the transition band below it. */
const ROLLOFF = 0.9;
/** Kernel table resolution: entries per input sample (linear interpolation in between). */
const TABLE_RES = 128;

/**
 * Streaming band-limited resampler (Blackman-windowed sinc, mono f32). State carries across `process()` calls,
 * so feeding arbitrary chunk sizes yields the same output as one contiguous block (no clicks at chunk edges).
 *
 * The low-pass cutoff is `ROLLOFF` x the lower of the two Nyquist rates, so decimation (48 kHz -> 16 kHz) removes
 * content above ~7.2 kHz before it can alias. Output is normalised by the kernel sum, so DC gain is exactly 1.
 *
 * Self-check (by hand, no test runner): 48 kHz or 44.1 kHz -> 16 kHz of a 1 kHz sine matches
 * `sin(2*pi*1000*m/16000)` to ~1e-5 away from the edges, identically for any chunking; tones >= 8.5 kHz come out
 * ~-77 dB; `inRate === outRate` is a passthrough. No time shift: output m sits at input time m/outRate (zero
 * history before the first sample); only the newest ~`halfWidth` input samples wait for the next chunk.
 */
export class StreamingResampler {
  readonly inRate: number;
  readonly outRate: number;
  readonly #passthrough: boolean;
  /** Kernel half-width in input samples. */
  readonly #halfWidth: number;
  /** kernel(|d|) sampled at d = i / TABLE_RES for i = 0 .. halfWidth * TABLE_RES + 1 (last entry 0). */
  readonly #table: Float32Array;
  /** Input history (zero-primed) plus samples not yet consumed. */
  #buf: Float32Array;
  #len: number;
  /** Next output position relative to `#buf[0]`, in input samples x `outRate` (exact integer arithmetic). */
  #posNum: number;

  constructor(inRate: number, outRate: number) {
    this.inRate = Math.max(1, Math.round(inRate));
    this.outRate = Math.max(1, Math.round(outRate));
    this.#passthrough = this.inRate === this.outRate;

    // Normalised cutoff in input-sample units (1 = input Nyquist).
    const cutoff = Math.min(1, this.outRate / this.inRate) * ROLLOFF;
    const halfWidth = Math.ceil(ZERO_CROSSINGS / cutoff);
    this.#halfWidth = halfWidth;

    const table = new Float32Array(halfWidth * TABLE_RES + 2);
    for (let i = 0; i <= halfWidth * TABLE_RES; i++) {
      const x = i / TABLE_RES;
      const t = cutoff * x;
      const sinc = t === 0 ? 1 : Math.sin(Math.PI * t) / (Math.PI * t);
      const r = x / halfWidth; // 0..1
      const blackman = 0.42 + 0.5 * Math.cos(Math.PI * r) + 0.08 * Math.cos(2 * Math.PI * r);
      table[i] = cutoff * sinc * blackman;
    }
    this.#table = table;

    this.#buf = new Float32Array(Math.max(4096, 4 * halfWidth));
    this.#len = halfWidth; // zero history before the first input sample
    this.#posNum = halfWidth * this.outRate;
  }

  /** Resample the next block of mono input; returns the outputs that are complete so far (may be empty). */
  process(input: Float32Array): Float32Array {
    if (this.#passthrough) return input;
    if (input.length === 0) return new Float32Array(0);

    // Append.
    if (this.#len + input.length > this.#buf.length) {
      const grown = new Float32Array(Math.max(this.#buf.length * 2, this.#len + input.length));
      grown.set(this.#buf.subarray(0, this.#len));
      this.#buf = grown;
    }
    this.#buf.set(input, this.#len);
    this.#len += input.length;

    const H = this.#halfWidth;
    const inRate = this.inRate;
    const outRate = this.outRate;
    const buf = this.#buf;
    const table = this.#table;
    const len = this.#len;

    const maxOut = Math.max(0, Math.ceil(((len - H) * outRate - this.#posNum) / inRate) + 1);
    const out = new Float32Array(maxOut);
    let produced = 0;
    let posNum = this.#posNum;

    for (;;) {
      const i0 = Math.floor(posNum / outRate);
      if (i0 + H >= len || produced >= maxOut) break;
      const frac = (posNum - i0 * outRate) / outRate; // [0, 1)
      let acc = 0;
      let wsum = 0;
      // Taps k = i0-H+1 .. i0+H, i.e. distance d = i0 + frac - k in (-H, H].
      for (let k = i0 - H + 1; k <= i0 + H; k++) {
        const pos = Math.abs(i0 + frac - k) * TABLE_RES;
        const j = pos | 0;
        const w = table[j]! + (table[j + 1]! - table[j]!) * (pos - j);
        acc += buf[k]! * w;
        wsum += w;
      }
      out[produced++] = wsum !== 0 ? acc / wsum : 0;
      posNum += inRate;
    }

    // Drop input no later output can reach (the next output's first tap is i0' - H + 1).
    const nextI0 = Math.floor(posNum / outRate);
    const drop = Math.max(0, Math.min(len, nextI0 - H + 1));
    if (drop > 0) {
      buf.copyWithin(0, drop, len);
      this.#len = len - drop;
      posNum -= drop * outRate;
    }
    this.#posNum = posNum;

    return produced === out.length ? out : out.subarray(0, produced);
  }
}

/**
 * Converts Rust `audio:chunk` payloads (device rate, interleaved channels) to mono `targetRate`. Keeps one
 * resampler across chunks; a chunk at a different device rate (new capture on another device) starts a fresh one.
 */
export class CaptureToMonoConverter {
  readonly targetRate: number;
  #resampler: StreamingResampler | null = null;

  constructor(targetRate: number) {
    this.targetRate = targetRate;
  }

  convert(samples: ArrayLike<number>, sampleRate: number, channels: number): Float32Array {
    const mono = toMonoFloat32(samples, channels);
    // Unknown rate: nothing to resample against; assume it is already the target rate.
    if (!(sampleRate > 0)) return mono;
    const rate = Math.round(sampleRate);
    if (!this.#resampler || this.#resampler.inRate !== rate) {
      this.#resampler = new StreamingResampler(rate, this.targetRate);
    }
    return this.#resampler.process(mono);
  }
}
