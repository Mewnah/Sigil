/**
 * PCM helpers for Rust `audio:chunk` payloads (Vosk). With `sampleRate: 16000` the capture hub sends 16 kHz mono;
 * otherwise payloads carry the device's own rate and channel count (each payload states its format). Engines that
 * need mono 16 kHz without IPC (Moonshine, HTTP servers, Groq, Whisper, `local`) run in the Rust speech pipeline
 * instead (`speech_core` + rubato).
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

/**
 * Linear-interpolation resampling of one mono block (stateless: each block is resampled on its own). Only a fallback
 * for a payload whose rate differs from what a recognizer was created with; the hub already delivers 16 kHz with a
 * proper low-pass (rubato), which this does not replace.
 */
export function resampleLinear(samples: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (!(fromRate > 0) || !(toRate > 0) || fromRate === toRate || samples.length === 0) return samples;
  const outLength = Math.max(1, Math.round((samples.length * toRate) / fromRate));
  const out = new Float32Array(outLength);
  const step = fromRate / toRate;
  const last = samples.length - 1;
  for (let i = 0; i < outLength; i++) {
    const pos = i * step;
    const j = Math.min(last, Math.floor(pos));
    const frac = pos - j;
    const a = samples[j]!;
    const b = samples[Math.min(last, j + 1)]!;
    out[i] = a + (b - a) * frac;
  }
  return out;
}
