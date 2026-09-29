/**
 * PCM helpers for Rust `audio:chunk` payloads (Vosk). With `sampleRate: 16000` the capture hub sends 16 kHz mono;
 * otherwise payloads carry the device's own rate and channel count (each payload states its format). Engines that
 * need mono 16 kHz without IPC (Moonshine, local HTTP STT, Groq) use Rust utterance capture instead
 * (`speech_core` + rubato).
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
