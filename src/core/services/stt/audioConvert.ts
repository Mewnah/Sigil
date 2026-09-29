/**
 * PCM helpers for Rust `audio:chunk` payloads (Vosk). Rust captures at the DEVICE rate and channel count (the rate
 * requested by `start_audio_capture` is ignored). Engines that need mono 16 kHz (Moonshine, local HTTP STT, Groq)
 * use Rust utterance capture instead, which already delivers 16 kHz mono PCM (`speech_core` + rubato).
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
