/**
 * PCM helpers for the TTS sink (P2-G): WAV parsing and 16-bit PCM conversion in TypeScript, so clips reach the
 * Rust sink as f32 samples over binary IPC (never as JSON number arrays). Compressed clips (MP3, Ogg) fall back
 * to the webview's decoder.
 */

/** Decoded interleaved f32 PCM. */
export interface DecodedPcm {
  samples: Float32Array;
  sampleRate: number;
  channels: number;
}

function view(bytes: ArrayBuffer | Uint8Array): DataView {
  return bytes instanceof Uint8Array
    ? new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
    : new DataView(bytes);
}

function fourcc(dv: DataView, at: number): string {
  return String.fromCharCode(dv.getUint8(at), dv.getUint8(at + 1), dv.getUint8(at + 2), dv.getUint8(at + 3));
}

/** Whether `bytes` start like a RIFF/WAVE file. */
export function isWav(bytes: ArrayBuffer | Uint8Array): boolean {
  const dv = view(bytes);
  return dv.byteLength >= 12 && fourcc(dv, 0) === "RIFF" && fourcc(dv, 8) === "WAVE";
}

const WAVE_FORMAT_PCM = 1;
const WAVE_FORMAT_IEEE_FLOAT = 3;
const WAVE_FORMAT_EXTENSIBLE = 0xfffe;

/**
 * Decode a WAV file (integer PCM 8/16/24/32-bit or float 32/64-bit, plain or WAVE_FORMAT_EXTENSIBLE). A `data`
 * chunk whose size runs past the end (streamed WAVs write 0xFFFFFFFF) is read to the end of the buffer. Throws a
 * readable error for anything else.
 */
export function decodeWav(bytes: ArrayBuffer | Uint8Array): DecodedPcm {
  const dv = view(bytes);
  if (!isWav(bytes)) throw new Error("not a WAV file");
  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bits = 0;
  let offset = 12;
  while (offset + 8 <= dv.byteLength) {
    const id = fourcc(dv, offset);
    const size = dv.getUint32(offset + 4, true);
    const body = offset + 8;
    if (id === "fmt ") {
      if (size < 16 || body + 16 > dv.byteLength) throw new Error("WAV: truncated fmt chunk");
      format = dv.getUint16(body, true);
      channels = dv.getUint16(body + 2, true);
      sampleRate = dv.getUint32(body + 4, true);
      bits = dv.getUint16(body + 14, true);
      if (format === WAVE_FORMAT_EXTENSIBLE && size >= 40 && body + 26 <= dv.byteLength) {
        // The sub-format GUID starts with the plain format code.
        format = dv.getUint16(body + 24, true);
      }
    } else if (id === "data") {
      if (!channels || !sampleRate) throw new Error("WAV: data before fmt");
      const end = Math.min(dv.byteLength, body + size);
      return { samples: readSamples(dv, body, end, format, bits), sampleRate, channels };
    }
    // Chunks are word-aligned.
    offset = body + size + (size & 1);
  }
  throw new Error("WAV: no data chunk");
}

function readSamples(dv: DataView, start: number, end: number, format: number, bits: number): Float32Array {
  const bytesPer = bits / 8;
  if (!Number.isInteger(bytesPer) || bytesPer < 1) throw new Error(`WAV: ${bits}-bit samples are not supported`);
  const count = Math.floor((end - start) / bytesPer);
  const out = new Float32Array(count);
  if (format === WAVE_FORMAT_IEEE_FLOAT) {
    if (bits === 32) for (let i = 0; i < count; i++) out[i] = dv.getFloat32(start + i * 4, true);
    else if (bits === 64) for (let i = 0; i < count; i++) out[i] = dv.getFloat64(start + i * 8, true);
    else throw new Error(`WAV: ${bits}-bit float is not supported`);
    for (let i = 0; i < count; i++) if (!Number.isFinite(out[i])) out[i] = 0;
    return out;
  }
  if (format !== WAVE_FORMAT_PCM) throw new Error(`WAV: format ${format} is not supported (PCM or float only)`);
  switch (bits) {
    case 8:
      for (let i = 0; i < count; i++) out[i] = (dv.getUint8(start + i) - 128) / 128;
      break;
    case 16:
      for (let i = 0; i < count; i++) out[i] = dv.getInt16(start + i * 2, true) / 32768;
      break;
    case 24:
      for (let i = 0; i < count; i++) {
        const at = start + i * 3;
        const v = dv.getUint8(at) | (dv.getUint8(at + 1) << 8) | (dv.getInt8(at + 2) << 16);
        out[i] = v / 8388608;
      }
      break;
    case 32:
      for (let i = 0; i < count; i++) out[i] = dv.getInt32(start + i * 4, true) / 2147483648;
      break;
    default:
      throw new Error(`WAV: ${bits}-bit PCM is not supported`);
  }
  return out;
}

/**
 * Little-endian 16-bit PCM to f32. Streams can split a sample across chunks: pass the previous call's `carry`
 * (an odd trailing byte) and keep the returned one.
 */
export function pcm16leToFloat32(
  bytes: ArrayBuffer | Uint8Array,
  carry: number | null = null,
): { samples: Float32Array; carry: number | null } {
  let src = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (carry !== null) {
    const merged = new Uint8Array(src.byteLength + 1);
    merged[0] = carry;
    merged.set(src, 1);
    src = merged;
  }
  const count = src.byteLength >> 1;
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    const v = src[2 * i] | (src[2 * i + 1] << 8);
    out[i] = (v >= 0x8000 ? v - 0x10000 : v) / 32768;
  }
  return { samples: out, carry: src.byteLength & 1 ? src[src.byteLength - 1] : null };
}

/** Rate the webview decodes compressed clips at (its decoder resamples to the context rate). */
export const WEB_DECODE_RATE = 48_000;

/**
 * Decode a clip of any format the webview understands: WAV in TypeScript (exact), anything else (MP3, Ogg, …)
 * with Web Audio's decoder at {@link WEB_DECODE_RATE}.
 */
export async function decodeAudioClip(bytes: ArrayBuffer | Uint8Array): Promise<DecodedPcm> {
  if (isWav(bytes)) return decodeWav(bytes);
  const Offline = (globalThis as { OfflineAudioContext?: typeof OfflineAudioContext }).OfflineAudioContext;
  if (!Offline) throw new Error("this audio format cannot be decoded here");
  const ctx = new Offline(1, 1, WEB_DECODE_RATE);
  // decodeAudioData detaches its argument: hand it a private copy.
  const copy = bytes instanceof Uint8Array ? bytes.slice().buffer : bytes.slice(0);
  const buffer = await ctx.decodeAudioData(copy);
  const channels = Math.max(1, buffer.numberOfChannels);
  const frames = buffer.length;
  const samples = new Float32Array(frames * channels);
  for (let c = 0; c < channels; c++) {
    const data = buffer.getChannelData(c);
    for (let f = 0; f < frames; f++) samples[f * channels + c] = data[f];
  }
  return { samples, sampleRate: buffer.sampleRate, channels };
}
