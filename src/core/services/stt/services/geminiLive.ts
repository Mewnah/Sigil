/**
 * The `geminiLive` STT backend (P6-K): Gemini 3.5 Transcribe Live on the user's own key, over a WebSocket from the
 * webview, and (preview) Gemini 3.5 Live Translate as a combined backend that publishes `text.stt` and
 * `text.translation` from the same audio.
 *
 * Audio: the capture hub's raw 16 kHz mono feed (`plugin:audio|start_audio_capture({ sampleRate: 16000 })`, the same
 * raw microphone every STT engine gets, withheld by Rust while STT is hard-muted), cut into 100 ms frames of 16-bit
 * PCM and sent as `realtimeInput.audio`. Muting sends `audioStreamEnd` so the open sentence is finalized.
 *
 * Keys: the API key only ever travels in the `x-goog-api-key` header of one HTTPS request that mints a single-use
 * ephemeral token (`POST /<version>/auth_tokens`); the WebSocket URL carries that short-lived token
 * (`BidiGenerateContentConstrained?access_token=`), never the key. Nothing logs the URL.
 *
 * Sessions end after 10 minutes, so {@link GeminiLiveRotator} rotates them without a caption gap:
 * 1. {@link PREPARE_LEAD_MS} before the rotation point (`rotateMinutes`, default 9.5) the next session is connected;
 * 2. at the rotation point, as soon as the old session has no open sentence and the microphone is quiet (or after
 *    {@link PAUSE_WAIT_MS} at the latest), both sessions hear the same audio (the overlap, from a short pre-roll on);
 * 3. after at least {@link OVERLAP_MS}, at the next pause (or after {@link MAX_OVERLAP_MS}), the old session gets
 *    `audioStreamEnd` and drains its last transcript, while the new one keeps listening;
 * 4. then the new session takes over: what it heard during the overlap is published through a {@link SeamGate} that
 *    drops the words the old session already published (suffix / prefix alignment of the two transcripts, tolerant
 *    of a cut word and small recognition differences, bounded by the overlap's duration; character-level for
 *    scripts without spaces). Only the old session's finals about speech the new one heard go into the gate (the
 *    newest loud frame the old session had got when the final arrived is not older than the new session's first
 *    frame). The gate follows the new session's first utterances for {@link SEAM_GRACE_MS} after the switch, so a
 *    repeat the new session reports a little late is dropped too.
 * A dropped connection reconnects with back-off and replays the buffered audio from shortly before the last final
 * (up to {@link MAX_REPLAY_MS}), and in any case all speech the lost session may never have got (frames it was not
 * sent, and speech sent after its last transcript event, which a dead connection swallows), through a seam gate of
 * the lost session's finals, so speech said while reconnecting is transcribed late instead of lost, and not twice.
 * Audio a session cannot take yet (a full send buffer, e.g. during such a replay) waits in the ring and goes out on
 * the next frame or tick; a buffer that stops draining replaces the session the same way. Failures of a rotation's
 * next session back off further each time without ending the running one. A `goAway` from the server starts the
 * rotation at once. While
 * muted (no audio), a session that reaches its rotation point is closed instead of replaced, and the next audio
 * reconnects.
 *
 * Pure parts (protocol, seam alignment, rotation state machine with injected sessions and clock, normalizers, output
 * stages) are exported for `__tests__/geminiRotation.test.ts`.
 */
import { invoke, isTauri } from "@tauri-apps/api/core";
import type { UnlistenFn } from "@tauri-apps/api/event";
import i18n from "i18next";
import { ServiceNetworkState, TextEventSource } from "@/types";
import { devLog } from "@/utils/devLog";
import { PROCESS_NONCE } from "@/utils/processNonce";
import { resolveRustInputDeviceForInvoke } from "@/utils/rustAudioInput";
import { httpError } from "@/core/services/llm/client";
import { redactKeys } from "@/core/services/llm/sanitize";
import {
  abortReason,
  createRustTransport,
  createWebviewTransport,
  LlmTransportError,
  type LlmHttpRequest,
  type LlmHttpTransport,
} from "@/core/services/llm/transport";
import { canonicalLanguage, UNSPACED_SCRIPT_RE } from "@/core/services/translation/languages";
import {
  buildLiveTranslateSetups,
  geminiLiveModelName,
  LiveTranslateNormalizer,
  LiveTranslationPublisher,
  LIVE_TRANSLATE_STT_STREAM,
  liveTranslateTargetCode,
  TRANSLATION_SERVICE_SKIPS_LIVE_TRANSLATE,
  type LiveChannelSink,
} from "@/core/services/translation/services/geminiLive";
import { isAbortError, listenWithAbort, raceWithAbort } from "../abort";
import { resampleLinear, toMonoFloat32 } from "../audioConvert";
import { SttStartLifecycle } from "../localEngineLifecycle";
import { baseLanguageTag, errorText, shortDetail } from "../rustCapturePolling";
import type { STT_State } from "../schema";
import {
  GEMINI_LIVE_API_VERSIONS,
  GEMINI_LIVE_ROTATE_DEFAULT,
  GEMINI_LIVE_ROTATE_MAX,
  GEMINI_LIVE_ROTATE_MIN,
  type GeminiLiveMode,
} from "../schema/geminiLive";
import { TRANSIENT_WARNING_MS } from "../statusChannel";
import type { ISTTReceiver, ISTTService, RustAudioChunkPayload, SttStatusOptions, SttTextMeta } from "../types";

// ---------------------------------------------------------------------------------------------------------------
// Constants

/** Gemini Live sessions end after 10 minutes. */
export const GEMINI_LIVE_SESSION_LIMIT_MS = 600_000;
/** A rotation finishes this long before the limit. */
export const SESSION_SAFETY_MS = 5_000;
/** The next session is connected this long before the rotation point (token, socket, setup). */
export const PREPARE_LEAD_MS = 10_000;
/** After the rotation point, the overlap waits up to this long for a pause before it starts anyway. */
export const PAUSE_WAIT_MS = 12_000;
/** Both sessions hear the same audio for at least this long. */
export const OVERLAP_MS = 1_000;
/** The old session stops hearing audio at the latest this long after the overlap started. */
export const MAX_OVERLAP_MS = 8_000;
/** Audio from before the overlap sent to the new session (it starts at a word boundary, not inside a word). */
export const PRE_ROLL_MS = 400;
/** Quiet microphone time that counts as a pause to start the overlap. */
export const QUIET_START_MS = 300;
/** Quiet microphone time that counts as a pause to end the old session's audio. */
export const QUIET_CUT_MS = 250;
/** After the old session's audio ended, its last transcript is awaited at least this long. */
export const DRAIN_MIN_MS = 600;
/** ... and at most this long, by mode and by whether it still had a sentence open (translation lags behind). */
export const DRAIN_MAX_MS: Readonly<Record<GeminiLiveMode, { idle: number; open: number }>> = {
  transcribe: { idle: 1_200, open: 3_000 },
  translate: { idle: 2_500, open: 8_000 },
};
/**
 * After a switch, an utterance of the new session that starts this long after it took over is new speech: the seam
 * gate no longer applies to it (by mode: translations lag behind the speech).
 */
export const SEAM_GRACE_MS: Readonly<Record<GeminiLiveMode, number>> = { transcribe: 1_500, translate: 5_000 };
/** Audio kept for overlaps and reconnect replays. */
export const AUDIO_RING_MS = 30_000;
/**
 * A reconnect replays at most this much audio the lost session had already transcribed (speech it may never have
 * got, e.g. during a stall, is replayed however old it is, up to {@link AUDIO_RING_MS}).
 */
export const MAX_REPLAY_MS = 15_000;
/**
 * A reconnect with a sentence open replays from this long before the last final arrived (the open sentence may have
 * begun while that final was still on its way); the seam gate drops what the lost session had already published.
 */
export const REPLAY_FINAL_MARGIN_MS = 1_000;
/** Without an open sentence, a reconnect replays this much audio from before the drop (speech not yet reported). */
export const REPLAY_IDLE_MS = 2_000;
/** Audio frame: 100 ms at 16 kHz, the documented chunk size. */
export const SAMPLE_RATE = 16_000;
export const FRAME_SAMPLES = 1_600;
/** Tick of the rotation state machine. */
export const TICK_MS = 100;
/** A session that ran this long resets the failure counts. */
export const STABLE_MS = 60_000;
/** Back-off of consecutive failed connections (the last value repeats). */
export const BACKOFF_MS = [500, 1_000, 2_000, 4_000, 8_000, 15_000] as const;
/** Back-off after a quota / rate-limit refusal without a server-given wait. */
export const QUOTA_BACKOFF_MS = [5_000, 15_000, 30_000, 60_000] as const;
export const MAX_FAILURES = 6;
export const MAX_QUOTA_FAILURES = 4;
/** No audio for this long (muted, capture paused) counts as idle: a session at its rotation point is closed, not replaced. */
export const AUDIO_IDLE_MS = 5_000;
/**
 * Above this many bytes waiting in the socket's send buffer, audio waits in the ring instead (a reconnect replay is a
 * burst of up to {@link AUDIO_RING_MS} of audio, about 1.3 MB; a slow uplink drains it over the next frames).
 */
export const SEND_BUFFER_HIGH_BYTES = 256 * 1024;
/** A send buffer that stays above that mark without draining for this long means the network stalled. */
export const SEND_STALL_MS = 10_000;
export const SETUP_TIMEOUT_MS = 15_000;
export const TOKEN_TIMEOUT_MS = 15_000;
/** Connection attempts of the first session before the start gives up. */
export const START_ATTEMPTS = 3;

export const GEMINI_LIVE_HOST = "generativelanguage.googleapis.com";
const PCM_MIME = `audio/pcm;rate=${SAMPLE_RATE}`;
const CHANNELS = ["source", "translation"] as const;

// ---------------------------------------------------------------------------------------------------------------
// Protocol (pure)

export type ApiVersion = (typeof GEMINI_LIVE_API_VERSIONS)[number];

function apiVersionOf(v: string): ApiVersion {
  return (GEMINI_LIVE_API_VERSIONS as readonly string[]).includes(v) ? (v as ApiVersion) : "v1beta";
}

/** Ephemeral-token endpoint (the key goes in the `x-goog-api-key` header). */
export function authTokenUrl(apiVersion: string): string {
  return `https://${GEMINI_LIVE_HOST}/${apiVersionOf(apiVersion)}/auth_tokens`;
}

/**
 * WebSocket URL for a session authorized by an ephemeral token (never a key). Never log it. The token name
 * (`auth_tokens/...`) keeps its `/`, as the official SDK sends it; everything else is percent-encoded.
 */
export function liveSocketUrl(apiVersion: string, token: string): string {
  const v = apiVersionOf(apiVersion);
  const accessToken = encodeURIComponent(token).replace(/%2F/gi, "/");
  return `wss://${GEMINI_LIVE_HOST}/ws/google.ai.generativelanguage.${v}.GenerativeService.BidiGenerateContentConstrained?access_token=${accessToken}`;
}

/** `languageCodes` hint: `[]` (detect) for auto / blank / malformed, else the BCP-47 tag. */
export function liveLanguageCodes(language: string): string[] {
  const tag = String(language ?? "").trim();
  if (!tag || tag.toLowerCase() === "auto") return [];
  return /^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$/.test(tag) ? [tag] : [];
}

/** Maximum custom vocabulary terms (the service limit). */
export const MAX_VOCABULARY = 1_000;

/** Terms of the vocabulary field: split on new lines, commas and semicolons (also CJK ones), trimmed, unique, capped. */
export function parseVocabulary(text: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of String(text ?? "").split(/[\n\r,;、，；]+/)) {
    const term = raw.trim().slice(0, 100);
    const key = term.toLowerCase();
    if (!term || seen.has(key)) continue;
    seen.add(key);
    out.push(term);
    if (out.length >= MAX_VOCABULARY) break;
  }
  return out;
}

export interface TranscribeSetupOptions {
  model: string;
  language: string;
  formatting: "verbatim" | "smart";
  vocabulary: string[];
}

/**
 * Setup messages for Transcribe Live, in order: with the custom vocabulary / smart formatting when set, then without
 * them (the driver falls back to it, with a warning, when the service rejects those fields). Verbatim is the service
 * default, so it is not sent.
 */
export function buildTranscribeSetups(o: TranscribeSetupOptions): Record<string, unknown>[] {
  const model = geminiLiveModelName(o.model);
  const languageCodes = liveLanguageCodes(o.language);
  const base = { model, generationConfig: { responseModalities: ["TEXT"] } };
  const minimal = { setup: { ...base, inputAudioTranscription: { languageCodes } } };
  const extras: Record<string, unknown> = {};
  if (o.vocabulary.length > 0) extras.customVocabulary = o.vocabulary.slice(0, MAX_VOCABULARY);
  if (o.formatting === "smart") extras.mode = "SMART";
  if (Object.keys(extras).length === 0) return [minimal];
  return [{ setup: { ...base, inputAudioTranscription: { languageCodes, ...extras } } }, minimal];
}

/** One parsed server message (a message may carry several parts). */
export type LiveServerMessage =
  | { kind: "setupComplete" }
  | {
      kind: "content";
      /** `interimInputTranscription.text`: the hypothesis since the last final (replaces the previous one). */
      interimInput?: string;
      /** `interimInputTranscription.languageCode` (the detected language), when sent. */
      interimLanguageCode?: string;
      /** `inputTranscription`: a complete segment (Transcribe Live) or a piece of the source (Live Translate). */
      input?: { text: string; languageCode?: string };
      /** `outputTranscription.text`: a piece of the translation. */
      output?: string;
      /** `turnComplete` / `generationComplete`. */
      turnComplete: boolean;
    }
  | { kind: "goAway"; timeLeftMs: number | null }
  | { kind: "other" };

/** Protobuf JSON duration ("12s", "1.5s", or `{ seconds, nanos }`) in ms; null when absent or malformed. */
export function parseDurationMs(value: unknown): number | null {
  if (typeof value === "string") {
    const m = /^\s*(\d+(?:\.\d+)?)s\s*$/.exec(value);
    return m ? Math.round(Number(m[1]) * 1000) : null;
  }
  if (value && typeof value === "object") {
    const rec = value as { seconds?: unknown; nanos?: unknown };
    const s = Number(rec.seconds ?? 0);
    const n = Number(rec.nanos ?? 0);
    if (Number.isFinite(s) && Number.isFinite(n)) return Math.round(s * 1000 + n / 1e6);
  }
  return null;
}

function textOf(value: unknown): string | undefined {
  if (!value || typeof value !== "object") return undefined;
  const text = (value as { text?: unknown }).text;
  return typeof text === "string" ? text : "";
}

function languageCodeOf(value: unknown): string | undefined {
  const code = (value as { languageCode?: unknown } | null | undefined)?.languageCode;
  return typeof code === "string" && code ? code : undefined;
}

/** Parses one server frame (JSON text). Unknown or malformed frames are `other`. */
export function parseServerMessage(raw: string): LiveServerMessage[] {
  let json: unknown;
  try {
    json = JSON.parse(raw);
  } catch {
    return [{ kind: "other" }];
  }
  if (!json || typeof json !== "object") return [{ kind: "other" }];
  const rec = json as Record<string, unknown>;
  const out: LiveServerMessage[] = [];
  if ("setupComplete" in rec) out.push({ kind: "setupComplete" });
  const content = rec.serverContent;
  if (content && typeof content === "object") {
    const c = content as Record<string, unknown>;
    const msg: Extract<LiveServerMessage, { kind: "content" }> = {
      kind: "content",
      turnComplete: c.turnComplete === true || c.generationComplete === true,
    };
    const interim = textOf(c.interimInputTranscription);
    if (interim !== undefined) {
      msg.interimInput = interim;
      const code = languageCodeOf(c.interimInputTranscription);
      if (code) msg.interimLanguageCode = code;
    }
    const input = textOf(c.inputTranscription);
    if (input !== undefined) {
      const code = languageCodeOf(c.inputTranscription);
      msg.input = code ? { text: input, languageCode: code } : { text: input };
    }
    const output = textOf(c.outputTranscription);
    if (output !== undefined) msg.output = output;
    out.push(msg);
  }
  if (rec.goAway && typeof rec.goAway === "object") {
    out.push({ kind: "goAway", timeLeftMs: parseDurationMs((rec.goAway as { timeLeft?: unknown }).timeLeft) });
  }
  return out.length > 0 ? out : [{ kind: "other" }];
}

/** Why a session failed: `auth` and `config` end the session; `quota` and `transient` are retried. */
export type LiveFailureKind = "auth" | "config" | "quota" | "transient";

/** A failed session open or a lost session, with a message that is safe to show (keys redacted). */
export class LiveSessionError extends Error {
  constructor(
    readonly kind: LiveFailureKind,
    message: string,
    readonly info: { code?: number; beforeSetup?: boolean; retryAfterMs?: number } = {},
  ) {
    super(message);
    this.name = "LiveSessionError";
  }
}

/** Kind of a WebSocket close by its code and reason (the Live API puts the error text in the reason). */
export function classifyClose(code: number, reason: string): LiveFailureKind {
  const r = String(reason ?? "").toLowerCase();
  if (/quota|resource[_ ]exhausted|rate limit|too many requests/.test(r)) return "quota";
  if (/api key|api_key|unauthenticated|permission[_ ]denied|not authorized|unauthorized|forbidden|billing/.test(r)) return "auth";
  if (
    code === 1007 ||
    code === 1008 ||
    /invalid argument|invalid_argument|not found|not_found|not supported|unsupported|unknown name|invalid json|cannot find field|invalid value/.test(r)
  )
    return "config";
  return "transient";
}

/** "1011: reason" (reason redacted, collapsed and capped), or the bare code. */
export function closeDetail(code: number, reason: string): string {
  const r = shortDetail(redactKeys(String(reason ?? "")), 160);
  return r ? `${code}: ${r}` : String(code);
}

// ---------------------------------------------------------------------------------------------------------------
// Audio (pure)

/** f32 [-1, 1] → 16-bit PCM (clamped). */
export function floatToPcm16(samples: Float32Array): Int16Array {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i] || 0));
    out[i] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
  }
  return out;
}

/** Little-endian bytes of `pcm`, base64-encoded. */
export function pcm16ToBase64(pcm: Int16Array): string {
  const bytes = new Uint8Array(pcm.length * 2);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < pcm.length; i++) view.setInt16(i * 2, pcm[i], true);
  let binary = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + CHUNK)));
  }
  return btoa(binary);
}

/** Cuts a stream of 16 kHz samples into {@link FRAME_SAMPLES}-sample PCM frames. */
export class PcmFramer {
  #buf = new Float32Array(FRAME_SAMPLES);
  #len = 0;

  push(samples: Float32Array, emit: (frame: Int16Array) => void): void {
    let i = 0;
    while (i < samples.length) {
      const n = Math.min(FRAME_SAMPLES - this.#len, samples.length - i);
      this.#buf.set(samples.subarray(i, i + n), this.#len);
      this.#len += n;
      i += n;
      if (this.#len === FRAME_SAMPLES) {
        emit(floatToPcm16(this.#buf));
        this.#len = 0;
      }
    }
  }

  reset(): void {
    this.#len = 0;
  }
}

/**
 * Tracks when the microphone was last active (energy above an adaptive noise floor), to time session switches at
 * pauses. A frame is active when it is 10 dB above the floor (and above -55 dBFS), or louder than -35 dBFS.
 */
export class QuietTracker {
  #floor = -60;
  #lastActive = Number.NEGATIVE_INFINITY;
  #lastAt: number | null = null;

  update(frame: Int16Array, now: number): void {
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    const rms = frame.length > 0 ? Math.sqrt(sum / frame.length) / 32768 : 0;
    const db = 20 * Math.log10(Math.max(rms, 1e-9));
    const dt = this.#lastAt === null ? 0 : Math.max(0, (now - this.#lastAt) / 1000);
    this.#lastAt = now;
    // The floor follows quiet frames at once and rises slowly (0.5 dB/s) under steady noise.
    this.#floor = db < this.#floor ? db : Math.min(db, this.#floor + 0.5 * dt);
    if (db > Math.max(this.#floor + 10, -55) || db > -35) this.#lastActive = now;
  }

  /** ms since the last active frame (infinite before any). */
  quietFor(now: number): number {
    return now - this.#lastActive;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Seam dedupe (pure)

/** A comparable unit of text: a word, or one character of a script written without spaces. */
export interface SeamUnit {
  /** Normalized form (NFKC, lower case, edge punctuation removed). */
  norm: string;
  /** Offsets in the original text. */
  start: number;
  end: number;
}

function normalizeUnit(s: string): string {
  return s.normalize("NFKC").toLowerCase().replace(/^[\p{P}\p{S}]+|[\p{P}\p{S}]+$/gu, "");
}

/** Units of `text`: whitespace-separated words, with runs of unspaced scripts (CJK, Thai, ...) split per character. */
export function seamUnits(text: string): SeamUnit[] {
  const units: SeamUnit[] = [];
  const push = (start: number, end: number) => {
    const norm = normalizeUnit(text.slice(start, end));
    if (norm) units.push({ norm, start, end });
  };
  const words = /\S+/g;
  let m: RegExpExecArray | null;
  while ((m = words.exec(text)) !== null) {
    const word = m[0];
    const base = m.index;
    if (!UNSPACED_SCRIPT_RE.test(word)) {
      push(base, base + word.length);
      continue;
    }
    let runStart = -1;
    let i = 0;
    for (const ch of word) {
      const at = base + i;
      if (UNSPACED_SCRIPT_RE.test(ch)) {
        if (runStart >= 0) push(runStart, at);
        runStart = -1;
        push(at, at + ch.length);
      }
      else if (runStart < 0) runStart = at;
      i += ch.length;
    }
    if (runStart >= 0) push(runStart, base + word.length);
  }
  return units;
}

/**
 * Edits tolerated when aligning `len` units (two sessions recognize the same audio a little differently): none for
 * one or two units, one up to seven, then a quarter of the units, at most 8. The anchor (last unit) must match too.
 */
export function allowedEdits(len: number): number {
  if (len < 3) return 0;
  return Math.max(1, Math.min(8, Math.floor(len / 4)));
}

/** Substitution cost: equal units, a new session starting inside a word, or the old one cut inside a word, are free. */
function unitCost(a: string, b: string, firstB: boolean, lastA: boolean): number {
  if (a === b) return 0;
  if (firstB && b.length >= 2 && a.length > b.length && a.endsWith(b)) return 0;
  if (lastA && a.length >= 2 && b.length > a.length && b.startsWith(a)) return 0;
  return 1;
}

/** How many units a seam may cover: the overlap's duration at a fast speaking rate, plus slack. */
export function seamMaxUnits(durationMs: number, unspaced: boolean): number {
  const s = Math.max(0, durationMs) / 1000;
  const units = unspaced ? Math.ceil(s * 10) + 6 : Math.ceil(s * 4) + 4;
  return Math.max(4, Math.min(80, units));
}

export interface SeamAlignment {
  /** Units at the start of the new text that repeat the end of the old one. */
  k: number;
  /** The old text's last word was cut short: keep the new text's complete word (drop `k - 1`). */
  keepLast: boolean;
  cost: number;
}

/**
 * The longest prefix of `b` (at most `maxUnits` units) that repeats the end of `a`: the last unit of `a` must match
 * the last unit of the prefix (anchor), and the units before it may differ by {@link allowedEdits}. When the old
 * session's last word was misheard (no anchor), its second-to-last word anchors instead, at one extra edit and for
 * prefixes of at least 3 units (the misheard word's counterpart in `b` is kept). null = no overlap.
 */
export function alignSeam(a: readonly SeamUnit[], b: readonly SeamUnit[], maxUnits: number): SeamAlignment | null {
  const exact = alignSeamEnding(a, b, maxUnits, 0, 1);
  if (exact || a.length < 2) return exact;
  const relaxed = alignSeamEnding(a.slice(0, -1), b, maxUnits, 1, 3);
  return relaxed ? { ...relaxed, keepLast: false } : null;
}

/** {@link alignSeam} anchored on `a`'s last unit, with `extraCost` edits already spent and a minimum prefix length. */
function alignSeamEnding(
  a: readonly SeamUnit[],
  b: readonly SeamUnit[],
  maxUnits: number,
  extraCost: number,
  minK: number,
): SeamAlignment | null {
  const W = Math.min(a.length, maxUnits + 2);
  const K = Math.min(b.length, maxUnits);
  if (W === 0 || K === 0) return null;
  const A = a.slice(a.length - W);
  // Row i: A[0..i) aligned (free start) with B[0..j). Only rows 0..W-1 are needed (the anchor is A[W-1]).
  let prev = new Array<number>(K + 1);
  for (let j = 0; j <= K; j++) prev[j] = j;
  for (let i = 1; i <= W - 1; i++) {
    const cur = new Array<number>(K + 1);
    cur[0] = 0;
    for (let j = 1; j <= K; j++) {
      cur[j] = Math.min(
        prev[j - 1] + unitCost(A[i - 1].norm, b[j - 1].norm, j === 1, false),
        prev[j] + 1,
        cur[j - 1] + 1,
      );
    }
    prev = cur;
  }
  const last = A[W - 1].norm;
  for (let k = K; k >= minK; k--) {
    const bk = b[k - 1].norm;
    if (unitCost(last, bk, k === 1, true) !== 0) continue;
    const cost = prev[k - 1] + extraCost;
    if (cost > allowedEdits(k)) continue;
    return { k, keepLast: last !== bk && bk.length > last.length && bk.startsWith(last), cost };
  }
  return null;
}

/**
 * Whether `b` lies entirely inside `a`'s last `maxUnits` units without reaching its end (a sentence the new session
 * split off earlier). Returns the index in `a` just after the match (the latest such match), or -1.
 */
export function containedSeam(a: readonly SeamUnit[], b: readonly SeamUnit[], maxUnits: number): number {
  const L = b.length;
  if (L === 0 || L > maxUnits) return -1;
  const W = Math.min(a.length, maxUnits + 2);
  if (W === 0) return -1;
  const offset = a.length - W;
  const A = a.slice(offset);
  // Rows: B[0..i); columns: A[..j) with a free start. Row L-1 is the cost before the anchor B[L-1].
  let prev = new Array<number>(W + 1).fill(0);
  for (let i = 1; i <= L - 1; i++) {
    const cur = new Array<number>(W + 1);
    cur[0] = i;
    for (let j = 1; j <= W; j++) {
      cur[j] = Math.min(prev[j - 1] + unitCost(A[j - 1].norm, b[i - 1].norm, i === 1, false), prev[j] + 1, cur[j - 1] + 1);
    }
    prev = cur;
  }
  const lastB = b[L - 1].norm;
  for (let j = W; j >= 1; j--) {
    if (unitCost(A[j - 1].norm, lastB, L === 1, false) !== 0) continue;
    if (prev[j - 1] <= allowedEdits(L)) return offset + j;
  }
  return -1;
}

/** `text` after its first `drop` units, without the separators the cut leaves at its start. */
export function remainderAfter(text: string, units: readonly SeamUnit[], drop: number): string {
  if (drop <= 0) return text;
  if (drop >= units.length) return "";
  return text.slice(units[drop].start).replace(/^[\s,.;:!?、。，．！？…・」』）)\]}]+/u, "");
}

/**
 * Drops the words of a new session's first utterance that the old session already published (one channel, one
 * session switch). Interims show only the new part; the first final disarms the gate (its duplicate prefix removed,
 * or dropped entirely when it lies inside the old text, the gate then following the rest of the old text).
 */
export class SeamGate {
  #a: SeamUnit[];
  armed = true;

  constructor(oldText: string, readonly maxUnits: number) {
    this.#a = seamUnits(oldText);
    if (this.#a.length === 0) this.armed = false;
  }

  /** Text to show for an interim ("" = only repeated words so far). */
  interim(text: string): string {
    if (!this.armed || !text) return text;
    const b = seamUnits(text);
    const al = alignSeam(this.#a, b, this.maxUnits);
    if (al) return remainderAfter(text, b, al.keepLast ? al.k - 1 : al.k);
    if (containedSeam(this.#a, b, this.maxUnits) >= 0) return "";
    return text;
  }

  /** Text to publish for a final ("" = all of it was already published). */
  final(text: string): string {
    if (!this.armed) return text;
    const b = seamUnits(text);
    const al = alignSeam(this.#a, b, this.maxUnits);
    if (al) {
      this.armed = false;
      return remainderAfter(text, b, al.keepLast ? al.k - 1 : al.k);
    }
    const end = containedSeam(this.#a, b, this.maxUnits);
    if (end >= 0) {
      this.#a = this.#a.slice(end);
      if (this.#a.length === 0) this.armed = false;
      return "";
    }
    this.armed = false;
    return text;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Session content → channel events (pure)

export type LiveChannel = (typeof CHANNELS)[number];

/** A transcript event of one session on one channel (before rotation handling). */
export interface LiveTranscriptEvent {
  channel: LiveChannel;
  kind: "interim" | "final";
  /** Trimmed; an interim "" abandons the open line. */
  text: string;
  lang?: string;
}

/** Turns a session's `content` messages into channel events. */
export interface LiveContentHandler {
  message(msg: Extract<LiveServerMessage, { kind: "content" }>): void;
  /** Idle rules (Live Translate lines). */
  poll(): void;
  /** Publish finished text still being joined (Live Translate source / translation lines). */
  commit(): void;
  /** Also promote the open hypothesis to a final (the session is being replaced). */
  flushOpen(): void;
}

/**
 * Transcribe Live: `interimInputTranscription` replaces the open hypothesis (unchanged ones are skipped) and every
 * `inputTranscription` is a complete segment; an empty final abandons the open hypothesis.
 */
export class TranscribeNormalizer {
  #interim = "";
  #lang: string | undefined;

  constructor(private readonly sink: LiveChannelSink, fixedLang?: string | null) {
    this.#lang = fixedLang ?? undefined;
  }

  get open(): boolean {
    return this.#interim !== "";
  }

  interim(text: string, languageCode?: string): void {
    const lang = baseLanguageTag(languageCode);
    if (lang) this.#lang = lang;
    const t = String(text ?? "").trim();
    if (t === this.#interim) return;
    this.#interim = t;
    this.sink.interim(t, this.#lang);
  }

  final(text: string, languageCode?: string): void {
    const lang = baseLanguageTag(languageCode);
    if (lang) this.#lang = lang;
    const t = String(text ?? "").trim();
    const wasOpen = this.#interim !== "";
    this.#interim = "";
    if (t) this.sink.final(t, this.#lang);
    else if (wasOpen) this.sink.interim("", this.#lang);
  }

  /** The open hypothesis as a final (its session is being replaced and will hear nothing more). */
  flushOpen(): void {
    if (this.#interim) this.final(this.#interim);
  }
}

/** Content handler of one session for `mode`, emitting channel events. */
export function createContentHandler(
  mode: GeminiLiveMode,
  emit: (ev: LiveTranscriptEvent) => void,
  options: { sourceLang?: string | null; now?: () => number } = {},
): LiveContentHandler {
  const sink = (channel: LiveChannel): LiveChannelSink => ({
    interim: (text, lang) => emit(lang ? { channel, kind: "interim", text, lang } : { channel, kind: "interim", text }),
    final: (text, lang) => emit(lang ? { channel, kind: "final", text, lang } : { channel, kind: "final", text }),
  });
  if (mode === "transcribe") {
    const norm = new TranscribeNormalizer(sink("source"), options.sourceLang);
    return {
      message: (msg) => {
        // A final closes the previous segment before a hypothesis for the next one in the same frame.
        if (msg.input) norm.final(msg.input.text, msg.input.languageCode);
        if (msg.interimInput !== undefined) norm.interim(msg.interimInput, msg.interimLanguageCode);
      },
      poll: () => {},
      commit: () => {},
      flushOpen: () => norm.flushOpen(),
    };
  }
  const norm = new LiveTranslateNormalizer({ source: sink("source"), translation: sink("translation") }, options.now);
  return {
    message: (msg) => {
      if (msg.input) norm.input(msg.input.text, msg.input.languageCode);
      if (msg.interimInput !== undefined) norm.interimInput(msg.interimInput);
      if (msg.output !== undefined) norm.output(msg.output);
      if (msg.turnComplete) norm.turnComplete();
    },
    poll: () => norm.poll(),
    commit: () => norm.commit(),
    flushOpen: () => norm.commit(),
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Rotation (pure: sessions, clock and timers are injected)

/** One open session as the rotator sees it. */
export interface LiveSessionHandle {
  /** Sends one PCM frame; false when the session cannot take it (closed or stalled; a close report follows). */
  sendAudio(frame: Int16Array): boolean;
  /** `audioStreamEnd`: no more audio for now (the open sentence is finalized). */
  endAudio(): void;
  poll(): void;
  commit(): void;
  flushOpen(): void;
  /** Closes without a close report. */
  close(): void;
}

export interface LiveSessionCallbacks {
  event(ev: LiveTranscriptEvent): void;
  /** The server ends the connection in `timeLeftMs` (null = soon). */
  goAway(timeLeftMs: number | null): void;
  /** The session ended on its own after it was ready (not after `close()`). */
  closed(error: LiveSessionError): void;
}

/** Opens a session; resolves once it is ready for audio, rejects with a {@link LiveSessionError} (or an abort). */
export type LiveSessionOpener = (callbacks: LiveSessionCallbacks, signal: AbortSignal) => Promise<LiveSessionHandle>;

/** Status reports of the rotator (the driver words them). */
export type LiveRotatorStatus =
  | { kind: "connecting" }
  | { kind: "reconnecting"; attempt: number; detail: string }
  | { kind: "waiting"; seconds: number; detail: string }
  | { kind: "gap" }
  | { kind: "connected" };

export interface LiveRotatorTimers {
  setInterval(fn: () => void, ms: number): unknown;
  clearInterval(handle: unknown): void;
  setTimeout(fn: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}

const defaultTimers: LiveRotatorTimers = {
  setInterval: (fn, ms) => setInterval(fn, ms),
  clearInterval: (h) => clearInterval(h as ReturnType<typeof setInterval>),
  setTimeout: (fn, ms) => setTimeout(fn, ms),
  clearTimeout: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
};

export interface LiveRotatorOptions {
  mode: GeminiLiveMode;
  open: LiveSessionOpener;
  /** Where transcript events go after rotation handling (`translation` only in translate mode). */
  output: { source: LiveChannelSink; translation?: LiveChannelSink };
  status(status: LiveRotatorStatus): void;
  /** The session cannot continue (refused key or settings, or too many failed reconnects). Not called after `stop()`. */
  fatal(error: LiveSessionError): void;
  /** Time from a session's connection to its rotation point. */
  rotateAfterMs: number;
  now?: () => number;
  timers?: LiveRotatorTimers;
  startAttempts?: number;
}

interface HeldEvent {
  text: string;
  lang?: string;
}

interface Slot {
  id: number;
  handle: LiveSessionHandle | null;
  abort: AbortController;
  connectAt: number;
  readyAt: number | null;
  rotateAt: number;
  hardDeadline: number;
  /** Next audio frame (sequence number) to send. */
  nextSeq: number;
  /** Receive time of the first audio frame this session got (null before any). */
  heardFrom: number | null;
  /** Receive time of the newest loud frame this session got (its transcripts cannot cover later speech). */
  lastLoudSentAt: number;
  /** Arrival of the session's last transcript event (null before any): audio sent later may never have reached it. */
  lastEventAt: number | null;
  /** Receives live audio. */
  fed: boolean;
  /** `audioStreamEnd` is due once the session has caught up with the buffered audio (the mute). */
  endPending: boolean;
  closed: boolean;
  open: Record<LiveChannel, boolean>;
  /** When the open utterance of each channel began (its first interim), or null. */
  openSince: Record<LiveChannel, number | null>;
  lastFinalAt: number | null;
  /**
   * Recent finals (for seams), with `heardUntil`: the newest loud frame the session had got when the final arrived.
   * The final's speech ended by then, so a successor that began listening later never heard it.
   */
  finals: { channel: LiveChannel; text: string; at: number; heardUntil: number }[];
  /** Events of a session that is not yet the primary. */
  held: Record<LiveChannel, { finals: HeldEvent[]; interim: HeldEvent | null }>;
}

interface Rotation {
  next: Slot;
  stage: "preparing" | "ready" | "overlap" | "draining";
  forceStartAt: number;
  overlapStart: number;
  audioStart: number;
  forceCutAt: number;
  cutAt: number;
  drainUntil: number;
}

/**
 * A session the next primary partly repeats: its finals whose speech the next primary also heard go into the seam
 * gates.
 */
interface SeamSource {
  slot: Slot;
  /**
   * The next primary heard the frames received from here on: only a final whose session had got a loud frame from
   * then on ({@link Slot.finals} `heardUntil`) can be repeated by it.
   */
  since: number;
  /** The old session heard nothing after this. */
  until: number;
}

interface Reconnect {
  /** The session connecting; null while the back-off runs. */
  slot: Slot | null;
  retryAt: number;
  replayStart: number;
  /** The session it replaces (lost, or still draining after a failed rotation), or null after a dormant stretch. */
  from: SeamSource | null;
}

/** An armed seam gate and the time after which a newly begun utterance is new speech (no longer gated). */
interface ActiveSeam {
  gate: SeamGate;
  until: number;
}

interface Frame {
  seq: number;
  at: number;
  pcm: Int16Array;
  /** Above the {@link QuietTracker} threshold (speech, or a noise worth transcribing). */
  loud: boolean;
}

const MAX_SLOT_FINALS = 24;

/**
 * Keeps one Gemini Live session running at all times with no caption gap: rotation with overlap and seam dedupe,
 * reconnects with audio replay (also seam-deduped), and dormancy while muted (see the module docs). All audio goes
 * through {@link pushAudio}; every frame is delivered to at least one session (or reported as a gap when a reconnect
 * took longer than the {@link AUDIO_RING_MS} buffer).
 */
export class GeminiLiveRotator {
  readonly #o: LiveRotatorOptions;
  readonly #now: () => number;
  readonly #timers: LiveRotatorTimers;
  #ids = 0;
  #primary: Slot | null = null;
  #rot: Rotation | null = null;
  #reconnect: Reconnect | null = null;
  #dormant = false;
  #started = false;
  #stopped = false;
  #muted = false;
  #tick: unknown = null;
  /** The first session while {@link start} opens it (so that {@link stop} can abort it). */
  #starting: Slot | null = null;
  /** Ends {@link start}'s back-off sleep early (stop). */
  #wake: (() => void) | null = null;
  /** Consecutive failures of replacements for a lost session (a stable session resets them; too many are fatal). */
  #failures = 0;
  #quotaFailures = 0;
  /**
   * Consecutive failures of the next session of a rotation: they lengthen the back-off but are never fatal while the
   * running session is fine (a completed rotation resets them).
   */
  #nextFailures = 0;
  #prepareRetryAt = 0;
  #seams: Partial<Record<LiveChannel, ActiveSeam>> = {};
  #ring: Frame[] = [];
  #seq = 0;
  /** Receive time of the newest frame dropped from the ring (audio before it can no longer be replayed). */
  #evictedUntil = Number.NEGATIVE_INFINITY;
  #lastAudioAt = Number.NEGATIVE_INFINITY;
  readonly #quiet = new QuietTracker();
  #evaluating = false;

  constructor(options: LiveRotatorOptions) {
    this.#o = options;
    this.#now = options.now ?? (() => Date.now());
    this.#timers = options.timers ?? defaultTimers;
  }

  /** Test / diagnostics view. */
  get snapshot(): {
    primary: number | null;
    next: number | null;
    stage: string;
    reconnecting: boolean;
    dormant: boolean;
    seams: LiveChannel[];
  } {
    return {
      primary: this.#primary?.id ?? null,
      next: this.#rot?.next.id ?? this.#reconnect?.slot?.id ?? null,
      stage: this.#rot?.stage ?? (this.#reconnect ? "reconnecting" : this.#dormant ? "dormant" : this.#primary ? "running" : "idle"),
      reconnecting: this.#reconnect !== null,
      dormant: this.#dormant,
      seams: CHANNELS.filter((ch) => this.#seams[ch]?.gate.armed),
    };
  }

  /** Opens the first session (with up to `startAttempts` tries for transient failures); resolves when it is ready. */
  async start(): Promise<void> {
    if (this.#started) throw new Error("GeminiLiveRotator.start called twice");
    this.#started = true;
    const attempts = Math.max(1, this.#o.startAttempts ?? START_ATTEMPTS);
    for (let attempt = 1; ; attempt++) {
      if (this.#stopped) throw new DOMException("Aborted", "AbortError");
      const slot = this.#newSlot();
      this.#starting = slot;
      try {
        await this.#openSlot(slot);
      } catch (e) {
        slot.closed = true;
        if (this.#stopped || isAbortError(e)) throw e instanceof Error ? e : new DOMException("Aborted", "AbortError");
        const err = asLiveError(e);
        if (err.kind === "auth" || err.kind === "config" || attempt >= attempts) {
          this.stop();
          throw err;
        }
        const delay = err.kind === "quota" ? err.info.retryAfterMs ?? QUOTA_BACKOFF_MS[0] : BACKOFF_MS[attempt - 1];
        this.#o.status(
          err.kind === "quota"
            ? { kind: "waiting", seconds: Math.ceil(delay / 1000), detail: err.message }
            : { kind: "reconnecting", attempt, detail: err.message },
        );
        await this.#sleep(delay);
        if (this.#stopped) throw new DOMException("Aborted", "AbortError");
        continue;
      } finally {
        if (this.#starting === slot) this.#starting = null;
      }
      if (this.#stopped) throw new DOMException("Aborted", "AbortError");
      this.#activate(slot, null, null, 0);
      this.#tick = this.#timers.setInterval(() => this.#evaluate(), TICK_MS);
      return;
    }
  }

  /** One {@link FRAME_SAMPLES}-sample frame of the microphone. */
  pushAudio(frame: Int16Array): void {
    if (this.#stopped || !this.#started) return;
    const now = this.#now();
    this.#quiet.update(frame, now);
    this.#ring.push({ seq: this.#seq++, at: now, pcm: frame, loud: this.#quiet.quietFor(now) <= 0 });
    while (this.#ring.length > 0 && now - this.#ring[0].at > AUDIO_RING_MS) this.#evictedUntil = this.#ring.shift()!.at;
    this.#lastAudioAt = now;
    if (this.#dormant) {
      // Audio again after a dormant stretch (unmuted): connect and replay from this frame on.
      this.#dormant = false;
      this.#reconnect = { slot: null, retryAt: now, replayStart: now, from: null };
      this.#o.status({ kind: "connecting" });
    }
    if (this.#primary) this.#flush(this.#primary);
    const next = this.#rot?.next;
    if (next && next.fed) this.#flush(next);
    this.#evaluate();
  }

  /**
   * The STT mute: open sentences are finalized (`audioStreamEnd`, sent once a session has caught up with the audio
   * still waiting for it); while muted no audio arrives.
   */
  setMuted(muted: boolean): void {
    if (muted === this.#muted) return;
    this.#muted = muted;
    for (const slot of [this.#primary, this.#rot?.next]) {
      if (!slot) continue;
      slot.endPending = muted && slot.fed;
      if (slot.endPending) this.#flush(slot);
    }
  }

  /** Publishes finished text the session still joins (translate mode) before the STT service closes the sentence. */
  commitPending(): void {
    this.#primary?.handle?.commit();
  }

  stop(): void {
    if (this.#stopped) return;
    this.#stopped = true;
    if (this.#tick !== null) this.#timers.clearInterval(this.#tick);
    this.#tick = null;
    // A first session still connecting is aborted (its token request and socket setup end now).
    const slots = [this.#starting, this.#primary, this.#rot?.next, this.#reconnect?.slot, this.#reconnect?.from?.slot];
    this.#starting = null;
    this.#primary = null;
    this.#rot = null;
    this.#reconnect = null;
    this.#seams = {};
    for (const s of slots) if (s) this.#closeSlot(s);
    this.#ring = [];
    // A start waiting out its back-off gives up at once.
    const wake = this.#wake;
    this.#wake = null;
    wake?.();
  }

  // --- slots --------------------------------------------------------------------------------------------------------

  #newSlot(): Slot {
    const now = this.#now();
    return {
      id: ++this.#ids,
      handle: null,
      abort: new AbortController(),
      connectAt: now,
      readyAt: null,
      rotateAt: now + this.#o.rotateAfterMs,
      hardDeadline: now + GEMINI_LIVE_SESSION_LIMIT_MS - SESSION_SAFETY_MS,
      nextSeq: this.#seq,
      heardFrom: null,
      lastLoudSentAt: Number.NEGATIVE_INFINITY,
      lastEventAt: null,
      fed: false,
      endPending: false,
      closed: false,
      open: { source: false, translation: false },
      openSince: { source: null, translation: null },
      lastFinalAt: null,
      finals: [],
      held: { source: { finals: [], interim: null }, translation: { finals: [], interim: null } },
    };
  }

  async #openSlot(slot: Slot): Promise<void> {
    const handle = await this.#o.open(
      {
        event: (ev) => this.#onEvent(slot, ev),
        goAway: (ms) => this.#onGoAway(slot, ms),
        closed: (err) => this.#onClosed(slot, err),
      },
      slot.abort.signal,
    );
    if (this.#stopped || slot.closed) {
      handle.close();
      throw new DOMException("Aborted", "AbortError");
    }
    slot.handle = handle;
    slot.readyAt = this.#now();
  }

  #closeSlot(slot: Slot): void {
    if (slot.closed) return;
    slot.closed = true;
    slot.fed = false;
    slot.abort.abort();
    try {
      slot.handle?.close();
    } catch {
      /* ignore */
    }
  }

  /** Starts opening `slot` in the background and routes the outcome by the slot's role at that time. */
  #launch(slot: Slot): void {
    this.#openSlot(slot).then(
      () => this.#onSlotReady(slot),
      (e: unknown) => this.#onSlotFailed(slot, e),
    );
  }

  #onSlotReady(slot: Slot): void {
    if (this.#stopped) return;
    const now = this.#now();
    const rot = this.#rot;
    if (rot && rot.next === slot && rot.stage === "preparing") {
      rot.stage = "ready";
      const p = this.#primary;
      rot.forceStartAt = p ? Math.min(p.rotateAt + PAUSE_WAIT_MS, p.hardDeadline - MAX_OVERLAP_MS - 1_000) : now;
      this.#evaluate();
      return;
    }
    const rc = this.#reconnect;
    if (rc && rc.slot === slot) {
      this.#reconnect = null;
      this.#activateAfterReconnect(slot, rc);
      return;
    }
    this.#closeSlot(slot);
  }

  #onSlotFailed(slot: Slot, e: unknown): void {
    if (this.#stopped || isAbortError(e)) return;
    slot.closed = true;
    const err = asLiveError(e);
    const now = this.#now();
    const rot = this.#rot;
    if (rot && rot.next === slot) {
      // The running session keeps going; the next attempt is made after the back-off.
      this.#rot = null;
      this.#nextFailed(err, now);
      return;
    }
    const rc = this.#reconnect;
    if (rc && rc.slot === slot) {
      rc.slot = null;
      const delay = this.#failureDelay(err);
      if (delay === null) return;
      rc.retryAt = now + delay;
      this.#reportRetry(err, delay);
    }
  }

  /** Counts a failure; the back-off before the next attempt, or null after a fatal error (already reported). */
  #failureDelay(err: LiveSessionError): number | null {
    if (err.kind === "auth" || err.kind === "config") {
      this.#fail(err);
      return null;
    }
    if (err.kind === "quota") {
      this.#quotaFailures += 1;
      if (this.#quotaFailures > MAX_QUOTA_FAILURES) {
        this.#fail(err);
        return null;
      }
      return err.info.retryAfterMs ?? QUOTA_BACKOFF_MS[Math.min(this.#quotaFailures, QUOTA_BACKOFF_MS.length) - 1];
    }
    this.#failures += 1;
    if (this.#failures > MAX_FAILURES) {
      this.#fail(err);
      return null;
    }
    return BACKOFF_MS[Math.min(this.#failures, BACKOFF_MS.length) - 1];
  }

  /**
   * The next session of a rotation failed while the running one is fine: a refused key or settings end the session,
   * anything else is retried after a back-off that grows with each consecutive failure (never fatal: when the running
   * session ends, the reconnect takes over with its own limits).
   */
  #nextFailed(err: LiveSessionError, now: number): void {
    if (err.kind === "auth" || err.kind === "config") {
      this.#fail(err);
      return;
    }
    this.#nextFailures += 1;
    const n = this.#nextFailures;
    const delay =
      err.kind === "quota"
        ? err.info.retryAfterMs ?? QUOTA_BACKOFF_MS[Math.min(n, QUOTA_BACKOFF_MS.length) - 1]
        : BACKOFF_MS[Math.min(n, BACKOFF_MS.length) - 1];
    this.#prepareRetryAt = now + delay;
    devLog(`[Gemini Live] next session failed (${err.kind}, ${n} in a row): ${err.message}; retry in ${delay} ms`);
  }

  /** When a lost session is retried: at once after the first loss, after the back-off when it keeps failing. */
  #retryAt(err: LiveSessionError, delay: number, now: number): number {
    return err.kind !== "quota" && this.#failures <= 1 ? now : now + delay;
  }

  #reportRetry(err: LiveSessionError, delay: number): void {
    this.#o.status(
      err.kind === "quota"
        ? { kind: "waiting", seconds: Math.ceil(delay / 1000), detail: err.message }
        : { kind: "reconnecting", attempt: Math.max(1, this.#failures), detail: err.message },
    );
  }

  #fail(err: LiveSessionError): void {
    if (this.#stopped) return;
    this.stop();
    this.#o.fatal(err);
  }

  // --- audio ---------------------------------------------------------------------------------------------------------

  /** Sequence number of the first buffered frame at or after `t` (the next frame when none). */
  #seqAt(t: number): number {
    for (const f of this.#ring) if (f.at >= t) return f.seq;
    return this.#seq;
  }

  /** The buffered frame `seq` (undefined when it is gone or not received yet). */
  #frame(seq: number): Frame | undefined {
    const first = this.#ring[0];
    return first ? this.#ring[seq - first.seq] : undefined;
  }

  #flush(slot: Slot): void {
    if (!slot.fed || !slot.handle || slot.closed) return;
    const first = this.#ring[0];
    if (first && slot.nextSeq < first.seq) {
      // The reconnect took longer than the buffer: the oldest audio is gone.
      this.#o.status({ kind: "gap" });
      slot.nextSeq = first.seq;
    }
    while (slot.nextSeq < this.#seq) {
      const frame = first ? this.#ring[slot.nextSeq - first.seq] : undefined;
      if (!frame) {
        slot.nextSeq = this.#seq;
        break;
      }
      // A failed send leaves the frame for later (the next frame or tick, or the session that replaces this one).
      if (!slot.handle.sendAudio(frame.pcm)) return;
      slot.heardFrom ??= frame.at;
      if (frame.loud) slot.lastLoudSentAt = frame.at;
      slot.nextSeq += 1;
      // Audio sent while muted (a replay, a backlog, an overlap's pre-roll) is followed by the end of the stream, so
      // the sentence in it is finalized; whichever way the session started hearing it.
      if (this.#muted) slot.endPending = true;
    }
    if (slot.endPending) {
      slot.endPending = false;
      slot.handle?.endAudio();
    }
  }

  /** The session got every buffered frame meant for it. */
  #caughtUp(slot: Slot): boolean {
    return !slot.fed || slot.nextSeq >= this.#seq;
  }

  #audioIdle(now: number): boolean {
    return this.#muted || now - this.#lastAudioAt >= AUDIO_IDLE_MS;
  }

  // --- events --------------------------------------------------------------------------------------------------------

  #onEvent(slot: Slot, ev: LiveTranscriptEvent): void {
    if (this.#stopped || slot.closed) return;
    const now = this.#now();
    const ch = ev.channel;
    const text = ev.text;
    slot.lastEventAt = now;
    // When the utterance this event belongs to began (seam gates apply only to utterances begun before their end).
    const startedAt = slot.openSince[ch] ?? now;
    if (ev.kind === "interim" && text) {
      if (!slot.open[ch]) slot.openSince[ch] = now;
      slot.open[ch] = true;
    }
    else {
      slot.open[ch] = false;
      slot.openSince[ch] = null;
      if (ev.kind === "final" && text) {
        slot.finals.push({ channel: ch, text, at: now, heardUntil: slot.lastLoudSentAt });
        if (slot.finals.length > MAX_SLOT_FINALS) slot.finals.shift();
        if (ch === "source") slot.lastFinalAt = now;
      }
    }
    if (slot === this.#primary) {
      this.#publish(ev, startedAt);
      return;
    }
    if (this.#rot?.next === slot) {
      const held = slot.held[ch];
      const item: HeldEvent = ev.lang ? { text, lang: ev.lang } : { text };
      if (ev.kind === "interim") held.interim = text ? item : null;
      else {
        held.interim = null;
        if (text) held.finals.push(item);
      }
    }
  }

  /**
   * Publishes a primary's event, through the channel's seam gate while it is armed and the event's utterance began
   * before the gate's grace ended (a later utterance is new speech: the gate is dropped).
   */
  #publish(ev: LiveTranscriptEvent, startedAt: number): void {
    const sink = this.#o.output[ev.channel];
    if (!sink) return;
    let seam = this.#seams[ev.channel];
    if (seam && (!seam.gate.armed || startedAt > seam.until)) {
      delete this.#seams[ev.channel];
      seam = undefined;
    }
    if (!seam) {
      if (ev.kind === "interim") sink.interim(ev.text, ev.lang);
      else sink.final(ev.text, ev.lang);
      return;
    }
    const gate = seam.gate;
    if (ev.kind === "interim") {
      sink.interim(ev.text ? gate.interim(ev.text) : "", ev.lang);
      return;
    }
    const out = ev.text ? gate.final(ev.text) : "";
    if (!gate.armed) delete this.#seams[ev.channel];
    if (out) sink.final(out, ev.lang);
    else sink.interim("", ev.lang);
  }

  #onGoAway(slot: Slot, timeLeftMs: number | null): void {
    if (this.#stopped || slot.closed) return;
    const now = this.#now();
    const left = timeLeftMs ?? 10_000;
    slot.hardDeadline = Math.min(slot.hardDeadline, now + Math.max(0, left - 1_000));
    slot.rotateAt = Math.min(slot.rotateAt, now);
    if (this.#rot?.next === slot) return;
    if (slot === this.#primary) {
      devLog(`[Gemini Live] session ${slot.id}: the server ends it in ${left} ms; rotating now`);
      this.#prepareRetryAt = 0;
      const rot = this.#rot;
      if (rot?.stage === "ready") rot.forceStartAt = Math.min(rot.forceStartAt, slot.hardDeadline - MAX_OVERLAP_MS / 2);
      if (rot?.stage === "overlap") rot.forceCutAt = Math.min(rot.forceCutAt, slot.hardDeadline - 500);
      if (rot?.stage === "draining") rot.drainUntil = Math.min(rot.drainUntil, slot.hardDeadline);
      this.#evaluate();
    }
  }

  #onClosed(slot: Slot, err: LiveSessionError): void {
    if (this.#stopped || slot.closed) return;
    slot.closed = true;
    slot.fed = false;
    slot.handle = null;
    const now = this.#now();
    devLog(`[Gemini Live] session ${slot.id} closed (${err.kind}): ${err.message}`);
    const rot = this.#rot;

    if (rot && rot.next === slot) {
      this.#rot = null;
      if (!(rot.stage === "draining" && this.#primary)) {
        // The running session still hears everything: it keeps going, the rotation is retried after a back-off.
        this.#nextFailed(err, now);
        return;
      }
      // The old session no longer hears audio: a new one replays everything since the overlap began.
      const p = this.#primary;
      const rc: Reconnect = {
        slot: null,
        retryAt: now,
        replayStart: rot.audioStart,
        from: { slot: p, since: rot.audioStart, until: rot.cutAt },
      };
      this.#reconnect = rc;
      const delay = this.#failureDelay(err);
      if (delay === null) return;
      rc.retryAt = this.#retryAt(err, delay, now);
      this.#reportRetry(err, delay);
      return;
    }

    if (slot === this.#reconnect?.from?.slot) {
      // The draining old session closed: the reconnect simply has no old session to wait for.
      return;
    }

    if (slot !== this.#primary) return;

    if (err.kind === "auth" || err.kind === "config") {
      this.#fail(err);
      return;
    }
    if (rot && (rot.stage === "overlap" || rot.stage === "draining") && !rot.next.closed) {
      // The new session already hears everything: hand over now.
      if (rot.stage === "overlap") rot.cutAt = now;
      this.#promote(now);
      return;
    }

    // Lost with no successor hearing the audio: reconnect and replay.
    this.#primary = null;
    this.#rot = null;
    const candidate = rot && !rot.next.closed ? rot.next : null;
    const open = slot.open.source || slot.open.translation;
    if (!open && this.#audioIdle(now) && !candidate && this.#undelivered(slot) === null) {
      this.#dormant = true;
      return;
    }
    const replayStart = this.#replayStart(slot, now, open);
    const rc: Reconnect = { slot: candidate, retryAt: now, replayStart, from: { slot, since: replayStart, until: now } };
    if (candidate?.readyAt != null) {
      // The next session is already connected: it takes over at once and replays the audio.
      this.#activateAfterReconnect(candidate, rc);
      return;
    }
    this.#reconnect = rc;
    // Sessions that keep dying count as failures (a stable minute resets the count).
    const delay = this.#failureDelay(err);
    if (delay === null) return;
    // The first reconnect goes at once (or waits for the next session already connecting); repeated losses back off.
    rc.retryAt = this.#retryAt(err, delay, now);
    this.#reportRetry(err, delay);
  }

  /**
   * Where a lost session's replacement starts listening: with a sentence open, from shortly before the last final
   * (the open sentence began after it), else the last {@link REPLAY_IDLE_MS}; never more than {@link MAX_REPLAY_MS}
   * back, and never before the lost session's own first audio (earlier speech was another session's and is
   * published). Speech the lost session may never have got ({@link #undelivered}) is replayed however old it is (up
   * to the ring; anything older is reported as a gap). The seam gate drops what the lost session had published.
   */
  #replayStart(slot: Slot, now: number, open: boolean): number {
    const idle = now - REPLAY_IDLE_MS;
    const recent = open ? Math.min(idle, (slot.lastFinalAt ?? slot.heardFrom ?? slot.connectAt) - REPLAY_FINAL_MARGIN_MS) : idle;
    const start = Math.max(recent, now - MAX_REPLAY_MS, slot.heardFrom ?? Number.NEGATIVE_INFINITY);
    const lost = this.#undelivered(slot);
    return lost === null ? start : Math.min(start, lost);
  }

  /**
   * Where speech that `slot` may never have got begins (null = none): frames it was never sent (send back-pressure),
   * and loud frames sent after its last transcript event (a connection that died silently swallowed what it still
   * buffered). That is the start of the run of speech holding the first such frame, a {@link PRE_ROLL_MS} early (but
   * not before {@link REPLAY_FINAL_MARGIN_MS} ahead of the last event, nor before the slot's own audio). Silence needs
   * no replay. Unsent frames already gone from the ring give the oldest time (the activation reports the gap).
   */
  #undelivered(slot: Slot): number | null {
    const ring = this.#ring;
    if (ring.length === 0) return null;
    if (slot.nextSeq < ring[0].seq) return this.#evictedUntil;
    const lastEvent = slot.lastEventAt ?? Number.NEGATIVE_INFINITY;
    // The slot's own audio begins here: earlier speech was another session's (and is published).
    const own = slot.heardFrom ?? this.#frame(slot.nextSeq)?.at ?? Number.POSITIVE_INFINITY;
    for (let i = 0; i < ring.length; i++) {
      const f = ring[i];
      if (!f.loud || f.at < own) continue;
      if (f.seq < slot.nextSeq && f.at <= lastEvent) continue;
      let run = i;
      while (run > 0 && ring[run - 1].loud && ring[run - 1].at >= own) run--;
      const from = Math.min(f.at, Math.max(ring[run].at, lastEvent - REPLAY_FINAL_MARGIN_MS)) - PRE_ROLL_MS;
      return Math.max(from, own);
    }
    return null;
  }

  // --- state machine -------------------------------------------------------------------------------------------------

  #busy(slot: Slot): boolean {
    return slot.open.source || slot.open.translation;
  }

  #evaluate(): void {
    if (this.#stopped || this.#evaluating) return;
    this.#evaluating = true;
    try {
      this.#step();
    } finally {
      this.#evaluating = false;
    }
  }

  #step(): void {
    const now = this.#now();
    for (const s of [this.#primary, this.#rot?.next, this.#reconnect?.from?.slot]) s?.handle?.poll();
    if (this.#stopped) return;
    // Audio a session could not take yet (send back-pressure) goes out on the tick too, also while no new audio comes.
    for (const s of [this.#primary, this.#rot?.next]) if (s?.fed) this.#flush(s);
    if (this.#stopped) return;

    const rc = this.#reconnect;
    if (rc) {
      if (!rc.slot && now >= rc.retryAt) {
        const slot = this.#newSlot();
        rc.slot = slot;
        this.#launch(slot);
      }
      return;
    }

    const p = this.#primary;
    if (!p || this.#dormant) return;
    if (p.readyAt !== null && now - p.readyAt >= STABLE_MS) {
      this.#failures = 0;
      this.#quotaFailures = 0;
    }

    const rot = this.#rot;
    if (!rot) {
      if (now < p.rotateAt - PREPARE_LEAD_MS || now < this.#prepareRetryAt) return;
      if (this.#audioIdle(now) && !this.#busy(p)) {
        // Muted or no audio: nothing to rotate for. Close at the rotation point (once the session has every frame
        // still waiting for it); the next audio reconnects.
        if (now >= p.rotateAt && this.#caughtUp(p)) {
          this.#closeSlot(p);
          this.#primary = null;
          this.#seams = {};
          this.#dormant = true;
          devLog("[Gemini Live] idle at the rotation point: session closed until audio returns");
        }
        return;
      }
      const next = this.#newSlot();
      this.#rot = {
        next,
        stage: "preparing",
        forceStartAt: Number.POSITIVE_INFINITY,
        overlapStart: 0,
        audioStart: 0,
        forceCutAt: 0,
        cutAt: 0,
        drainUntil: 0,
      };
      this.#launch(next);
      return;
    }

    switch (rot.stage) {
      case "preparing":
        return;
      case "ready": {
        if (now < p.rotateAt) return;
        const quiet = this.#quiet.quietFor(now);
        const atPause = !this.#busy(p) && quiet >= QUIET_START_MS;
        if (atPause || now >= rot.forceStartAt) this.#startOverlap(now, atPause ? quiet : 0);
        return;
      }
      case "overlap": {
        const atPause = !this.#busy(p) && this.#quiet.quietFor(now) >= QUIET_CUT_MS;
        if ((now - rot.overlapStart >= OVERLAP_MS && atPause) || now >= rot.forceCutAt) this.#cut(now);
        return;
      }
      case "draining":
        if ((!this.#busy(p) && now - rot.cutAt >= DRAIN_MIN_MS) || now >= rot.drainUntil) this.#promote(now);
        return;
    }
  }

  /** Both sessions hear the audio from now (plus a pre-roll that stays inside the pause when there is one). */
  #startOverlap(now: number, quietMs: number): void {
    const rot = this.#rot;
    const p = this.#primary;
    if (!rot || !p) return;
    const preRoll = quietMs > 0 ? Math.max(0, Math.min(PRE_ROLL_MS, quietMs - 50)) : PRE_ROLL_MS;
    rot.stage = "overlap";
    rot.overlapStart = now;
    rot.audioStart = now - preRoll;
    rot.forceCutAt = Math.min(now + MAX_OVERLAP_MS, p.hardDeadline - 500);
    const next = rot.next;
    next.nextSeq = this.#seqAt(rot.audioStart);
    if (p.nextSeq < next.nextSeq) {
      // The old session is behind (send back-pressure) and may never get those frames before its cut: the new one
      // takes them too, so every frame reaches a session.
      next.nextSeq = p.nextSeq;
      rot.audioStart = Math.min(rot.audioStart, this.#frame(p.nextSeq)?.at ?? this.#evictedUntil);
    }
    next.fed = true;
    devLog(`[Gemini Live] rotation: session ${next.id} hears the audio with session ${p.id}`);
    this.#flush(next);
  }

  /** The old session hears no more audio and finalizes what it has. */
  #cut(now: number): void {
    const rot = this.#rot;
    const p = this.#primary;
    if (!rot || !p) return;
    p.fed = false;
    p.handle?.endAudio();
    rot.stage = "draining";
    rot.cutAt = now;
    const limits = DRAIN_MAX_MS[this.#o.mode];
    // Never past the session's own end: the switch then still flushes what the old session holds open.
    rot.drainUntil = Math.min(now + (this.#busy(p) ? limits.open : limits.idle), p.hardDeadline);
  }

  /** The new session takes over; its overlap transcript goes out through the seam gates. */
  #promote(now: number): void {
    const rot = this.#rot;
    const old = this.#primary;
    if (!rot || !old) return;
    this.#rot = null;
    const next = rot.next;
    const cutAt = rot.cutAt || now;
    // Whatever the old session still holds open is all it will ever say about that speech.
    if (!old.closed) old.handle?.flushOpen();
    this.#closeSlot(old);
    const seams = this.#seamsFrom({ slot: old, since: rot.audioStart, until: cutAt });
    this.#primary = next;
    next.fed = true;
    this.#seams = {};
    // The rotation is complete: the next one starts with a fresh back-off.
    this.#nextFailures = 0;
    this.#releaseHeld(next, seams, now + SEAM_GRACE_MS[this.#o.mode]);
    devLog(`[Gemini Live] rotation: session ${next.id} took over from ${old.id}`);
    this.#flush(next);
  }

  /**
   * Seam gates from what `src` published about audio the next primary heard too. A final counts only when its
   * session had got a loud frame from `since` on when it arrived (`heardUntil`): a final arrives after its speech
   * ended, so one that arrives during the pause where the switch happens, or a moment into a reconnect's replay, is
   * about speech the next primary never heard, and gating on it would drop the next primary's own new words.
   */
  #seamsFrom(src: SeamSource): Partial<Record<LiveChannel, SeamGate>> {
    const seams: Partial<Record<LiveChannel, SeamGate>> = {};
    for (const ch of CHANNELS) {
      const text = src.slot.finals
        .filter((f) => f.channel === ch && f.heardUntil >= src.since)
        .map((f) => f.text)
        .join(" ")
        .trim();
      if (!text) continue;
      const gate = new SeamGate(text, seamMaxUnits(src.until - src.since, UNSPACED_SCRIPT_RE.test(text)));
      if (gate.armed) seams[ch] = gate;
    }
    return seams;
  }

  /** Publishes what `slot` said while it was not the primary, through `seams`; armed gates stay until `until`. */
  #releaseHeld(slot: Slot, seams: Partial<Record<LiveChannel, SeamGate>>, until: number): void {
    for (const ch of CHANNELS) {
      const sink = this.#o.output[ch];
      const held = slot.held[ch];
      const gate = seams[ch];
      if (sink) {
        for (const f of held.finals) {
          const out = gate?.armed ? gate.final(f.text) : f.text;
          if (out) sink.final(out, f.lang);
        }
        if (held.interim) {
          const out = gate?.armed ? gate.interim(held.interim.text) : held.interim.text;
          if (out) sink.interim(out, held.interim.lang);
        }
      }
      // The utterance the new session is still in, or one it reports a little late, may repeat the old text too.
      if (gate?.armed) this.#seams[ch] = { gate, until };
      held.finals = [];
      held.interim = null;
    }
  }

  /** `slot` becomes the primary, hearing the audio from `replayStart` (null = from the next frame). */
  #activate(
    slot: Slot,
    replayStart: number | null,
    seams: Partial<Record<LiveChannel, SeamGate>> | null,
    seamUntil: number,
  ): void {
    this.#primary = slot;
    this.#rot = null;
    this.#seams = {};
    this.#nextFailures = 0;
    for (const ch of CHANNELS) {
      const gate = seams?.[ch];
      if (gate?.armed) this.#seams[ch] = { gate, until: seamUntil };
    }
    // The reconnect took longer than the buffer: audio from before its oldest frame is gone.
    if (replayStart !== null && replayStart <= this.#evictedUntil) this.#o.status({ kind: "gap" });
    slot.nextSeq = replayStart === null ? this.#seq : this.#seqAt(replayStart);
    slot.fed = true;
    this.#flush(slot);
  }

  #activateAfterReconnect(slot: Slot, rc: Reconnect): void {
    let seams: Partial<Record<LiveChannel, SeamGate>> | null = null;
    const from = rc.from;
    if (from) {
      if (!from.slot.closed) from.slot.handle?.flushOpen();
      this.#closeSlot(from.slot);
      seams = this.#seamsFrom(from);
    }
    // Any session still around (a draining old one) is replaced now.
    if (this.#primary && this.#primary !== slot) this.#closeSlot(this.#primary);
    const now = this.#now();
    // Before the activation, which may report a gap (that warning must stay on the status line).
    this.#o.status({ kind: "connected" });
    // The replayed audio is transcribed after it is sent: its utterances may begin up to its length later.
    const replayMs = Math.min(AUDIO_RING_MS, Math.max(0, now - rc.replayStart));
    this.#activate(slot, rc.replayStart, seams, now + replayMs + SEAM_GRACE_MS[this.#o.mode]);
    devLog(`[Gemini Live] session ${slot.id} connected; replaying from ${Math.round(now - rc.replayStart)} ms ago`);
  }

  /** Waits `ms`, or less when {@link stop} is called meanwhile. */
  #sleep(ms: number): Promise<void> {
    if (this.#stopped) return Promise.resolve();
    return new Promise((resolve) => {
      let done = false;
      const wake = () => {
        if (done) return;
        done = true;
        if (this.#wake === wake) this.#wake = null;
        this.#timers.clearTimeout(handle);
        resolve();
      };
      this.#wake = wake;
      const handle = this.#timers.setTimeout(wake, ms);
    });
  }
}

function asLiveError(e: unknown): LiveSessionError {
  if (e instanceof LiveSessionError) return e;
  return new LiveSessionError("transient", shortDetail(redactKeys(errorText(e))) || "error");
}

// ---------------------------------------------------------------------------------------------------------------
// Connection (WebSocket)

/** The part of a browser WebSocket the connection uses (injectable for tests). */
export interface WebSocketLike {
  readonly readyState: number;
  readonly bufferedAmount: number;
  binaryType: string;
  onopen: ((ev: unknown) => void) | null;
  onmessage: ((ev: { data: unknown }) => void) | null;
  onerror: ((ev: unknown) => void) | null;
  onclose: ((ev: { code: number; reason: string; wasClean?: boolean }) => void) | null;
  send(data: string): void;
  close(code?: number, reason?: string): void;
}

export type WebSocketFactory = (url: string) => WebSocketLike;

const WS_OPEN = 1;
const textDecoder = typeof TextDecoder !== "undefined" ? new TextDecoder("utf-8") : null;

function frameText(data: unknown): string | null {
  if (typeof data === "string") return data;
  if (data instanceof ArrayBuffer) return textDecoder ? textDecoder.decode(new Uint8Array(data)) : null;
  if (ArrayBuffer.isView(data)) return textDecoder ? textDecoder.decode(data as ArrayBufferView) : null;
  return null;
}

export interface GeminiLiveConnectionOptions {
  url: string;
  setup: Record<string, unknown>;
  createSocket: WebSocketFactory;
  /** Content messages after `setupComplete`. */
  onContent(msg: Extract<LiveServerMessage, { kind: "content" }>): void;
  onGoAway(timeLeftMs: number | null): void;
  /** The session ended on its own after it was ready. */
  onClosed(error: LiveSessionError): void;
  setupTimeoutMs?: number;
  /** Send-buffer mark above which audio waits ({@link SEND_BUFFER_HIGH_BYTES}). */
  maxBufferedBytes?: number;
  /** How long the buffer may stay above the mark without draining ({@link SEND_STALL_MS}). */
  stallTimeoutMs?: number;
  now?: () => number;
}

/** One Gemini Live WebSocket session: setup, audio out, server messages in. */
export class GeminiLiveConnection {
  #ws: WebSocketLike | null = null;
  #ready = false;
  #done = false;
  /** Blob frames still being decoded (later frames queue behind them), or null. */
  #decoding: Promise<void> | null = null;
  /** The send buffer is above the mark: since when without draining, and its lowest level in that time. */
  #pressure: { since: number; lowest: number } | null = null;

  constructor(private readonly o: GeminiLiveConnectionOptions) {}

  /** Connects and sends the setup; resolves on `setupComplete`. */
  open(signal: AbortSignal): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      if (signal.aborted) {
        reject(new DOMException("Aborted", "AbortError"));
        return;
      }
      let settled = false;
      let timer: ReturnType<typeof setTimeout> | undefined;
      const settle = (error: unknown | null) => {
        if (settled) return;
        settled = true;
        if (timer !== undefined) clearTimeout(timer);
        signal.removeEventListener("abort", onAbort);
        if (error === null) resolve();
        else reject(error);
      };
      const onAbort = () => {
        this.#shutdown(1000, "cancelled");
        settle(new DOMException("Aborted", "AbortError"));
      };
      let ws: WebSocketLike;
      try {
        ws = this.o.createSocket(this.o.url);
      } catch (e) {
        // Never echo the error text: it may quote the URL (with the token).
        settle(new LiveSessionError("transient", e instanceof Error ? e.name : "WebSocket error", { beforeSetup: true }));
        return;
      }
      this.#ws = ws;
      signal.addEventListener("abort", onAbort, { once: true });
      timer = setTimeout(() => {
        this.#shutdown(1000, "setup timeout");
        settle(new LiveSessionError("transient", i18n.t("stt.gemini_detail_setup_timeout") || "setup timeout", { beforeSetup: true }));
      }, this.o.setupTimeoutMs ?? SETUP_TIMEOUT_MS);
      try {
        ws.binaryType = "arraybuffer";
      } catch {
        /* ignore */
      }
      let opened = false;
      ws.onopen = () => {
        opened = true;
        try {
          ws.send(JSON.stringify(this.o.setup));
        } catch {
          /* the close report follows */
        }
      };
      ws.onerror = () => {
        /* onclose follows with the code */
      };
      ws.onmessage = (ev) => {
        if (this.#done) return;
        this.#deliver(ev.data, () => settle(null));
      };
      ws.onclose = (ev) => {
        const wasDone = this.#done;
        this.#done = true;
        this.#ws = null;
        const code = typeof ev?.code === "number" ? ev.code : 1006;
        const reason = typeof ev?.reason === "string" ? ev.reason : "";
        // The handshake itself failed (the browser hides the HTTP status): no network, or an endpoint that moved.
        const detail =
          !opened && !reason.trim() ? i18n.t("stt.gemini_detail_handshake", { code }) || closeDetail(code, reason) : closeDetail(code, reason);
        const error = new LiveSessionError(classifyClose(code, reason), detail, {
          code,
          beforeSetup: !this.#ready,
        });
        if (!this.#ready) {
          settle(error);
          return;
        }
        if (!wasDone) this.o.onClosed(error);
      };
    });
  }

  /**
   * Hands one received frame to {@link #onFrame} in arrival order. Frames are ArrayBuffers (`binaryType`) or text;
   * a Blob (a socket that ignores `binaryType`) is decoded asynchronously, and later frames wait for it.
   */
  #deliver(data: unknown, ready: () => void): void {
    const text = frameText(data);
    if (text !== null && this.#decoding === null) {
      this.#onFrame(text, ready);
      return;
    }
    const blob = data as { text?: () => Promise<string> } | null;
    const step = (this.#decoding ?? Promise.resolve())
      .then(async () => {
        const t = text ?? (blob && typeof blob.text === "function" ? await blob.text() : null);
        if (t !== null) this.#onFrame(t, ready);
      })
      .catch(() => {});
    this.#decoding = step;
    void step.then(() => {
      if (this.#decoding === step) this.#decoding = null;
    });
  }

  #onFrame(text: string, ready: () => void): void {
    if (this.#done) return;
    for (const msg of parseServerMessage(text)) {
      if (msg.kind === "setupComplete") {
        if (!this.#ready) {
          this.#ready = true;
          ready();
        }
        continue;
      }
      if (!this.#ready) {
        // Content before setupComplete (a service that skips it): the session is evidently ready.
        if (msg.kind !== "content") continue;
        this.#ready = true;
        ready();
      }
      if (msg.kind === "content") this.o.onContent(msg);
      else if (msg.kind === "goAway") this.o.onGoAway(msg.timeLeftMs);
    }
  }

  /**
   * Sends one frame; false leaves it for later. Above the send-buffer mark the frame waits (back-pressure); a buffer
   * that does not drain for {@link SEND_STALL_MS} gives the session up (a close report follows at once) so the
   * rotator replays the audio on a new one.
   */
  sendAudio(frame: Int16Array): boolean {
    const ws = this.#ws;
    if (!ws || !this.#ready || this.#done || ws.readyState !== WS_OPEN) return false;
    const buffered = ws.bufferedAmount;
    if (buffered > (this.o.maxBufferedBytes ?? SEND_BUFFER_HIGH_BYTES)) {
      const now = (this.o.now ?? Date.now)();
      const p = this.#pressure;
      if (!p || buffered < p.lowest) this.#pressure = { since: now, lowest: buffered };
      else if (now - p.since >= (this.o.stallTimeoutMs ?? SEND_STALL_MS)) {
        this.#done = true;
        this.#shutdown(4000, "send buffer stalled");
        this.o.onClosed(new LiveSessionError("transient", i18n.t("stt.gemini_detail_stalled") || "the connection stalled"));
      }
      return false;
    }
    this.#pressure = null;
    try {
      ws.send(JSON.stringify({ realtimeInput: { audio: { data: pcm16ToBase64(frame), mimeType: PCM_MIME } } }));
      return true;
    } catch {
      return false;
    }
  }

  endAudio(): void {
    const ws = this.#ws;
    if (!ws || !this.#ready || this.#done || ws.readyState !== WS_OPEN) return;
    try {
      ws.send(JSON.stringify({ realtimeInput: { audioStreamEnd: true } }));
    } catch {
      /* ignore */
    }
  }

  close(): void {
    this.#done = true;
    this.#shutdown(1000, "done");
  }

  #shutdown(code: number, reason: string): void {
    const ws = this.#ws;
    this.#ws = null;
    if (!ws) return;
    ws.onmessage = null;
    try {
      ws.close(code, reason);
    } catch {
      /* ignore */
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Ephemeral token

/** Maps a failed token request to a session error (the shared LLM client's HTTP error mapping). */
export function tokenHttpError(status: number, body: string, headers: Record<string, string>, apiKey: string): LiveSessionError {
  const e = httpError(status, body, (s) => redactKeys(s, [apiKey]), headers);
  const info = { code: status, retryAfterMs: e.retryAfterMs };
  switch (e.kind) {
    case "unauthorized":
      return new LiveSessionError("auth", e.message, info);
    case "rate_limited":
      return new LiveSessionError("quota", e.message, info);
    case "bad_request":
      return new LiveSessionError(/api[ _]?key/i.test(e.message) ? "auth" : "config", e.message, info);
    default:
      return new LiveSessionError("transient", e.message, info);
  }
}

const MAX_TOKEN_REPLY = 64 * 1024;

export interface MintTokenOptions {
  apiKey: string;
  apiVersion: string;
  signal: AbortSignal;
  /** Tried in order; the next one only when a transport could not reach the server at all (CORS, network). */
  transports: LlmHttpTransport[];
  timeoutMs?: number;
}

/**
 * Mints a single-use ephemeral token for one Live session (`POST /<version>/auth_tokens`, the key in the
 * `x-goog-api-key` header). The service defaults apply: the token must start its session within 1 minute (the driver
 * connects right away) and the session may run for 30 minutes (the driver rotates before 10). No times are sent, so
 * a PC whose clock is off cannot mint an already-expired token.
 */
export async function mintGeminiLiveToken(o: MintTokenOptions): Promise<string> {
  const request: LlmHttpRequest = {
    url: authTokenUrl(o.apiVersion),
    method: "POST",
    headers: { "content-type": "application/json", "x-goog-api-key": o.apiKey },
    body: JSON.stringify({ uses: 1 }),
    stream: false,
    timeoutMs: o.timeoutMs ?? TOKEN_TIMEOUT_MS,
  };
  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    ctrl.abort();
  }, o.timeoutMs ?? TOKEN_TIMEOUT_MS);
  const onAbort = () => ctrl.abort();
  o.signal.addEventListener("abort", onAbort, { once: true });
  const redact = (s: string) => shortDetail(redactKeys(s, [o.apiKey]));
  try {
    let unreachable = "";
    for (const transport of o.transports) {
      if (o.signal.aborted) throw abortReason(o.signal);
      let response;
      try {
        response = await transport(request, ctrl.signal);
      } catch (e) {
        if (o.signal.aborted) throw abortReason(o.signal);
        if (timedOut) throw new LiveSessionError("transient", i18n.t("stt.gemini_detail_token_timeout") || "the token request timed out");
        if (e instanceof LlmTransportError && e.beforeResponse) {
          unreachable = redact(e.message);
          continue;
        }
        throw new LiveSessionError("transient", redact(errorText(e)));
      }
      let body = "";
      try {
        for await (const chunk of response.body) {
          body += chunk;
          if (body.length > MAX_TOKEN_REPLY) {
            response.cancel();
            break;
          }
        }
      } catch (e) {
        if (o.signal.aborted) throw abortReason(o.signal);
        throw new LiveSessionError("transient", redact(errorText(e)));
      }
      if (response.status < 200 || response.status >= 300) throw tokenHttpError(response.status, body, response.headers, o.apiKey);
      let name: unknown;
      try {
        name = (JSON.parse(body) as { name?: unknown }).name;
      } catch {
        name = undefined;
      }
      if (typeof name !== "string" || !/^[^\s]{4,1024}$/.test(name)) {
        throw new LiveSessionError("config", i18n.t("stt.gemini_detail_no_token") || "the reply had no token");
      }
      return name;
    }
    throw new LiveSessionError("transient", unreachable || "the server could not be reached");
  } finally {
    clearTimeout(timer);
    o.signal.removeEventListener("abort", onAbort);
  }
}

/**
 * {@link mintGeminiLiveToken} on `apiVersion`, or on the other Live API version when that one has no token endpoint
 * (HTTP 404: a preview service moved). Resolves with the version that worked, which the session's socket uses too.
 */
export async function mintGeminiLiveTokenAnyVersion(o: MintTokenOptions): Promise<{ token: string; apiVersion: ApiVersion }> {
  const preferred = apiVersionOf(o.apiVersion);
  let first: LiveSessionError | null = null;
  for (const apiVersion of [preferred, ...GEMINI_LIVE_API_VERSIONS.filter((v) => v !== preferred)]) {
    try {
      return { token: await mintGeminiLiveToken({ ...o, apiVersion }), apiVersion };
    } catch (e) {
      if (!(e instanceof LiveSessionError) || e.info.code !== 404) throw e;
      first ??= e;
    }
  }
  throw first ?? new LiveSessionError("config", "no API version");
}

// ---------------------------------------------------------------------------------------------------------------
// Output stages

/** Publishes the source channel to the STT receiver with segment ids `<stream>:<n>` and rising revisions. */
export class LiveSegmentOutput implements LiveChannelSink {
  #n = 0;
  #open: { id: string; revision: number; text: string } | null = null;

  constructor(
    private readonly stream: string,
    private readonly sink: { interim(text: string, meta?: SttTextMeta): void; final(text: string, meta?: SttTextMeta): void },
    private readonly interims: () => boolean,
    private readonly fallbackLang: string | null = null,
  ) {}

  interim(text: string, lang?: string): void {
    if (!text) {
      const open = this.#open;
      this.#open = null;
      if (open && open.text) this.sink.interim("", this.#meta(open, lang));
      return;
    }
    const open = (this.#open ??= this.#newSegment());
    if (!this.interims() || text === open.text) return;
    open.text = text;
    open.revision += 1;
    this.sink.interim(text, this.#meta(open, lang));
  }

  final(text: string, lang?: string): void {
    const seg = this.#open ?? this.#newSegment();
    this.#open = null;
    if (!text) {
      if (seg.text) this.sink.interim("", this.#meta(seg, lang));
      return;
    }
    seg.revision += 1;
    this.sink.final(text, this.#meta(seg, lang));
  }

  #newSegment(): { id: string; revision: number; text: string } {
    this.#n += 1;
    return { id: `${this.stream}:${this.#n}`, revision: 0, text: "" };
  }

  #meta(seg: { id: string; revision: number }, lang?: string): SttTextMeta {
    const meta: SttTextMeta = { segmentId: seg.id, revision: seg.revision };
    const l = baseLanguageTag(lang) ?? this.fallbackLang;
    if (l) meta.lang = l;
    return meta;
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Settings → session configuration (pure)

export interface GeminiLiveConfig {
  mode: GeminiLiveMode;
  apiKey: string;
  apiVersion: string;
  /** Setup messages to try, in order (see {@link buildTranscribeSetups} / {@link buildLiveTranslateSetups}). */
  setups: Record<string, unknown>[];
  /** Transcribe mode: the second setup drops the vocabulary / formatting. */
  setupFallbackDropsOptions: boolean;
  rotateAfterMs: number;
  interim: boolean;
  /** Base language of the captions when it is fixed in the settings, else null. */
  sourceLang: string | null;
  /** Canonical target language (translate mode), else "". */
  targetLang: string;
}

/** `rotateMinutes` as ms, clamped to the supported range (blank, zero or invalid = the default). */
export function rotateAfterMs(rotateMinutes: string | number): number {
  const n = typeof rotateMinutes === "number" ? rotateMinutes : parseFloat(String(rotateMinutes ?? ""));
  const minutes = Number.isFinite(n) && n > 0 ? n : GEMINI_LIVE_ROTATE_DEFAULT;
  return Math.round(Math.min(GEMINI_LIVE_ROTATE_MAX, Math.max(GEMINI_LIVE_ROTATE_MIN, minutes)) * 60_000);
}

/** The session configuration of the settings, or the message to stop with. `sharedKey` = translation's Gemini key. */
export function resolveGeminiLiveConfig(params: Pick<STT_State, "geminiLive">, sharedKey: string): GeminiLiveConfig | string {
  const g = params.geminiLive;
  const apiKey = String(g.apiKey ?? "").trim() || String(sharedKey ?? "").trim();
  if (!apiKey) return i18n.t("stt.gemini_error_no_key");
  const mode: GeminiLiveMode = g.mode === "translate" ? "translate" : "transcribe";
  const common = {
    mode,
    apiKey,
    apiVersion: apiVersionOf(g.apiVersion),
    rotateAfterMs: rotateAfterMs(g.rotateMinutes),
    interim: g.interim !== false,
  };
  if (mode === "translate") {
    const model = geminiLiveModelName(g.translateModel);
    const target = liveTranslateTargetCode(g.translateTo);
    if (!model) return i18n.t("stt.gemini_error_model");
    if (!target) return i18n.t("stt.gemini_error_target");
    return {
      ...common,
      setups: buildLiveTranslateSetups({ model, targetLanguageCode: target, echoTargetLanguage: g.echoTargetLanguage !== false }),
      setupFallbackDropsOptions: false,
      sourceLang: null,
      targetLang: canonicalLanguage(g.translateTo),
    };
  }
  const model = geminiLiveModelName(g.model);
  if (!model) return i18n.t("stt.gemini_error_model");
  const vocabulary = parseVocabulary(g.vocabulary);
  const formatting = g.formatting === "smart" ? "smart" : "verbatim";
  const setups = buildTranscribeSetups({ model, language: g.language, formatting, vocabulary });
  return {
    ...common,
    setups,
    setupFallbackDropsOptions: setups.length > 1,
    sourceLang: liveLanguageCodes(g.language).length > 0 ? baseLanguageTag(g.language) : null,
    targetLang: "",
  };
}

/**
 * The stop message for a fatal session error. `starting`: the first session never opened (a connection problem is
 * then "could not connect", not "lost the connection").
 */
export function geminiLiveErrorMessage(err: LiveSessionError, starting = false): string {
  const detail = err.message || String(err.info.code ?? "");
  switch (err.kind) {
    case "auth":
      return i18n.t("stt.gemini_error_key", { detail });
    case "config":
      return i18n.t("stt.gemini_error_rejected", { detail });
    case "quota":
      return i18n.t("stt.gemini_error_quota", { detail });
    default:
      return starting ? i18n.t("stt.gemini_error_connect", { detail }) : i18n.t("stt.gemini_error_lost", { detail });
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Service

/** Test hooks: the WebSocket constructor and the token transports. */
export const geminiLiveDeps: { createSocket: WebSocketFactory; transports: () => LlmHttpTransport[] } = {
  createSocket: (url) => new WebSocket(url) as unknown as WebSocketLike,
  transports: () => (isTauri() ? [createWebviewTransport(), createRustTransport()] : [createWebviewTransport()]),
};

/** Session runs of this process (the epoch part of segment and line ids). */
let geminiRuns = 0;
/** How often the driver follows the STT mute (the Rust gate withholds audio; `audioStreamEnd` is sent from here). */
const MUTE_POLL_MS = 250;

class CaptureError extends Error {}

function sharedGeminiKey(): string {
  try {
    return String(window.ApiServer?.state?.services?.translation?.data?.llm?.geminiKey ?? "");
  } catch {
    return "";
  }
}

function translationServiceRunning(): boolean {
  try {
    const status = window.ApiServer?.translation?.serviceState?.status;
    return status !== undefined && status !== ServiceNetworkState.disconnected;
  } catch {
    return false;
  }
}

export class STT_GeminiLiveService implements ISTTService {
  startAbortSignal: AbortSignal | null = null;
  startAutomatic?: boolean;

  readonly #lifecycle: SttStartLifecycle;
  #rotator: GeminiLiveRotator | null = null;
  #unlisten: UnlistenFn | null = null;
  #capturing = false;
  #framer = new PcmFramer();
  #muted = false;
  #muteTimer: ReturnType<typeof setInterval> | null = null;
  /** A warning that stays on the status line while the session runs (e.g. the translation service also runs). */
  #note: [string, SttStatusOptions] | null = null;
  /** Index of the setup layout that the service accepted in this run (later sessions start from it). */
  #variant = 0;
  /** API version whose token endpoint answered in this run (the settings' one unless it had moved). */
  #apiVersion = "";

  constructor(private readonly receiver: ISTTReceiver) {
    this.#lifecycle = new SttStartLifecycle((error) => this.receiver.onStop(error));
  }

  async start(params: STT_State): Promise<void> {
    this.#lifecycle.beginStart();
    try {
      const config = resolveGeminiLiveConfig(params, sharedGeminiKey());
      if (typeof config === "string") {
        this.receiver.onStatus("");
        this.#lifecycle.notifyStopOnce(config);
        return;
      }
      if (!isTauri()) {
        this.receiver.onStatus("");
        this.#lifecycle.notifyStopOnce(i18n.t("stt.gemini_error_desktop_only"));
        return;
      }
      this.#variant = 0;
      this.#apiVersion = config.apiVersion;
      this.#note = null;
      this.receiver.onStatus(i18n.t("stt.status_connecting_engine", { engine: "Gemini" }));

      const epoch = ++geminiRuns;
      // Live Translate's captions carry their own stream name: the Translation service recognizes them
      // (`isLiveTranslatedStt`) and does not translate them a second time.
      const stream = config.mode === "translate" ? LIVE_TRANSLATE_STT_STREAM : "gemini";
      const source = new LiveSegmentOutput(
        `${stream}-${PROCESS_NONCE}:${epoch}`,
        { interim: (t, m) => this.receiver.onInterim(t, m), final: (t, m) => this.receiver.onFinal(t, m) },
        () => config.interim,
        config.sourceLang,
      );
      let translation: LiveChannelSink | undefined;
      if (config.mode === "translate") {
        const publisher = new LiveTranslationPublisher({
          epoch,
          lang: config.targetLang,
          // Not scoped by the STT receiver: nothing is published once this session is stopping.
          publish: (event) => {
            if (!this.#lifecycle.aborted) window.ApiShared.pubsub.publishText(TextEventSource.translation, event);
          },
          interims: () => config.interim,
          muted: () => this.receiver.isMuted(),
        });
        translation = { interim: (t) => publisher.interim(t), final: (t) => publisher.final(t) };
      }

      const rotator = new GeminiLiveRotator({
        mode: config.mode,
        open: (callbacks, signal) => this.#openSession(config, callbacks, signal),
        output: { source, translation },
        status: (s) => this.#onRotatorStatus(s),
        fatal: (err) => void this.#onFatal(err),
        rotateAfterMs: config.rotateAfterMs,
      });
      this.#rotator = rotator;
      const starting = rotator.start();
      // A start given up on (stop while connecting) still settles in the background: never an unhandled rejection.
      starting.catch(() => {});
      await raceWithAbort(starting, this.startAbortSignal);
      if (this.#lifecycle.aborted) {
        await this.#teardown();
        this.#lifecycle.notifyStopOnce();
        return;
      }

      this.#unlisten = await listenWithAbort<RustAudioChunkPayload>(
        "audio:chunk",
        (event) => this.#onChunk(event.payload),
        this.startAbortSignal,
      );
      this.#capturing = true;
      try {
        await raceWithAbort(
          invoke("plugin:audio|start_audio_capture", {
            deviceName: resolveRustInputDeviceForInvoke(params.geminiLive.device),
            sampleRate: SAMPLE_RATE,
          }),
          this.startAbortSignal,
        );
      } catch (e) {
        if (isAbortError(e)) throw e;
        throw new CaptureError(shortDetail(errorText(e)));
      }
      if (this.#lifecycle.aborted) {
        await this.#teardown();
        this.#lifecycle.notifyStopOnce();
        return;
      }
      this.#muted = this.receiver.isMuted();
      if (this.#muted) rotator.setMuted(true);
      this.#muteTimer = setInterval(() => this.#followMute(), MUTE_POLL_MS);

      if (config.mode === "translate" && !TRANSLATION_SERVICE_SKIPS_LIVE_TRANSLATE && translationServiceRunning()) {
        this.#note = [i18n.t("stt.gemini_status_translation_running"), { level: "warning" }];
      }
      // After onStart, which clears the setup status: the warning stays while the session runs.
      this.receiver.onStart();
      this.#showNote();
      devLog(`[Gemini Live] Listening (${config.mode})`);
    } catch (error) {
      if (isAbortError(error) || this.#lifecycle.aborted) {
        await this.#teardown();
        return;
      }
      await this.#teardown();
      this.receiver.onStatus("");
      const message =
        error instanceof LiveSessionError ? geminiLiveErrorMessage(error, true)
        : error instanceof CaptureError ? i18n.t("stt.error_capture", { detail: error.message })
        : i18n.t("stt.error_engine_start", { engine: "Gemini", detail: shortDetail(redactKeys(errorText(error))) });
      console.error("[Gemini Live] Start failed:", message);
      this.#lifecycle.notifyStopOnce(message);
    } finally {
      this.#lifecycle.finishStart();
    }
  }

  async stop(): Promise<void> {
    this.#lifecycle.abort();
    const live = this.#rotator !== null || this.#capturing;
    await this.#teardown();
    if (live) this.#lifecycle.notifyStopOnce();
  }

  dispose(): void {
    void this.stop();
  }

  commitPending(): void {
    this.#rotator?.commitPending();
  }

  // --- sessions ------------------------------------------------------------------------------------------------------

  async #openSession(config: GeminiLiveConfig, callbacks: LiveSessionCallbacks, signal: AbortSignal): Promise<LiveSessionHandle> {
    let first: LiveSessionError | null = null;
    for (let v = this.#variant; v < config.setups.length; v++) {
      const { token, apiVersion } = await mintGeminiLiveTokenAnyVersion({
        apiKey: config.apiKey,
        apiVersion: this.#apiVersion || config.apiVersion,
        signal,
        transports: geminiLiveDeps.transports(),
      });
      if (apiVersion !== this.#apiVersion) {
        devLog(`[Gemini Live] the ${this.#apiVersion} token endpoint was not found; using ${apiVersion}`);
        this.#apiVersion = apiVersion;
      }
      const content = createContentHandler(config.mode, (ev) => callbacks.event(ev), { sourceLang: config.sourceLang });
      const conn = new GeminiLiveConnection({
        url: liveSocketUrl(apiVersion, token),
        setup: config.setups[v],
        createSocket: geminiLiveDeps.createSocket,
        onContent: (msg) => content.message(msg),
        onGoAway: (ms) => callbacks.goAway(ms),
        onClosed: (err) => callbacks.closed(err),
      });
      try {
        await conn.open(signal);
      } catch (e) {
        if (isAbortError(e)) throw e;
        const err = asLiveError(e);
        if (err.kind === "config" && err.info.beforeSetup && v + 1 < config.setups.length) {
          first ??= err;
          devLog(`[Gemini Live] setup ${v + 1} refused (${err.message}); trying the next layout`);
          continue;
        }
        throw err.kind === "config" && first ? first : err;
      }
      if (v !== this.#variant) {
        this.#variant = v;
        if (config.setupFallbackDropsOptions) {
          this.#note = [i18n.t("stt.gemini_status_options_dropped", { detail: first?.message ?? "" }), { level: "warning" }];
          this.#showNote();
        }
      }
      return {
        sendAudio: (frame) => conn.sendAudio(frame),
        endAudio: () => conn.endAudio(),
        poll: () => content.poll(),
        commit: () => content.commit(),
        flushOpen: () => content.flushOpen(),
        close: () => conn.close(),
      };
    }
    throw first ?? new LiveSessionError("config", "no setup");
  }

  #onRotatorStatus(s: LiveRotatorStatus): void {
    switch (s.kind) {
      case "connecting":
        this.receiver.onStatus(i18n.t("stt.status_connecting_engine", { engine: "Gemini" }));
        return;
      case "reconnecting":
        this.receiver.onStatus(i18n.t("stt.gemini_status_reconnecting", { attempt: Math.max(1, s.attempt), detail: s.detail }), { level: "warning" });
        return;
      case "waiting":
        this.receiver.onStatus(i18n.t("stt.gemini_status_wait", { seconds: Math.max(1, s.seconds), detail: s.detail }), { level: "warning" });
        return;
      case "gap":
        this.receiver.onStatus(i18n.t("stt.gemini_status_gap"), { level: "warning", expireMs: TRANSIENT_WARNING_MS });
        return;
      case "connected":
        this.#showNote();
        return;
    }
  }

  /** The persistent warning of this session on the status line, or a clear line. */
  #showNote(): void {
    if (this.#note) this.receiver.onStatus(this.#note[0], this.#note[1]);
    else this.receiver.onStatus("");
  }

  async #onFatal(err: LiveSessionError): Promise<void> {
    if (this.#lifecycle.aborted) return;
    const message = geminiLiveErrorMessage(err);
    console.error("[Gemini Live]", message);
    this.#lifecycle.abort();
    await this.#teardown();
    this.receiver.onStatus("");
    this.#lifecycle.notifyStopOnce(message);
  }

  // --- audio ---------------------------------------------------------------------------------------------------------

  #onChunk(payload: RustAudioChunkPayload): void {
    const rotator = this.#rotator;
    if (!rotator || !this.#capturing) return;
    // Follow the mute at once (the poll lags up to 250 ms): the first audio after an unmute is not dropped.
    if (this.receiver.isMuted() !== this.#muted) this.#followMute();
    if (this.#muted) return;
    const { samples, sample_rate, channels } = payload ?? ({} as RustAudioChunkPayload);
    if (!(sample_rate > 0) || !samples?.length) return;
    const mono = toMonoFloat32(samples, channels || 1);
    this.#framer.push(resampleLinear(mono, sample_rate, SAMPLE_RATE), (frame) => rotator.pushAudio(frame));
  }

  #followMute(): void {
    const muted = this.receiver.isMuted();
    if (muted === this.#muted) return;
    this.#muted = muted;
    this.#framer.reset();
    this.#rotator?.setMuted(muted);
  }

  async #teardown(): Promise<void> {
    if (this.#muteTimer !== null) clearInterval(this.#muteTimer);
    this.#muteTimer = null;
    const rotator = this.#rotator;
    this.#rotator = null;
    rotator?.stop();
    this.#unlisten?.();
    this.#unlisten = null;
    this.#framer.reset();
    if (this.#capturing) {
      this.#capturing = false;
      try {
        await invoke("plugin:audio|stop_audio_capture");
      } catch {
        /* ignore */
      }
    }
  }
}
