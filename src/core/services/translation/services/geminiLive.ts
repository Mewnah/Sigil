/**
 * Gemini 3.5 Live Translate (preview) helpers for the `geminiLive` STT backend's `translate` mode (P6-K).
 *
 * Live Translate is a combined backend: one WebSocket session takes the microphone audio and returns the source
 * transcript and its translation (plus translated speech, which Sigil does not play: the TTS service can speak
 * `text.translation`). The STT driver (`stt/services/geminiLive.ts`) owns the session, rotation and audio; this module
 * holds what is specific to translation:
 *
 * - the setup message ({@link buildLiveTranslateSetups}): the documented layout first, then the layout of the general
 *   Live API (transcription configs at the top of `setup`) in case the preview service rejects the first one;
 * - the target language code ({@link liveTranslateTargetCode}, canonical BCP-47 → Gemini's code);
 * - assembling the streamed transcription pieces into lines ({@link LiveTranslateNormalizer}): Live API transcriptions
 *   arrive as incremental pieces that are joined until the model's turn completes (or the line goes quiet);
 * - publishing translated lines on `text.translation` the way the translation pipeline does
 *   ({@link LiveTranslationPublisher}: a generated line id per line, rising `revision`, the target `lang`, and
 *   `stableLength` on interims; the committed part only grows);
 * - {@link geminiLiveTranslatesStt}: whether the Translation service should leave `text.stt` alone (it would translate
 *   the same speech a second time).
 *
 * Pure (no `window`, no Tauri): unit-tested in `stt/__tests__/geminiRotation.test.ts`. Owned by P6-K.
 */
import { TextEventType } from "@/types";
import { PROCESS_NONCE } from "@/utils/processNonce";
import { canonicalLanguage } from "../languages";
import { completeTokens } from "../localAgreement";
import type { TranslationOutputEvent } from "../pipeline";

/** A source line without new transcription for this long is final (translate mode has no per-line final event). */
export const LIVE_TRANSLATE_INPUT_IDLE_MS = 2_500;
/** A translated line without new output for this long is final (when the turn-complete signal does not come). */
export const LIVE_TRANSLATE_OUTPUT_IDLE_MS = 3_000;
/**
 * A piece that starts with the whole text so far replaces it (a cumulative update) only when the text so far is at
 * least this long: Live API pieces are normally deltas, and a short delta can repeat the previous words ("no", "no").
 */
const CUMULATIVE_MIN_CHARS = 6;

/** `models/<id>` for a model id typed with or without the prefix; "" for a blank id. */
export function geminiLiveModelName(id: string): string {
  const trimmed = String(id ?? "").trim().replace(/^models\//, "").trim();
  return trimmed ? `models/${trimmed}` : "";
}

/**
 * Gemini's `targetLanguageCode` (BCP-47) for a canonical translation language: Simplified Chinese is `zh-Hans`
 * (Gemini names both scripts), everything else keeps its canonical code (`en`, `pt-PT`, `zh-Hant`). "" when the code
 * is unknown or `auto`.
 */
export function liveTranslateTargetCode(code: string): string {
  const canonical = canonicalLanguage(code);
  if (!canonical) return "";
  return canonical === "zh" ? "zh-Hans" : canonical;
}

export interface LiveTranslateSetupOptions {
  /** Model id, with or without `models/`. */
  model: string;
  /** Gemini target code ({@link liveTranslateTargetCode}). */
  targetLanguageCode: string;
  /** Speech already in the target language is repeated (true) instead of skipped. */
  echoTargetLanguage: boolean;
}

/**
 * Setup messages to try, in order. The first is the layout of the Live Translate guide (transcription configs inside
 * `generationConfig`); the second puts them at the top of `setup` like the rest of the Live API. A preview service
 * that rejects a field closes the socket before `setupComplete`, and the driver then tries the next layout (and keeps
 * the one that worked). Only `AUDIO` output is supported by the model; the audio is ignored.
 */
export function buildLiveTranslateSetups(o: LiveTranslateSetupOptions): Record<string, unknown>[] {
  const model = geminiLiveModelName(o.model);
  const translationConfig = { targetLanguageCode: o.targetLanguageCode, echoTargetLanguage: !!o.echoTargetLanguage };
  return [
    {
      setup: {
        model,
        generationConfig: {
          responseModalities: ["AUDIO"],
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          translationConfig,
        },
      },
    },
    {
      setup: {
        model,
        generationConfig: { responseModalities: ["AUDIO"], translationConfig },
        inputAudioTranscription: {},
        outputAudioTranscription: {},
      },
    },
  ];
}

/**
 * Joins a streamed transcription piece onto the text so far. Live API pieces are deltas that carry their own leading
 * space where the language needs one ("Hello", " there"); a piece that repeats the whole text so far and goes on is
 * a cumulative update and replaces it.
 */
export function mergeTranscriptPiece(acc: string, piece: string): string {
  if (!piece) return acc;
  if (!acc) return piece;
  if (acc.length >= CUMULATIVE_MIN_CHARS && piece.length > acc.length && piece.startsWith(acc)) return piece;
  return acc + piece;
}

/** Normalized transcript events of one channel (source speech or its translation). */
export interface LiveChannelSink {
  /** The open line's text so far (replaces the previous interim); "" abandons the open line. */
  interim(text: string, lang?: string): void;
  /** The line is complete. */
  final(text: string, lang?: string): void;
}

/** Lower-case base language of a BCP-47 tag (`en-US` → `en`), or undefined. */
function baseLang(code: unknown): string | undefined {
  const base = String(code ?? "").trim().split(/[-_]/)[0].toLowerCase();
  return /^[a-z]{2,3}$/.test(base) ? base : undefined;
}

/**
 * Turns one Live Translate session's server content into lines on two channels:
 *
 * - source: `inputTranscription` pieces are joined into the open line (published as interims) and the line is final
 *   when the model's turn completes, or after {@link LIVE_TRANSLATE_INPUT_IDLE_MS} without a new piece. A session that
 *   sends `interimInputTranscription` (the transcription model's protocol) is treated like it: interims replace each
 *   other and each `inputTranscription` is a complete line.
 * - translation: `outputTranscription` pieces are joined the same way; final on turn completion or after
 *   {@link LIVE_TRANSLATE_OUTPUT_IDLE_MS} of silence.
 *
 * `poll()` applies the idle rules (the driver calls it on its tick); `commit()` finalizes whatever is open (stop, mute).
 */
export class LiveTranslateNormalizer {
  #source = "";
  #sourceLang: string | undefined;
  #sourceAt = 0;
  /** The session sends complete source segments (it also sent `interimInputTranscription`). */
  #segmented = false;
  #segmentedOpen = false;
  #output = "";
  #outputAt = 0;

  constructor(
    private readonly sinks: { source: LiveChannelSink; translation: LiveChannelSink },
    private readonly now: () => number = () => Date.now(),
  ) {}

  /** A line is open on either channel. */
  get open(): boolean {
    return this.#source.trim() !== "" || this.#segmentedOpen || this.#output.trim() !== "";
  }

  /** `serverContent.interimInputTranscription.text`. */
  interimInput(text: string): void {
    // Text joined before the session showed its protocol was a complete segment: it stays a line of its own.
    if (!this.#segmented) this.#finishSource();
    this.#segmented = true;
    const t = String(text ?? "").trim();
    this.#segmentedOpen = t !== "";
    this.sinks.source.interim(t, this.#sourceLang);
  }

  /** `serverContent.inputTranscription` (`text`, and the detected `languageCode` when sent). */
  input(text: string, languageCode?: string): void {
    const lang = baseLang(languageCode);
    if (lang) this.#sourceLang = lang;
    if (this.#segmented) {
      this.#segmentedOpen = false;
      const t = String(text ?? "").trim();
      if (t) this.sinks.source.final(t, this.#sourceLang);
      else this.sinks.source.interim("", this.#sourceLang);
      return;
    }
    const piece = String(text ?? "");
    if (!piece) return;
    const before = this.#source.trim();
    this.#source = mergeTranscriptPiece(this.#source, piece);
    this.#sourceAt = this.now();
    const shown = this.#source.trim();
    if (shown && shown !== before) this.sinks.source.interim(shown, this.#sourceLang);
  }

  /** `serverContent.outputTranscription.text`. */
  output(text: string): void {
    const piece = String(text ?? "");
    if (!piece) return;
    const before = this.#output.trim();
    this.#output = mergeTranscriptPiece(this.#output, piece);
    this.#outputAt = this.now();
    const shown = this.#output.trim();
    if (shown && shown !== before) this.sinks.translation.interim(shown);
  }

  /** `turnComplete` / `generationComplete`: the model finished translating what it heard. */
  turnComplete(): void {
    this.#finishSource();
    this.#finishOutput();
  }

  /** Idle rules (called on the driver's tick). */
  poll(): void {
    const now = this.now();
    if (this.#source.trim() && now - this.#sourceAt >= LIVE_TRANSLATE_INPUT_IDLE_MS) this.#finishSource();
    if (this.#output.trim() && now - this.#outputAt >= LIVE_TRANSLATE_OUTPUT_IDLE_MS) this.#finishOutput();
  }

  /** Finalize the open lines now (stop, mute, session switch). */
  commit(): void {
    this.turnComplete();
  }

  #finishSource(): void {
    const text = this.#source.trim();
    this.#source = "";
    if (text) this.sinks.source.final(text, this.#sourceLang);
  }

  #finishOutput(): void {
    const text = this.#output.trim();
    this.#output = "";
    if (text) this.sinks.translation.final(text);
  }
}

export interface LiveTranslationPublisherOptions {
  /** Session run number of the STT service (part of the line ids, with the process nonce). */
  epoch: number;
  /** Canonical target language published as `lang` ("" = not published). */
  lang: string;
  publish(event: TranslationOutputEvent): void;
  /** Whether interims are published (finals always are). */
  interims(): boolean;
  /** Nothing is published while true (the STT mute). */
  muted(): boolean;
}

interface OpenTranslationLine {
  id: string;
  revision: number;
  shown: string;
  committed: string;
}

/**
 * Publishes translated lines on `text.translation` in the shape of the translation pipeline (`pipeline.ts` `#emit`):
 * a line id `gemini.tr.<process nonce>.<epoch>:<n>` shared by its interims and its final, a rising `revision`, the
 * target `lang`, and on interims `stableLength` (the complete words so far; it only grows, the last word may still be
 * streaming). The final carries the whole line without `stableLength`.
 */
export class LiveTranslationPublisher {
  #n = 0;
  #open: OpenTranslationLine | null = null;

  constructor(private readonly o: LiveTranslationPublisherOptions) {}

  interim(text: string): void {
    const value = String(text ?? "").trim();
    if (!value) {
      // An abandoned line: nothing to publish (the next line gets a new id).
      this.#open = null;
      return;
    }
    if (!this.o.interims()) return;
    const line = this.#line();
    if (value === line.shown) return;
    const complete = completeTokens(value, this.o.lang).trimEnd();
    let stableLength: number | undefined;
    if (complete.startsWith(line.committed)) {
      line.committed = complete;
      stableLength = complete.length;
    }
    else if (value.startsWith(line.committed)) stableLength = line.committed.length;
    line.shown = value;
    this.#emit(line, TextEventType.interim, value, stableLength);
  }

  final(text: string): void {
    const value = String(text ?? "").trim();
    const line = this.#open;
    this.#open = null;
    if (!value) return;
    this.#emit(line ?? this.#newLine(), TextEventType.final, value);
  }

  #line(): OpenTranslationLine {
    this.#open ??= this.#newLine();
    return this.#open;
  }

  #newLine(): OpenTranslationLine {
    this.#n += 1;
    return { id: `gemini.tr.${PROCESS_NONCE}.${this.o.epoch}:${this.#n}`, revision: 0, shown: "", committed: "" };
  }

  #emit(line: OpenTranslationLine, type: TextEventType, value: string, stableLength?: number): void {
    line.revision += 1;
    if (this.o.muted()) return;
    const event: TranslationOutputEvent = { type, value, segmentId: line.id, revision: line.revision };
    if (this.o.lang) event.lang = this.o.lang;
    if (stableLength !== undefined) event.stableLength = Math.max(0, Math.min(value.length, stableLength));
    this.o.publish(event);
  }
}

/**
 * True when the STT backend is Gemini Live in `translate` mode: it publishes its own `text.translation`, so the
 * Translation service should not translate that backend's `text.stt` again (see {@link isLiveTranslatedStt}; until
 * `Service_Translation` skips those lines, the STT inspector and status line warn while both run).
 */
export function geminiLiveTranslatesStt(stt: { backend?: unknown; geminiLive?: { mode?: unknown } } | null | undefined): boolean {
  return stt?.backend === "geminiLive" && stt.geminiLive?.mode === "translate";
}

/**
 * Whether `Service_Translation` skips Live Translate's `text.stt` lines ({@link isLiveTranslatedStt}). While false,
 * the Gemini Live status line and inspector warn that a running Translation service translates those lines a second
 * time. True since P6-I0 added the skip to `translation/index.ts` (`#onText`).
 */
export const TRANSLATION_SERVICE_SKIPS_LIVE_TRANSLATE: boolean = true;

/** Stream part of the `text.stt` segment ids that Live Translate publishes (`geminilt-<nonce>:<epoch>:<n>`). */
export const LIVE_TRANSLATE_STT_STREAM = "geminilt";

/**
 * True for a `text.stt` event published by Gemini Live Translate, whose translation is already on
 * `text.translation`: the Translation service skips it instead of translating the line a second time. Decided by the
 * segment id, so it also holds for lines relayed from a linked host.
 */
export function isLiveTranslatedStt(event: { segmentId?: unknown } | null | undefined): boolean {
  const id = event?.segmentId;
  return typeof id === "string" && id.startsWith(`${LIVE_TRANSLATE_STT_STREAM}-`);
}
