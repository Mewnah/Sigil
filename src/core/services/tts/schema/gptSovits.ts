import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/**
 * How `api_v2.py` streams when `streaming` is on: `fragments` sends `true` (every api_v2 version: whole fragments, best
 * quality), `medium` and `fast` send the v2ProPlus modes 2 and 3 (sooner first audio, lower quality).
 */
export const GPT_SOVITS_STREAM_MODES = ["fragments", "medium", "fast"] as const;
export type GptSovitsStreamMode = (typeof GPT_SOVITS_STREAM_MODES)[number];

/** `text_lang` / `prompt_lang` choices (GPT-SoVITS's codes; `auto` detects, `all_*` reads everything as one language). */
export const GPT_SOVITS_LANGUAGES = ["auto", "ja", "zh", "en", "ko", "yue", "all_ja", "all_zh", "all_ko", "all_yue", "auto_yue"] as const;

/**
 * Settings of the `gptSovits` TTS backend (P6-G): a user-run GPT-SoVITS `api_v2` server (`gpt-sovits` plugin): weight
 * switching, a reference clip (voice library or a server path), streaming.
 * `device` / `volume` are the TTS sink target (the route is the top-level `route`). Persisted values are never renamed
 * or removed.
 */
export const TTS_GptSovits_Schema = z.object({
  baseUrl: zSafe(z.coerce.string(), "http://127.0.0.1:9880"),
  /** Server-side weight paths ("" = keep the server's). */
  gptWeights: zSafe(z.coerce.string(), ""),
  sovitsWeights: zSafe(z.coerce.string(), ""),
  /** Voice-library voice used as the reference ("" = `refAudioPath`). */
  referenceVoiceId: zSafe(z.coerce.string(), ""),
  /** Reference clip path on the server's machine. */
  refAudioPath: zSafe(z.coerce.string(), ""),
  /** Transcript of the reference ("" = the library voice's). */
  promptText: zSafe(z.coerce.string(), ""),
  promptLang: zSafe(z.coerce.string(), "auto"),
  /** Language of the spoken text ("auto" = the caption's language when known, else the server detects). */
  textLang: zSafe(z.coerce.string(), "auto"),
  streaming: zSafe(z.coerce.boolean(), true),
  /** P6-G: the streaming mode when `streaming` is on. */
  streamingMode: zSafe(z.enum(GPT_SOVITS_STREAM_MODES), "fragments"),
  speed: zSafe(zStringNumber(), "1.0"),
  /** JSON object of extra request members ("" = none). */
  extra: zSafe(z.coerce.string(), ""),
  /** Time to the first audio in seconds (0 = automatic). */
  timeout: zSafe(zStringNumber(), "0"),
  /** Output device (cpal name); "" = the system default output. */
  device: zSafe(z.coerce.string(), ""),
  /** Sink gain 0..2. */
  volume: zSafe(zStringNumber(), "1.0"),
}).default({});

export type TTS_GptSovits_State = z.infer<typeof TTS_GptSovits_Schema>;
