import { CaptionEventSourceSchema, TextEventSource } from "@/types";
import { zSafe } from "@/utils";
import { z } from "zod";

/** How pinyin shows tones (Rust `text_readings::PinyinStyle`). */
export const READINGS_PINYIN_STYLES = ["marks", "numbers", "none"] as const;

/**
 * Root key `readings` (P6-S0; owned by P6-L): readings of Japanese and Chinese captions published on `readings.stt`
 * (`ReadingsEvent`), keyed by the line's `utteranceId`. Off by default.
 *
 * - `sources`: caption topics whose finals are annotated (own speech `text.stt`, other players `others.stt`).
 * - Japanese: `hiragana`, `romaji`; Chinese: `pinyin` with `pinyinStyle`.
 * - `interim`: annotate interims too (more CPU; finals only by default).
 */
export const Readings_Schema = z.object({
  enable: zSafe(z.coerce.boolean(), false),
  sources: zSafe(z.array(CaptionEventSourceSchema).max(4), [TextEventSource.stt]),
  hiragana: zSafe(z.coerce.boolean(), true),
  romaji: zSafe(z.coerce.boolean(), true),
  pinyin: zSafe(z.coerce.boolean(), true),
  pinyinStyle: zSafe(z.enum(READINGS_PINYIN_STYLES), "marks"),
  interim: zSafe(z.coerce.boolean(), false),
}).default({});

export type ReadingsState = z.infer<typeof Readings_Schema>;
