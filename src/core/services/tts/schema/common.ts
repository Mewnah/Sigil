import { TextEventSource, TextEventSourceSchema } from "@/types";
import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

/** Settings shared by every TTS backend (spread into `Service_TTS_Schema`). */
export const TTS_CommonShape = {
  source: zSafe(TextEventSourceSchema, TextEventSource.stt),
  inputField: zSafe(z.coerce.boolean(), true),
  autoStart: zSafe(z.coerce.boolean(), false),
  stopWithStream: zSafe(z.coerce.boolean(), false),
  replaceWords: zSafe(z.record(z.coerce.string(), z.coerce.string()), {}),
  replaceWordsIgnoreCase: zSafe(z.coerce.boolean(), true),
  /** Smart Turn commit (P6-C/P6-I0): hold sentences of an unfinished turn and speak them joined. Off by default. */
  commitOnTurnEnd: zSafe(z.coerce.boolean(), false),
  /** Longest wait after a pause before held text is spoken anyway (ms, clamped to 500-10000). */
  turnHoldMs: zSafe(zStringNumber(), "2500"),
};
