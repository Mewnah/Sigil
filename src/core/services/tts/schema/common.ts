import { TextEventSource, TextEventSourceSchema } from "@/types";
import { zSafe } from "@/utils";
import { z } from "zod";

/** Settings shared by every TTS backend (spread into `Service_TTS_Schema`). */
export const TTS_CommonShape = {
  source: zSafe(TextEventSourceSchema, TextEventSource.stt),
  inputField: zSafe(z.coerce.boolean(), true),
  autoStart: zSafe(z.coerce.boolean(), false),
  stopWithStream: zSafe(z.coerce.boolean(), false),
  replaceWords: zSafe(z.record(z.coerce.string(), z.coerce.string()), {}),
  replaceWordsIgnoreCase: zSafe(z.coerce.boolean(), true),
};
