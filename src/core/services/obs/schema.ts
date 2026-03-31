import { TextEventSource, TextEventSourceSchema } from "@/types";
import { zSafe } from "@/utils";
import { z } from "zod";
import { ObsCaptionOutputMode } from "./types";

export const Service_OBS_Schema = z.object({
  enable: zSafe(z.coerce.boolean(), false),
  wsHost: zSafe(z.coerce.string(), ""),
  wsAutoStart: zSafe(z.coerce.boolean(), false),
  wsPort: zSafe(z.coerce.string(), ""),
  wsPassword: zSafe(z.coerce.string(), ""),

  browserCaptionsEnable: zSafe(z.coerce.boolean(), true),
  browserSource: zSafe(TextEventSourceSchema, TextEventSource.stt),
  browserInputField: zSafe(z.coerce.boolean(), true),
  browserInterim: zSafe(z.coerce.boolean(), true),
  browserOutputMode: zSafe(z.nativeEnum(ObsCaptionOutputMode), ObsCaptionOutputMode.styled),
  browserFontSizePx: zSafe(z.coerce.string(), "54"),
  browserMaxLines: zSafe(z.coerce.string(), "2"),

  captionsEnable: zSafe(z.coerce.boolean(), false),
  source: zSafe(TextEventSourceSchema, TextEventSource.stt),
  inputField: zSafe(z.coerce.boolean(), false),
  interim: zSafe(z.coerce.boolean(), false),

  scenesEnable: zSafe(z.coerce.boolean(), false),
  scenesFallback: zSafe(z.string(), "Main"),
  scenesMap: zSafe(z.record(z.string(), z.string()), {}),
}).default({});

export type OBS_State = z.infer<typeof Service_OBS_Schema>
