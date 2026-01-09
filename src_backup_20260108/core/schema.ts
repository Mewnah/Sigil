import { customAlphabet, urlAlphabet } from "nanoid";
import { Service_Discord_Schema } from "./services/discord/schema";
import { Service_OBS_Schema } from "./services/obs/schema";
import { Service_Kick_Schema } from "./services/kick/schema";
import { Service_STT_Schema } from "./services/stt/schema";
import { Service_Translation_Schema } from "./services/translation/schema";
import { Service_Transform_Schema } from "./services/transform/schema";
import { Service_TTS_Schema } from "./services/tts/schema";
import { Service_Twitch_Schema } from "./services/twitch/schema";
import { Service_VRC_Schema } from "./services/vrc/schema";

import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

const zodServiceSchemaFactory = <Data extends z.ZodDefault<z.AnyZodObject>>(schema: Data) => {
  return z.object({
    showActionButton: zSafe(z.coerce.boolean(), false),
    data: schema
  });
}

export const BackendSchema = z.object({
  id: zSafe(z.string(), () => customAlphabet(urlAlphabet, 42)()),
  linkAddress: zSafe(z.string(), ""),
  clientTheme: zSafe(z.string(), "sigil"),
  uiScale: zSafe(z.number(), 1),
  uiLanguage: zSafe(z.string(), "en"),
  showOverlay: zSafe(z.coerce.boolean(), false),
  showLogs: zSafe(z.coerce.boolean(), false),
  muteSoundEffects: zSafe(z.coerce.boolean(), false),
  audioInputDevice: zSafe(z.string(), ""),
  audioOutputDevice: zSafe(z.string(), ""),
  onboardingComplete: zSafe(z.coerce.boolean(), false),
  showOverlayLogs: zSafe(z.coerce.boolean(), false),
  backgroundInputTimer: zSafe(zStringNumber(), "5000"),
  recentSnapshots: zSafe(z.array(z.object({
    id: z.string(),
    name: z.string(),
    path: z.string(),
    date: z.string()
  })), []),
  elementTemplates: zSafe(z.array(z.object({
    id: z.string(),
    name: z.string(),
    type: z.string(),
    data: z.any()
  })), []),
  shortcuts: z.object({
    bgInput: zSafe(z.string(), ""),
    start: zSafe(z.string(), ""),
    muteMic: zSafe(z.string(), ""),
    muteSound: zSafe(z.string(), ""),
  }).default({}),
  services: z.object({
    vrc: zodServiceSchemaFactory(Service_VRC_Schema).default({}),
    stt: zodServiceSchemaFactory(Service_STT_Schema).default({}),
    tts: zodServiceSchemaFactory(Service_TTS_Schema).default({}),
    translation: zodServiceSchemaFactory(Service_Translation_Schema).default({}),
    transform: zodServiceSchemaFactory(Service_Transform_Schema).default({}),
    twitch: zodServiceSchemaFactory(Service_Twitch_Schema).default({}),
    kick: zodServiceSchemaFactory(Service_Kick_Schema).default({}),
    discord: zodServiceSchemaFactory(Service_Discord_Schema).default({}),
    obs: zodServiceSchemaFactory(Service_OBS_Schema).default({}),
  }).default({})
}).default({});
export type BackendState = z.infer<typeof BackendSchema>;
