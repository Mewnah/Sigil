import { TextEventSource, TextEventSourceSchema } from "@/types";
import { zSafe } from "@/utils";
import { z } from "zod";

function migrateDiscordPersisted(input: unknown): unknown {
  if (input == null || typeof input !== "object" || Array.isArray(input)) return input;
  const o = { ...(input as Record<string, unknown>) };
  if ("postWithTwitchLive" in o && !("postOnlyWhenLive" in o)) {
    o.postOnlyWhenLive = o.postWithTwitchLive;
    delete o.postWithTwitchLive;
  }
  return o;
}

const discordDataObject = z
  .object({
    channelHook: zSafe(z.coerce.string(), ""),
    channelBotName: zSafe(z.coerce.string(), ""),
    channelAvatarUrl: zSafe(z.coerce.string(), ""),
    postEnable: zSafe(z.coerce.boolean(), false),
    postOnlyWhenLive: zSafe(z.coerce.boolean(), false),
    webhookThreadId: zSafe(z.coerce.string(), ""),
    webhookThreadName: zSafe(z.coerce.string(), ""),
    postSource: zSafe(TextEventSourceSchema, TextEventSource.stt),
    postInput: zSafe(z.coerce.boolean(), false),
  })
  .default({});

export const Service_Discord_Schema = z.preprocess(migrateDiscordPersisted, discordDataObject);

export type Discord_State = z.infer<typeof discordDataObject>;
