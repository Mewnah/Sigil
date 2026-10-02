/**
 * Host integration / inspector tab ids — kept free of core/client imports to avoid circular deps.
 * Not every value has persisted settings: see `PersistedHostServiceKey` in `src/core/schema.ts`
 * and `docs/DEVELOPER_GUIDE.md` (host services section).
 */
export enum Services {
  vrc = "vrc",
  stt = "stt",
  tts = "tts",
  translation = "translation",
  transform = "transform",
  twitch = "twitch",
  kick = "kick",
  discord = "discord",
  obs = "obs",
  voice_changer = "voice_changer",
  /** Other players: loopback transcription and reverse translation (P6-A; persisted slice `others`). */
  others = "others",
}
