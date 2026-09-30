import { zSafe } from "@/utils";
import { z } from "zod";
import { STT_Azure_Schema } from "./azure";
import { STT_Deepgram_Schema } from "./deepgram";
import { STT_Local_Schema } from "./local";
import { STT_Moonshine_Schema } from "./moonshine";
import { STT_Native_Schema } from "./native";
import { STT_OpenAiAudio_Schema } from "./openaiAudio";
import { STT_Vad_Schema } from "./vad";
import { STT_Vosk_Schema } from "./vosk";
import { STT_Whisper_Schema } from "./whisper";

// One slice per backend (P2-S0 split of the former stt/schema.ts): index + local + whisper (P2-E); vosk, moonshine,
// openaiAudio, azure, deepgram (P2-F); native and vad are shared.
export { OpenAiAudioPreset } from "./openaiAudio";

/** Persisted backend ids: add values, never rename or remove them (zSafe falls back to the default). */
export enum STT_Backends {
  native = "native",
  chrome = "chrome",
  edge = "edge",
  browser = "browser",
  azure = "azure",
  deepgram = "deepgram",
  whisper = "whisper",
  vosk = "vosk",
  moonshine = "moonshine",
  /** Local server: POST /v1/audio/transcriptions (same request shape as OpenAI; e.g. speaches, LocalAI). */
  openai_audio = "openai_audio",
  /** Sigil's in-process speech pipeline (Rust `speech` plugin) with automatic model routing (P2). */
  local = "local",
}

export const zodSTT_Backends = z.nativeEnum(STT_Backends);

export const Service_STT_Schema = z.object({
  backend: zSafe(zodSTT_Backends, STT_Backends.native),
  /**
   * Set once the one-time desktop default migration ran (`native`, which cannot work in the desktop webview, is
   * switched to local Whisper). Never reset, so a later deliberate choice of `native` is kept.
   */
  desktopDefaultMigrated: zSafe(z.coerce.boolean(), false),
  /** Set once the one-time offer to switch to the `local` backend was answered (`Service_STT.offerLocalBackend`). */
  localOfferShown: zSafe(z.coerce.boolean(), false),
  autoStart: zSafe(z.coerce.boolean(), false),
  uwu: zSafe(z.coerce.boolean(), false),
  stopWithStream: zSafe(z.coerce.boolean(), false),
  replaceWords: zSafe(z.record(z.coerce.string(), z.coerce.string()), {}),
  replaceWordsIgnoreCase: zSafe(z.coerce.boolean(), false),
  replaceWordsPreserveCase: zSafe(z.coerce.boolean(), false),
  native: STT_Native_Schema,
  azure: STT_Azure_Schema,
  vad: STT_Vad_Schema,
  whisper: STT_Whisper_Schema,
  deepgram: STT_Deepgram_Schema,
  vosk: STT_Vosk_Schema,
  moonshine: STT_Moonshine_Schema,
  openai_audio: STT_OpenAiAudio_Schema,
  local: STT_Local_Schema,
}).default({});

export type STT_State = z.infer<typeof Service_STT_Schema>;
