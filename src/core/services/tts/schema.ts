import { TextEventSource, TextEventSourceSchema } from "@/types";
import { zSafe, zStringNumber } from "@/utils";
import { z } from "zod";

export enum TTS_Backends {
  native = "native",
  windows = "windows",
  azure = "azure",
  uberduck = "uberduck",
  voicevox = "voicevox",
  kokoro = "kokoro",
  melo = "melo",
  chatterbox = "chatterbox",
  fishSpeech = "fishSpeech",
}

const zodTTS_Backends = z.nativeEnum(TTS_Backends);

export const Service_TTS_Schema = z.object({
  source: zSafe(TextEventSourceSchema, TextEventSource.stt),
  inputField: zSafe(z.coerce.boolean(), true),
  backend: zSafe(zodTTS_Backends, TTS_Backends.native),
  autoStart: zSafe(z.coerce.boolean(), false),
  stopWithStream: zSafe(z.coerce.boolean(), false),
  replaceWords: zSafe(z.record(z.coerce.string(), z.coerce.string()), {}),
  replaceWordsIgnoreCase: zSafe(z.coerce.boolean(), true),
  native: z.object({
    voice: zSafe(z.coerce.string(), ""),
    pitch: zSafe(zStringNumber(), "1"),
    rate: zSafe(zStringNumber(), "1"),
    volume: zSafe(zStringNumber(), "1"),
  }).default({}),
  windows: z.object({
    device: zSafe(z.coerce.string(), ""),
    voice: zSafe(z.coerce.string(), ""),
    volume: zSafe(zStringNumber(), "1"),
    rate: zSafe(zStringNumber(), "1"),
  }).default({}),
  uberduck: z.object({
    api_key: zSafe(z.coerce.string(), ""),
    secret_key: zSafe(z.coerce.string(), ""),
    device: zSafe(z.coerce.string(), ""),
    voice: zSafe(z.coerce.string(), ""),
    volume: zSafe(zStringNumber(), "1"),
    rate: zSafe(zStringNumber(), "1"),
  }).default({}),
  azure: z.object({
    device: zSafe(z.coerce.string(), ""),
    language: zSafe(z.coerce.string(), "English (United States)"),
    voice: zSafe(z.coerce.string(), ""),
    voiceStyle: zSafe(z.coerce.string(), ""),
    voiceRole: zSafe(z.coerce.string(), ""),
    voiceVolume: zSafe(z.coerce.string(), "default"),
    voiceRate: zSafe(z.coerce.string(), "default"),
    voicePitch: zSafe(z.coerce.string(), "default"),
    voiceRange: zSafe(z.coerce.string(), "default"),
    volume: zSafe(zStringNumber(), "1"),
    rate: zSafe(zStringNumber(), "1"),
    key: zSafe(z.coerce.string(), ""),
    location: zSafe(z.coerce.string(), ""),
  }).default({}),
  voicevox: z.object({
    host: zSafe(z.coerce.string(), "http://localhost:50021"),
    speaker: zSafe(z.coerce.string(), "0"),
    speedScale: zSafe(zStringNumber(), "1.0"),
    pitchScale: zSafe(zStringNumber(), "0.0"),
    intonationScale: zSafe(zStringNumber(), "1.0"),
    volumeScale: zSafe(zStringNumber(), "1.0"),
    device: zSafe(z.coerce.string(), ""),
  }).default({}),
  kokoro: z.object({
    endpoint: zSafe(z.coerce.string(), "http://localhost:8880"),
    voice: zSafe(z.coerce.string(), "af_bella"),
    speed: zSafe(zStringNumber(), "1.0"),
    device: zSafe(z.coerce.string(), ""),
  }).default({}),
  melo: z.object({
    endpoint: zSafe(z.coerce.string(), "http://localhost:8888"),
    speaker: zSafe(z.coerce.string(), ""),
    speed: zSafe(zStringNumber(), "1.0"),
    device: zSafe(z.coerce.string(), ""),
  }).default({}),
  chatterbox: z.object({
    endpoint: zSafe(z.coerce.string(), "http://localhost:5555"),
    voice: zSafe(z.coerce.string(), "default"),
    speed: zSafe(zStringNumber(), "1.0"),
    exaggeration: zSafe(zStringNumber(), "0.5"),
    device: zSafe(z.coerce.string(), ""),
  }).default({}),
  fishSpeech: z.object({
    endpoint: zSafe(z.coerce.string(), "http://localhost:8080"),
    referenceId: zSafe(z.coerce.string(), ""),
    device: zSafe(z.coerce.string(), ""),
  }).default({})
}).default({});

export type TTS_State = z.infer<typeof Service_TTS_Schema>


