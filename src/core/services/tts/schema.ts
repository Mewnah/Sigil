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
  /** Local ONNX speech (`neural_onnx_tts` plugin; Hub or local speech package — see docs/LOCAL_SPEECH_PACKAGE.md). */
  neuralOnnx = "neuralOnnx",
  /** User-run HTTP server (`/speech`, `/health`). */
  neuralSidecar = "neuralSidecar",
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
  }).default({}),
  neuralOnnx: z.object({
    /** `hub` = download from Hugging Face; `local` = folder matching supported package (see docs/LOCAL_SPEECH_PACKAGE.md). */
    modelSource: zSafe(z.enum(["hub", "local"]), "hub"),
    /** On-disk layout when `modelSource` is `local`. Only `speech_package_v1` is implemented today. */
    localPackageFormat: zSafe(z.enum(["speech_package_v1"]), "speech_package_v1"),
    localModelDir: zSafe(z.coerce.string(), ""),
    /** Default Hugging Face repo id for Hub mode (example model). */
    hubId: zSafe(z.coerce.string(), "KittenML/kitten-tts-mini-0.8"),
    voice: zSafe(z.coerce.string(), "Jasper"),
    speed: zSafe(zStringNumber(), "1.0"),
    /** Hint for ORT / GPU (`cpu`, `cuda`, `dml`). Actual EPs depend on the bundled ONNX Runtime. */
    executionProvider: zSafe(z.enum(["cpu", "cuda", "dml"]), "cpu"),
    device: zSafe(z.coerce.string(), ""),
    /** Mix TTS output into the live voice changer input (start VC first). */
    routeThroughVoiceChanger: zSafe(z.coerce.boolean(), false),
  }).default({}),
  neuralSidecar: z.object({
    endpoint: zSafe(z.coerce.string(), "http://127.0.0.1:8766"),
    voice: zSafe(z.coerce.string(), "default"),
    device: zSafe(z.coerce.string(), ""),
    routeThroughVoiceChanger: zSafe(z.coerce.boolean(), false),
  }).default({}),
}).default({});

export type TTS_State = z.infer<typeof Service_TTS_Schema>


