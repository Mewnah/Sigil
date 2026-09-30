import { zSafe } from "@/utils";
import { z } from "zod";
import { TTS_Azure_Schema } from "./azure";
import { TTS_Chatterbox_Schema } from "./chatterbox";
import { TTS_CommonShape } from "./common";
import { TTS_FishSpeech_Schema } from "./fishSpeech";
import { TTS_Kokoro_Schema } from "./kokoro";
import { TTS_LocalNeural_Schema } from "./localNeural";
import { TTS_Melo_Schema } from "./melo";
import { TTS_Native_Schema } from "./native";
import { TTS_NeuralOnnx_Schema } from "./neuralOnnx";
import { TTS_NeuralSidecar_Schema } from "./neuralSidecar";
import { TTS_OpenAiSpeech_Schema } from "./openaiSpeech";
import { TTS_Route_Schema, migrateTtsRoute } from "./route";
import { TTS_Uberduck_Schema } from "./uberduck";
import { TTS_Voicevox_Schema } from "./voicevox";
import { TTS_Windows_Schema } from "./windows";

export { TTS_Routes } from "./route";

// One slice per backend (P2-S0 split of the former tts/schema.ts): index, common, route, azure, voicevox, uberduck,
// native, windows (P2-G); neuralOnnx, localNeural (P2-H); openaiSpeech, kokoro, chatterbox, melo, fishSpeech,
// neuralSidecar (P2-I).

/** Persisted backend ids: add values, never rename or remove them (zSafe falls back to the default). */
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
  /** Local ONNX speech (`neural_onnx_tts` plugin; Hub or local speech package — see docs/OPERATOR_REFERENCE.md). */
  neuralOnnx = "neuralOnnx",
  /** User-run HTTP server (`/speech`, `/health`). */
  neuralSidecar = "neuralSidecar",
  /** In-process Kokoro / Piper / Kitten (Rust `local-tts` plugin, P2). */
  localNeural = "localNeural",
  /** Any OpenAI-compatible `/v1/audio/speech` server, streaming (Rust `openai-speech` plugin, P2). */
  openaiSpeech = "openaiSpeech",
}

const zodTTS_Backends = z.nativeEnum(TTS_Backends);

/**
 * Persisted TTS settings. `route` (device / voice changer / both, see `route.ts`) is top-level and applies to every
 * backend that plays through the TTS sink; saved settings from before it existed are migrated from the active
 * backend's old `routeThroughVoiceChanger` flag ({@link migrateTtsRoute}).
 */
export const Service_TTS_Schema = z.preprocess(migrateTtsRoute, z.object({
  ...TTS_CommonShape,
  backend: zSafe(zodTTS_Backends, TTS_Backends.native),
  /**
   * The P2 desktop default ran once (`defaults.ts`): new desktop installs start on `localNeural`; existing profiles
   * on Web Speech / SAPI get a one-time offer instead of a silent switch.
   */
  desktopDefaultMigrated: zSafe(z.coerce.boolean(), false),
  /** The one-time "speak with Kokoro" offer was shown (answered or not). */
  localOfferShown: zSafe(z.coerce.boolean(), false),
  route: TTS_Route_Schema,
  native: TTS_Native_Schema,
  windows: TTS_Windows_Schema,
  uberduck: TTS_Uberduck_Schema,
  azure: TTS_Azure_Schema,
  voicevox: TTS_Voicevox_Schema,
  kokoro: TTS_Kokoro_Schema,
  melo: TTS_Melo_Schema,
  chatterbox: TTS_Chatterbox_Schema,
  fishSpeech: TTS_FishSpeech_Schema,
  neuralOnnx: TTS_NeuralOnnx_Schema,
  neuralSidecar: TTS_NeuralSidecar_Schema,
  localNeural: TTS_LocalNeural_Schema,
  openaiSpeech: TTS_OpenAiSpeech_Schema,
})).default({});

export type TTS_State = z.infer<typeof Service_TTS_Schema>
