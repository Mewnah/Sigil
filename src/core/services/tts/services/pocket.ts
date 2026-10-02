import { invoke } from "@tauri-apps/api/core";
import i18n from "i18next";
import { ServiceNetworkState } from "@/types";
import type { RustCatalogEntry, RustCloningTtsRequest, RustModelEntryStatus, RustTtsSinkTarget, RustTtsSpeakReport } from "@/types";
import { entryTitle } from "../../models/format";
import { isSinkCancelled } from "../../sound/sink";
import { clampedOr } from "../num";
import { TTS_State } from "../schema";
import { ITTSReceiver, ITTSService, TtsSinkContext, errorMessage } from "../types";
import { missingDownloads } from "./localNeural";

/**
 * The in-process voice-cloning engines of the Rust `local-tts` plugin (P6-I): Kyutai Pocket TTS (`pocket`, CPU) and
 * Chatterbox Multilingual V3 on Sigil's ONNX Runtime (`chatterboxLocal`, `./chatterboxLocal.ts`). Both clone a voice
 * from the shared voice library and share one TS base ({@link CloningTtsBackend}) and one Rust worker design.
 */
export type CloningEngine = "pocket" | "chatterbox";

/** IPC commands per engine (frozen by P6-S0, `local_tts::{pocket, chatterbox_ort}`). */
export const CLONING_COMMANDS: Readonly<Record<CloningEngine, { speak: string; cancel: string; unload: string }>> = {
  pocket: {
    speak: "plugin:local-tts|pocket_tts_speak",
    cancel: "plugin:local-tts|pocket_tts_cancel",
    unload: "plugin:local-tts|pocket_tts_unload",
  },
  chatterbox: {
    speak: "plugin:local-tts|chatterbox_local_speak",
    cancel: "plugin:local-tts|chatterbox_local_cancel",
    unload: "plugin:local-tts|chatterbox_local_unload",
  },
};

/** Pocket TTS models (`model-catalog/tts-cloning.json`; Rust `pocket::MODELS`), the default first. */
export const POCKET_MODELS = ["tts/pocket-tts-en-int8", "tts/pocket-tts-en"] as const;
export const POCKET_DEFAULT_MODEL = POCKET_MODELS[0];
/** The panel's range for Pocket's sampling temperature (Rust accepts 0.05-2; 0.7 is sherpa's default). */
export const POCKET_TEMPERATURE = { min: 0.1, max: 1.5, default: 0.7 } as const;

/** Fire-and-forget IPC: never throws, synchronously or later. */
function send(command: string, args?: Record<string, unknown>): void {
  try {
    void Promise.resolve(invoke(command, args)).catch(() => {});
  } catch {
    // Without Tauri (web build, tests) there is nothing to tell.
  }
}

/** Threads setting → the request's `threads` (`null` = Rust's default; `1..max`). */
export function cloningThreads(setting: unknown, max = 4): number | null {
  const n = Number(setting);
  return Number.isInteger(n) && n >= 1 && n <= max ? n : null;
}

/** The request `model` of the Pocket settings (an unknown id falls back to the default model). */
export function pocketModel(c: TTS_State["pocket"]): string {
  const m = (c.model ?? "").trim();
  return (POCKET_MODELS as readonly string[]).includes(m) ? m : POCKET_DEFAULT_MODEL;
}

/** The Pocket request for one sentence (`text` "" = load and prepare the voice). */
export function pocketRequest(c: TTS_State["pocket"], text: string): RustCloningTtsRequest {
  return {
    model: pocketModel(c),
    referenceVoiceId: (c.referenceVoiceId ?? "").trim(),
    text,
    // Pocket speaks English and has no rate control: neither is sent.
    language: null,
    speed: 1,
    threads: cloningThreads(c.threads),
    options: {
      temperature: clampedOr(c.temperature, POCKET_TEMPERATURE.default, POCKET_TEMPERATURE.min, POCKET_TEMPERATURE.max),
    },
  };
}

/**
 * A localized "download it first" message when `ids` (or what they require) are not installed, else `null` (also
 * when the catalog is unknown: Rust then reports the missing model itself).
 */
export async function cloningDownloadRefusal(ids: readonly string[]): Promise<string | null> {
  const models = window.ApiServer?.models;
  if (!models || !ids.length) return null;
  try {
    await models.load();
  } catch {
    return null;
  }
  const missing = missingDownloads(
    ids,
    models.state.catalog as RustCatalogEntry[],
    models.state.states as Record<string, RustModelEntryStatus>,
  );
  if (!missing?.length) return null;
  return i18n.t("tts.cloning_not_downloaded", { title: missing.map(entryTitle).join(", ") });
}

/**
 * Speak `request.text` once into a new sink utterance on `target` (the panels' voice preview). Resolves after the
 * audio is queued; a stop / skip meanwhile is not an error. The model stays loaded (the engine frees it when idle).
 */
export async function previewCloningVoice(engine: CloningEngine, request: RustCloningTtsRequest, target: RustTtsSinkTarget): Promise<void> {
  const sink = window.ApiServer.sound.sink;
  const id = `${engine}-preview-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  await sink.begin(id, target);
  try {
    await invoke<RustTtsSpeakReport>(CLONING_COMMANDS[engine].speak, { request, utteranceId: id });
    await sink.end(id);
  } catch (error) {
    await sink.cancel(id).catch(() => {});
    if (!isSinkCancelled(error)) throw error;
  }
}

/**
 * Free `engine`'s model after a panel preview, unless the running TTS backend is `backend` (it keeps the model
 * loaded). Chatterbox holds a few GB; Pocket a few hundred MB.
 */
export function unloadCloningIfIdle(engine: CloningEngine, backend: string): void {
  const server = window.ApiServer;
  const running = server?.tts?.serviceState?.status !== ServiceNetworkState.disconnected;
  const active = server?.state?.services?.tts?.data?.backend === backend;
  if (!(running && active)) send(CLONING_COMMANDS[engine].unload);
}

/**
 * Base of the cloning backends (ITTSService v2 `synthesizeToSink`): the Rust engine speaks each sentence straight into
 * the sink utterance the sentence pipeline began (no audio crosses IPC); stop / skip abort the signal, which cancels
 * the Rust request at once.
 *
 * - `start` refuses with a clear message when the model is not downloaded, then sends an empty-text request: the Rust
 *   worker loads the model and prepares the reference voice, so a missing voice or a driver problem is reported at
 *   Start and the first sentence does not wait for the load. That request also holds the model until `stop` (no idle
 *   unload between sentences, however far apart).
 * - Each sentence reads the current settings (a new voice or option applies from the next sentence on).
 * - `stop` cancels everything and frees the model; a start right after keeps it (the worker skips an unload with a
 *   request waiting behind it).
 */
export abstract class CloningTtsBackend implements ITTSService {
  #running = false;
  #stopped = false;

  constructor(
    protected readonly receiver: ITTSReceiver,
    protected readonly engine: CloningEngine,
    /** Prefix of error messages. */
    protected readonly label: string,
  ) {}

  /** The request for one sentence, or a localized problem to refuse with. `lang` is the sentence's language tag. */
  protected abstract request(state: TTS_State, text: string, lang: string | null): RustCloningTtsRequest | string;

  /** Catalog ids the current settings need installed. */
  protected abstract requirementIds(state: TTS_State): string[];

  get running(): boolean {
    return this.#running;
  }

  async start(state: TTS_State): Promise<void> {
    const request = this.request(state, "", null);
    if (typeof request === "string") {
      this.receiver.onStop(request);
      return;
    }
    try {
      const refusal = await cloningDownloadRefusal(this.requirementIds(state));
      if (this.#stopped) return;
      if (refusal) {
        this.receiver.onStop(refusal);
        return;
      }
      await invoke<RustTtsSpeakReport>(CLONING_COMMANDS[this.engine].speak, { request, utteranceId: "" });
      if (this.#stopped) return;
      this.#running = true;
      this.receiver.onStart();
    } catch (error) {
      if (this.#stopped) return;
      this.receiver.onStop(`${this.label}: ${errorMessage(error)}`);
    }
  }

  /** v2: the Rust engine speaks one sentence into the begun sink utterance `sink.utteranceId`. */
  async synthesizeToSink(text: string, sink: TtsSinkContext, signal: AbortSignal): Promise<void> {
    if (!this.#running || signal.aborted || !text.trim()) return;
    const request = this.request(window.ApiServer.state.services.tts.data, text, sink.lang ?? null);
    if (typeof request === "string") throw new Error(request);
    const commands = CLONING_COMMANDS[this.engine];
    const onAbort = () => send(commands.cancel, { utteranceId: sink.utteranceId });
    signal.addEventListener("abort", onAbort, { once: true });
    try {
      await invoke<RustTtsSpeakReport>(commands.speak, { request, utteranceId: sink.utteranceId });
    } catch (error) {
      // Stop / skip cancel the utterance; Rust then answers "cancelled".
      if (signal.aborted || isSinkCancelled(error)) return;
      throw new Error(`${this.label}: ${errorMessage(error)}`);
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
  }

  /** Legacy whole-line path; unused (a `synthesizeToSink` backend always speaks through the sentence pipeline). */
  play(_value: string): void {}

  stop(): void {
    this.#halt();
    this.receiver.onStop();
  }

  dispose(): void {
    this.#halt();
  }

  #halt() {
    const wasActive = !this.#stopped;
    this.#stopped = true;
    this.#running = false;
    if (!wasActive) return;
    // Cancel what is queued, then free the model (in this order, without waiting: a start right after queues its
    // warm-up behind the unload, which then keeps the model).
    const commands = CLONING_COMMANDS[this.engine];
    send(commands.cancel, { utteranceId: null });
    send(commands.unload);
  }
}

/**
 * `pocket` backend: Kyutai Pocket TTS, voice cloning on the processor (about two cores) in process
 * (`plugin:local-tts|pocket_tts_speak`). English only; a voice-library voice, or the model's built-in voice when none
 * is chosen.
 */
export class TTS_PocketService extends CloningTtsBackend {
  constructor(receiver: ITTSReceiver) {
    super(receiver, "pocket", "Pocket TTS");
  }

  protected request(state: TTS_State, text: string): RustCloningTtsRequest {
    return pocketRequest(state.pocket, text);
  }

  protected requirementIds(state: TTS_State): string[] {
    return [pocketModel(state.pocket)];
  }
}
