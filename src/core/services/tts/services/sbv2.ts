import { invoke } from "@tauri-apps/api/core";
import i18n from "i18next";
import { proxy } from "valtio";
import { ServiceNetworkState } from "@/types";
import type {
  RustCatalogEntry,
  RustModelEntryStatus,
  RustSbv2Model,
  RustTtsHelperEndpoint,
  RustTtsSinkTarget,
  RustTtsSpeakReport,
} from "@/types";
import { entryTitle } from "../../models/format";
import { isSinkCancelled } from "../../sound/sink";
import { clampedOr } from "../num";
import type { TTS_State } from "../schema";
import type { TTS_Sbv2_State } from "../schema/sbv2";
import { ITTSReceiver, ITTSService, TtsSinkContext, errorMessage } from "../types";
import { missingDownloads } from "./localNeural";

/**
 * `sbv2` backend (P6-G): Style-Bert-VITS2 JP-Extra and AivisSpeech `.aivmx` voices running inside Sigil on its own ONNX
 * Runtime (the Rust `tts-helpers` plugin's `sbv2` engine: a Japanese OpenJTalk frontend, a DeBERTa BERT and the voice's
 * VITS graph, on a worker thread). No server: `tts_helper_start { engine: "sbv2", config }` loads the engine and the
 * chosen voice, and every sentence is one `sbv2_speak` that pushes its audio straight into the begun TTS sink utterance
 * (no audio crosses IPC). Voices are the user's files, imported once (`sbv2_import_model`: a copy into Sigil's library
 * after a format check).
 *
 * - Speaker, style, style strength, tempo variation and speed apply to the next sentence; the voice model, the device
 *   (processor / GPU) and the threads restart the engine (the panel locks them while it runs).
 * - Stop cancels what is queued and frees the engine (the BERT holds about 0.7 GB of RAM). Stop and start are
 *   serialized, so a restart never cancels its own load.
 */

/** Catalog id of the Japanese BERT (Rust `tts_helpers::sbv2::BERT_MODEL_ID`). */
export const SBV2_BERT_ID = "tts/sbv2-bert-ja";
/** Catalog id of the jpreprocess dictionary (Rust `tts_helpers::sbv2::DICTIONARY_ID`). */
export const SBV2_DICTIONARY_ID = "text/jpreprocess-naist-jdic";
/** Everything the engine downloads (the voices are the user's own files). */
export const SBV2_REQUIREMENTS: readonly string[] = [SBV2_BERT_ID, SBV2_DICTIONARY_ID];
/** File types the import accepts (a folder of an ONNX export works too). */
export const SBV2_IMPORT_EXTENSIONS = ["aivmx", "onnx"] as const;

/** A style of a speaker (Rust `format::StyleInfo`). */
export interface Sbv2Style {
  name: string;
  /** Row of the style vectors. */
  id: number;
}

/** A speaker of a model (Rust `format::SpeakerInfo`). */
export interface Sbv2Speaker {
  name: string;
  id: number;
  styles: Sbv2Style[];
  languages: string[];
}

/** Row of `sbv2_list_models` with the fields P6-G added (optional on the frozen `RustSbv2Model`). */
export type Sbv2Model = RustSbv2Model & {
  voices?: Sbv2Speaker[];
  description?: string;
  creators?: string[];
  version?: string;
  architecture?: string;
  sampleRate?: number;
  sizeBytes?: number;
  sourceName?: string;
};

/** `sbv2_speak`'s `request` (Rust `tts_helpers::sbv2::Sbv2SpeakRequest`). */
export interface Sbv2SpeakRequest {
  /** Imported model id ("" = the started one). */
  model: string;
  text: string;
  speaker: string;
  style: string;
  styleWeight: number;
  sdpRatio: number;
  noise: number;
  noiseW: number;
  speed: number;
}

/** `tts_helper_start`'s `config` for the `sbv2` engine (Rust `tts_helpers::sbv2::StartConfig`). */
export interface Sbv2StartConfig {
  model: string;
  ep: string;
  threads: number | null;
}

// -------------------------------------------------------------------------------------------------
// Settings (pure)
// -------------------------------------------------------------------------------------------------

/** Engine settings that need a restart. */
export function sbv2StartConfig(c: TTS_Sbv2_State): Sbv2StartConfig {
  const threads = Number(c.threads);
  return {
    model: c.model.trim(),
    ep: c.executionProvider,
    threads: Number.isInteger(threads) && threads >= 1 && threads <= 8 ? threads : null,
  };
}

/** Everything but the text of a sentence (applies to the next sentence). */
export function sbv2SpeakSettings(c: TTS_Sbv2_State): Omit<Sbv2SpeakRequest, "text"> {
  return {
    model: c.model.trim(),
    speaker: c.speaker.trim(),
    style: c.style.trim(),
    styleWeight: clampedOr(c.styleWeight, 1, 0, 20),
    sdpRatio: clampedOr(c.sdpRatio, 0.2, 0, 1),
    noise: clampedOr(c.noise, 0.6, 0, 2),
    noiseW: clampedOr(c.noiseW, 0.8, 0, 2),
    speed: clampedOr(c.speed, 1, 0.5, 2),
  };
}

/**
 * The speakers of `model` with their styles: the detailed list when Rust sent it, else one speaker per name with the
 * model's style names.
 */
export function sbv2Speakers(model: Sbv2Model | null | undefined): Sbv2Speaker[] {
  if (!model) return [];
  if (model.voices?.length) return model.voices;
  const styles = model.styles.map((name, id) => ({ name, id }));
  return model.speakers.map((name, id) => ({ name, id, styles, languages: ["ja"] }));
}

/** The speaker a setting picks (its name, else the first). */
export function pickSpeaker(model: Sbv2Model | null | undefined, speaker: string): Sbv2Speaker | null {
  const all = sbv2Speakers(model);
  return all.find((s) => s.name === speaker) ?? all[0] ?? null;
}

/** The style a setting picks within `speaker` (its name, else the neutral style, id 0). */
export function pickStyle(speaker: Sbv2Speaker | null, style: string): Sbv2Style | null {
  if (!speaker) return null;
  return speaker.styles.find((s) => s.name === style) ?? speaker.styles.find((s) => s.id === 0) ?? speaker.styles[0] ?? null;
}

/** Whether `style` is the neutral (mean) style, where the style strength has no effect. */
export function isNeutralStyle(style: Sbv2Style | null): boolean {
  return !style || style.id === 0;
}

// -------------------------------------------------------------------------------------------------
// The voice library (imported models)
// -------------------------------------------------------------------------------------------------

/** Imported voice models, for the panel. */
export const sbv2Library = proxy({
  models: null as Sbv2Model[] | null,
  loading: false,
  importing: false,
  error: null as string | null,
});

/** Reload the imported models. */
export async function refreshSbv2Models(): Promise<Sbv2Model[]> {
  sbv2Library.loading = true;
  try {
    const list = await invoke<Sbv2Model[]>("plugin:tts-helpers|sbv2_list_models");
    sbv2Library.models = Array.isArray(list) ? list : [];
    sbv2Library.error = null;
    return sbv2Library.models;
  } catch (e) {
    sbv2Library.error = errorMessage(e);
    throw e;
  } finally {
    sbv2Library.loading = false;
  }
}

/** The import running now (imports run one after another; Rust serializes them too). */
let importQueue: Promise<unknown> = Promise.resolve();
/** Imports queued or running; `sbv2Library.importing` while any is. */
let importsPending = 0;

/**
 * Import the voice at `path` (a `.aivmx`, an ONNX export or its folder); resolves with the model (an existing one when
 * it is already imported). A second drop during an import waits for the first, and `importing` stays set until the
 * last one ends.
 */
export function importSbv2Model(path: string): Promise<Sbv2Model> {
  importsPending += 1;
  sbv2Library.importing = true;
  const run = importQueue.then(async () => {
    const model = await invoke<Sbv2Model>("plugin:tts-helpers|sbv2_import_model", { path });
    await refreshSbv2Models().catch(() => {});
    return model;
  });
  const done = run.finally(() => {
    importsPending -= 1;
    if (importsPending === 0) sbv2Library.importing = false;
  });
  importQueue = done.catch(() => {});
  return done;
}

/** Delete an imported model. */
export async function removeSbv2Model(id: string): Promise<void> {
  await invoke("plugin:tts-helpers|sbv2_remove_model", { id });
  await refreshSbv2Models().catch(() => {});
}

// -------------------------------------------------------------------------------------------------
// IPC
// -------------------------------------------------------------------------------------------------

/** Fire-and-forget IPC: never throws, synchronously or later. */
function send(command: string, args?: Record<string, unknown>): Promise<void> {
  try {
    return Promise.resolve(invoke(command, args)).then(
      () => {},
      () => {},
    );
  } catch {
    return Promise.resolve();
  }
}

/**
 * Engine starts and stops in call order (a stop's cancel must never hit the next start's load). Each step waits for
 * the previous one, whatever its outcome.
 */
let lifecycle: Promise<unknown> = Promise.resolve();

function inOrder<T>(step: () => Promise<T>): Promise<T> {
  const run = lifecycle.then(step, step);
  lifecycle = run.catch(() => {});
  return run;
}

/** Speak `text` with `settings` into the begun sink utterance `utteranceId`. */
export function speakSbv2(settings: Omit<Sbv2SpeakRequest, "text">, text: string, utteranceId: string): Promise<RustTtsSpeakReport> {
  const request: Sbv2SpeakRequest = { ...settings, text };
  return invoke<RustTtsSpeakReport>("plugin:tts-helpers|sbv2_speak", { request, utteranceId });
}

/** A localized "download it first" message when the engine's files are not installed, else `null`. */
async function downloadRefusal(): Promise<string | null> {
  const models = window.ApiServer?.models;
  if (!models) return null;
  try {
    await models.load();
  } catch {
    return null;
  }
  const missing = missingDownloads(
    SBV2_REQUIREMENTS,
    models.state.catalog as RustCatalogEntry[],
    models.state.states as Record<string, RustModelEntryStatus>,
  );
  if (!missing?.length) return null;
  return i18n.t("tts.sbv2_not_downloaded", { title: missing.map(entryTitle).join(", ") });
}

/** Start the engine with `config` (serialized with stops). */
export function startSbv2Engine(config: Sbv2StartConfig): Promise<RustTtsHelperEndpoint> {
  return inOrder(() => invoke<RustTtsHelperEndpoint>("plugin:tts-helpers|tts_helper_start", { engine: "sbv2", config }));
}

/** Cancel everything and free the engine (serialized with starts). */
export function stopSbv2Engine(): Promise<void> {
  return inOrder(async () => {
    await send("plugin:tts-helpers|sbv2_cancel", { utteranceId: null });
    await send("plugin:tts-helpers|tts_helper_stop", { engine: "sbv2" });
  });
}

/**
 * Speak once into a new sink utterance on `target` with the settings of `c` (the panel's preview): starts the engine
 * for the chosen voice when needed. A stop / skip meanwhile is not an error.
 */
export async function previewSbv2(c: TTS_Sbv2_State, text: string, target: RustTtsSinkTarget): Promise<void> {
  await startSbv2Engine(sbv2StartConfig(c));
  const sink = window.ApiServer.sound.sink;
  const id = `sbv2-preview-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  await sink.ready();
  await sink.begin(id, target);
  try {
    await speakSbv2(sbv2SpeakSettings(c), text, id);
    await sink.end(id);
  } catch (error) {
    await sink.cancel(id).catch(() => {});
    if (!isSinkCancelled(error)) throw error;
  }
}

/** Free the engine after a preview, unless the `sbv2` backend is running. */
export function releaseSbv2IfIdle(): void {
  const server = window.ApiServer;
  const status = server?.tts?.serviceState?.status;
  const running = status !== undefined && status !== ServiceNetworkState.disconnected;
  if (!(running && server?.state?.services?.tts?.data?.backend === "sbv2")) void stopSbv2Engine();
}

// -------------------------------------------------------------------------------------------------
// The backend
// -------------------------------------------------------------------------------------------------

export class TTS_Sbv2Service implements ITTSService {
  #running = false;
  #stopped = false;

  constructor(private readonly receiver: ITTSReceiver) {}

  get running(): boolean {
    return this.#running;
  }

  async start(state: TTS_State): Promise<void> {
    const config = sbv2StartConfig(state.sbv2);
    if (!config.model) {
      this.receiver.onStop(i18n.t("tts.sbv2_no_voice"));
      return;
    }
    try {
      const refusal = await downloadRefusal();
      if (this.#stopped) return;
      if (refusal) {
        this.receiver.onStop(refusal);
        return;
      }
      await startSbv2Engine(config);
      if (this.#stopped) return;
      this.#running = true;
      this.receiver.onStart();
    } catch (error) {
      if (this.#stopped) return;
      this.receiver.onStop(`Style-Bert-VITS2: ${errorMessage(error)}`);
    }
  }

  /** v2: the engine speaks one sentence into the begun sink utterance `sink.utteranceId`. */
  async synthesizeToSink(text: string, sink: TtsSinkContext, signal: AbortSignal): Promise<void> {
    if (!this.#running || signal.aborted || !text.trim()) return;
    const settings = sbv2SpeakSettings(window.ApiServer.state.services.tts.data.sbv2);
    const onAbort = () => void send("plugin:tts-helpers|sbv2_cancel", { utteranceId: sink.utteranceId });
    signal.addEventListener("abort", onAbort, { once: true });
    try {
      await speakSbv2(settings, text, sink.utteranceId);
    } catch (error) {
      // Stop / skip cancel the utterance; Rust then answers "cancelled".
      if (signal.aborted || isSinkCancelled(error) || !this.#running) return;
      throw new Error(`Style-Bert-VITS2: ${errorMessage(error)}`);
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
  }

  /** Legacy whole-line path. Unused: a backend with `synthesizeToSink` always speaks through the sentence pipeline. */
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
    if (wasActive) void stopSbv2Engine();
  }
}
