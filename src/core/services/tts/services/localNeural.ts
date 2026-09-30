import { invoke } from "@tauri-apps/api/core";
import i18n from "i18next";
import { proxy } from "valtio";
import { ServiceNetworkState } from "@/types";
import type {
  RustCatalogEntry,
  RustLocalTtsModelInfo,
  RustLocalTtsRequest,
  RustModelEntryStatus,
  RustTtsSinkTarget,
  RustTtsSpeakReport,
} from "@/types";
import { entryTitle, summarizeRequirement } from "../../models/format";
import { isSinkCancelled } from "../../sound/sink";
import { clampedOr } from "../num";
import { TTS_State } from "../schema";
import { LOCAL_NEURAL_FOLDER } from "../schema/localNeural";
import { ITTSReceiver, ITTSService, TtsSinkContext, errorMessage } from "../types";

/** The default local model (Kokoro v1.0 fp32, the zero-setup voice; Rust `local_tts::catalog::DEFAULT_MODEL`). */
export const LOCAL_TTS_DEFAULT_MODEL = "tts/kokoro-v1.0";
/** Prefix of a KittenML Hugging Face repo in a request's `model` (legacy `neuralOnnx` Hub setting). */
export const LOCAL_TTS_HUB_PREFIX = "hub:";
/** Organisation assumed for a bare Hub repo name (Rust `kitten::hub::DEFAULT_ORG`). */
export const KITTEN_HUB_DEFAULT_ORG = "KittenML";

/**
 * KittenML Hub repos whose model is also a catalog package: once that package is installed, a `hub:` request runs
 * it (sherpa-onnx). Mirror of Rust `local_tts::models::HUB_CATALOG` (a vitest drift test compares the two).
 */
export const HUB_CATALOG: readonly (readonly [repo: string, catalogId: string])[] = [
  ["KittenML/kitten-tts-mini-0.8", "tts/kitten-mini-en-v0.8"],
  ["KittenML/kitten-tts-micro-0.8", "tts/kitten-micro-en-v0.8"],
  ["KittenML/kitten-tts-nano-0.8", "tts/kitten-nano-en-v0.8"],
  ["KittenML/kitten-tts-nano-0.8-fp32", "tts/kitten-nano-en-v0.8"],
  ["KittenML/kitten-tts-nano-0.8-int8", "tts/kitten-nano-en-v0.8"],
];

/** `load_local_tts` arguments (P2-H added the optional `voice` / `language`). */
export interface LocalTtsLoadArgs {
  model: string;
  threads: number | null;
  voice: string | null;
  language: string | null;
}

/** A voice row with the field P2-H added (optional on the frozen `RustLocalTtsVoice`). */
export type LocalTtsVoiceInfo = RustLocalTtsModelInfo["voices"][number] & { gender?: "female" | "male" | string };

/** Row of `list_local_tts_models` with the fields P2-H added (optional on the frozen `RustLocalTtsModelInfo`). */
export type LocalTtsModelInfo = Omit<RustLocalTtsModelInfo, "voices"> & {
  defaultVoice?: string;
  voices: LocalTtsVoiceInfo[];
};

/**
 * What the running local backend loaded (its info from `load_local_tts`), for the panels: a voice folder or Hub repo
 * lists its voices only once loaded. Kept after stop so the voice list stays while nothing runs.
 */
export const localTtsState = proxy({
  loaded: null as LocalTtsModelInfo | null,
});

/** Fire-and-forget IPC: never throws, synchronously or later. */
function send(command: string, args?: Record<string, unknown>): void {
  try {
    void Promise.resolve(invoke(command, args)).catch(() => {});
  } catch {
    // Without Tauri (web build, tests) there is nothing to tell.
  }
}

/** Threads setting → the request's `threads` (`null` = Rust's automatic choice). */
export function threadsOf(setting: unknown): number | null {
  const n = Number(setting);
  return Number.isInteger(n) && n >= 1 && n <= 4 ? n : null;
}

/** The request `model` of the `localNeural` settings ("" when a folder is chosen but empty). */
export function localNeuralModel(c: TTS_State["localNeural"]): string {
  if (c.model === LOCAL_NEURAL_FOLDER) return c.folder.trim();
  return c.model.trim() || LOCAL_TTS_DEFAULT_MODEL;
}

/** Catalog id of a request `model`, when it is one (`tts/...`). */
export function catalogIdOf(model: string): string | null {
  return model.startsWith("tts/") ? model : null;
}

const REPO_PART = /^[A-Za-z0-9_.-]{1,96}$/;

/** `org/name` for a Hub repo id (a bare name gets {@link KITTEN_HUB_DEFAULT_ORG}), or `null` when invalid. Same rules as Rust `kitten::hub::normalize_repo_id`. */
export function normalizeHubRepo(repo: string): string | null {
  const id = repo.trim();
  const slash = id.indexOf("/");
  const [org, name] = slash < 0 ? [KITTEN_HUB_DEFAULT_ORG, id] : [id.slice(0, slash), id.slice(slash + 1)];
  const ok = (part: string) => REPO_PART.test(part) && !part.startsWith(".");
  return ok(org) && ok(name) ? `${org}/${name}` : null;
}

/** The catalog package of a KittenML Hub repo ({@link HUB_CATALOG}), if any. */
export function hubCatalogId(repo: string): string | null {
  const norm = normalizeHubRepo(repo)?.toLowerCase();
  if (!norm) return null;
  return HUB_CATALOG.find(([r]) => r.toLowerCase() === norm)?.[1] ?? null;
}

/**
 * The catalog entries a request `model` needs for its download check: a catalog id itself, or the catalog package of
 * a mapped Hub repo; empty for folders and other repos.
 */
export function requirementIdsOf(model: string): string[] {
  const id = catalogIdOf(model);
  if (id) return [id];
  if (model.startsWith(LOCAL_TTS_HUB_PREFIX)) {
    const mapped = hubCatalogId(model.slice(LOCAL_TTS_HUB_PREFIX.length));
    return mapped ? [mapped] : [];
  }
  return [];
}

/** Everything but the text of a `speak_local_tts` / `synthesize_local_tts` request. */
export type LocalTtsVoiceSettings = Omit<RustLocalTtsRequest, "text">;

const KANA = /[\p{scx=Hiragana}\p{scx=Katakana}]/u;
const HAN = /\p{Script=Han}/u;
const DEVANAGARI = /\p{Script=Devanagari}/u;
const LETTER = /[\p{L}\p{M}]/u;

/**
 * The Kokoro language a sentence's script shows: `ja` (kana, with or without kanji), `zh` (Han without kana) or `hi`
 * (Devanagari) when that script is most of the sentence; `null` for anything else (Latin: the letters do not tell
 * English from Spanish); `undefined` for a sentence without letters (digits, punctuation, emoji). A kana or Han
 * character counts like a word, so a brand name in Latin letters does not outweigh a Chinese sentence.
 */
export function scriptLanguage(text: string): "ja" | "zh" | "hi" | null | undefined {
  const n = { kana: 0, han: 0, deva: 0, other: 0 };
  let run: keyof typeof n | null = null;
  for (const ch of text) {
    if (!LETTER.test(ch)) {
      run = null;
      continue;
    }
    const script: keyof typeof n = KANA.test(ch) ? "kana" : HAN.test(ch) ? "han" : DEVANAGARI.test(ch) ? "deva" : "other";
    // Kana and Han: one character is about one word. Other scripts: one word per run of letters.
    if (script === "kana" || script === "han" || run !== script) n[script]++;
    run = script;
  }
  const total = n.kana + n.han + n.deva + n.other;
  if (!total) return undefined;
  if (n.kana && (n.kana + n.han) * 2 > total) return "ja";
  if (n.han * 2 > total) return "zh";
  if (n.deva * 2 > total) return "hi";
  return null;
}

/**
 * The request `language` of a sentence spoken with the model's default voice (`voice` ""): what its script shows
 * ({@link scriptLanguage}), a sentence without letters keeping `previous`. Rust then picks that language's default
 * voice and phonemizer (Kokoro: `jf_alpha` with the Japanese frontend for Japanese, a Mandarin voice with the Mandarin
 * rules for Chinese); `null` keeps the model's own default. A language switch reloads Kokoro (about 1 s); a sentence
 * without letters ("123!") keeps the previous language so it does not cause one. Latin-script languages
 * (es / fr / it / pt) need the text event's language: {@link LocalTtsBackend} uses the sentence's `lang` tag
 * (`TtsSinkContext.lang`) when the publisher set one, and this guess otherwise.
 */
export function defaultVoiceLanguage(text: string, previous: string | null): string | null {
  const lang = scriptLanguage(text);
  return lang === undefined ? previous : lang;
}

/** The voice settings of the `localNeural` backend. */
export function localNeuralSettings(c: TTS_State["localNeural"]): LocalTtsVoiceSettings {
  return {
    model: localNeuralModel(c),
    voice: c.voice.trim(),
    speed: clampedOr(c.speed, 1, 0.5, 2),
    language: null,
    threads: threadsOf(c.threads),
  };
}

/**
 * The catalog entries (with requirements) that `ids` still need, from the models service state, or `null` when that
 * is unknown (no ids, or the catalog did not load). Used to refuse Start with a clear "download it first" message
 * instead of Rust's.
 */
export function missingDownloads(
  ids: readonly string[],
  catalog: readonly RustCatalogEntry[],
  states: Readonly<Record<string, RustModelEntryStatus | undefined>>,
): RustCatalogEntry[] | null {
  if (!ids.length || !catalog.length) return null;
  const summary = summarizeRequirement(catalog, states, ids);
  if (summary.unknown.length) return null;
  return summary.missing;
}

/** A localized "download it first" message when `ids` (or the runtime they need) are not installed, else `null`. */
async function downloadRefusal(ids: readonly string[]): Promise<string | null> {
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
  return i18n.t("tts.local_neural_not_downloaded", { title: missing.map(entryTitle).join(", ") });
}

/**
 * Speak `text` once with `settings` into a new sink utterance on `target` (the panels' voice preview). Resolves after
 * the audio is queued in the sink; a stop / skip meanwhile is not an error.
 */
export async function previewLocalVoice(settings: LocalTtsVoiceSettings, text: string, target: RustTtsSinkTarget): Promise<void> {
  const sink = window.ApiServer.sound.sink;
  const id = `local-tts-preview-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  await sink.begin(id, target);
  try {
    const request: RustLocalTtsRequest = { ...settings, text };
    await invoke<RustTtsSpeakReport>("plugin:local-tts|speak_local_tts", { request, utteranceId: id });
    await sink.end(id);
  } catch (error) {
    await sink.cancel(id).catch(() => {});
    if (!isSinkCancelled(error)) throw error;
  }
}

/** Backends that speak through the local-tts engine (their running instance keeps its model loaded). */
export const LOCAL_TTS_BACKENDS: readonly string[] = ["localNeural", "neuralOnnx"];

/** Free the local model after a preview, unless a running local backend is using it. */
export function unloadLocalTtsIfIdle(): void {
  const server = window.ApiServer;
  const running = server?.tts?.serviceState?.status !== ServiceNetworkState.disconnected;
  const localBackend = LOCAL_TTS_BACKENDS.includes(server?.state?.services?.tts?.data?.backend ?? "");
  if (!(running && localBackend)) send("plugin:local-tts|unload_local_tts");
}

/**
 * Base of the backends that speak through the Rust `local-tts` engine (`localNeural`, and `neuralOnnx` mapped onto
 * it): ITTSService v2 `synthesizeToSink`. The engine pushes each sentence straight into the sink utterance the
 * sentence pipeline began (no audio crosses IPC); stop / skip abort the signal, which cancels the Rust job at once
 * (the sink cancel also stops it at the next chunk).
 *
 * - `start` loads and warms up the model (about 1 s for Kokoro) so the first sentence starts quickly. A catalog model
 *   that is not downloaded refuses to start with a message naming it (the panel offers the download); a Hub repo is
 *   checked the same way when its load fails (an earlier download of the repo may still serve it).
 * - `stop` cancels everything and frees the model (Kokoro holds about 0.5-0.8 GB of RAM); a start right after a stop
 *   keeps it loaded (the engine skips an unload with a load waiting behind it).
 * - With the default voice, each sentence carries its text event's language tag, else the language its script shows
 *   ({@link defaultVoiceLanguage}), so Japanese, Chinese and Hindi captions (and tagged es / fr / it / pt text) get a
 *   voice of their language instead of the English default.
 */
export abstract class LocalTtsBackend implements ITTSService {
  #running = false;
  #stopped = false;
  /** Language of the previous default-voice sentence ({@link defaultVoiceLanguage}). */
  #sentenceLanguage: string | null = null;

  constructor(
    protected readonly bindings: ITTSReceiver,
    /** Prefix of error messages. */
    protected readonly label: string,
  ) {}

  /** The voice settings from the persisted state, or an error message to refuse Start with. */
  protected abstract settings(state: TTS_State): LocalTtsVoiceSettings | string;

  get running(): boolean {
    return this.#running;
  }

  async start(state: TTS_State): Promise<void> {
    const settings = this.settings(state);
    if (typeof settings === "string") {
      this.bindings.onStop(settings);
      return;
    }
    const needs = requirementIdsOf(settings.model);
    try {
      if (catalogIdOf(settings.model)) {
        const refusal = await downloadRefusal(needs);
        if (this.#stopped) return;
        if (refusal) {
          this.bindings.onStop(refusal);
          return;
        }
      }
      const args: LocalTtsLoadArgs = {
        model: settings.model,
        threads: settings.threads ?? null,
        voice: settings.voice || null,
        language: settings.language ?? null,
      };
      const info = await invoke<LocalTtsModelInfo>("plugin:local-tts|load_local_tts", { ...args });
      if (this.#stopped) return;
      if (info && typeof info === "object") localTtsState.loaded = { ...info, voices: [...(info.voices ?? [])] };
      this.#running = true;
      this.bindings.onStart();
    } catch (error) {
      if (this.#stopped) return;
      // A Hub repo whose catalog package is missing (and no earlier download): say what to download.
      const refusal = catalogIdOf(settings.model) ? null : await downloadRefusal(needs);
      if (this.#stopped) return;
      this.bindings.onStop(refusal ?? `${this.label}: ${errorMessage(error)}`);
    }
  }

  /** v2: the Rust engine speaks one sentence into the begun sink utterance `sink.utteranceId`. */
  async synthesizeToSink(text: string, sink: TtsSinkContext, signal: AbortSignal): Promise<void> {
    if (!this.#running || signal.aborted || !text.trim()) return;
    const settings = this.settings(window.ApiServer.state.services.tts.data);
    if (typeof settings === "string") throw new Error(settings);
    const request: RustLocalTtsRequest = { ...settings, text };
    if (!settings.voice && !settings.language) {
      // The text event's language tag wins over the script guess (Latin-script es / fr / it / pt need it).
      this.#sentenceLanguage = sink.lang || defaultVoiceLanguage(text, this.#sentenceLanguage);
      request.language = this.#sentenceLanguage;
    }
    const onAbort = () => send("plugin:local-tts|cancel_local_tts", { utteranceId: sink.utteranceId });
    signal.addEventListener("abort", onAbort, { once: true });
    try {
      await invoke<RustTtsSpeakReport>("plugin:local-tts|speak_local_tts", { request, utteranceId: sink.utteranceId });
    } catch (error) {
      // Stop / skip cancel the utterance; Rust then answers "cancelled".
      if (signal.aborted || isSinkCancelled(error)) return;
      throw new Error(`${this.label}: ${errorMessage(error)}`);
    } finally {
      signal.removeEventListener("abort", onAbort);
    }
  }

  /**
   * Legacy whole-line path. Unused: a backend with `synthesizeToSink` always speaks through the sentence pipeline
   * (Service_TTS creates it for every non-self-playing backend).
   */
  play(_value: string): void {}

  stop(): void {
    this.#halt();
    this.bindings.onStop();
  }

  dispose(): void {
    this.#halt();
  }

  #halt() {
    const wasActive = !this.#stopped;
    this.#stopped = true;
    this.#running = false;
    if (!wasActive) return;
    // Cancel whatever the engine still has queued, then free the model (sent in this order, without waiting, so a
    // start right after this stop queues its load behind the unload and keeps the model).
    send("plugin:local-tts|cancel_local_tts", { utteranceId: null });
    send("plugin:local-tts|unload_local_tts");
  }
}

/**
 * In-process Kokoro / Kokoro v1.1-zh / Piper / Kitten through the Rust `local-tts` plugin (backend `localNeural`):
 * natural voices on the processor, no server to run. Models come from the model store (the panel's
 * `<ModelRequirement>` asks before downloading) or a local voice folder.
 */
export class TTS_LocalNeuralService extends LocalTtsBackend {
  constructor(bindings: ITTSReceiver) {
    super(bindings, "Local voice");
  }

  protected settings(state: TTS_State): LocalTtsVoiceSettings | string {
    const c = state.localNeural;
    if (c.model === LOCAL_NEURAL_FOLDER && !c.folder.trim()) return i18n.t("tts.local_neural_folder_missing");
    return localNeuralSettings(c);
  }
}

// -------------------------------------------------------------------------------------------------
// Panel helpers (pure)
// -------------------------------------------------------------------------------------------------

/** A short sample line per language (base code) for the voice preview; English for the rest. */
export const PREVIEW_TEXT: Readonly<Record<string, string>> = {
  en: "Hello! This is how my voice sounds.",
  ja: "こんにちは。これが私の声です。",
  zh: "你好！这就是我的声音。",
  es: "¡Hola! Así suena mi voz.",
  fr: "Bonjour ! Voici ma voix.",
  hi: "नमस्ते! मेरी आवाज़ ऐसी सुनाई देती है।",
  it: "Ciao! Questa è la mia voce.",
  pt: "Olá! Esta é a minha voz.",
  ru: "Привет! Так звучит мой голос.",
  uk: "Привіт! Так звучить мій голос.",
  ar: "مرحبا! هكذا يبدو صوتي.",
  de: "Hallo! So klingt meine Stimme.",
  pl: "Cześć! Tak brzmi mój głos.",
  nl: "Hallo! Zo klinkt mijn stem.",
  tr: "Merhaba! Sesim böyle duyuluyor.",
  vi: "Xin chào! Đây là giọng nói của tôi.",
  id: "Halo! Beginilah suara saya.",
  ur: "ہیلو! میری آواز ایسی ہے۔",
  sv: "Hej! Så här låter min röst.",
  cs: "Ahoj! Takhle zní můj hlas.",
  fi: "Hei! Tältä ääneni kuulostaa.",
  el: "Γεια σας! Έτσι ακούγεται η φωνή μου.",
  hu: "Szia! Így hangzik a hangom.",
  ro: "Salut! Așa sună vocea mea.",
  fa: "سلام! صدای من این‌طور است.",
  ca: "Hola! Així sona la meva veu.",
  da: "Hej! Sådan lyder min stemme.",
  no: "Hei! Slik høres stemmen min ut.",
  sk: "Ahoj! Takto znie môj hlas.",
  sl: "Živjo! Tako zveni moj glas.",
  sr: "Здраво! Овако звучи мој глас.",
};

/** The preview line for a voice language (`pt-BR` → Portuguese), English when there is none. */
export function previewText(language: string | null | undefined): string {
  const base = (language ?? "").trim().toLowerCase().split(/[-_]/)[0];
  return PREVIEW_TEXT[base] ?? PREVIEW_TEXT.en;
}

/** Model families in picker order. */
export const LOCAL_TTS_FAMILIES = ["kokoro", "kitten", "piper"] as const;

/** Catalog models grouped by family (Kokoro, Kitten, Piper, then anything else), Piper voices sorted by title. */
export function groupLocalModels(models: readonly LocalTtsModelInfo[]): { family: string; models: LocalTtsModelInfo[] }[] {
  const families: string[] = [...LOCAL_TTS_FAMILIES];
  for (const m of models) if (!families.includes(m.family)) families.push(m.family);
  return families
    .map((family) => {
      const list = models.filter((m) => m.family === family);
      if (family === "piper") list.sort((a, b) => a.title.localeCompare(b.title));
      return { family, models: list };
    })
    .filter((g) => g.models.length > 0);
}

/** Voices grouped by language, in the order the languages first appear (Kokoro lists voices by language). */
export function groupVoicesByLanguage<V extends { language: string | null }>(voices: readonly V[]): { language: string | null; voices: V[] }[] {
  const groups: { language: string | null; voices: V[] }[] = [];
  for (const v of voices) {
    const language = v.language ?? null;
    let g = groups.find((x) => x.language === language);
    if (!g) {
      g = { language, voices: [] };
      groups.push(g);
    }
    g.voices.push(v);
  }
  return groups;
}

/**
 * The voices to offer for request `model`: what the running backend loaded when it is that model (folders and Hub
 * repos know their voices only once loaded), else the model list's row (catalog voices, Piper speakers once
 * installed). `null` when neither knows (a folder not loaded yet: the voice is typed).
 */
export function voicesFor(
  model: string,
  list: readonly LocalTtsModelInfo[],
  loaded: LocalTtsModelInfo | null,
): { voices: LocalTtsVoiceInfo[]; defaultVoice: string } | null {
  const pick = (m: LocalTtsModelInfo) => (m.voices.length ? { voices: m.voices, defaultVoice: m.defaultVoice ?? "" } : null);
  const row = list.find((m) => m.id === model);
  if (row) return pick(row);
  if (loaded && loaded.id === model) return pick(loaded);
  return null;
}
