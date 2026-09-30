import type {
  RustSpeechEngineKind,
  RustSpeechEngineSpec,
  RustSpeechRoutePlan,
  RustSpeechRouteRequest,
  RustSpeechTier,
} from "@/types";
import routingDocument from "../../../../src-tauri/model-catalog/stt-routing.json";

/**
 * TypeScript mirror of the Rust speech router (`src-tauri/src/services/speech/router.rs`, P2-C) and of its language
 * normalization (`speech/lang.rs`). Pure: no `window`, no Tauri.
 *
 * The app asks Rust for the authoritative plan (`plugin:speech|preview_route`, which knows the GPU, the VRAM the
 * resource governor grants, the voice changer and what is installed). This mirror plans on an explicit
 * {@link RouteContext} with the same table (`model-catalog/stt-routing.json`, imported here), so that:
 * - `__tests__/router.test.ts` runs the shared fixture (`src-tauri/test-fixtures/routing.json`) against it, which
 *   keeps the two implementations in step (the Rust test runs the same cases);
 * - the UI can list the languages the router handles and explain plans without a round trip.
 *
 * Differences from Rust, by design: Rust additionally refuses a sherpa model for a language its sherpa catalog entry
 * lacks (`speech_rt::sherpa::catalog`). The Rust test `table_models_match_the_catalogs` guarantees the table's
 * `languages` of every sherpa model are a subset of its catalog languages (and empty only when the catalog's are), so
 * with the shipped table the table rule already implies the catalog rule. Logging is left to Rust.
 */

// ---------------------------------------------------------------------------------------------------------------
// Languages (mirror of speech::lang)

/** "Detect the language". */
export const AUTO_LANGUAGE = "auto";

/** True for `auto` and empty input. */
export function isAutoLanguage(code: string | null | undefined): boolean {
  const c = String(code ?? "").trim();
  return c === "" || c.toLowerCase() === AUTO_LANGUAGE;
}

/** Deprecated or three-letter codes → the code Sigil uses (`lang.rs` `ALIASES`). */
const LANGUAGE_ALIASES: Readonly<Record<string, string>> = {
  iw: "he",
  in: "id",
  ji: "yi",
  jw: "jv",
  mo: "ro",
  eng: "en",
  jpn: "ja",
  zho: "zh",
  chi: "zh",
  cmn: "zh",
  kor: "ko",
  rus: "ru",
  deu: "de",
  ger: "de",
  fra: "fr",
  fre: "fr",
  spa: "es",
  por: "pt",
  ita: "it",
  nld: "nl",
  dut: "nl",
  pol: "pl",
  ukr: "uk",
  tur: "tr",
  ara: "ar",
  hin: "hi",
  vie: "vi",
  tha: "th",
  ind: "id",
  msa: "ms",
  may: "ms",
  tgl: "fil",
};

/**
 * Lower-case base subtag of a BCP-47 / locale tag, as Sigil's engines, router and events use it (`en-US` → `en`,
 * `zh-Hant-TW` → `zh`, `pt_BR` → `pt`, `zh-yue` → `yue`, `iw` → `he`, `jpn` → `ja`); `null` for `auto`, empty or
 * malformed input (including `x-` / `i-` private-use tags).
 */
export function normalizeLanguage(code: string | null | undefined): string | null {
  if (isAutoLanguage(code)) return null;
  const tag = String(code).trim().replace(/_/g, "-").toLowerCase();
  const parts = tag.split("-");
  const base = parts[0] ?? "";
  if (!/^[a-z]{2,3}$/.test(base)) return null;
  // Cantonese: `yue`, or a `zh-*` tag with a `yue` subtag; every other `zh-*` tag (including `zh-HK`) stays `zh`.
  if ((base === "zh" || base === "zho" || base === "chi") && parts.slice(1).includes("yue")) return "yue";
  return LANGUAGE_ALIASES[base] ?? base;
}

/** Whether two tags name the same language (normalized; `auto` matches nothing). */
export function sameLanguage(a: string | null | undefined, b: string | null | undefined): boolean {
  const x = normalizeLanguage(a);
  return x !== null && x === normalizeLanguage(b);
}

// ---------------------------------------------------------------------------------------------------------------
// Context and table

/** What the router knows about the PC (Rust `speech::router::RouteContext`; camelCase like the shared fixture). */
export interface RouteContext {
  /** A GPU-tier engine may run here (a supported GPU and an allowed transcribe GPU variant). */
  gpu: boolean;
  /** Best allowed GPU runtime variant (`cuda13`, `vulkan`), if any. */
  gpuVariant: string | null;
  /** VRAM the resource governor would grant to the STT final now (MB). */
  vramAvailableMb: number;
  /** The voice changer is running. */
  vcRunning: boolean;
  /** 0 = unknown (enough for two passes). */
  physicalCores: number;
  /** Installed catalog ids (models and runtime packs). */
  installed: string[];
  /** Dedicated VRAM of the primary GPU (MB; 0 = unknown). */
  vramTotalMb: number;
  /** Whether the running voice changer uses a GPU EP (`null`: not running, or not known yet). */
  vcGpu: boolean | null;
  /** Catalog ids that cannot be installed here (never planned). */
  unavailable: string[];
  /** Why `gpu` is false (the resource governor's reason code). */
  gpuReason: string | null;
}

const EMPTY_CONTEXT: RouteContext = Object.freeze({
  gpu: false,
  gpuVariant: null,
  vramAvailableMb: 0,
  vcRunning: false,
  physicalCores: 0,
  installed: [],
  vramTotalMb: 0,
  vcGpu: null,
  unavailable: [],
  gpuReason: null,
}) as RouteContext;

/** A full context from a partial one (missing fields: false / 0 / empty / null, like Rust's `#[serde(default)]`). */
export function routeContext(partial: Partial<RouteContext> | null | undefined = {}): RouteContext {
  const p = partial ?? {};
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) && v > 0 ? Math.floor(v) : 0);
  const list = (v: unknown) => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []);
  const str = (v: unknown) => (typeof v === "string" && v !== "" ? v : null);
  return {
    ...EMPTY_CONTEXT,
    gpu: p.gpu === true,
    gpuVariant: str(p.gpuVariant),
    vramAvailableMb: num(p.vramAvailableMb),
    vcRunning: p.vcRunning === true,
    physicalCores: num(p.physicalCores),
    installed: list(p.installed),
    vramTotalMb: num(p.vramTotalMb),
    vcGpu: typeof p.vcGpu === "boolean" ? p.vcGpu : null,
    unavailable: list(p.unavailable),
    gpuReason: str(p.gpuReason),
  };
}

/** One model the table may route to (`stt-routing.json` → `routing.models`). */
export interface RoutingModel {
  engine: RustSpeechEngineKind;
  /** Runs on the GPU (transcribe.cpp; its runtimes come from `gpuRuntimes`). */
  gpu: boolean;
  /** Runtime packs a CPU model needs. */
  runtimes: string[];
  /** Languages it may be routed for (normalized); empty = any. */
  languages: string[];
  needsLanguage: boolean;
  detectsLanguage: boolean;
  /** Never in `auto` mode. */
  requiresLanguage: boolean;
  neverLanguages: string[];
  /** An explicit language goes to the engine as `options.language`. */
  languageOption: boolean;
  vramMb: number;
  minTotalVramMb: number;
  minCores: number;
}

/** One language row. */
export interface RouteRow {
  name: string;
  /** Normalized codes, or `auto` / `*`. */
  languages: string[];
  streaming: string[];
  /** The streaming model's own finals are caption quality (a CPU single pass uses it alone). */
  streamFinal: boolean;
  cpuFinal: string[];
  gpuFinal: string[];
}

/** The `routing` object of `stt-routing.json`. */
export interface RoutingTable {
  version: number;
  /** Spoken-language-ID engine for `auto` with a final that cannot detect the language. */
  languageId: string | null;
  cpuTwoPassMinCores: number;
  vcShareGpuMinTotalMb: number;
  /** GPU variant → runtime packs a GPU model needs on it. */
  gpuRuntimes: Record<string, string[]>;
  models: Record<string, RoutingModel>;
  routes: RouteRow[];
}

const ENGINE_KINDS: readonly RustSpeechEngineKind[] = ["whisper", "http", "sherpa-online", "sherpa-offline", "transcribe"];

function asRecord(v: unknown, what: string): Record<string, unknown> {
  if (v === null || typeof v !== "object" || Array.isArray(v)) throw new Error(`stt-routing.json: ${what} must be an object`);
  return v as Record<string, unknown>;
}

function asStrings(v: unknown, what: string): string[] {
  if (v === undefined) return [];
  if (!Array.isArray(v) || v.some((x) => typeof x !== "string")) throw new Error(`stt-routing.json: ${what} must be a list of strings`);
  return [...(v as string[])];
}

function asCount(v: unknown, what: string, required = false): number {
  if (v === undefined && !required) return 0;
  if (typeof v !== "number" || !Number.isInteger(v) || v < 0) throw new Error(`stt-routing.json: ${what} must be a whole number`);
  return v;
}

function asFlag(v: unknown, what: string): boolean {
  if (v === undefined) return false;
  if (typeof v !== "boolean") throw new Error(`stt-routing.json: ${what} must be true or false`);
  return v;
}

/**
 * Parse and check a whole `stt-routing.json` document (the checks of Rust `RoutingTable::validate`: version 1, every
 * referenced model defined with a kind that fits its role, `auto` and `*` rows, normalized row languages, a CPU
 * language-ID model that detects languages). Throws with a message on a bad document.
 */
export function parseRoutingTable(doc: unknown): RoutingTable {
  const routing = asRecord(asRecord(doc, "the document").routing, "'routing'");
  const version = asCount(routing.version, "version", true);
  if (version !== 1) throw new Error(`stt-routing.json: unsupported routing version ${version}`);
  const models: Record<string, RoutingModel> = {};
  for (const [id, raw] of Object.entries(asRecord(routing.models, "models"))) {
    const m = asRecord(raw, `model ${id}`);
    const engine = m.engine as RustSpeechEngineKind;
    if (!ENGINE_KINDS.includes(engine)) throw new Error(`stt-routing.json: model ${id} has an unknown engine`);
    models[id] = {
      engine,
      gpu: asFlag(m.gpu, `${id}.gpu`),
      runtimes: asStrings(m.runtimes, `${id}.runtimes`),
      languages: asStrings(m.languages, `${id}.languages`),
      needsLanguage: asFlag(m.needsLanguage, `${id}.needsLanguage`),
      detectsLanguage: asFlag(m.detectsLanguage, `${id}.detectsLanguage`),
      requiresLanguage: asFlag(m.requiresLanguage, `${id}.requiresLanguage`),
      neverLanguages: asStrings(m.neverLanguages, `${id}.neverLanguages`),
      languageOption: asFlag(m.languageOption, `${id}.languageOption`),
      vramMb: asCount(m.vramMb, `${id}.vramMb`),
      minTotalVramMb: asCount(m.minTotalVramMb, `${id}.minTotalVramMb`),
      minCores: asCount(m.minCores, `${id}.minCores`),
    };
  }
  const gpuRuntimes: Record<string, string[]> = {};
  for (const [variant, packs] of Object.entries(asRecord(routing.gpuRuntimes, "gpuRuntimes")))
    gpuRuntimes[variant] = asStrings(packs, `gpuRuntimes.${variant}`);
  if (!Array.isArray(routing.routes)) throw new Error("stt-routing.json: routes must be a list");
  const routes: RouteRow[] = routing.routes.map((raw, i) => {
    const r = asRecord(raw, `route ${i}`);
    if (typeof r.name !== "string") throw new Error(`stt-routing.json: route ${i} has no name`);
    return {
      name: r.name,
      languages: asStrings(r.languages, `${r.name}.languages`),
      streaming: asStrings(r.streaming, `${r.name}.streaming`),
      streamFinal: asFlag(r.streamFinal, `${r.name}.streamFinal`),
      cpuFinal: asStrings(r.cpuFinal, `${r.name}.cpuFinal`),
      gpuFinal: asStrings(r.gpuFinal, `${r.name}.gpuFinal`),
    };
  });
  const languageId = routing.languageId === undefined || routing.languageId === null ? null : String(routing.languageId);
  const table: RoutingTable = {
    version,
    languageId,
    cpuTwoPassMinCores: asCount(routing.cpuTwoPassMinCores, "cpuTwoPassMinCores", true),
    vcShareGpuMinTotalMb: asCount(routing.vcShareGpuMinTotalMb, "vcShareGpuMinTotalMb", true),
    gpuRuntimes,
    models,
    routes,
  };
  validateTable(table);
  return table;
}

function validateTable(table: RoutingTable): void {
  const model = (id: string, row: string) => {
    const m = table.models[id];
    if (!m) throw new Error(`stt-routing.json: row '${row}' names unknown model ${id}`);
    return m;
  };
  let hasAuto = false;
  let hasOther = false;
  for (const row of table.routes) {
    for (const l of row.languages) {
      if (l === AUTO_LANGUAGE) hasAuto = true;
      else if (l === "*") hasOther = true;
      else if (normalizeLanguage(l) !== l)
        throw new Error(`stt-routing.json: row '${row.name}' has a non-normalized language '${l}'`);
    }
    for (const id of row.streaming) {
      const m = model(id, row.name);
      if (m.engine !== "sherpa-online" || m.gpu)
        throw new Error(`stt-routing.json: ${id} in '${row.name}' is not a CPU streaming model`);
    }
    for (const id of row.cpuFinal)
      if (model(id, row.name).gpu) throw new Error(`stt-routing.json: GPU model ${id} in the CPU finals of '${row.name}'`);
    for (const id of row.gpuFinal)
      if (!model(id, row.name).gpu) throw new Error(`stt-routing.json: CPU model ${id} in the GPU finals of '${row.name}'`);
  }
  if (!hasAuto || !hasOther) throw new Error("stt-routing.json: the 'auto' and '*' rows are required");
  if (table.languageId !== null) {
    const m = table.models[table.languageId];
    if (!m) throw new Error(`stt-routing.json: unknown languageId model ${table.languageId}`);
    if (!m.detectsLanguage || m.gpu)
      throw new Error(`stt-routing.json: languageId ${table.languageId} must be a CPU model that detects languages`);
  }
}

/** The shipped routing table (the same file Rust embeds); `null` when it fails to parse (a build bug). */
export const ROUTING_TABLE: RoutingTable | null = (() => {
  try {
    return parseRoutingTable(routingDocument);
  } catch (e) {
    console.error("[STT] routing table:", e);
    return null;
  }
})();

/** Every catalog id the table can put in a plan (models and runtime packs), sorted. */
export function referencedIds(table: RoutingTable): string[] {
  const ids = new Set<string>(Object.keys(table.models));
  for (const m of Object.values(table.models)) m.runtimes.forEach((r) => ids.add(r));
  for (const packs of Object.values(table.gpuRuntimes)) packs.forEach((p) => ids.add(p));
  if (table.languageId) ids.add(table.languageId);
  return [...ids].sort();
}

/**
 * Languages the table routes explicitly (normalized, sorted; without `auto` / `*`). Any other well-formed language
 * is still routed (the `*` row: a general multilingual model).
 */
export function routedLanguages(table: RoutingTable | null = ROUTING_TABLE): string[] {
  if (!table) return [];
  const out = new Set<string>();
  for (const row of table.routes) for (const l of row.languages) if (l !== AUTO_LANGUAGE && l !== "*") out.add(l);
  return [...out].sort();
}

// ---------------------------------------------------------------------------------------------------------------
// Planning (mirror of speech::router::plan_table)

/** Machine codes of `RoutePlan.notes` (Rust `speech::router::notes`). UI text: `stt.route_note_<code>`. */
export const ROUTE_NOTES = {
  autoLanguage: "auto_language",
  languageId: "language_id",
  genericLanguageModel: "generic_language_model",
  finalsOnly: "finals_only",
  singlePass: "single_pass",
  fewCoresSinglePass: "few_cores_single_pass",
  vcCpuSinglePass: "vc_running_single_pass",
  vcUnknownSinglePass: "vc_unknown_single_pass",
  noCpuFallback: "no_cpu_fallback",
  gpuUnavailable: "gpu_unavailable",
  gpuRuntimeUnavailable: "gpu_runtime_unavailable",
  gpuDeniedVcRunning: "gpu_denied_vc_running",
  vcSharesGpu: "vc_shares_gpu",
  gpuFull: "gpu_full",
  noGpuModelForLanguage: "no_gpu_model_for_language",
  nemotronNeedsLanguage: "nemotron_needs_language",
  neverRoute: "never_route",
  noEngine: "no_engine",
  routingTableInvalid: "routing_table_invalid",
} as const;

/**
 * Resource-governor GPU reason codes a plan may carry for a PC without a usable GPU (UI text:
 * `models.gpu.rec_reason_<code>`).
 */
export const GOVERNOR_REASONS: readonly string[] = ["no_gpu", "gpu_unknown", "integrated_gpu", "small_gpu", "no_gpu_runtime"];

/** Never-route rules enforced in code whatever the table says: model id prefix → languages it must never transcribe. */
export const NEVER_ROUTES: ReadonlyArray<readonly [prefix: string, languages: readonly string[]]> = [
  ["stt/cohere-", ["ja", "zh", "ko", "ru", "yue"]],
];
/** Model id prefixes that never run in `auto` mode (no language ID of their own). */
export const NEVER_AUTO: readonly string[] = ["stt/cohere-"];
/** Model id prefixes that must always be given a language (Nemotron 3.5 streaming). */
export const NEEDS_LANGUAGE: readonly string[] = ["stt/nemotron-"];

/**
 * Option of an `auto` plan's streaming spec (Rust `speech::router::STREAM_LANGUAGES_OPTION`): the detected languages
 * (normalized, sorted) the pipeline may open the stream with; every other utterance goes to the final alone.
 */
export const STREAM_LANGUAGES_OPTION = "streamLanguages";

/** A chosen model. */
interface ModelPick {
  id: string;
  row: RoutingModel;
}

function tablePermits(m: RoutingModel, language: string | null): boolean {
  if (language === null) return !m.requiresLanguage;
  return (m.languages.length === 0 || m.languages.includes(language)) && !m.neverLanguages.includes(language);
}

function codePermits(id: string, language: string | null): boolean {
  if (language === null) return !NEVER_AUTO.some((p) => id.startsWith(p));
  return !NEVER_ROUTES.some(([prefix, langs]) => id.startsWith(prefix) && langs.includes(language));
}

function pickNeedsLanguage(p: ModelPick): boolean {
  return p.row.needsLanguage || NEEDS_LANGUAGE.some((prefix) => p.id.startsWith(prefix));
}

class Planner {
  readonly notes: string[] = [];
  private readonly unavailable: ReadonlySet<string>;

  constructor(
    readonly table: RoutingTable,
    readonly ctx: RouteContext,
    /** Normalized language; `null` = auto. */
    readonly language: string | null,
  ) {
    this.unavailable = new Set(ctx.unavailable);
  }

  note(code: string): void {
    if (!this.notes.includes(code)) this.notes.push(code);
  }

  available(id: string): boolean {
    return !this.unavailable.has(id);
  }

  usable(id: string): ModelPick | null {
    const row = this.table.models[id];
    if (!row) return null;
    if (!tablePermits(row, this.language)) return null;
    if (!codePermits(id, this.language)) {
      this.note(ROUTE_NOTES.neverRoute);
      return null;
    }
    if (!this.available(id) || !row.runtimes.every((r) => this.available(r))) return null;
    if (row.minCores > 0 && this.ctx.physicalCores > 0 && this.ctx.physicalCores < row.minCores) return null;
    return { id, row };
  }

  firstUsable(ids: readonly string[], gpu: boolean): ModelPick | null {
    for (const id of ids) {
      const p = this.usable(id);
      if (p && p.row.gpu === gpu) return p;
    }
    return null;
  }

  vcOnGpu(): boolean {
    return this.ctx.vcRunning && this.ctx.vcGpu !== false;
  }

  /** The voice changer runs and may use the CPU (unknown counts as yes, as in {@link vcOnGpu}). */
  vcOnCpu(): boolean {
    return this.ctx.vcRunning && this.ctx.vcGpu !== true;
  }

  /** The CPU may run two speech engines: enough physical cores (0 = unknown) and no voice changer that may use it. */
  cpuRunsTwoEngines(): boolean {
    const cores = this.ctx.physicalCores;
    return (cores === 0 || cores >= this.table.cpuTwoPassMinCores) && !this.vcOnCpu();
  }

  gpuVariant(tier: RustSpeechTier): string | null {
    if (tier === "cpu") return null;
    const variant = this.ctx.gpu ? this.ctx.gpuVariant : null;
    if (!variant) {
      if (tier === "gpu") this.note(ROUTE_NOTES.gpuUnavailable);
      if (this.ctx.gpuReason) this.note(this.ctx.gpuReason);
      return null;
    }
    const packs = this.table.gpuRuntimes[variant];
    if (!packs || !packs.every((id) => this.available(id))) {
      this.note(ROUTE_NOTES.gpuRuntimeUnavailable);
      return null;
    }
    if (tier === "auto" && this.vcOnGpu() && this.ctx.vramTotalMb < this.table.vcShareGpuMinTotalMb) {
      this.note(ROUTE_NOTES.gpuDeniedVcRunning);
      return null;
    }
    return variant;
  }

  gpuFinal(ids: readonly string[]): ModelPick | null {
    const candidates: ModelPick[] = [];
    for (const id of ids) {
      const p = this.usable(id);
      if (p && p.row.gpu) candidates.push(p);
    }
    if (candidates.length === 0) {
      this.note(ROUTE_NOTES.noGpuModelForLanguage);
      return null;
    }
    const fit = candidates.find(
      (p) =>
        p.row.vramMb <= this.ctx.vramAvailableMb &&
        (p.row.minTotalVramMb === 0 || this.ctx.vramTotalMb >= p.row.minTotalVramMb),
    );
    if (!fit) this.note(ROUTE_NOTES.gpuFull);
    return fit ?? null;
  }

  spec(pick: ModelPick, variant: string | null): RustSpeechEngineSpec {
    return {
      engine: pick.row.engine,
      model: pick.id,
      threads: null,
      variant: pick.row.gpu ? variant : null,
      options:
        this.language !== null && (pick.row.languageOption || pickNeedsLanguage(pick)) ? { language: this.language } : null,
    };
  }

  idsOf(pick: ModelPick, variant: string | null, out: string[]): void {
    const runtimes = pick.row.gpu ? (variant ? this.table.gpuRuntimes[variant] ?? [] : []) : pick.row.runtimes;
    for (const id of [pick.id, ...runtimes]) if (!out.includes(id)) out.push(id);
  }
}

function rowFor(table: RoutingTable, language: string | null): RouteRow | null {
  const key = language ?? AUTO_LANGUAGE;
  return (
    table.routes.find((r) => r.languages.includes(key)) ??
    (language !== null ? table.routes.find((r) => r.languages.includes("*")) ?? null : null)
  );
}

/**
 * `auto` mode: the detected languages the streaming model may be opened with (Rust `auto_stream_languages`): the
 * model's languages whose own row streams with it (never zh / yue). The candidates are the model's table languages,
 * else the languages of every explicit row; `null` (no limit) for a model without a language list that the `*` row
 * streams with. (Rust also falls back to the sherpa catalog languages, which the shipped table never needs.)
 */
function autoStreamLanguages(table: RoutingTable, pick: ModelPick): string[] | null {
  const streams = (row: RouteRow | null) => !!row && row.streaming.includes(pick.id);
  let candidates: string[];
  if (pick.row.languages.length > 0) candidates = [...pick.row.languages];
  else if (table.routes.some((r) => r.languages.includes("*") && streams(r))) return null;
  else candidates = table.routes.flatMap((r) => r.languages).filter((l) => l !== AUTO_LANGUAGE && l !== "*");
  const allowed = new Set(
    candidates.filter((l) => tablePermits(pick.row, l) && codePermits(pick.id, l) && streams(rowFor(table, l))),
  );
  return [...allowed].sort();
}

function emptyPlan(language: string, notes: string[]): RustSpeechRoutePlan {
  return {
    language,
    tier: "cpu",
    streaming: null,
    final: null,
    fallback: null,
    languageId: null,
    models: [],
    missing: [],
    vramMb: 0,
    notes,
  };
}

/** The request with Rust's defaults applied (`language` "auto", `tier` "auto", `twoPass` true). */
export function routeRequest(request: RustSpeechRouteRequest | null | undefined): Required<RustSpeechRouteRequest> {
  const tier = request?.tier === "cpu" || request?.tier === "gpu" ? request.tier : "auto";
  return {
    language: typeof request?.language === "string" ? request.language : AUTO_LANGUAGE,
    tier,
    twoPass: request?.twoPass !== false,
  };
}

/** An auto streaming spec carries its language limit ({@link STREAM_LANGUAGES_OPTION}). */
function streamingSpec(spec: RustSpeechEngineSpec, languages: string[] | null): RustSpeechEngineSpec {
  if (languages === null) return spec;
  return { ...spec, options: { ...(spec.options ?? {}), [STREAM_LANGUAGES_OPTION]: languages } };
}

/**
 * The routing decision for `request` on the PC described by `context` (pure and deterministic; mirror of Rust
 * `plan_table` / `plan_with`). With no table (`null`) the plan is empty with the note `routing_table_invalid`.
 */
export function planRoute(
  request: RustSpeechRouteRequest | null | undefined,
  context: Partial<RouteContext> | null | undefined,
  table: RoutingTable | null = ROUTING_TABLE,
): RustSpeechRoutePlan {
  const req = routeRequest(request);
  const normalized = normalizeLanguage(req.language);
  const language = normalized ?? AUTO_LANGUAGE;
  if (!table) return emptyPlan(language, [ROUTE_NOTES.routingTableInvalid]);
  const ctx = routeContext(context);
  const p = new Planner(table, ctx, normalized);
  const auto = normalized === null;
  if (auto) p.note(ROUTE_NOTES.autoLanguage);
  const row = rowFor(table, normalized);
  if (!row) {
    p.note(ROUTE_NOTES.noEngine);
    return emptyPlan(language, p.notes);
  }
  if (row.languages.includes("*")) p.note(ROUTE_NOTES.genericLanguageModel);

  // 1. Candidates. In auto mode the stream runs only for detected languages whose own row streams with it.
  let streamingPick = p.firstUsable(row.streaming, false);
  let autoLanguages: string[] | null = null;
  if (auto && streamingPick) {
    autoLanguages = autoStreamLanguages(table, streamingPick);
    if (autoLanguages !== null && autoLanguages.length === 0) {
      p.note(ROUTE_NOTES.nemotronNeedsLanguage);
      streamingPick = null;
      autoLanguages = null;
    }
  }
  const cpuFinal = p.firstUsable(row.cpuFinal, false);
  const gpuVariant = p.gpuVariant(req.tier);
  const gpuFinal = gpuVariant !== null ? p.gpuFinal(row.gpuFinal) : null;
  const tier: RustSpeechTier = gpuFinal ? "gpu" : "cpu";
  const variant = gpuFinal ? gpuVariant : null;
  if (tier === "gpu" && p.vcOnGpu()) p.note(ROUTE_NOTES.vcSharesGpu);
  const bestFinal = gpuFinal ?? cpuFinal;

  // 2. Passes.
  const coresOk = ctx.physicalCores === 0 || ctx.physicalCores >= table.cpuTwoPassMinCores;
  const cpuTwoEngines = p.cpuRunsTwoEngines();
  let streaming: ModelPick | null = streamingPick;
  let finalPick: ModelPick | null = bestFinal;
  if (streamingPick && bestFinal) {
    if (!(req.twoPass && (tier === "gpu" || cpuTwoEngines))) {
      p.note(
        !req.twoPass
          ? ROUTE_NOTES.singlePass
          : !coresOk
            ? ROUTE_NOTES.fewCoresSinglePass
            : ctx.vcGpu === false
              ? ROUTE_NOTES.vcCpuSinglePass
              : ROUTE_NOTES.vcUnknownSinglePass,
      );
      // The stream alone only where its own finals are caption quality, and never a stream that would have to
      // guess its language.
      const streamAlone = tier === "cpu" && row.streamFinal && !(auto && pickNeedsLanguage(streamingPick));
      if (streamAlone) finalPick = null;
      else streaming = null;
    }
  }

  // 3. Language: auto mode uses spoken language ID; a model that needs a language always gets one.
  let languageId: ModelPick | null = null;
  if (auto) {
    if (finalPick && !finalPick.row.detectsLanguage) {
      languageId = table.languageId ? p.usable(table.languageId) : null;
      if (languageId) p.note(ROUTE_NOTES.languageId);
    }
    if (streaming && pickNeedsLanguage(streaming)) {
      const detected = (finalPick?.row.detectsLanguage ?? false) || languageId !== null;
      if (!finalPick || !detected) {
        p.note(ROUTE_NOTES.nemotronNeedsLanguage);
        streaming = null;
      }
    }
  }
  if (!streaming && finalPick) p.note(ROUTE_NOTES.finalsOnly);
  else if (!streaming && !finalPick) p.note(ROUTE_NOTES.noEngine);

  // 4. The CPU final behind a GPU final: next to a CPU stream only where the CPU may run two engines (else the
  //    first-pass finals stand after a GPU failure); a finals-only GPU plan always keeps it.
  let fallback = finalPick && cpuFinal && finalPick.row.gpu && cpuFinal.id !== finalPick.id ? cpuFinal : null;
  if (fallback && streaming && !cpuTwoEngines) {
    p.note(ROUTE_NOTES.noCpuFallback);
    fallback = null;
  }

  const models: string[] = [];
  const parts: [ModelPick | null, string | null][] = [
    [streaming, null],
    [finalPick, variant],
    [fallback, null],
    [languageId, null],
  ];
  for (const [pick, v] of parts) if (pick) p.idsOf(pick, v, models);
  const installed = new Set(ctx.installed);
  return {
    language,
    tier,
    streaming: streaming ? streamingSpec(p.spec(streaming, null), autoLanguages) : null,
    final: finalPick ? p.spec(finalPick, variant) : null,
    fallback: fallback ? p.spec(fallback, null) : null,
    languageId: languageId ? p.spec(languageId, null) : null,
    models,
    missing: models.filter((id) => !installed.has(id)),
    vramMb: finalPick?.row.gpu ? finalPick.row.vramMb : 0,
    notes: p.notes,
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Plan helpers for the driver and the UI

/** Rust `speech::smart_turn::MODEL_ID`: the Smart Turn model a pipeline with `smartTurn` loads. */
export const SMART_TURN_MODEL_ID = "vad/smart-turn-v3.2";

/** The plan decodes twice: streaming partials, then the final re-decodes each closed utterance. */
export function isTwoPassPlan(plan: Pick<RustSpeechRoutePlan, "streaming" | "final"> | null | undefined): boolean {
  return !!plan?.streaming && !!plan?.final;
}

/** The plan has at least one engine. */
export function planHasEngine(plan: Pick<RustSpeechRoutePlan, "streaming" | "final"> | null | undefined): boolean {
  return !!plan?.streaming || !!plan?.final;
}

/** Catalog ids a session needs: the plan's models (and packs), plus Smart Turn when it is on. */
export function requiredModelIds(plan: Pick<RustSpeechRoutePlan, "models"> | null | undefined, smartTurn: boolean): string[] {
  const ids = [...(plan?.models ?? [])].filter((id) => typeof id === "string" && id.trim() !== "");
  if (smartTurn && !ids.includes(SMART_TURN_MODEL_ID)) ids.push(SMART_TURN_MODEL_ID);
  return ids;
}

/** What an engine of a plan does (for the route preview). */
export type PlanRole = "streaming" | "final" | "fallback" | "languageId";

/** The plan's engines in display order. */
export function planRoles(plan: RustSpeechRoutePlan | null | undefined): { role: PlanRole; spec: RustSpeechEngineSpec }[] {
  if (!plan) return [];
  const out: { role: PlanRole; spec: RustSpeechEngineSpec }[] = [];
  if (plan.streaming) out.push({ role: "streaming", spec: plan.streaming });
  if (plan.final) out.push({ role: "final", spec: plan.final });
  if (plan.fallback) out.push({ role: "fallback", spec: plan.fallback });
  if (plan.languageId) out.push({ role: "languageId", spec: plan.languageId });
  return out;
}

/** The engine runs on the graphics card (a GPU runtime variant). */
export function specOnGpu(spec: Pick<RustSpeechEngineSpec, "variant"> | null | undefined): boolean {
  const v = String(spec?.variant ?? "").toLowerCase();
  return v.startsWith("cuda") || v === "vulkan" || v === "directml" || v === "dml";
}
