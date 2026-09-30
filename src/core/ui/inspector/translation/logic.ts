/**
 * Pure helpers of the translation inspector panels (P4-F): language pickers, glossary editing and bulk import, router
 * chain editing, per-engine setup checks, Bergamot directions, server presets and prompt-template checks. No React,
 * no `window`, no Tauri: covered by `__tests__/logic.test.ts`.
 *
 * Owned by P4-F.
 */
import { canonicalLanguage, languageInfo, TRANSLATION_LANGUAGES } from "@/core/services/translation/languages";
import { PROMPT_FORMAT_TRAITS, type ResolvedPromptFormat } from "@/core/services/translation/prompt";
import {
  TRANSLATION_ENGINE_IDS,
  TRANSLATION_GLOSSARY_MAX,
  Translation_Backends,
  type Translation_GlossaryEntry,
  type Translation_State,
  type TranslationEngineId,
} from "@/core/services/translation/schema";

// ---------------------------------------------------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------------------------------------------------

/**
 * A typed number in stored units, or null while the text is empty, partial or out of range. `scale` converts the
 * shown unit to the stored one (seconds shown, milliseconds stored: 1000); stored integers are rounded.
 */
export function parseBoundedNumber(
  text: string,
  bounds: { min: number; max: number; scale?: number; integer?: boolean },
): number | null {
  const trimmed = text.trim().replace(",", ".");
  if (!trimmed || !/^-?\d*(?:\.\d*)?$/.test(trimmed) || trimmed === "-" || trimmed === ".") return null;
  const shown = Number(trimmed);
  if (!Number.isFinite(shown)) return null;
  const scale = bounds.scale ?? 1;
  let stored = shown * scale;
  if (bounds.integer ?? true) {
    if (scale === 1 && !Number.isInteger(shown)) return null;
    stored = Math.round(stored);
  }
  if (stored < bounds.min || stored > bounds.max) return null;
  return stored;
}

/** A stored value in shown units without float noise (`1500` ms → `1.5`). */
export function formatScaled(value: number, scale = 1): string {
  const shown = value / scale;
  return String(Math.round(shown * 1000) / 1000);
}

// ---------------------------------------------------------------------------------------------------------------------
// Languages
// ---------------------------------------------------------------------------------------------------------------------

export interface LanguageOption {
  value: string;
  label: string;
}

const displayNamesCache = new Map<string, Intl.DisplayNames | null>();

function displayNames(uiLanguage: string): Intl.DisplayNames | null {
  const key = (uiLanguage || "en").replace(/_/g, "-");
  if (!displayNamesCache.has(key)) {
    let names: Intl.DisplayNames | null = null;
    try {
      names = new Intl.DisplayNames([key, "en"], { type: "language", fallback: "none" });
    } catch {
      names = null;
    }
    displayNamesCache.set(key, names);
  }
  return displayNamesCache.get(key) ?? null;
}

/** `Intl.DisplayNames` input for a canonical code: canonical `zh` is Simplified Chinese. */
function displayCode(code: string): string {
  return code === "zh" ? "zh-Hans" : code;
}

/**
 * Name of a language in the UI language ("Japanese", "Japonés"); the table's English name when the runtime has
 * none; the raw input for codes that are not languages.
 */
export function languageDisplayName(code: string, uiLanguage: string): string {
  const canonical = canonicalLanguage(code);
  if (!canonical) return code.trim();
  let name: string | undefined;
  try {
    name = displayNames(uiLanguage)?.of(displayCode(canonical));
  } catch {
    name = undefined;
  }
  if (!name || name.toLowerCase() === canonical.toLowerCase()) name = languageInfo(canonical)?.name ?? canonical;
  return name;
}

const optionsCache = new Map<string, readonly LanguageOption[]>();

/**
 * Every language of the translation table as picker options, `Name (code)` in the UI language, sorted by name.
 * Cached per UI language.
 */
export function languageOptions(uiLanguage: string): readonly LanguageOption[] {
  const key = uiLanguage || "en";
  const cached = optionsCache.get(key);
  if (cached) return cached;
  let collator: Intl.Collator;
  try {
    collator = new Intl.Collator([key.replace(/_/g, "-"), "en"]);
  } catch {
    collator = new Intl.Collator("en");
  }
  const rows = TRANSLATION_LANGUAGES.map((l) => ({ code: l.code, name: languageDisplayName(l.code, key), english: l.name }));
  // Some runtimes give two codes one name (Twi is shown as "Akan"): the table's English name tells them apart.
  const uses = new Map<string, number>();
  for (const r of rows) uses.set(r.name, (uses.get(r.name) ?? 0) + 1);
  for (const r of rows) if ((uses.get(r.name) ?? 0) > 1 && r.english !== r.name) r.name = `${r.name} – ${r.english}`;
  rows.sort((a, b) => collator.compare(a.name, b.name) || a.code.localeCompare(b.code));
  const options = rows.map((r) => ({ value: r.code, label: `${r.name} (${r.code})` }));
  optionsCache.set(key, options);
  return options;
}

/**
 * The value a language picker shows for a stored code: the canonical code when it is a known language (a
 * hand-edited `ja-JP` shows as Japanese), else the stored text itself (kept selectable by the picker).
 */
export function pickerValue(stored: string): string {
  const raw = stored.trim();
  return canonicalLanguage(raw) || raw;
}

/** Both sides name the same (known) language. */
export function isSameLanguage(a: string, b: string): boolean {
  const ca = canonicalLanguage(a);
  return ca !== "" && ca === canonicalLanguage(b);
}

// ---------------------------------------------------------------------------------------------------------------------
// Glossary
// ---------------------------------------------------------------------------------------------------------------------

export type GlossaryEntry = Translation_GlossaryEntry;

export function newGlossaryEntry(source: string, target: string, from = "", to = ""): GlossaryEntry {
  return { source: source.trim(), target: target.trim(), from: from.trim(), to: to.trim(), caseSensitive: false, enabled: true };
}

/** A side of an entry's language pair for comparisons: the canonical code, else the text lowercased; "" = any. */
function entryLanguage(code: string): string {
  const raw = code.trim();
  return raw ? (canonicalLanguage(raw) || raw).toLowerCase() : "";
}

/** Identity of an entry for duplicate checks on add and import: the normalized source term plus its language pair. */
export function glossaryKey(e: Pick<GlossaryEntry, "source" | "from" | "to" | "caseSensitive">): string {
  const source = e.source.trim().normalize("NFKC");
  return `${e.caseSensitive ? source : source.toLowerCase()}\u0000${entryLanguage(e.from)}\u0000${entryLanguage(e.to)}`;
}

/** The term as the pipeline's `filterGlossary` deduplicates it: one entry per normalized source, whatever the case. */
function termKey(source: string): string {
  return source.trim().normalize("NFKC").toLowerCase();
}

/** Where an earlier entry for the same term is sent instead of a later one. */
export interface GlossaryOverlap {
  /** Index of the earlier entry. */
  by: number;
  /** The languages this happens for ("" = any language). */
  from: string;
  to: string;
  /** Only in lines that spell the term exactly like this (the earlier entry is case-sensitive); null = any spelling. */
  spelling: string | null;
}

/** An entry that an earlier one replaces: always (it is never sent), or only for some languages or spellings. */
export type GlossaryShadow = { kind: "always"; by: number } | { kind: "partly"; overlaps: GlossaryOverlap[] };

export interface GlossaryIssues {
  /** Entries an earlier enabled entry for the same term replaces (the pipeline sends only the first one that matches). */
  shadowed: Map<number, GlossaryShadow>;
  /** Indexes of entries missing the term or its translation (never sent). */
  incomplete: Set<number>;
  /** Entries whose `from` / `to` is not a language Sigil knows, with those values (they never match a known language). */
  unknownLanguage: Map<number, string[]>;
}

/** Where `earlier` is used instead of `later` (both enabled, same term): everywhere ("all"), partly, or nowhere (null). */
function overlapOf(earlier: GlossaryEntry, later: GlossaryEntry): "all" | Omit<GlossaryOverlap, "by"> | null {
  const side = (e: string, l: string): { scope: string; all: boolean } | null => {
    const el = entryLanguage(e);
    const ll = entryLanguage(l);
    if (!el) return { scope: l.trim(), all: true };
    // The earlier entry is limited to one language, the later one applies to any: they collide only for that language.
    if (!ll) return { scope: e.trim(), all: false };
    return el === ll ? { scope: l.trim(), all: true } : null;
  };
  const from = side(earlier.from, later.from);
  const to = side(earlier.to, later.to);
  if (!from || !to) return null;
  let spelling: string | null = null;
  if (earlier.caseSensitive) {
    const exact = earlier.source.trim().normalize("NFKC");
    if (later.caseSensitive) {
      // Two case-sensitive spellings of one term match different text.
      if (exact !== later.source.trim().normalize("NFKC")) return null;
    } else {
      spelling = earlier.source.trim();
    }
  }
  if (from.all && to.all && spelling === null) return "all";
  return { from: from.scope, to: to.scope, spelling };
}

/**
 * Problems of a glossary, modelled on how the pipeline's `filterGlossary` picks terms: it skips disabled and incomplete
 * entries and entries for other languages, then sends the first entry for each normalized term (whatever its case or
 * language scope), so a later entry for the same term is used only where no earlier one applies. Lines whose source
 * language is unknown match every language scope; that case is not reported.
 */
export function glossaryIssues(entries: readonly GlossaryEntry[]): GlossaryIssues {
  const shadowed = new Map<number, GlossaryShadow>();
  const incomplete = new Set<number>();
  const unknownLanguage = new Map<number, string[]>();
  const byTerm = new Map<string, number[]>();
  entries.forEach((e, i) => {
    if (!e.source.trim() || !e.target.trim()) {
      incomplete.add(i);
      return;
    }
    const unknown = [e.from, e.to].map((s) => s.trim()).filter((s) => s !== "" && !canonicalLanguage(s));
    if (unknown.length) unknownLanguage.set(i, unknown);
    if (!e.enabled) return;
    const key = termKey(e.source);
    const earlier = byTerm.get(key) ?? [];
    const overlaps: GlossaryOverlap[] = [];
    let always = -1;
    for (const j of earlier) {
      const o = overlapOf(entries[j], e);
      if (o === null) continue;
      if (o === "all") {
        always = j;
        break;
      }
      if (!overlaps.some((x) => x.from === o.from && x.to === o.to && x.spelling === o.spelling)) overlaps.push({ by: j, ...o });
    }
    if (always >= 0) shadowed.set(i, { kind: "always", by: always });
    else if (overlaps.length) shadowed.set(i, { kind: "partly", overlaps });
    earlier.push(i);
    byTerm.set(key, earlier);
  });
  return { shadowed, incomplete, unknownLanguage };
}

/** Indexes of entries whose term or translation contains `query` (case-insensitive); all when the query is blank. */
export function searchGlossary(entries: readonly GlossaryEntry[], query: string): number[] {
  const q = query.trim().normalize("NFKC").toLowerCase();
  const out: number[] = [];
  entries.forEach((e, i) => {
    if (!q || e.source.normalize("NFKC").toLowerCase().includes(q) || e.target.normalize("NFKC").toLowerCase().includes(q)) {
      out.push(i);
    }
  });
  return out;
}

/**
 * Rows the editor lists: the search hits plus `keep`, the row last edited, so a row does not vanish while its term is
 * retyped into something the search no longer matches.
 */
export function visibleGlossaryRows(entries: readonly GlossaryEntry[], query: string, keep: number | null): number[] {
  const hits = searchGlossary(entries, query);
  if (keep === null || keep < 0 || keep >= entries.length || hits.includes(keep)) return hits;
  return [...hits, keep].sort((a, b) => a - b);
}

/** The row being edited after row `removed` is deleted (rows below it move up by one). */
export function keptRowAfterRemove(keep: number | null, removed: number): number | null {
  if (keep === null || keep === removed) return null;
  return keep > removed ? keep - 1 : keep;
}

export interface ParsedGlossaryLine {
  source: string;
  target: string;
  from: string;
  to: string;
}

/** Separators of a pasted `term = translation` line, tried in this order after tabs. */
const PAIR_SEPARATORS = ["=>", "->", "→", "⇒", "="];

function unquote(s: string): string {
  const t = s.trim();
  if (t.length >= 2 && ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'")))) {
    return t.slice(1, -1).replace(/""/g, '"').trim();
  }
  return t;
}

/** Splits a CSV line on `delimiter`, honouring double quotes. */
function splitCsv(line: string, delimiter: string): string[] {
  const out: string[] = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (c === '"') {
      if (quoted && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        quoted = !quoted;
      }
    } else if (c === delimiter && !quoted) {
      out.push(cur);
      cur = "";
    } else {
      cur += c;
    }
  }
  out.push(cur);
  return out.map((s) => s.trim());
}

/** A pasted language column as a canonical code; null when it is not a language (a notes or category column). */
function languageColumn(value: string): string | null {
  const raw = value.trim();
  if (!raw) return "";
  return canonicalLanguage(raw) || null;
}

function parseLine(line: string): (ParsedGlossaryLine & { ignoredColumns: number }) | null {
  let fields: string[];
  if (line.includes("\t")) {
    fields = line.split("\t").map(unquote);
  } else {
    const sep = PAIR_SEPARATORS.find((s) => line.includes(s));
    if (sep) {
      const at = line.indexOf(sep);
      fields = [unquote(line.slice(0, at)), unquote(line.slice(at + sep.length))];
    } else if (line.includes(",") || line.includes(";")) {
      fields = splitCsv(line, line.includes(",") ? "," : ";");
    } else {
      return null;
    }
  }
  const [source = "", target = "", fromRaw = "", toRaw = ""] = fields;
  if (!source.trim() || !target.trim()) return null;
  // Only language codes are kept: another column (notes, a category) would limit the term to a pair that never occurs.
  const from = languageColumn(fromRaw);
  const to = languageColumn(toRaw);
  return {
    source: source.trim(),
    target: target.trim(),
    from: from ?? "",
    to: to ?? "",
    ignoredColumns: (from === null ? 1 : 0) + (to === null ? 1 : 0),
  };
}

export interface ParsedGlossaryText {
  entries: ParsedGlossaryLine[];
  /** Lines without both a term and a translation. */
  skipped: number;
  /** Third or fourth columns that are not languages, dropped (the term then applies to any language on that side). */
  ignoredColumns: number;
}

/**
 * Glossary lines pasted as text: `term<TAB>translation[<TAB>from<TAB>to]` (what "Copy as text" and spreadsheets
 * produce), `term = translation` (also `=>`, `->`, `→`) or CSV (`term,translation`). Blank lines and lines starting
 * with `#` are comments (a term starting with `#` is written in quotes); lines without both sides are counted as
 * skipped. Language columns are stored as canonical codes.
 */
export function parseGlossaryText(text: string): ParsedGlossaryText {
  const entries: ParsedGlossaryLine[] = [];
  let skipped = 0;
  let ignoredColumns = 0;
  for (const raw of text.replace(/^﻿/, "").split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const parsed = parseLine(raw.replace(/^\s+|\s+$/g, ""));
    if (!parsed) {
      skipped++;
      continue;
    }
    const { ignoredColumns: ignored, ...entry } = parsed;
    ignoredColumns += ignored;
    entries.push(entry);
  }
  return { entries, skipped, ignoredColumns };
}

/**
 * A copied field, quoted when the parser would otherwise misread it: a first column starting with `#` (a comment
 * line) or a value wrapped in quotes (they would be stripped).
 */
function exportField(value: string, first: boolean): string {
  const quotedLooking = value.length >= 2 && ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'")));
  if ((first && value.startsWith("#")) || quotedLooking) return `"${value.replace(/"/g, '""')}"`;
  return value;
}

/** Tab-separated text of the glossary (language columns only when an entry has them); `parseGlossaryText` reads it. */
export function formatGlossaryText(entries: readonly Pick<GlossaryEntry, "source" | "target" | "from" | "to">[]): string {
  const withLanguages = entries.some((e) => e.from.trim() || e.to.trim());
  const clean = (s: string) => s.replace(/[\t\r\n]+/g, " ").trim();
  return entries
    .filter((e) => e.source.trim() || e.target.trim())
    .map((e) => {
      const cols = [exportField(clean(e.source), true), exportField(clean(e.target), false)];
      if (withLanguages) cols.push(clean(e.from), clean(e.to));
      return cols.join("\t").replace(/\t+$/, "");
    })
    .join("\n");
}

export interface GlossaryMerge {
  entries: GlossaryEntry[];
  added: number;
  /** Incoming terms already in the glossary (same term and pair) or repeated in the input. */
  duplicates: number;
  /** Incoming terms dropped because the glossary is full. */
  overflow: number;
}

/** Appends new terms, skipping ones the glossary already has, up to `max` entries in total. */
export function mergeGlossary(
  existing: readonly GlossaryEntry[],
  incoming: readonly ParsedGlossaryLine[],
  max: number = TRANSLATION_GLOSSARY_MAX,
): GlossaryMerge {
  const entries = existing.map((e) => ({ ...e }));
  const seen = new Set(entries.map(glossaryKey));
  let added = 0;
  let duplicates = 0;
  let overflow = 0;
  for (const line of incoming) {
    const entry = newGlossaryEntry(line.source, line.target, line.from, line.to);
    const key = glossaryKey(entry);
    if (seen.has(key)) {
      duplicates++;
      continue;
    }
    if (entries.length >= max) {
      overflow++;
      continue;
    }
    seen.add(key);
    entries.push(entry);
    added++;
  }
  return { entries, added, duplicates, overflow };
}

// ---------------------------------------------------------------------------------------------------------------------
// Router chains
// ---------------------------------------------------------------------------------------------------------------------

/** i18n key of a backend's name (same keys as the provider list and the router's messages). */
export function backendLabelKey(id: Translation_Backends | TranslationEngineId): string {
  return `transl.backend_${id === Translation_Backends.onDevice ? "ondevice" : id}`;
}

/** `list` with the item at `index` moved by `delta` (clamped); a copy. */
export function moveItem<T>(list: readonly T[], index: number, delta: number): T[] {
  const out = [...list];
  if (index < 0 || index >= out.length) return out;
  const to = Math.max(0, Math.min(out.length - 1, index + delta));
  if (to === index) return out;
  const [item] = out.splice(index, 1);
  out.splice(to, 0, item);
  return out;
}

/** The chain without `id`; a chain keeps at least one engine. */
export function removeFromChain(chain: readonly TranslationEngineId[], id: TranslationEngineId): TranslationEngineId[] {
  if (chain.length <= 1) return [...chain];
  return chain.filter((c) => c !== id);
}

/** The chain with `id` appended (no duplicates, known engines only). */
export function addToChain(chain: readonly TranslationEngineId[], id: TranslationEngineId): TranslationEngineId[] {
  if (chain.includes(id) || !(TRANSLATION_ENGINE_IDS as readonly string[]).includes(id)) return [...chain];
  return [...chain, id];
}

/** Engines a chain does not use yet, in the default preference order. */
export function enginesNotInChain(chain: readonly TranslationEngineId[]): TranslationEngineId[] {
  return TRANSLATION_ENGINE_IDS.filter((id) => !chain.includes(id));
}

/** What an engine still needs before the router can use it (an i18n key), or null when it looks configured. */
export function engineSetupIssue(id: TranslationEngineId, state: Translation_State): string | null {
  switch (id) {
    case Translation_Backends.llm: {
      const cfg = state.llm;
      switch (cfg.provider) {
        case "bundled":
          return null;
        case "gemini":
          return !cfg.geminiKey.trim() ? "transl.route_needs_key" : !cfg.geminiModel.trim() ? "transl.route_needs_model" : null;
        case "openrouter":
          return !cfg.openRouterKey.trim() ? "transl.route_needs_key" : !cfg.openRouterModel.trim() ? "transl.route_needs_model" : null;
        case "openai":
          return !cfg.openaiKey.trim() ? "transl.route_needs_key" : !cfg.openaiModel.trim() ? "transl.route_needs_model" : null;
        default:
          return !cfg.customUrl.trim() ? "transl.route_needs_url" : !cfg.customModel.trim() ? "transl.route_needs_model" : null;
      }
    }
    case Translation_Backends.deepl:
      return state.deepl.apiKey.trim() ? null : "transl.route_needs_key";
    case Translation_Backends.azure:
      return state.azure.key.trim() ? null : "transl.route_needs_key";
    case Translation_Backends.libretranslate:
      return state.libretranslate.endpoint.trim() ? null : "transl.route_needs_url";
    default:
      return null;
  }
}

/**
 * Engines that send the text over the internet: the cloud services, and LibreTranslate or a custom LLM server at an
 * address outside this PC and the local network. (The router's `isCloudEngine` is narrower on purpose: it decides the
 * local fallback after failures of the known cloud services.)
 */
export function isOnlineEngine(id: TranslationEngineId, state: Pick<Translation_State, "llm" | "libretranslate">): boolean {
  switch (id) {
    case Translation_Backends.deepl:
    case Translation_Backends.azure:
      return true;
    case Translation_Backends.libretranslate:
      return isInternetAddress(state.libretranslate.endpoint);
    case Translation_Backends.llm:
      if (state.llm.provider === "bundled") return false;
      if (state.llm.provider === "custom") return isInternetAddress(state.llm.customUrl);
      return true;
    default:
      return false;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Bergamot directions
// ---------------------------------------------------------------------------------------------------------------------

/** Catalog family of the Bergamot direction models (`mt-bergamot/<src>_<tgt>`). */
export const BERGAMOT_FAMILY = "mt-bergamot";

export interface BergamotDirection {
  /** Canonical codes. */
  source: string;
  target: string;
  /** Catalog id. */
  id: string;
}

/** Directions of the catalog's Bergamot entries (`mt-bergamot/ja_en`, `mt-bergamot/en_zh-hant`). */
export function bergamotDirections(entries: readonly { id: string; family?: string }[]): BergamotDirection[] {
  const out: BergamotDirection[] = [];
  for (const e of entries) {
    if (e.family !== undefined && e.family !== BERGAMOT_FAMILY) continue;
    const m = /^mt-bergamot\/([A-Za-z-]+)_([A-Za-z-]+)$/.exec(e.id);
    if (!m) continue;
    const source = canonicalLanguage(m[1]);
    const target = canonicalLanguage(m[2]);
    if (source && target && source !== target) out.push({ source, target, id: e.id });
  }
  return out;
}

export type BergamotPlan =
  | { kind: "no_target" }
  | { kind: "unknown_source" }
  | { kind: "same" }
  | { kind: "direct"; ids: string[] }
  | { kind: "pivot"; ids: string[] }
  | { kind: "unsupported" };

/** The direction models a pair needs: one direct model, or two through English. */
export function planBergamot(directions: readonly BergamotDirection[], from: string, to: string): BergamotPlan {
  const target = canonicalLanguage(to);
  if (!target) return { kind: "no_target" };
  const source = canonicalLanguage(from);
  if (!source) return { kind: "unknown_source" };
  if (source === target) return { kind: "same" };
  const find = (s: string, t: string) => directions.find((d) => d.source === s && d.target === t);
  const direct = find(source, target);
  if (direct) return { kind: "direct", ids: [direct.id] };
  if (source !== "en" && target !== "en") {
    const first = find(source, "en");
    const second = find("en", target);
    if (first && second) return { kind: "pivot", ids: [first.id, second.id] };
  }
  return { kind: "unsupported" };
}

/** Directions from Rust `bergamot_pairs` rows when it answered (the engine's own view), else from the catalog. */
export function bergamotPlanFromPairs(
  pairs: readonly { source: string; target: string; ids: readonly string[]; pivot: boolean }[],
  from: string,
  to: string,
): BergamotPlan | null {
  const source = canonicalLanguage(from);
  const target = canonicalLanguage(to);
  if (!source || !target || source === target) return null;
  const row = pairs.find((p) => canonicalLanguage(p.source) === source && canonicalLanguage(p.target) === target);
  if (!row || row.ids.length === 0) return null;
  return { kind: row.pivot ? "pivot" : "direct", ids: [...row.ids] };
}

// ---------------------------------------------------------------------------------------------------------------------
// Server presets
// ---------------------------------------------------------------------------------------------------------------------

export type LibreServerKind = "libretranslate" | "ltengine" | "custom";

/**
 * LibreTranslate-compatible server presets. LTEngine (LibreTranslate's local LLM server, Gemma 3 GGUF models) listens
 * on port 5050 and detects the source language itself (`source: "auto"`). It needs no API key; a stored key is kept
 * (not cleared), so switching back to a hosted LibreTranslate server still has it.
 */
export const LIBRE_PRESETS: Readonly<Record<Exclude<LibreServerKind, "custom">, Partial<Translation_State["libretranslate"]>>> = {
  libretranslate: { endpoint: "http://localhost:5000" },
  ltengine: { endpoint: "http://localhost:5050", languageFrom: "auto", autoDetect: false },
};

function hostPort(url: string): { host: string; port: string } | null {
  if (!url.trim()) return null;
  try {
    const u = new URL(/^[a-z][a-z0-9+.-]*:\/\//i.test(url.trim()) ? url.trim() : `http://${url.trim()}`);
    const host = u.hostname.toLowerCase().replace(/\.$/, "");
    if (!host) return null;
    const port = u.port || (u.protocol === "https:" ? "443" : "80");
    return { host, port };
  } catch {
    return null;
  }
}

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);

/** Private, loopback, link-local and shared (CGNAT, e.g. Tailscale) IPv4 ranges. */
function isPrivateIpv4(host: string): boolean {
  const m = /^(\d{1,3})\.(\d{1,3})\.\d{1,3}\.\d{1,3}$/.exec(host);
  if (!m) return false;
  const a = Number(m[1]);
  const b = Number(m[2]);
  return (
    a === 0 ||
    a === 10 ||
    a === 127 ||
    (a === 100 && b >= 64 && b <= 127) ||
    (a === 169 && b === 254) ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168)
  );
}

/** Whether a (lowercased, URL-normalized) host is this PC or the local network. */
function isLocalNetworkHost(host: string): boolean {
  if (LOCAL_HOSTS.has(host) || host.endsWith(".localhost")) return true;
  if (isPrivateIpv4(host)) return true;
  if (host.startsWith("[") && host.endsWith("]")) {
    const v6 = host.slice(1, -1);
    // IPv4-mapped (`::ffff:c0a8:101`, as the URL parser writes it).
    const mapped = /^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/.exec(v6);
    if (mapped) {
      const hi = parseInt(mapped[1], 16);
      const lo = parseInt(mapped[2], 16);
      return isPrivateIpv4(`${hi >> 8}.${hi & 255}.${lo >> 8}.${lo & 255}`);
    }
    // Loopback, unspecified, unique local (fc00::/7) and link-local (fe80::/10).
    return v6 === "::1" || v6 === "::" || /^f[cd][0-9a-f]{2}:/.test(v6) || /^fe[89ab][0-9a-f]:/.test(v6);
  }
  if (/^[\d.]+$/.test(host)) return false;
  // Single-label intranet names (`gpu-box`) and local-network suffixes.
  return !host.includes(".") || /\.(local|lan|home\.arpa|internal|localdomain)$/.test(host);
}

/** Whether a server URL points outside this PC and the local network (false for an empty or unreadable URL). */
export function isInternetAddress(url: string): boolean {
  const hp = hostPort(url);
  return hp !== null && !isLocalNetworkHost(hp.host);
}

/** Which preset an endpoint matches (a local server on the preset's port). */
export function libreServerKind(endpoint: string): LibreServerKind {
  const hp = hostPort(endpoint);
  if (!hp || !LOCAL_HOSTS.has(hp.host)) return "custom";
  if (hp.port === "5050") return "ltengine";
  if (hp.port === "5000") return "libretranslate";
  return "custom";
}

export type LlmServerPresetId = "ollama" | "lmstudio" | "llamacpp" | "vllm";

/** OpenAI-compatible servers people run themselves, with their default local URLs. */
export const LLM_SERVER_PRESETS: readonly { id: LlmServerPresetId; url: string }[] = [
  { id: "ollama", url: "http://localhost:11434/v1" },
  { id: "lmstudio", url: "http://localhost:1234/v1" },
  { id: "llamacpp", url: "http://localhost:8080/v1" },
  { id: "vllm", url: "http://localhost:8000/v1" },
];

/** The preset a custom server URL matches (a local server on the preset's port), else "other". */
export function llmServerPresetOf(url: string): LlmServerPresetId | "other" {
  const hp = hostPort(url);
  if (!url.trim() || !hp || !LOCAL_HOSTS.has(hp.host)) return "other";
  const hit = LLM_SERVER_PRESETS.find((p) => hostPort(p.url)?.port === hp.port);
  return hit?.id ?? "other";
}

/** Stored fields of the key and the model name of each remote LLM provider. */
export const LLM_PROVIDER_FIELDS = {
  gemini: { key: "geminiKey", model: "geminiModel" },
  openrouter: { key: "openRouterKey", model: "openRouterModel" },
  openai: { key: "openaiKey", model: "openaiModel" },
  custom: { key: "customKey", model: "customModel" },
} as const satisfies Record<string, { key: keyof Translation_State["llm"]; model: keyof Translation_State["llm"] }>;

/** Whether a server's model list has `model` (ids may carry a prefix: Gemini's `models/…`, OpenRouter's `org/…`). */
export function listsModel(models: readonly string[], model: string): boolean {
  const want = model.trim();
  if (!want) return false;
  return models.some((m) => m === want || m.endsWith(`/${want}`));
}

// ---------------------------------------------------------------------------------------------------------------------
// Prompts
// ---------------------------------------------------------------------------------------------------------------------

/** Which live settings a prompt family sends (the others are ignored by it). */
export function promptFormatUses(format: ResolvedPromptFormat): { context: boolean; glossary: boolean; background: boolean } {
  const t = PROMPT_FORMAT_TRAITS[format];
  return { context: t.context, glossary: t.glossary, background: t.background };
}

/** Placeholders of the custom prompt template (`prompt.ts` `customPrompt`). */
export const CUSTOM_PROMPT_PLACEHOLDERS = ["source_lang", "target_lang", "text", "context", "glossary", "background"] as const;

/** Problems of a custom template: no `{text}` (the line is appended), unknown `{placeholders}` (sent as typed). */
export function customPromptIssues(template: string): { missingText: boolean; unknown: string[] } {
  const known = new Set<string>(CUSTOM_PROMPT_PLACEHOLDERS);
  const unknown: string[] = [];
  for (const m of template.matchAll(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g)) {
    if (!known.has(m[1]) && !unknown.includes(m[1])) unknown.push(m[1]);
  }
  return { missingText: template.trim() !== "" && !/\{text\}/.test(template), unknown };
}

// ---------------------------------------------------------------------------------------------------------------------
// DeepL
// ---------------------------------------------------------------------------------------------------------------------

/** The formality choice shown: the engine sends `prefer_*` for both the strict and the `prefer_` values. */
export function deeplFormalityChoice(value: Translation_State["deepl"]["formality"]): "default" | "more" | "less" {
  if (value === "more" || value === "prefer_more") return "more";
  if (value === "less" || value === "prefer_less") return "less";
  return "default";
}

/** Share of the monthly quota used, 0–100 (rounded down), or null when the limit is unknown. */
export function usagePercent(usage: { characterCount: number; characterLimit: number } | null | undefined): number | null {
  if (!usage || !(usage.characterLimit > 0)) return null;
  return Math.max(0, Math.min(100, Math.floor((usage.characterCount / usage.characterLimit) * 100)));
}
