import { filterGlossary } from "@/core/services/translation/glossary";
import { canonicalLanguage } from "@/core/services/translation/languages";
import { TRANSLATION_GLOSSARY_MAX } from "@/core/services/translation/schema";
import { FC, memo, useCallback, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { RiAddLine, RiDeleteBin6Line, RiMore2Fill } from "react-icons/ri";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import { InputBaseText, InputCheckbox } from "../components/input";
import { Hint, LanguageSelect, translationData } from "./common";
import {
  formatGlossaryText,
  glossaryIssues,
  keptRowAfterRemove,
  languageDisplayName,
  mergeGlossary,
  newGlossaryEntry,
  parseGlossaryText,
  visibleGlossaryRows,
  type GlossaryEntry,
  type GlossaryOverlap,
  type GlossaryShadow,
} from "./logic";

/** Rows shown before "Show more" (a glossary can hold 1000 terms). */
const PAGE = 40;
/** Show the search box from this many terms on. */
const SEARCH_FROM = 8;

/**
 * Glossary editor (live: the pipeline reads `glossary` for every line, so edits apply to the next line without a
 * restart). Rows edit the stored entries in place; each has an on/off switch, the term and its translation, and
 * details for the language pair it applies to and case sensitivity. Bulk paste and copy use tab-separated text.
 *
 * Owned by P4-F.
 */
export const GlossaryEditor: FC = memo(() => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.translation.data);
  const entries = data.glossary as readonly GlossaryEntry[];
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(PAGE);
  const [source, setSource] = useState("");
  const [target, setTarget] = useState("");
  const [bulk, setBulk] = useState<string | null>(null);
  const [probe, setProbe] = useState("");
  // The row last edited stays listed while a search runs, even once its text no longer matches (it would otherwise
  // unmount mid-typing); a new search starts from the plain results.
  const [keep, setKeep] = useState<number | null>(null);

  const issues = useMemo(() => glossaryIssues(entries), [entries]);
  const visible = useMemo(() => visibleGlossaryRows(entries, query, keep), [entries, query, keep]);
  const full = entries.length >= TRANSLATION_GLOSSARY_MAX;
  // The box stays while a search is typed, so shrinking the list below SEARCH_FROM never leaves a filter nobody can clear.
  const showSearch = entries.length >= SEARCH_FROM || query !== "";

  const focusRow = useCallback((index: number) => setKeep(index), []);
  const removeRow = useCallback((index: number) => {
    setKeep((cur) => keptRowAfterRemove(cur, index));
    translationData().glossary.splice(index, 1);
  }, []);

  const add = () => {
    if (!source.trim() || !target.trim() || full) return;
    const merged = mergeGlossary(translationData().glossary, [{ source, target, from: "", to: "" }]);
    if (merged.added === 0) {
      toast(t("transl.glossary_exists", { term: source.trim() }), { type: "warning", autoClose: 4000 });
      return;
    }
    translationData().glossary.push(newGlossaryEntry(source, target));
    setSource("");
    setTarget("");
  };

  const importText = () => {
    if (bulk === null) return;
    const parsed = parseGlossaryText(bulk);
    const merged = mergeGlossary(translationData().glossary, parsed.entries);
    if (merged.added > 0) translationData().glossary = merged.entries;
    toast(
      t("transl.glossary_import_result", {
        added: merged.added,
        duplicates: merged.duplicates,
        skipped: parsed.skipped + merged.overflow,
      }),
      { type: merged.added > 0 ? "success" : "warning", autoClose: 5000 },
    );
    if (parsed.ignoredColumns > 0) {
      toast(t("transl.glossary_import_ignored_columns", { count: parsed.ignoredColumns }), { type: "info", autoClose: 6000 });
    }
    if (merged.added > 0) setBulk(null);
  };

  const copyAll = async () => {
    try {
      await navigator.clipboard.writeText(formatGlossaryText(translationData().glossary));
      toast(t("transl.glossary_copied", { count: entries.length }), { type: "success", autoClose: 2500 });
    } catch (e) {
      toast(t("transl.glossary_copy_failed", { error: e instanceof Error ? e.message : String(e) }), { type: "error", autoClose: 5000 });
    }
  };

  const pairFrom = canonicalLanguage(data.sourceLanguage);
  const pairTo = canonicalLanguage(data.targetLanguage);
  const matches = probe.trim() ? filterGlossary(entries, probe, pairFrom, pairTo) : null;

  return (
    <div className="flex flex-col gap-2">
      <Hint>{t("transl.glossary_desc")}</Hint>

      {showSearch && (
        <InputBaseText
          fieldWidth={false}
          className="w-full"
          type="search"
          placeholder={t("transl.glossary_search")}
          aria-label={t("transl.glossary_search")}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setLimit(PAGE);
            setKeep(null);
          }}
        />
      )}

      {entries.length > 0 && (
        <ul className="flex flex-col gap-1" aria-label={t("transl.glossary_title")}>
          {visible.slice(0, limit).map((index) => (
            <GlossaryRow
              key={index}
              index={index}
              entry={entries[index]}
              shadow={issues.shadowed.get(index)}
              incomplete={issues.incomplete.has(index)}
              unknownLanguage={issues.unknownLanguage.get(index)}
              onFocusRow={focusRow}
              onRemove={removeRow}
            />
          ))}
        </ul>
      )}
      {visible.length > limit && (
        <button type="button" className="btn btn-ghost btn-xs normal-case" onClick={() => setLimit((n) => n + PAGE)}>
          {t("transl.glossary_show_more", { count: visible.length - limit })}
        </button>
      )}
      {entries.length > 0 && query.trim() && visible.length === 0 && <Hint>{t("transl.glossary_no_match")}</Hint>}

      <form
        className="flex items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          add();
        }}
      >
        <InputBaseText
          fieldWidth={false}
          className="w-full min-w-0"
          placeholder={t("transl.glossary_term")}
          aria-label={t("transl.glossary_term")}
          value={source}
          onChange={(e) => setSource(e.target.value)}
        />
        <InputBaseText
          fieldWidth={false}
          className="w-full min-w-0"
          placeholder={t("transl.glossary_translation")}
          aria-label={t("transl.glossary_translation")}
          value={target}
          onChange={(e) => setTarget(e.target.value)}
        />
        <button type="submit" className="btn btn-sm btn-primary btn-outline gap-1 shrink-0" disabled={!source.trim() || !target.trim() || full}>
          <RiAddLine aria-hidden /> {t("common.btn_add")}
        </button>
      </form>
      <Hint tone={full ? "warning" : "muted"}>{t("transl.glossary_count", { count: entries.length, max: TRANSLATION_GLOSSARY_MAX })}</Hint>

      <div className="flex flex-wrap gap-2">
        <button type="button" className="btn btn-xs btn-ghost normal-case" onClick={() => setBulk((v) => (v === null ? "" : null))}>
          {bulk === null ? t("transl.glossary_paste") : t("transl.glossary_paste_cancel")}
        </button>
        <button type="button" className="btn btn-xs btn-ghost normal-case" disabled={entries.length === 0} onClick={() => void copyAll()}>
          {t("transl.glossary_copy")}
        </button>
      </div>
      {bulk !== null && (
        <div className="flex flex-col gap-2">
          <textarea
            className="w-full h-28 p-2 bg-base-100 border border-base-content/20 rounded font-mono text-xs"
            dir="auto"
            value={bulk}
            placeholder={t("transl.glossary_paste_placeholder")}
            aria-label={t("transl.glossary_paste")}
            onChange={(e) => setBulk(e.target.value)}
          />
          <Hint>{t("transl.glossary_paste_help")}</Hint>
          <div>
            <button type="button" className="btn btn-sm btn-primary" disabled={!bulk.trim()} onClick={importText}>
              {t("transl.glossary_import")}
            </button>
          </div>
        </div>
      )}

      {entries.length > 0 && (
        <div className="flex flex-col gap-1">
          <InputBaseText
            fieldWidth={false}
            className="w-full"
            placeholder={t("transl.glossary_probe")}
            aria-label={t("transl.glossary_probe")}
            value={probe}
            onChange={(e) => setProbe(e.target.value)}
          />
          {matches && (
            <Hint tone={matches.length ? "success" : "muted"}>
              {matches.length
                ? t("transl.glossary_probe_match", { list: matches.map((m) => `${m.source} → ${m.target}`).join(", ") })
                : t("transl.glossary_probe_none")}
            </Hint>
          )}
        </div>
      )}
    </div>
  );
});

/** One glossary entry, edited in place. */
const GlossaryRow: FC<{
  index: number;
  entry: GlossaryEntry;
  shadow: GlossaryShadow | undefined;
  incomplete: boolean;
  unknownLanguage: readonly string[] | undefined;
  /** Focus entered the row (it stays listed while a search runs). */
  onFocusRow: (index: number) => void;
  onRemove: (index: number) => void;
}> = memo(
  ({ index, entry, shadow, incomplete, unknownLanguage, onFocusRow, onRemove }) => {
    const { t, i18n } = useTranslation();
    const [open, setOpen] = useState(false);
    const edit = <K extends keyof GlossaryEntry>(key: K, value: GlossaryEntry[K]) => {
      const row = translationData().glossary[index];
      if (row) row[key] = value;
    };
    const scoped = entry.from.trim() !== "" || entry.to.trim() !== "" || entry.caseSensitive;
    const languageName = (code: string) => (code ? languageDisplayName(code, i18n.language || "en") : t("transl.route_any_language"));
    const overlapText = (o: GlossaryOverlap) => {
      if (!o.from && !o.to) return t("transl.glossary_shadowed_spelling", { spelling: o.spelling ?? "" });
      const pair = `${languageName(o.from)} → ${languageName(o.to)}`;
      return o.spelling === null
        ? t("transl.glossary_shadowed_pair", { pair })
        : t("transl.glossary_shadowed_pair_spelling", { pair, spelling: o.spelling });
    };

    return (
      // Focus events from the row's own select menus (portals) reach the row through React too.
      <li className="flex flex-col gap-1" onFocus={() => onFocusRow(index)}>
        <div className="flex items-center gap-1">
          <input
            type="checkbox"
            className="checkbox checkbox-sm checkbox-primary shrink-0"
            checked={entry.enabled}
            aria-label={t("transl.glossary_enabled", { term: entry.source })}
            title={t("transl.glossary_enabled", { term: entry.source })}
            onChange={(e) => edit("enabled", e.target.checked)}
          />
          <InputBaseText
            fieldWidth={false}
            className={`w-full min-w-0 ${entry.enabled ? "" : "opacity-60"}`}
            placeholder={t("transl.glossary_term")}
            aria-label={t("transl.glossary_term")}
            value={entry.source}
            onChange={(e) => edit("source", e.target.value)}
          />
          <InputBaseText
            fieldWidth={false}
            className={`w-full min-w-0 ${entry.enabled ? "" : "opacity-60"}`}
            placeholder={t("transl.glossary_translation")}
            aria-label={t("transl.glossary_translation")}
            value={entry.target}
            onChange={(e) => edit("target", e.target.value)}
          />
          <button
            type="button"
            className={`btn btn-ghost btn-xs btn-square shrink-0 ${scoped ? "text-primary" : ""}`}
            aria-expanded={open}
            aria-label={t("transl.glossary_details")}
            title={t("transl.glossary_details")}
            onClick={() => setOpen((v) => !v)}
          >
            <RiMore2Fill aria-hidden />
          </button>
          <button
            type="button"
            className="btn btn-ghost btn-xs btn-square shrink-0 text-error"
            aria-label={t("transl.glossary_remove", { term: entry.source })}
            title={t("transl.glossary_remove", { term: entry.source })}
            onClick={() => onRemove(index)}
          >
            <RiDeleteBin6Line aria-hidden />
          </button>
        </div>
        {incomplete && <Hint tone="warning">{t("transl.glossary_incomplete")}</Hint>}
        {shadow?.kind === "always" && <Hint tone="warning">{t("transl.glossary_shadowed")}</Hint>}
        {shadow?.kind === "partly" &&
          shadow.overlaps.map((o) => (
            <Hint key={`${o.from}\u0000${o.to}\u0000${o.spelling ?? ""}`} tone="warning">
              {overlapText(o)}
            </Hint>
          ))}
        {unknownLanguage && unknownLanguage.length > 0 && (
          <Hint tone="warning">{t("transl.glossary_unknown_language", { list: unknownLanguage.map((c) => `"${c}"`).join(", ") })}</Hint>
        )}
        {open && (
          <div className="flex flex-col gap-2 rounded-lg border border-base-content/10 p-2 ms-6">
            <LanguageSelect label="transl.glossary_from" value={entry.from} emptyLabel={t("transl.route_any_language")} onChange={(v) => edit("from", v)} />
            <LanguageSelect label="transl.glossary_to" value={entry.to} emptyLabel={t("transl.route_any_language")} onChange={(v) => edit("to", v)} />
            <InputCheckbox label="transl.glossary_case_sensitive" value={entry.caseSensitive} onChange={(v) => edit("caseSensitive", v)} />
          </div>
        )}
      </li>
    );
  },
);
