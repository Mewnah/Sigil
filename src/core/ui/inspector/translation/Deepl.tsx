import { canonicalLanguage } from "@/core/services/translation/languages";
import type { Translation_State } from "@/core/services/translation/schema";
import {
  DEEPL_FALLBACK_SOURCES,
  DEEPL_FALLBACK_TARGETS,
  deeplSourceCode,
  Translation_DeeplService,
} from "@/core/services/translation/services/deepl";
import type { RustDeeplLanguage, RustDeeplUsage } from "@/types";
import { FC, memo, useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { RiRefreshLine } from "react-icons/ri";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputCheckbox, InputSelect } from "../components/input";
import { Hint, LANGUAGE_PROBE_DEBOUNCE_MS, LiveArea, sliceSetter, TextField } from "./common";
import { LanguagePair } from "./LanguagePair";
import { deeplFormalityChoice, languageDisplayName, usagePercent } from "./logic";

type DeeplSlice = Translation_State["deepl"];

/** DeepL's language lists for the key (built-in lists until they load or when they cannot be fetched). */
interface DeeplLists {
  source: readonly string[];
  target: readonly RustDeeplLanguage[];
  /** Fetched from DeepL (false = the built-in lists). */
  live: boolean;
}

const FALLBACK_LISTS: DeeplLists = {
  source: DEEPL_FALLBACK_SOURCES,
  target: DEEPL_FALLBACK_TARGETS.map((code) => ({ code, name: code, supportsFormality: false })),
  live: false,
};

type UsageState = { kind: "idle" } | { kind: "loading" } | { kind: "done"; usage: RustDeeplUsage | null };

/**
 * DeepL panel (`deepl` backend): key and plan, the shared pair with the DeepL codes it maps to, formality, model type,
 * context and interims, plus this month's character usage (refreshable while translation runs).
 *
 * Owned by P4-F.
 */
export const Deepl: FC = memo(() => {
  const { t, i18n } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.translation.data.deepl);
  const data = useSnapshot(window.ApiServer.state.services.translation.data);
  const up = sliceSetter("deepl");
  const key = pr.apiKey.trim();
  const plan = pr.plan;
  const [lists, setLists] = useState<DeeplLists>(FALLBACK_LISTS);

  useEffect(() => {
    setLists(FALLBACK_LISTS);
    if (!key) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      void Promise.all([
        Translation_DeeplService.getLanguages(key, plan, "source"),
        Translation_DeeplService.getLanguages(key, plan, "target"),
      ]).then(([source, target]) => {
        if (cancelled || !source.length || !target.length) return;
        setLists({ source: source.map((l) => l.code.toUpperCase()), target, live: true });
      });
    }, LANGUAGE_PROBE_DEBOUNCE_MS);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [key, plan]);

  const uiLang = i18n.language || "en";
  const target = canonicalLanguage(data.targetLanguage);
  const source = canonicalLanguage(data.sourceLanguage);
  const targetCode = target ? Translation_DeeplService.targetCode(target, lists.target) : null;
  const sourceCode = source ? deeplSourceCode(source) : null;
  const sourceOk = !source || (sourceCode !== null && lists.source.includes(sourceCode));
  const targetRow = targetCode ? lists.target.find((l) => l.code.toUpperCase() === targetCode) : undefined;
  const formality = deeplFormalityChoice(pr.formality);

  return (
    <>
      <Inspector.SubHeader>{t("transl.deepl_title")}</Inspector.SubHeader>
      <Hint>{t("transl.deepl_desc")}</Hint>
      <TextField
        label="transl.deepl_key"
        type="password"
        autoComplete="new-password"
        value={pr.apiKey}
        placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:fx"
        onChange={(e) => up("apiKey", e.target.value)}
      />
      <InputSelect
        label="transl.deepl_field_plan"
        options={(["auto", "free", "pro"] as const).map((v) => ({ value: v, label: t(`transl.deepl_plan_${v}`) }))}
        value={pr.plan}
        onValueChange={(v) => up("plan", v as DeeplSlice["plan"])}
      />
      <DeeplUsage apiKey={key} plan={plan} />

      <LanguagePair>
        {target && !targetCode && (
          <Hint tone="warning">{t("transl.error_unsupported_target", { engine: "DeepL", language: languageDisplayName(target, uiLang) })}</Hint>
        )}
        {target && targetCode && <Hint>{t("transl.deepl_target_code", { code: targetCode })}</Hint>}
        {source && !sourceOk && (
          <Hint tone="warning">{t("transl.pair_source_unsupported", { engine: "DeepL", language: languageDisplayName(source, uiLang) })}</Hint>
        )}
        {!lists.live && key && <Hint>{t("transl.deepl_lists_builtin")}</Hint>}
      </LanguagePair>

      <InputSelect
        label="transl.deepl_field_formality"
        options={(["default", "more", "less"] as const).map((v) => ({ value: v, label: t(`transl.deepl_formality_${v}`) }))}
        value={formality}
        onValueChange={(v) => up("formality", v as DeeplSlice["formality"])}
      />
      {formality !== "default" && lists.live && targetRow && !targetRow.supportsFormality && (
        <Hint>{t("transl.deepl_formality_unsupported", { language: languageDisplayName(target, uiLang) })}</Hint>
      )}
      <InputSelect
        label="transl.deepl_field_model"
        options={(["default", "latency_optimized", "quality_optimized", "prefer_quality_optimized"] as const).map((v) => ({
          value: v,
          label: t(`transl.deepl_model_${v}`),
        }))}
        value={pr.modelType}
        onValueChange={(v) => up("modelType", v as DeeplSlice["modelType"])}
      />
      <Hint>{t("transl.deepl_model_hint")}</Hint>
      <InputCheckbox label="transl.deepl_field_context" value={pr.sendContext} onChange={(v) => up("sendContext", v)} />
      <Hint>{t("transl.deepl_context_hint")}</Hint>
      <InputCheckbox label="common.field_interim_results" onChange={(v) => up("interim", v)} value={pr.interim} />
      <Hint>{t("transl.deepl_interim_hint")}</Hint>
    </>
  );
});

/** Characters used this billing period, loaded when the key changes and on request. */
const DeeplUsage: FC<{ apiKey: string; plan: DeeplSlice["plan"] }> = ({ apiKey, plan }) => {
  const { t, i18n } = useTranslation();
  const [state, setState] = useState<UsageState>({ kind: "idle" });
  const request = useRef(0);

  const load = useCallback(() => {
    const id = ++request.current;
    if (!apiKey) {
      setState({ kind: "idle" });
      return;
    }
    setState({ kind: "loading" });
    void Translation_DeeplService.getUsage(apiKey, plan).then((usage) => {
      if (id === request.current) setState({ kind: "done", usage });
    });
  }, [apiKey, plan]);

  useEffect(() => {
    const timer = setTimeout(load, LANGUAGE_PROBE_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      request.current++;
    };
  }, [load]);

  if (!apiKey) return null;
  const nf = new Intl.NumberFormat(i18n.language || undefined);
  const usage = state.kind === "done" ? state.usage : null;
  const percent = usagePercent(usage);

  return (
    <LiveArea>
      <div className="flex items-center gap-2 text-xs px-1">
        <span className="flex-grow min-w-0">
          {state.kind === "loading" && t("transl.deepl_usage_loading")}
          {state.kind === "done" && !usage && <span className="text-warning">{t("transl.deepl_usage_failed")}</span>}
          {usage &&
            (percent === null
              ? t("transl.deepl_usage_unlimited", { used: nf.format(usage.characterCount) })
              : t("transl.deepl_usage", {
                  used: nf.format(usage.characterCount),
                  limit: nf.format(usage.characterLimit),
                  percent,
                }))}
        </span>
        <button
          type="button"
          className="btn btn-ghost btn-xs btn-square"
          aria-label={t("transl.deepl_usage_refresh")}
          title={t("transl.deepl_usage_refresh")}
          disabled={state.kind === "loading"}
          onClick={load}
        >
          <RiRefreshLine aria-hidden />
        </button>
      </div>
      {percent !== null && (
        <progress
          className={`progress w-full ${percent >= 90 ? "progress-warning" : "progress-primary"}`}
          value={percent}
          max={100}
          aria-hidden
        />
      )}
    </LiveArea>
  );
};
