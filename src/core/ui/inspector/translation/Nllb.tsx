import { formatBytes } from "@/core/services/diagnostics/format";
import { entrySize, entryTitle } from "@/core/services/models/format";
import { canonicalLanguage, toFlores } from "@/core/services/translation/languages";
import { NLLB_MODELS } from "@/core/services/translation/schema";
import type { RustCatalogEntry } from "@/types";
import { FC, memo, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { ModelRequirement, useModelsLoaded } from "../components/ModelRequirement";
import { InputCheckbox, InputSelect, type InputSelectOption } from "../components/input";
import { countOptions, Hint, LiveArea, mergeOption, sliceSetter } from "./common";
import { LanguagePair } from "./LanguagePair";
import { languageDisplayName } from "./logic";

/** Largest thread count and beam width offered (schema max). */
const MAX_THREADS = 4;
const MAX_BEAM = 4;

/**
 * NLLB panel (`nllb` backend): the model (600M or 1.3B) with its `<ModelRequirement>`, the shared pair checked against
 * NLLB's languages, threads, beam width and interims.
 *
 * Owned by P4-F.
 */
export const Nllb: FC = memo(() => {
  const { t, i18n } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.translation.data.nllb);
  const data = useSnapshot(window.ApiServer.state.services.translation.data);
  const up = sliceSetter("nllb");
  const uiLang = i18n.language || "en";

  useModelsLoaded();
  const models = useSnapshot(window.ApiServer.models.state);
  const catalog = models.catalog as RustCatalogEntry[];

  const modelOptions = useMemo<InputSelectOption[]>(() => {
    const opts = NLLB_MODELS.map((id, i) => {
      const entry = catalog.find((e) => e.id === id);
      const size = entry ? entrySize(entry) : null;
      const title = entry ? entryTitle(entry) : t(i === 0 ? "transl.nllb_model_small" : "transl.nllb_model_large");
      return { value: id, label: size && size.bytes > 0 ? `${title} · ${formatBytes(size.bytes)}` : title };
    });
    return mergeOption(opts, pr.model);
  }, [catalog, pr.model, t]);

  // Known languages NLLB lacks (codes Sigil does not know at all are flagged by the pair itself).
  const unsupported = [data.sourceLanguage, data.targetLanguage]
    .map((code) => canonicalLanguage(code))
    .filter((code) => code !== "" && !toFlores(code));
  const model = (NLLB_MODELS as readonly string[]).includes(pr.model) ? pr.model : NLLB_MODELS[0];

  return (
    <>
      <Inspector.SubHeader>{t("transl.nllb_title")}</Inspector.SubHeader>
      <Hint>{t("transl.nllb_desc")}</Hint>
      <LanguagePair>
        {unsupported.map((code) => (
          <Hint key={code} tone="warning">
            {t("transl.nllb_error_language", { language: languageDisplayName(code, uiLang) })}
          </Hint>
        ))}
      </LanguagePair>

      <InputSelect label="transl.nllb_field_model" options={modelOptions} value={pr.model} onValueChange={(v) => up("model", v)} />
      <Hint>{t("transl.nllb_model_hint")}</Hint>
      <LiveArea>
        <ModelRequirement ids={[model]} reason={t("transl.nllb_requires")} />
      </LiveArea>

      <InputSelect
        label="transl.mt_field_threads"
        options={countOptions(MAX_THREADS, (n) => (n === 1 ? t("transl.mt_threads_one") : String(n)))}
        value={String(pr.threads)}
        onValueChange={(v) => up("threads", Number(v))}
      />
      <Hint>{t("transl.mt_threads_hint")}</Hint>
      <InputSelect
        label="transl.nllb_field_beam"
        options={countOptions(MAX_BEAM, (n) => (n === 1 ? t("transl.nllb_beam_greedy") : String(n)))}
        value={String(pr.beam)}
        onValueChange={(v) => up("beam", Number(v))}
      />
      <Hint>{t("transl.nllb_beam_hint")}</Hint>
      <InputCheckbox label="common.field_interim_results" onChange={(v) => up("interim", v)} value={pr.interim} />
      <Hint>{t("transl.nllb_interim_hint")}</Hint>
    </>
  );
});
