import { isSatisfied } from "@/core/services/models/format";
import { Translation_BergamotService } from "@/core/services/translation/services/bergamot";
import type { RustCatalogEntry, RustModelEntryStatus, RustMtPair } from "@/types";
import { FC, memo, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { ModelRequirement, useModelsLoaded } from "../components/ModelRequirement";
import { InputCheckbox, InputSelect } from "../components/input";
import { countOptions, Hint, LiveArea, sliceSetter } from "./common";
import { LanguagePair } from "./LanguagePair";
import { bergamotDirections, bergamotPlanFromPairs, languageDisplayName, planBergamot, type BergamotPlan } from "./logic";

/** Largest thread count offered (schema max). */
const MAX_THREADS = 4;

/**
 * Bergamot panel (`bergamot` backend): the shared pair with the direction models it needs (one direct model, or two
 * through English) as a `<ModelRequirement>`, the directions already installed, worker threads and interims.
 *
 * Owned by P4-F.
 */
export const Bergamot: FC = memo(() => {
  const { t, i18n } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.translation.data.bergamot);
  const data = useSnapshot(window.ApiServer.state.services.translation.data);
  const up = sliceSetter("bergamot");
  const uiLang = i18n.language || "en";

  useModelsLoaded();
  const models = useSnapshot(window.ApiServer.models.state);
  const catalog = models.catalog as RustCatalogEntry[];
  const states = models.states as Record<string, RustModelEntryStatus>;
  const directions = useMemo(() => bergamotDirections(catalog), [catalog]);
  const installed = useMemo(() => directions.filter((d) => isSatisfied(states[d.id])), [directions, states]);
  const installedKey = installed.map((d) => d.id).join(",");

  // The engine's own view of the directions (P4-E); the catalog answers until it does.
  const [pairs, setPairs] = useState<RustMtPair[]>([]);
  useEffect(() => {
    let cancelled = false;
    Translation_BergamotService.getPairs()
      .then((rows) => !cancelled && setPairs(Array.isArray(rows) ? rows : []))
      .catch(() => !cancelled && setPairs([]));
    return () => {
      cancelled = true;
    };
  }, [installedKey]);

  const plan: BergamotPlan =
    bergamotPlanFromPairs(pairs, data.sourceLanguage, data.targetLanguage) ??
    planBergamot(directions, data.sourceLanguage, data.targetLanguage);
  const name = (code: string) => languageDisplayName(code, uiLang);

  return (
    <>
      <Inspector.SubHeader>{t("transl.bergamot_title")}</Inspector.SubHeader>
      <Hint>{t("transl.bergamot_desc")}</Hint>
      <LanguagePair>
        {plan.kind === "unsupported" && (
          <Hint tone="warning">
            {t("transl.error_unsupported_pair", {
              engine: t("transl.backend_bergamot"),
              pair: `${name(data.sourceLanguage)} → ${name(data.targetLanguage)}`,
            })}
          </Hint>
        )}
        {plan.kind === "unknown_source" && <Hint>{t("transl.bergamot_unknown_source")}</Hint>}
        {plan.kind === "pivot" && <Hint>{t("transl.bergamot_pivot")}</Hint>}
      </LanguagePair>

      {(plan.kind === "direct" || plan.kind === "pivot") && (
        <LiveArea>
          <ModelRequirement ids={plan.ids} reason={t("transl.bergamot_requires")} />
        </LiveArea>
      )}
      {directions.length > 0 && (
        <Hint>
          {installed.length
            ? t("transl.bergamot_installed", { list: installed.map((d) => `${name(d.source)} → ${name(d.target)}`).join(", ") })
            : t("transl.bergamot_installed_none")}
        </Hint>
      )}

      <InputSelect
        label="transl.mt_field_threads"
        options={countOptions(MAX_THREADS, (n) => (n === 1 ? t("transl.mt_threads_one") : String(n)))}
        value={String(pr.threads)}
        onValueChange={(v) => up("threads", Number(v))}
      />
      <Hint>{t("transl.mt_threads_hint")}</Hint>
      <InputCheckbox label="common.field_interim_results" onChange={(v) => up("interim", v)} value={pr.interim} />
      <Hint>{t("transl.bergamot_interim_hint")}</Hint>
    </>
  );
});
