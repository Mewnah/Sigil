import {
  TRANSLATION_DEFAULT_CHAIN,
  TRANSLATION_INTERIM_ENGINES,
  type Translation_State,
  type TranslationEngineId,
  type TranslationInterimEngine,
} from "@/core/services/translation/schema";
import { FC, memo } from "react";
import { useTranslation } from "react-i18next";
import { RiAddLine, RiDeleteBin6Line } from "react-icons/ri";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputCheckbox, InputSelect } from "../components/input";
import { ChainEditor } from "./ChainEditor";
import { Hint, LanguageSelect, NumberField, sliceSetter, translationData } from "./common";
import { LanguagePair } from "./LanguagePair";
import { backendLabelKey, languageDisplayName } from "./logic";

/** Schema limit of per-pair rules. */
const MAX_PAIR_RULES = 50;

/**
 * Automatic routing panel (`auto` backend): the engine chain, per-pair chains, the interim engine, the attempt timeout,
 * the local fallback after cloud failures and interims. The chain's engines are configured in their own panels.
 *
 * Owned by P4-F.
 */
export const Router: FC = memo(() => {
  const { t, i18n } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.translation.data);
  const pr = data.auto;
  const up = sliceSetter("auto");
  const state = data as Translation_State;
  const uiLang = i18n.language || "en";

  const setChain = (chain: TranslationEngineId[]) => {
    translationData().auto.chain = chain;
  };
  const addRule = () => {
    const auto = translationData().auto;
    if (auto.pairOverrides.length >= MAX_PAIR_RULES) return;
    auto.pairOverrides.push({ from: translationData().sourceLanguage, to: translationData().targetLanguage, chain: [...auto.chain] });
  };

  const interimOptions = TRANSLATION_INTERIM_ENGINES.map((id) => ({
    value: id,
    label: id === "auto" ? t("transl.route_interim_auto") : t(backendLabelKey(id)),
  }));

  return (
    <>
      <Inspector.SubHeader>{t("transl.auto_title")}</Inspector.SubHeader>
      <Hint>{t("transl.route_desc")}</Hint>
      <LanguagePair />

      <Inspector.SubHeader>{t("transl.route_chain_title")}</Inspector.SubHeader>
      <Hint>{t("transl.route_chain_hint")}</Hint>
      <ChainEditor chain={pr.chain} state={state} onChange={setChain} label={t("transl.route_chain_title")} />
      <div className="flex justify-end">
        <button
          type="button"
          className="btn btn-ghost btn-xs normal-case"
          onClick={() => setChain([...TRANSLATION_DEFAULT_CHAIN])}
          disabled={pr.chain.join() === TRANSLATION_DEFAULT_CHAIN.join()}
        >
          {t("transl.route_chain_reset")}
        </button>
      </div>

      <Inspector.SubHeader>{t("transl.route_rules_title")}</Inspector.SubHeader>
      <Hint>{t("transl.route_rules_hint")}</Hint>
      {pr.pairOverrides.map((rule, i) => (
        <div key={i} className="flex flex-col gap-2 rounded-lg border border-base-content/10 p-2">
          <div className="flex items-center gap-2">
            <span className="flex-grow text-xs font-semibold truncate">
              {t("transl.route_rule_name", {
                from: rule.from.trim() ? languageDisplayName(rule.from, uiLang) : t("transl.route_any_language"),
                to: rule.to.trim() ? languageDisplayName(rule.to, uiLang) : t("transl.route_any_language"),
              })}
            </span>
            <button
              type="button"
              className="btn btn-ghost btn-xs btn-square text-error"
              aria-label={t("transl.route_rule_remove")}
              title={t("transl.route_rule_remove")}
              onClick={() => translationData().auto.pairOverrides.splice(i, 1)}
            >
              <RiDeleteBin6Line aria-hidden />
            </button>
          </div>
          <LanguageSelect
            label="transl.route_rule_from"
            value={rule.from}
            emptyLabel={t("transl.route_any_language")}
            onChange={(v) => (translationData().auto.pairOverrides[i].from = v)}
          />
          <LanguageSelect
            label="transl.route_rule_to"
            value={rule.to}
            emptyLabel={t("transl.route_any_language")}
            onChange={(v) => (translationData().auto.pairOverrides[i].to = v)}
          />
          <ChainEditor
            chain={rule.chain}
            state={state}
            label={t("transl.route_rule_chain")}
            onChange={(chain) => (translationData().auto.pairOverrides[i].chain = chain)}
          />
        </div>
      ))}
      <div>
        <button type="button" className="btn btn-sm btn-outline gap-1 normal-case" disabled={pr.pairOverrides.length >= MAX_PAIR_RULES} onClick={addRule}>
          <RiAddLine aria-hidden /> {t("transl.route_rule_add")}
        </button>
      </div>

      <Inspector.SubHeader>{t("transl.route_behaviour_title")}</Inspector.SubHeader>
      <InputSelect
        label="transl.route_field_interim_engine"
        options={interimOptions}
        value={pr.interimEngine}
        onValueChange={(v) => up("interimEngine", v as TranslationInterimEngine)}
      />
      <Hint>{t("transl.route_interim_hint")}</Hint>
      <NumberField
        label="transl.route_field_timeout"
        value={pr.attemptTimeoutMs}
        min={500}
        max={60_000}
        scale={1000}
        step={0.5}
        onCommit={(v) => up("attemptTimeoutMs", v)}
      />
      <Hint>{t("transl.route_timeout_hint")}</Hint>
      <InputCheckbox label="transl.route_field_cloud_fallback" value={pr.cloudFallback} onChange={(v) => up("cloudFallback", v)} />
      <Hint>{t("transl.route_cloud_fallback_hint")}</Hint>
      <InputCheckbox label="common.field_interim_results" onChange={(v) => up("interim", v)} value={pr.interim} />
    </>
  );
});
