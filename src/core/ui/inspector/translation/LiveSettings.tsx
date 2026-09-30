import { FC, memo } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputCheckbox } from "../components/input";
import { Details, Hint, NumberField, translationData } from "./common";
import { GlossaryEditor } from "./GlossaryEditor";

/**
 * Settings that apply to the next line without a restart, rendered by the shell outside `Inspector.Deactivatable`
 * (the pipeline reads them per line): glossary, rolling context and background, the interim (partial caption) policy
 * and typed input.
 *
 * Owned by P4-F.
 */
export const TranslationLiveSettings: FC = memo(() => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.translation.data);
  const context = data.context;
  const partials = data.partials;
  const setContext = <K extends keyof typeof context>(key: K, value: (typeof context)[K]) => {
    translationData().context[key] = value;
  };
  const setPartials = <K extends keyof typeof partials>(key: K, value: (typeof partials)[K]) => {
    translationData().partials[key] = value;
  };

  return (
    <>
      <Inspector.SubHeader>{t("transl.live_title")}</Inspector.SubHeader>
      <Hint>{t("transl.live_desc")}</Hint>

      <Details title={t("transl.glossary_title_count", { count: data.glossary.length })} defaultOpen={data.glossary.length > 0}>
        <GlossaryEditor />
      </Details>

      <Details title={t("transl.context_title")}>
        <InputCheckbox label="transl.context_field_enabled" value={context.enabled} onChange={(v) => setContext("enabled", v)} />
        <NumberField
          label="transl.context_field_turns"
          value={context.turns}
          min={0}
          max={10}
          disabled={!context.enabled}
          onCommit={(v) => setContext("turns", v)}
        />
        <Hint>{t("transl.context_turns_hint")}</Hint>
        <label className="flex flex-col gap-2">
          <span className="font-medium text-base-content/80 text-xs">{t("transl.context_field_background")}</span>
          <textarea
            className="w-full h-20 p-2 bg-base-100 border border-base-content/20 rounded text-sm"
            dir="auto"
            value={context.background}
            placeholder={t("transl.context_background_placeholder")}
            onChange={(e) => setContext("background", e.target.value)}
          />
        </label>
        <Hint>{t("transl.context_background_hint")}</Hint>
      </Details>

      <Details title={t("transl.partials_title")}>
        <InputCheckbox
          label="transl.partials_field_local_agreement"
          value={partials.localAgreement}
          onChange={(v) => setPartials("localAgreement", v)}
        />
        <Hint>{t("transl.partials_local_agreement_hint")}</Hint>
        {/* Only LocalAgreement splits an interim into a committed part and a tentative tail. */}
        <div className={partials.localAgreement ? "flex flex-col gap-2" : "flex flex-col gap-2 opacity-50 pointer-events-none"} aria-disabled={!partials.localAgreement}>
          <InputCheckbox
            label="transl.partials_field_show_tentative"
            value={partials.showTentative}
            onChange={(v) => setPartials("showTentative", v)}
          />
          <Hint>{t("transl.partials_show_tentative_hint")}</Hint>
        </div>
        <NumberField
          label="transl.partials_field_throttle"
          value={partials.throttleMs}
          min={0}
          max={5000}
          step={50}
          onCommit={(v) => setPartials("throttleMs", v)}
        />
        <Hint>{t("transl.partials_throttle_hint")}</Hint>
        <InputCheckbox
          label="transl.partials_field_turn_end"
          value={partials.turnEndCommit}
          onChange={(v) => setPartials("turnEndCommit", v)}
        />
        <Hint>{t("transl.partials_turn_end_hint")}</Hint>
        <NumberField
          label="transl.partials_field_turn_hold"
          value={partials.turnHoldMs}
          min={500}
          max={10_000}
          scale={1000}
          step={0.5}
          disabled={!partials.turnEndCommit}
          onCommit={(v) => setPartials("turnHoldMs", v)}
        />
      </Details>

      <InputCheckbox
        label="common.field_use_keyboard_input"
        value={data.inputField}
        onChange={(v) => {
          translationData().inputField = v;
        }}
      />
      <Hint>{t("transl.input_field_hint")}</Hint>
    </>
  );
});
