import { Translation_State } from "@/core/services/translation/schema";
import { FC, memo } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputCheckbox, InputSelect, InputSelectOption } from "../components/input";
import { Hint, mergeOption, TextField } from "./common";

/**
 * Azure Translator panel: key and region, its own language pair from Azure's language list (blank source = Azure
 * detects it), profanity handling and interims.
 *
 * Owned by P4-F (moved from inspector_translation.tsx by P4-S0).
 */
export const Azure: FC = memo(() => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.translation.data.azure);
  const { azure: azureLanguages } = useSnapshot(window.ApiServer.translation.lib);
  const up = <K extends keyof Translation_State["azure"]>(key: K, v: Translation_State["azure"][K]) => window.ApiServer.state.services.translation.data.azure[key] = v;
  // Stored codes stay selectable while the language list is loading or offline. A blank source
  // language makes Azure auto-detect it (azureTranslateUrl leaves `from` out).
  const languages = azureLanguages as InputSelectOption[];
  const fromOpts: InputSelectOption[] = [
    { label: t("transl.azure_from_auto"), value: "" },
    ...mergeOption(languages, pr.languageFrom.trim()),
  ];
  const toOpts = mergeOption(languages, pr.language);

  return <>
    <Inspector.SubHeader>{t('transl.azure_title')}</Inspector.SubHeader>
    <TextField label="transl.azure_key" type="password" autoComplete="new-password" value={pr.key} onChange={e => up("key", e.target.value)} />
    <TextField label="transl.azure_location_optional" value={pr.location} placeholder="westeurope" onChange={e => up("location", e.target.value)} />
    <Hint>{t("transl.azure_location_hint")}</Hint>
    <InputSelect
      label="transl.azure_from_language"
      options={fromOpts}
      value={pr.languageFrom.trim()}
      onValueChange={e => up("languageFrom", e)}
    />
    <InputSelect
      label="transl.azure_to_language"
      options={toOpts}
      value={pr.language}
      onValueChange={e => up("language", e)}
    />
    <InputSelect
      label="transl.azure_profanity"
      options={[
        { label: t('transl.azure_profanity_masked'), value: 'Marked' },
        { label: t('transl.azure_profanity_removed'), value: 'Deleted' },
        { label: t('transl.azure_profanity_raw'), value: 'NoAction' },
      ]}
      value={pr.profanity}
      onValueChange={e => up("profanity", e as Translation_State["azure"]["profanity"])}
    />
    <InputCheckbox label="common.field_interim_results" onChange={e => up("interim", e)} value={pr.interim} />
  </>
})
