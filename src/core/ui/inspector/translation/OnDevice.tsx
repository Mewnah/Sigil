import { Translation_State } from "@/core/services/translation/schema";
import { FC, memo } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputCheckbox } from "../components/input";
import { Hint, LanguageSelect } from "./common";

type OnDeviceSlice = Translation_State["onDevice"];

/**
 * On-device panel (the WebView's built-in Translator API): its own language pair (BCP-47, picked from the translation
 * language table; a stored code that is not in the table stays selectable), source detection and interims.
 *
 * Owned by P4-F (moved from inspector_translation.tsx by P4-S0).
 */
export const OnDevice: FC = memo(() => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.translation.data.onDevice);
  const up = <K extends keyof OnDeviceSlice>(key: K, v: OnDeviceSlice[K]) => {
    window.ApiServer.state.services.translation.data.onDevice[key] = v;
  };

  return <>
    <Inspector.SubHeader>{t("transl.ondevice_title")}</Inspector.SubHeader>
    <Hint>{t("transl.ondevice_desc")}</Hint>
    <LanguageSelect label="transl.field_source_language" value={pr.languageFrom} onChange={(v) => up("languageFrom", v)} />
    <LanguageSelect label="transl.field_target_language" value={pr.language} onChange={(v) => up("language", v)} />
    <InputCheckbox
      label="transl.ondevice_auto_detect"
      value={pr.autoDetect}
      onChange={(v) => up("autoDetect", v)}
    />
    <InputCheckbox label="common.field_interim_results" onChange={(v) => up("interim", v)} value={pr.interim} />
  </>;
});
