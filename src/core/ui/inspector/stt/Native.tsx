import { nativeLangs } from "@/core/services/stt/stt_data";
import { isTauri } from "@tauri-apps/api/core";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputMappedGroupSelect } from "../components/input";

/** Web Speech in the webview (`native` / legacy `browser`). */
export const Native: FC = () => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.native);
  const updateLanguage = (value: { group: string, option: string }) => {
    window.ApiServer.state.services.stt.data.native.language = value.option;
    window.ApiServer.state.services.stt.data.native.language_group = value.group;
  };
  return <>
    <Inspector.SubHeader>{t('stt.native_title')}</Inspector.SubHeader>
    {isTauri() && <Inspector.Description>{t("stt.native_desktop_hint")}</Inspector.Description>}
    <InputMappedGroupSelect
      labelGroup="common.field_language"
      labelOption="common.field_dialect"
      value={{ option: pr.language, group: pr.language_group }}
      onChange={updateLanguage}
      library={nativeLangs} />
  </>
}
