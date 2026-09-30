import type { STT_State } from "@/core/services/stt/schema";
import { MOONSHINE_DEFAULT_ENDPOINT } from "@/core/services/stt/services/moonshine";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputRustAudioInput, InputText } from "../components/input";
import { VoiceDetection } from "./VoiceDetection";

/** Legacy: a Moonshine server the user runs. The `local` backend runs Moonshine v2 in-process. */
export const Moonshine: FC = () => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.moonshine);
  const up = <K extends keyof STT_State["moonshine"]>(key: K, v: STT_State["moonshine"][K]) =>
    window.ApiServer.state.services.stt.data.moonshine[key] = v;

  return <>
    <Inspector.SubHeader>{t("stt.moonshine_legacy_title")}</Inspector.SubHeader>
    <div role="note" className="alert alert-info alert-soft text-xs leading-snug mb-2">
      <span>{t("stt.moonshine_legacy_notice")}</span>
    </div>
    <div className="text-xs text-base-content/70 mb-2">
      {t("stt.moonshine_desc")}
    </div>
    <InputText
      label="stt.moonshine_endpoint"
      value={pr.endpoint}
      placeholder={MOONSHINE_DEFAULT_ENDPOINT}
      onChange={e => up("endpoint", e.target.value)}
    />
    <InputRustAudioInput
      label="common.field_input_device"
      value={pr.device}
      onChange={e => up("device", e)}
      hint="stt.rust_input_device_hint"
    />
    <InputText
      label="stt.moonshine_language"
      value={pr.language}
      onChange={e => up("language", e.target.value)}
    />
    <Inspector.Description>{t("stt.moonshine_language_desc")}</Inspector.Description>
    <VoiceDetection />
  </>;
};
