import type { TTS_State } from "@/core/services/tts/schema";
import { normalizeAzureVoiceName } from "@/core/services/tts/ssml";
import { azureVoices } from "@/core/services/tts/tts_data";
import { ServiceNetworkState } from "@/types";
import { FC, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputMappedGroupSelect, InputRange, InputSelect, InputText } from "../components/input";
import { TtsOutputDevice } from "./Routing";

export const Azure: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.azure);
  const handleUpdate = <K extends keyof TTS_State["azure"]>(key: K, v: TTS_State["azure"][K]) =>
    window.ApiServer.patchService("tts", s => s.data.azure[key] = v);
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  const [voiceStyles, setVoiceStyles] = useState<{ label: string, value: string }[]>([]);
  const [voiceRoles, setVoiceRoles] = useState<{ label: string, value: string }[]>([]);

  const updateVoice = (value: { group: string, option: string }) => {
    window.ApiServer.patchService("tts", s => {
      s.data.azure.language = value.group;
      s.data.azure.voice = value.option;
      // Styles and roles differ per voice: clear them.
      s.data.azure.voiceStyle = "";
      s.data.azure.voiceRole = "";
    });
  };

  useEffect(() => {
    // find voice
    // set styles and roles
    if (!data.language || !data.voice)
      return;

    const voiceName = normalizeAzureVoiceName(data.voice);
    const voice = azureVoices[data.language]?.find(voice => voice[0] === voiceName);
    const voiceParams = voice?.[2];
    setVoiceStyles(voiceParams?.styles ? [{ label: "None", value: "" }, ...voiceParams.styles.map(style => ({ label: style, value: style }))] : []);
    setVoiceRoles(voiceParams?.roles ? [{ label: "None", value: "" }, ...voiceParams.roles.map(roles => ({ label: roles, value: roles }))] : []);

  }, [data.language, data.voice]);

  return <>
    <Inspector.SubHeader>{t('tts.azure_title')}</Inspector.SubHeader>
    <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
      <InputText type="password" label="tts.azure_key" value={data.key} onChange={e => handleUpdate("key", e.target.value)} />
      <InputText type="password" label="tts.azure_location" value={data.location} onChange={e => handleUpdate("location", e.target.value)} />
    </Inspector.Deactivatable>

    <InputMappedGroupSelect
      labelGroup="common.field_language"
      labelOption="common.field_dialect"
      value={{ group: data.language, option: data.voice }}
      onChange={updateVoice}
      library={azureVoices} />

    {voiceStyles.length > 0 && <InputSelect value={data.voiceStyle} options={voiceStyles} label="tts.azure_voice_style" onValueChange={e => handleUpdate("voiceStyle", e)} />}
    {voiceRoles.length > 0 && <InputSelect value={data.voiceRole} options={voiceRoles} label="tts.azure_voice_role" onValueChange={e => handleUpdate("voiceRole", e)} />}
    <InputSelect options={[
      { label: "silent", value: "silent" },
      { label: "x-soft", value: "x-soft" },
      { label: "soft", value: "soft" },
      { label: "medium", value: "medium" },
      { label: "loud", value: "loud" },
      { label: "x-loud", value: "x-loud" },
      { label: "default", value: "default" }
    ]} label="tts.azure_voice_volume" value={data.voiceVolume} onValueChange={e => handleUpdate("voiceVolume", e)} />
    <InputSelect options={[
      { label: "x-slow", value: "x-slow" },
      { label: "slow", value: "slow" },
      { label: "medium", value: "medium" },
      { label: "fast", value: "fast" },
      { label: "x-fast", value: "x-fast" },
      { label: "default", value: "default" },
    ]} label="tts.azure_voice_rate" value={data.voiceRate} onValueChange={e => handleUpdate("voiceRate", e)} />
    <InputSelect options={[
      { label: "x-low", value: "x-low" },
      { label: "low", value: "low" },
      { label: "medium", value: "medium" },
      { label: "high", value: "high" },
      { label: "x-high", value: "x-high" },
      { label: "default", value: "default" },
    ]} label="tts.azure_voice_pitch" value={data.voicePitch} onValueChange={e => handleUpdate("voicePitch", e)} />
    <InputSelect options={[
      { label: "x-low", value: "x-low" },
      { label: "low", value: "low" },
      { label: "medium", value: "medium" },
      { label: "high", value: "high" },
      { label: "x-high", value: "x-high" },
      { label: "default", value: "default" },
    ]} label="tts.azure_voice_range" value={data.voiceRange} onValueChange={e => handleUpdate("voiceRange", e)} />
    <InputRange value={data.volume} onChange={e => handleUpdate("volume", e.target.value)} label="common.field_volume" step="0.05" min="0" max="1" />
    {/* <InputRange value={data.rate} onChange={e => handleUpdate("rate", e.target.value)} label={`Rate (${data.rate})`} step="0.05" min="0.1" max="5" /> */}
  </>
}
