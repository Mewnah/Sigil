import type { TTS_State } from "@/core/services/tts/schema";
import { invoke } from "@tauri-apps/api/core";
import { FC, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputRange, InputSelect } from "../components/input";

type WindowsToken = {
  id: string;
  label: string;
}
type WindowsConfig = { devices: WindowsToken[], voices: WindowsToken[] };

export const Windows: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.windows);
  const handleUpdate = <K extends keyof TTS_State["windows"]>(key: K, v: TTS_State["windows"][K]) => window.ApiServer.patchService("tts", s => s.data.windows[key] = v);

  const [config, setConfig] = useState<WindowsConfig>();

  useEffect(() => {
    invoke<WindowsConfig>("plugin:windows-tts|get_voices").then(setConfig);
  }, []);

  return <>
    <Inspector.SubHeader>{t('tts.windows_title')}</Inspector.SubHeader>
    <InputSelect
      value={data.device}
      onValueChange={e => handleUpdate("device", e)}
      options={config?.devices.map(d => ({ label: d.label, value: d.id })) || []}
      label="common.field_output_device" />
    {/* <InputNativeAudioOutput label="Audio Output" value={pr.device} onChange={e => handleUpdate("device", e)} /> */}
    <InputSelect
      value={data.voice}
      onValueChange={e => handleUpdate("voice", e)}
      options={config?.voices.map(d => ({ ...d, value: d.id })) || []}
      label="tts.field_voice" />
    <InputRange value={data.volume} onChange={e => handleUpdate("volume", e.target.value)} label="common.field_volume" step="0.05" min="0" max="1" />
    <InputRange value={data.rate} onChange={e => handleUpdate("rate", e.target.value)} label="common.field_rate" step="0.05" min="0.1" max="5" />
  </>
}
