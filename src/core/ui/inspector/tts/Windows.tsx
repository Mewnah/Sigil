import type { TTS_State } from "@/core/services/tts/schema";
import { invoke } from "@tauri-apps/api/core";
import { FC, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputRange, InputSelect } from "../components/input";
import { TtsOutputDevice } from "./Routing";

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

  // Windows voices play through the TTS sink since P6-J, so the output is the same device picker as every other sink
  // backend. A device saved from the old SAPI output list (a token id ending in `\MMAudioOut\{endpoint}`) still plays
  // (the sink resolves it), and is migrated once to the endpoint's name (the label SAPI reports is the cpal name).
  useEffect(() => {
    const saved = window.ApiServer.state.services.tts.data.windows.device;
    if (!config || !/\\MMAudioOut\\/i.test(saved)) return;
    const match = config.devices.find(d => d.id === saved);
    if (match?.label) handleUpdate("device", match.label);
  }, [config]); // eslint-disable-line react-hooks/exhaustive-deps

  return <>
    <Inspector.SubHeader>{t('tts.windows_title')}</Inspector.SubHeader>
    <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
    <InputSelect
      value={data.voice}
      onValueChange={e => handleUpdate("voice", e)}
      options={config?.voices.map(d => ({ ...d, value: d.id })) || []}
      label="tts.field_voice" />
    <InputRange value={data.volume} onChange={e => handleUpdate("volume", e.target.value)} label="common.field_volume" step="0.05" min="0" max="2" />
    <InputRange value={data.rate} onChange={e => handleUpdate("rate", e.target.value)} label="common.field_rate" step="0.05" min="0.1" max="5" />
  </>
}
