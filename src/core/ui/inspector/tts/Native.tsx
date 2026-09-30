import type { TTS_State } from "@/core/services/tts/schema";
import { ServiceNetworkState } from "@/types";
import { FC, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputRange, InputSelect } from "../components/input";

export const Native: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.native);
  const handleUpdate = <K extends keyof TTS_State["native"]>(key: K, v: TTS_State["native"][K]) => window.ApiServer.patchService("tts", s => s.data.native[key] = v);
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  const [voices, setVoices] = useState<{ label: string, value: string }[]>([]);

  useEffect(() => {
    let voices = window.speechSynthesis.getVoices();

    if (voices.length) {
      setVoices(voices.map(v => ({ label: v.name, value: v.voiceURI })));
    }
    else {
      window.speechSynthesis.onvoiceschanged = () => {
        let options = window.speechSynthesis.getVoices().map(v => ({ label: v.name, value: v.voiceURI }))
        setVoices(options);
      }
    }

  }, []);

  return <>
    <Inspector.SubHeader>{t('tts.native_title')}</Inspector.SubHeader>
    <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
      <InputSelect
        value={data.voice}
        onValueChange={e => handleUpdate("voice", e)}
        options={voices}
        label="tts.field_voice" />
    </Inspector.Deactivatable>

    <InputRange value={data.pitch} onChange={e => handleUpdate("pitch", e.target.value)} label="common.field_pitch" step="0.1" min="0" max="2" />
    <InputRange value={data.rate} onChange={e => handleUpdate("rate", e.target.value)} label="common.field_rate" step="0.1" min="0.1" max="10" />
    <InputRange value={data.volume} onChange={e => handleUpdate("volume", e.target.value)} label="common.field_volume" step="0.05" min="0" max="1" />
  </>
}
