import type { TTS_State } from "@/core/services/tts/schema";
import { ServiceNetworkState } from "@/types";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputRange, InputText } from "../components/input";
import { TtsOutputDevice } from "./Routing";

export const Chatterbox: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.chatterbox);
  const handleUpdate = <K extends keyof TTS_State["chatterbox"]>(key: K, v: TTS_State["chatterbox"][K]) =>
    window.ApiServer.patchService("tts", s => s.data.chatterbox[key] = v);
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  return <>
    <Inspector.SubHeader>{t("tts.chatterbox_title")}</Inspector.SubHeader>
    <div className="text-xs text-base-content/70 mb-2">{t("tts.chatterbox_desc")}</div>
    <div className="text-xs text-base-content/60 mb-2">{t("tts.http_legacy_openai_hint")}</div>
    <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
      <InputText label="tts.chatterbox_endpoint" value={data.endpoint} onChange={e => handleUpdate("endpoint", e.target.value)} />
      <InputText label="tts.field_voice" value={data.voice} onChange={e => handleUpdate("voice", e.target.value)} />
      <InputRange value={data.speed} onChange={e => handleUpdate("speed", e.target.value)} label="tts.chatterbox_speed" step="0.05" min="0.5" max="2" />
      <InputRange value={data.exaggeration} onChange={e => handleUpdate("exaggeration", e.target.value)} label="tts.chatterbox_exaggeration" step="0.05" min="0" max="1" />
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
      <InputRange label="common.field_volume" value={data.volume} onChange={e => handleUpdate("volume", e.target.value)} step="0.05" min="0" max="2" />
    </Inspector.Deactivatable>
  </>;
};
