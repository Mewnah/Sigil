import type { TTS_State } from "@/core/services/tts/schema";
import { ServiceNetworkState } from "@/types";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputRange, InputText } from "../components/input";
import { TtsOutputDevice } from "./Routing";

export const NeuralSidecar: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.neuralSidecar);
  const handleUpdate = <K extends keyof TTS_State["neuralSidecar"]>(key: K, v: TTS_State["neuralSidecar"][K]) =>
    window.ApiServer.patchService("tts", s => s.data.neuralSidecar[key] = v);
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  return <>
    <Inspector.SubHeader>{t("tts.neural_sidecar_title")}</Inspector.SubHeader>
    <div className="text-xs text-base-content/70 mb-2">{t("tts.neural_sidecar_desc")}</div>
    <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
      <InputText label="tts.neural_sidecar_endpoint" value={data.endpoint} onChange={e => handleUpdate("endpoint", e.target.value)} />
      <InputText label="tts.field_voice" value={data.voice} onChange={e => handleUpdate("voice", e.target.value)} />
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
      <InputRange label="common.field_volume" value={data.volume} onChange={e => handleUpdate("volume", e.target.value)} step="0.05" min="0" max="2" />
    </Inspector.Deactivatable>
  </>;
};
