import type { STT_State } from "@/core/services/stt/schema";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputText } from "../components/input";
import { Advanced } from "./common";

/** Silero VAD settings shared by Whisper, Moonshine and the HTTP / Groq engines. */
export const VoiceDetection: FC = () => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.vad);
  const up = <K extends keyof STT_State["vad"]>(key: K, v: STT_State["vad"][K]) => window.ApiServer.state.services.stt.data.vad[key] = v;
  const reset = () => window.ApiServer.patchService("stt", s => {
    s.data.vad.threshold = "0.5";
    s.data.vad.minSilenceMs = "500";
    s.data.vad.preRollMs = "400";
    s.data.vad.minSpeechMs = "250";
    s.data.vad.maxUtteranceMs = "18000";
  });

  return <Advanced title={t("stt.vad_title")}>
    <Inspector.Description>{t("stt.vad_desc")}</Inspector.Description>
    <InputText type="number" step="0.05" min="0.05" max="0.95" label="stt.vad_threshold" value={pr.threshold} onChange={e => up("threshold", e.target.value)} />
    <Inspector.Description>{t("stt.vad_threshold_desc")}</Inspector.Description>
    <InputText type="number" step="50" min="100" max="5000" label="stt.vad_min_silence" value={pr.minSilenceMs} onChange={e => up("minSilenceMs", e.target.value)} />
    <Inspector.Description>{t("stt.vad_min_silence_desc")}</Inspector.Description>
    <InputText type="number" step="50" min="0" max="2000" label="stt.vad_pre_roll" value={pr.preRollMs} onChange={e => up("preRollMs", e.target.value)} />
    <InputText type="number" step="50" min="0" max="5000" label="stt.vad_min_speech" value={pr.minSpeechMs} onChange={e => up("minSpeechMs", e.target.value)} />
    <InputText type="number" step="1000" min="2000" max="60000" label="stt.vad_max_utterance" value={pr.maxUtteranceMs} onChange={e => up("maxUtteranceMs", e.target.value)} />
    <Inspector.Description>{t("stt.vad_max_utterance_desc")}</Inspector.Description>
    <button type="button" className="btn btn-xs btn-ghost self-end" onClick={reset}>{t("stt.vad_reset")}</button>
  </Advanced>;
};
