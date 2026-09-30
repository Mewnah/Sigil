import { OpenAiAudioPreset, type STT_State } from "@/core/services/stt/schema";
import { normalizeEndpointUrl } from "@/core/services/stt/rustCapturePolling";
import { GROQ_KEYS_URL, GROQ_MODELS } from "@/core/services/stt/sttOptions";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputRustAudioInput, InputSelect, InputText } from "../components/input";
import { VoiceDetection } from "./VoiceDetection";

/**
 * OpenAI-compatible transcription server, or the Groq preset of the same backend. Requests are sent by Sigil's Rust
 * speech pipeline (one per sentence), so servers on the local network work as well as ones on this PC.
 */
export const OpenAIAudio: FC = () => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.openai_audio);
  const up = <K extends keyof STT_State["openai_audio"]>(key: K, v: STT_State["openai_audio"][K]) =>
    window.ApiServer.state.services.stt.data.openai_audio[key] = v;
  const groq = pr.preset === OpenAiAudioPreset.groq;
  const baseUrlInvalid = !groq && pr.baseUrl.trim() !== "" && normalizeEndpointUrl(pr.baseUrl) === null;

  return <>
    {groq ? <>
      <Inspector.SubHeader>{t("stt.groq_title")}</Inspector.SubHeader>
      <div className="text-xs text-base-content/70 mb-2">
        {t("stt.groq_desc")}{" "}
        <a className="link link-primary link-hover" target="_blank" rel="noopener noreferrer" href={GROQ_KEYS_URL}>{t("stt.groq_get_key")}</a>
      </div>
      <InputText label="stt.groq_api_key" type="password" value={pr.groqApiKey} onChange={e => up("groqApiKey", e.target.value)} />
      <InputSelect
        label="stt.groq_model"
        value={pr.groqModel || GROQ_MODELS[0]}
        options={[
          { value: "whisper-large-v3-turbo", label: t("stt.groq_model_turbo") },
          { value: "whisper-large-v3", label: t("stt.groq_model_large") },
        ]}
        onValueChange={e => up("groqModel", e)}
      />
      <Inspector.Description>{t("stt.groq_limits_paced")}</Inspector.Description>
      <Inspector.Description>{t("stt.groq_limits_merge")}</Inspector.Description>
    </> : <>
      <Inspector.SubHeader>{t("stt.openai_audio_title")}</Inspector.SubHeader>
      <div className="text-xs text-base-content/70 mb-2">
        {t("stt.openai_audio_desc")}
      </div>
      <InputText
        label="stt.openai_audio_base_url"
        value={pr.baseUrl}
        onChange={e => up("baseUrl", e.target.value)}
        placeholder="http://127.0.0.1:8000/v1"
      />
      {baseUrlInvalid
        ? <span role="alert" className="text-xs text-warning">{t("stt.error_http_base_url_invalid", { url: pr.baseUrl.trim() })}</span>
        : <Inspector.Description>{t("stt.openai_audio_lan_hint")}</Inspector.Description>}
      <InputText
        label="stt.openai_audio_model"
        value={pr.model}
        onChange={e => up("model", e.target.value)}
        placeholder={t("stt.openai_audio_model_placeholder")}
      />
      <InputText
        label="stt.openai_audio_api_key"
        type="password"
        value={pr.apiKey}
        onChange={e => up("apiKey", e.target.value)}
      />
    </>}
    <InputRustAudioInput
      label="common.field_input_device"
      value={pr.device}
      onChange={e => up("device", e)}
      hint="stt.rust_input_device_hint"
    />
    <InputText
      label="stt.openai_audio_language"
      value={pr.language}
      onChange={e => up("language", e.target.value)}
    />
    <Inspector.Description>{t("stt.openai_audio_language_desc")}</Inspector.Description>
    <InputText
      label="stt.openai_audio_prompt"
      value={pr.prompt}
      onChange={e => up("prompt", e.target.value)}
    />
    <Inspector.Description>{t("stt.openai_audio_prompt_desc")}</Inspector.Description>
    <VoiceDetection />
  </>;
};
