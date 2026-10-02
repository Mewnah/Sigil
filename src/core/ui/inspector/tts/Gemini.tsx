import { ttsSinkTarget } from "@/core/services/tts/pipeline";
import { TTS_Backends, type TTS_State } from "@/core/services/tts/schema";
import {
  GEMINI_DEFAULT_MODEL,
  GEMINI_DEFAULT_VOICE,
  GEMINI_KEY_URL,
  GEMINI_STYLE_PRESETS,
  GEMINI_TTS_MODELS,
  GEMINI_VOICES,
  checkGemini,
  previewGeminiVoice,
} from "@/core/services/tts/services/gemini";
import { errorMessage } from "@/core/services/tts/types";
import { ServiceNetworkState } from "@/types";
import { isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-shell";
import { FC, useCallback, useId, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { RiExternalLinkLine } from "react-icons/ri";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputRange, InputSelect, InputText } from "../components/input";
import { CloudFallbackSection, CloudPreview, Hint } from "./Edge";
import { TtsOutputDevice } from "./Routing";

type GeminiState = TTS_State["gemini"];

/** Select value of "another model id" (never a real model id: they have no spaces). */
const CUSTOM_MODEL = "custom model";
/** Select value of "another voice" (never a real voice name or id: they have no spaces). */
const CUSTOM_VOICE = "custom voice";

/**
 * Panel of the `gemini` backend (P6-H): Google Gemini TTS on the user's own (free-tier) key. Key (with a link to get
 * one and a check), model (Gemini 3.8 Flash / Flash-Lite TTS or any model id), one of 30 voices (or any voice name or
 * id: Google's extended voice library, a designed voice), a style instruction, a test sentence, the local fallback
 * voice, the API URL for a proxy, and the sink output. Everything applies to the next sentence.
 */
export const Gemini: FC = () => {
  const { t } = useTranslation();
  const tauri = isTauri();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.gemini) as GeminiState;
  const service = useSnapshot(window.ApiServer.tts.serviceState);
  const model = data.model.trim() || GEMINI_DEFAULT_MODEL;
  const listedModel = GEMINI_TTS_MODELS.some((m) => m.id === model);
  const [customModel, setCustomModel] = useState(!listedModel);
  const voice = data.voice.trim() || GEMINI_DEFAULT_VOICE;
  const [customVoice, setCustomVoice] = useState(!GEMINI_VOICES.some((v) => v.name === voice));
  const [checking, setChecking] = useState(false);
  const styleId = useId();

  const up = <K extends keyof GeminiState>(key: K, v: GeminiState[K]) =>
    window.ApiServer.patchService("tts", (s) => {
      s.data.gemini[key] = v;
    });

  const stopped = service.status === ServiceNetworkState.disconnected;
  const hasKey = !!data.apiKey.trim();

  const modelOptions = useMemo(
    () => [
      ...GEMINI_TTS_MODELS.map((m) => ({ label: m.label ? t(m.label) : m.id, value: m.id })),
      { label: t("tts.gemini_model_custom"), value: CUSTOM_MODEL },
    ],
    [t],
  );

  const voiceOptions = useMemo(
    () => [
      ...GEMINI_VOICES.map((v) => ({
        label: t("tts.gemini_voice_option", { name: v.name, style: t(`tts.gemini_voice_style_${v.style}`) }),
        value: v.name,
      })),
      { label: t("tts.gemini_voice_custom"), value: CUSTOM_VOICE },
    ],
    [t],
  );

  const chooseVoice = (v: string) => {
    if (v === CUSTOM_VOICE) {
      setCustomVoice(true);
      return;
    }
    setCustomVoice(false);
    up("voice", v);
  };

  const chooseModel = (v: string) => {
    if (v === CUSTOM_MODEL) {
      setCustomModel(true);
      return;
    }
    setCustomModel(false);
    up("model", v);
  };

  const check = async () => {
    setChecking(true);
    try {
      const ok = await checkGemini(window.ApiServer.state.services.tts.data.gemini);
      if (ok) toast.success(t("tts.gemini_check_ok"));
      else toast.warning(t("tts.gemini_unreachable"));
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setChecking(false);
    }
  };

  const getKey = () => {
    open(GEMINI_KEY_URL).catch((e) => toast.error(errorMessage(e)));
  };

  const preview = useCallback(async (text: string) => {
    const state = window.ApiServer.state.services.tts.data;
    const target = ttsSinkTarget({ ...state, backend: TTS_Backends.gemini }, window.ApiServer.state.audioOutputDevice ?? "");
    await previewGeminiVoice(state.gemini, text, target);
  }, []);

  return (
    <>
      <Inspector.SubHeader>{t("tts.gemini_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("tts.gemini_desc")}</Inspector.Description>
      {!tauri && <Hint>{t("tts.cloud_desktop_only")}</Hint>}

      <InputText
        label="tts.gemini_api_key"
        type="password"
        autoComplete="off"
        value={data.apiKey}
        placeholder={t("tts.gemini_api_key_placeholder")}
        onChange={(e) => up("apiKey", e.target.value)}
      />
      <div className="flex flex-wrap gap-2 justify-end">
        <button type="button" className="btn btn-xs btn-ghost gap-1" disabled={!tauri} onClick={getKey}>
          <RiExternalLinkLine /> {t("tts.gemini_get_key")}
        </button>
        <button type="button" className="btn btn-xs btn-outline" disabled={!tauri || !hasKey || checking} onClick={() => void check()}>
          {checking && <span className="loading loading-spinner loading-xs" />}
          {t("tts.gemini_check")}
        </button>
      </div>
      {!hasKey && <Hint>{t("tts.gemini_key_hint")}</Hint>}

      <InputSelect label="tts.gemini_model" value={customModel ? CUSTOM_MODEL : model} options={modelOptions} onValueChange={chooseModel} />
      {customModel && (
        <InputText
          label="tts.gemini_model_id"
          value={data.model}
          placeholder={GEMINI_DEFAULT_MODEL}
          dir="ltr"
          onChange={(e) => up("model", e.target.value)}
        />
      )}
      <Hint>{t("tts.gemini_model_hint")}</Hint>

      <InputSelect label="tts.field_voice" value={customVoice ? CUSTOM_VOICE : voice} options={voiceOptions} onValueChange={chooseVoice} />
      {customVoice && (
        <>
          <InputText
            label="tts.gemini_voice_id"
            value={data.voice}
            placeholder={GEMINI_DEFAULT_VOICE}
            dir="ltr"
            onChange={(e) => up("voice", e.target.value)}
          />
          <Hint>{t("tts.gemini_voice_id_hint")}</Hint>
        </>
      )}
      <div className="flex flex-col gap-1">
        <label className="font-medium text-base-content/80 text-xs" htmlFor={styleId}>
          {t("tts.gemini_style")}
        </label>
        <input
          id={styleId}
          type="text"
          className="input input-bordered input-sm w-full text-xs"
          value={data.style}
          placeholder={t("tts.gemini_style_placeholder")}
          onChange={(e) => up("style", e.target.value)}
        />
        <div className="flex flex-wrap gap-1">
          {GEMINI_STYLE_PRESETS.map((p) => (
            <button
              key={p.key}
              type="button"
              className={data.style.trim() === p.style ? "btn btn-xs btn-primary" : "btn btn-xs btn-ghost"}
              onClick={() => up("style", data.style.trim() === p.style ? "" : p.style)}
            >
              {t(`tts.gemini_style_preset_${p.key}`)}
            </button>
          ))}
        </div>
        <Hint>{t("tts.gemini_style_hint")}</Hint>
      </div>

      <CloudPreview
        placeholder={t("tts.gemini_preview_default")}
        disabled={!tauri || !hasKey}
        title={hasKey ? undefined : t("tts.gemini_key_hint")}
        onPreview={preview}
      />
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={(v) => up("device", v)} />
      <InputRange label="common.field_volume" value={data.volume} min="0" max="2" step="0.05" onChange={(e) => up("volume", e.target.value)} />

      <CloudFallbackSection value={data} stopped={stopped} onChange={(key, v) => up(key, v as GeminiState[typeof key])} />

      <Inspector.SubHeader>{t("tts.gemini_advanced")}</Inspector.SubHeader>
      <InputText
        label="tts.gemini_base_url"
        value={data.baseUrl}
        placeholder="https://generativelanguage.googleapis.com"
        dir="ltr"
        onChange={(e) => up("baseUrl", e.target.value)}
      />
      <Hint>{t("tts.gemini_base_url_hint")}</Hint>
    </>
  );
};
