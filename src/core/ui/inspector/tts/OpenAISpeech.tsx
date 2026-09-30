import { ttsSinkTarget } from "@/core/services/tts/pipeline";
import { TTS_Backends, type TTS_State } from "@/core/services/tts/schema";
import { OPENAI_SPEECH_PRESET_IDS, type OpenAiSpeechPresetId } from "@/core/services/tts/schema/openaiSpeech";
import {
  applyOpenAiSpeechPreset,
  listOpenAiVoices,
  openAiSpeechPreset,
  openAiSpeechSettings,
  parseExtra,
  previewToSink,
  speakOpenAi,
} from "@/core/services/tts/services/openaiSpeech";
import { errorMessage } from "@/core/services/tts/types";
import { ServiceNetworkState } from "@/types";
import { isTauri } from "@tauri-apps/api/core";
import { FC, useId, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { RiPlayLine } from "react-icons/ri";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputCheckbox, InputCode, InputRange, InputSelect, InputText } from "../components/input";
import { TtsOutputDevice } from "./Routing";

type OpenAiSpeechState = TTS_State["openaiSpeech"];

/** i18n key of each preset's name. */
const PRESET_LABELS: Record<OpenAiSpeechPresetId, string> = {
  kokoroFastapi: "tts.openai_speech_preset_kokoro",
  chatterbox: "tts.openai_speech_preset_chatterbox",
  voxtral: "tts.openai_speech_preset_voxtral",
  indexTts: "tts.openai_speech_preset_indextts",
  voxcpm2: "tts.openai_speech_preset_voxcpm2",
  qwentts: "tts.openai_speech_preset_qwentts",
  custom: "tts.openai_speech_preset_custom",
};

/**
 * Panel of the `openaiSpeech` backend: any OpenAI-compatible `/v1/audio/speech` server (Rust `openai-speech` plugin).
 * A server preset fills URL, model, voice, format, sample rate, streaming and extra fields; the voice list comes from
 * the server when it offers one (typing a voice id always works); "Test voice" speaks one line through the sink. The
 * connection fields are locked while the backend runs; voice, speed, device and volume apply to the next sentence.
 */
export const OpenAISpeech: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.openaiSpeech);
  const service = useSnapshot(window.ApiServer.tts.serviceState);
  const [voices, setVoices] = useState<string[] | null>(null);
  const [loadingVoices, setLoadingVoices] = useState(false);
  const [testing, setTesting] = useState(false);
  const voiceListId = useId();
  const tauri = isTauri();

  const up = <K extends keyof OpenAiSpeechState>(key: K, v: OpenAiSpeechState[K]) =>
    window.ApiServer.patchService("tts", (s) => {
      s.data.openaiSpeech[key] = v;
    });

  const stopped = service.status === ServiceNetworkState.disconnected;
  const preset = openAiSpeechPreset(data.preset);
  const extraInvalid = parseExtra(data.extra) === undefined;

  const presetOptions = useMemo(
    () => OPENAI_SPEECH_PRESET_IDS.map((id) => ({ label: t(PRESET_LABELS[id]), value: id })),
    [t],
  );

  const choosePreset = (id: string) => {
    if (!(OPENAI_SPEECH_PRESET_IDS as readonly string[]).includes(id)) return;
    window.ApiServer.patchService("tts", (s) => {
      // Mutate the slice in place: replacing the object would detach the panel's snapshot from it.
      Object.assign(s.data.openaiSpeech, applyOpenAiSpeechPreset(s.data.openaiSpeech, id as OpenAiSpeechPresetId));
    });
    setVoices(null);
  };

  const refreshVoices = async () => {
    setLoadingVoices(true);
    try {
      const list = await listOpenAiVoices(data.baseUrl, data.apiKey);
      setVoices(list);
      if (!list.length) toast.info(t("tts.openai_speech_no_voices"));
    } catch {
      toast.error(t("tts.http_list_fetch_failed"));
    } finally {
      setLoadingVoices(false);
    }
  };

  const testVoice = async () => {
    const state = window.ApiServer.state.services.tts.data;
    const settings = openAiSpeechSettings(state.openaiSpeech);
    if (typeof settings === "string") {
      toast.error(settings);
      return;
    }
    setTesting(true);
    try {
      const target = ttsSinkTarget({ ...state, backend: TTS_Backends.openaiSpeech }, window.ApiServer.state.audioOutputDevice ?? "");
      await previewToSink(target, (id) => speakOpenAi(settings, t("tts.openai_speech_test_text"), id));
    } catch (e) {
      toast.error(t("tts.openai_speech_test_failed", { error: errorMessage(e) }));
    } finally {
      setTesting(false);
    }
  };

  return (
    <>
      <Inspector.SubHeader>{t("tts.openai_speech_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("tts.openai_speech_desc")}</Inspector.Description>
      <Inspector.Deactivatable active={stopped}>
        <InputSelect label="tts.openai_speech_preset" value={data.preset} options={presetOptions} onValueChange={choosePreset} />
        {preset?.command && (
          <div className="text-xs text-base-content/70 flex flex-wrap gap-1 items-baseline">
            <span>{t("tts.openai_speech_start_with")}</span>
            <code className="font-mono select-all break-all" dir="ltr">{preset.command}</code>
          </div>
        )}
        <InputText
          label="tts.openai_speech_base_url"
          value={data.baseUrl}
          placeholder="http://127.0.0.1:8880/v1"
          dir="ltr"
          onChange={(e) => up("baseUrl", e.target.value)}
        />
        <InputText label="tts.openai_speech_api_key" type="password" autoComplete="off" value={data.apiKey} onChange={(e) => up("apiKey", e.target.value)} />
        <InputText label="tts.openai_speech_model" value={data.model} dir="ltr" onChange={(e) => up("model", e.target.value)} />
      </Inspector.Deactivatable>
      <div className="flex flex-wrap gap-2 items-end">
        <div className="flex-1 min-w-[12rem]">
          <InputText
            label="tts.field_voice"
            value={data.voice}
            list={voiceListId}
            placeholder={t("tts.openai_speech_voice_placeholder")}
            dir="ltr"
            onChange={(e) => up("voice", e.target.value)}
          />
        </div>
        <button
          type="button"
          className="btn btn-sm btn-primary mb-1"
          onClick={() => void refreshVoices()}
          disabled={!tauri || loadingVoices}
        >
          {loadingVoices && <span className="loading loading-spinner loading-xs" />}
          {t("tts.http_refresh_list")}
        </button>
      </div>
      <datalist id={voiceListId}>
        {(voices ?? []).map((v) => (
          <option key={v} value={v} />
        ))}
      </datalist>
      {voices && voices.length > 0 && (
        <div className="text-xs text-base-content/60">{t("tts.openai_speech_voices_loaded", { n: voices.length })}</div>
      )}
      <InputRange label="tts.openai_speech_speed" value={data.speed} min="0.5" max="2" step="0.05" onChange={(e) => up("speed", e.target.value)} />
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={(v) => up("device", v)} />
      <InputRange label="common.field_volume" value={data.volume} min="0" max="2" step="0.05" onChange={(e) => up("volume", e.target.value)} />
      <div className="flex justify-end">
        <button type="button" className="btn btn-sm btn-outline gap-1" disabled={!tauri || testing} onClick={() => void testVoice()}>
          {testing ? <span className="loading loading-spinner loading-xs" /> : <RiPlayLine />}
          {t("tts.openai_speech_test")}
        </button>
      </div>

      <Inspector.SubHeader>{t("tts.openai_speech_advanced")}</Inspector.SubHeader>
      <Inspector.Deactivatable active={stopped}>
        <InputSelect
          label="tts.openai_speech_format"
          value={data.format}
          options={[
            { label: t("tts.openai_speech_format_pcm"), value: "pcm" },
            { label: t("tts.openai_speech_format_wav"), value: "wav" },
          ]}
          onValueChange={(v) => up("format", v === "wav" ? "wav" : "pcm")}
        />
        {data.format === "pcm" && (
          <InputText
            label="tts.openai_speech_pcm_rate"
            type="number"
            min="8000"
            max="192000"
            step="50"
            value={data.pcmSampleRate}
            onChange={(e) => up("pcmSampleRate", e.target.value)}
          />
        )}
        <InputCheckbox label="tts.openai_speech_stream" value={data.stream} onChange={(v) => up("stream", v)} />
        <InputText
          label="tts.openai_speech_timeout"
          type="number"
          min="0"
          max="600"
          step="1"
          value={data.timeout}
          onChange={(e) => up("timeout", e.target.value)}
        />
        <div className="text-xs text-base-content/60">{t("tts.openai_speech_timeout_hint")}</div>
        <InputCode
          label="tts.openai_speech_extra"
          language="json"
          value={data.extra}
          placeholder='{"lang_code": "a"}'
          onChange={(v) => up("extra", v ?? "")}
        />
        {extraInvalid && <div className="text-xs text-error">{t("tts.openai_speech_extra_invalid")}</div>}
        <div className="text-xs text-base-content/60">{t("tts.openai_speech_extra_hint")}</div>
      </Inspector.Deactivatable>
    </>
  );
};
