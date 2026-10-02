import { ttsSinkTarget } from "@/core/services/tts/pipeline";
import { TTS_Backends, type TTS_State } from "@/core/services/tts/schema";
import { GPT_SOVITS_LANGUAGES, GPT_SOVITS_STREAM_MODES } from "@/core/services/tts/schema/gptSovits";
import {
  GPT_SOVITS_DEFAULT_URL,
  GPT_SOVITS_REFERENCE_EXTENSIONS,
  checkGptSovits,
  gptSovitsSettings,
  isLocalServer,
  setGptSovitsWeights,
  speakGptSovits,
} from "@/core/services/tts/services/gptSovits";
import { parseExtra, previewToSink } from "@/core/services/tts/services/openaiSpeech";
import { errorMessage } from "@/core/services/tts/types";
import { ServiceNetworkState } from "@/types";
import { isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FC, useState } from "react";
import { useTranslation } from "react-i18next";
import { RiCheckLine, RiFolderOpenLine, RiPlayLine } from "react-icons/ri";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { ReferenceVoicePicker } from "../components/ReferenceVoiceRecorder";
import { InputCheckbox, InputCode, InputRange, InputSelect, InputText } from "../components/input";
import { TtsOutputDevice } from "./Routing";

type GptSovitsState = TTS_State["gptSovits"];

/** Label key of a GPT-SoVITS language code. */
function languageLabel(code: string): string {
  return `tts.gpt_sovits_lang_${code}`;
}

/**
 * Panel of the `gptSovits` backend (P6-G): a user-run GPT-SoVITS `api_v2.py` server. Server URL with a check, the GPT /
 * SoVITS weight paths (applied at start, or now with Apply), the reference (a voice-library voice for a server on this
 * PC, or a clip path on the server with its transcript), languages, streaming, speed, extra request fields, a test, and
 * the sink output device and volume. The URL and the weights restart the backend; the rest applies to the next sentence.
 */
export const GptSovits: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.gptSovits);
  const service = useSnapshot(window.ApiServer.tts.serviceState);
  const [checking, setChecking] = useState(false);
  const [applying, setApplying] = useState(false);
  const [testing, setTesting] = useState(false);
  const tauri = isTauri();

  const up = <K extends keyof GptSovitsState>(key: K, v: GptSovitsState[K]) =>
    window.ApiServer.patchService("tts", (s) => {
      s.data.gptSovits[key] = v;
    });

  const stopped = service.status === ServiceNetworkState.disconnected;
  const url = data.baseUrl.trim() || GPT_SOVITS_DEFAULT_URL;
  const local = isLocalServer(url);
  const extraInvalid = parseExtra(data.extra) === undefined;
  const languageOptions = GPT_SOVITS_LANGUAGES.map((code) => ({ label: t(languageLabel(code)), value: code }));

  const check = async () => {
    setChecking(true);
    try {
      const ok = await checkGptSovits(url);
      if (ok) toast.success(t("tts.gpt_sovits_check_ok", { url }));
      else toast.error(t("tts.gpt_sovits_unreachable", { url }));
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setChecking(false);
    }
  };

  const applyWeights = async () => {
    setApplying(true);
    try {
      await setGptSovitsWeights(url, data.gptWeights, data.sovitsWeights);
      toast.success(t("tts.gpt_sovits_weights_applied"));
    } catch (e) {
      toast.error(t("tts.gpt_sovits_weights_failed", { error: errorMessage(e) }));
    } finally {
      setApplying(false);
    }
  };

  /** Pick a clip on this PC (a server on this PC reads it from disk; api_v2 has no upload). */
  const browseReference = async () => {
    try {
      const picked = await open({
        multiple: false,
        directory: false,
        title: t("tts.gpt_sovits_ref_browse_title"),
        filters: [{ name: t("tts.gpt_sovits_ref_browse_filter"), extensions: [...GPT_SOVITS_REFERENCE_EXTENSIONS] }],
      });
      if (typeof picked === "string" && picked) up("refAudioPath", picked);
    } catch (e) {
      toast.error(errorMessage(e));
    }
  };

  const test = async () => {
    const state = window.ApiServer.state.services.tts.data;
    const settings = gptSovitsSettings(state.gptSovits);
    if (typeof settings === "string") {
      toast.error(settings);
      return;
    }
    setTesting(true);
    try {
      const target = ttsSinkTarget({ ...state, backend: TTS_Backends.gptSovits }, window.ApiServer.state.audioOutputDevice ?? "");
      const text = t("tts.openai_speech_test_text");
      await previewToSink(target, (id) => speakGptSovits(settings, text, id, null));
    } catch (e) {
      toast.error(t("tts.openai_speech_test_failed", { error: errorMessage(e) }));
    } finally {
      setTesting(false);
    }
  };

  return (
    <>
      <Inspector.SubHeader>{t("tts.gpt_sovits_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("tts.gpt_sovits_desc")}</Inspector.Description>
      <div className="text-xs text-base-content/70 flex flex-wrap gap-1 items-baseline">
        <span>{t("tts.openai_speech_start_with")}</span>
        <code className="font-mono select-all break-all" dir="ltr">
          python api_v2.py -a 127.0.0.1 -p 9880
        </code>
      </div>
      <Inspector.Deactivatable active={stopped}>
        <div className="flex gap-2 items-end">
          <div className="flex-1 min-w-0">
            <InputText
              label="tts.openai_speech_base_url"
              value={data.baseUrl}
              placeholder={GPT_SOVITS_DEFAULT_URL}
              dir="ltr"
              onChange={(e) => up("baseUrl", e.target.value)}
            />
          </div>
          <button type="button" className="btn btn-sm btn-outline mb-1 gap-1" disabled={!tauri || checking} onClick={() => void check()}>
            {checking ? <span className="loading loading-spinner loading-xs" /> : <RiCheckLine aria-hidden />}
            {t("tts.gpt_sovits_check")}
          </button>
        </div>
        <InputText
          label="tts.gpt_sovits_gpt_weights"
          value={data.gptWeights}
          placeholder="GPT_weights_v2Pro/MyVoice-e15.ckpt"
          dir="ltr"
          onChange={(e) => up("gptWeights", e.target.value)}
        />
        <InputText
          label="tts.gpt_sovits_sovits_weights"
          value={data.sovitsWeights}
          placeholder="SoVITS_weights_v2Pro/MyVoice_e8_s200.pth"
          dir="ltr"
          onChange={(e) => up("sovitsWeights", e.target.value)}
        />
        <div className="flex items-center justify-between gap-2">
          <div className="text-xs text-base-content/60">{t("tts.gpt_sovits_weights_hint")}</div>
          <button
            type="button"
            className="btn btn-xs btn-outline shrink-0"
            disabled={!tauri || applying || (!data.gptWeights.trim() && !data.sovitsWeights.trim())}
            onClick={() => void applyWeights()}
          >
            {applying && <span className="loading loading-spinner loading-xs" />}
            {t("tts.gpt_sovits_apply_weights")}
          </button>
        </div>
      </Inspector.Deactivatable>

      <Inspector.SubHeader>{t("tts.gpt_sovits_reference")}</Inspector.SubHeader>
      <div className={local ? undefined : "pointer-events-none opacity-50"}>
        <ReferenceVoicePicker value={data.referenceVoiceId} onChange={(id) => up("referenceVoiceId", id)} needsTranscript />
      </div>
      {!local && <div className="text-xs text-base-content/60">{t("tts.gpt_sovits_library_needs_local")}</div>}
      {(!data.referenceVoiceId || !local) && (
        <div className="flex gap-2 items-end">
          <div className="flex-1 min-w-0">
            <InputText
              label="tts.gpt_sovits_ref_path"
              value={data.refAudioPath}
              placeholder="D:/GPT-SoVITS/refs/voice.wav"
              dir="ltr"
              onChange={(e) => up("refAudioPath", e.target.value)}
            />
          </div>
          {local && (
            <button type="button" className="btn btn-sm btn-outline mb-1 gap-1" disabled={!tauri} onClick={() => void browseReference()}>
              <RiFolderOpenLine aria-hidden />
              {t("tts.gpt_sovits_ref_browse")}
            </button>
          )}
        </div>
      )}
      <InputText
        label="tts.gpt_sovits_prompt_text"
        value={data.promptText}
        placeholder={data.referenceVoiceId && local ? t("tts.gpt_sovits_prompt_from_library") : ""}
        onChange={(e) => up("promptText", e.target.value)}
      />
      <InputSelect label="tts.gpt_sovits_prompt_lang" value={data.promptLang || "auto"} options={languageOptions} onValueChange={(v) => up("promptLang", v)} />
      <div className="text-xs text-base-content/60">{t("tts.gpt_sovits_reference_hint")}</div>

      <Inspector.SubHeader>{t("tts.gpt_sovits_speech")}</Inspector.SubHeader>
      <InputSelect label="tts.gpt_sovits_text_lang" value={data.textLang || "auto"} options={languageOptions} onValueChange={(v) => up("textLang", v)} />
      <InputCheckbox label="tts.gpt_sovits_streaming" value={data.streaming} onChange={(v) => up("streaming", v)} />
      {data.streaming && (
        <InputSelect
          label="tts.gpt_sovits_streaming_mode"
          value={data.streamingMode}
          options={GPT_SOVITS_STREAM_MODES.map((m) => ({ label: t(`tts.gpt_sovits_streaming_${m}`), value: m }))}
          onValueChange={(v) => up("streamingMode", v as GptSovitsState["streamingMode"])}
        />
      )}
      <InputRange label="tts.openai_speech_speed" value={data.speed} min="0.5" max="2" step="0.05" onChange={(e) => up("speed", e.target.value)} />
      <div className="flex justify-end">
        <button type="button" className="btn btn-sm btn-outline gap-1" disabled={!tauri || testing} onClick={() => void test()}>
          {testing ? <span className="loading loading-spinner loading-xs" /> : <RiPlayLine />}
          {t("tts.openai_speech_test")}
        </button>
      </div>
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={(v) => up("device", v)} />
      <InputRange label="common.field_volume" value={data.volume} min="0" max="2" step="0.05" onChange={(e) => up("volume", e.target.value)} />

      <Inspector.SubHeader>{t("tts.openai_speech_advanced")}</Inspector.SubHeader>
      <InputText label="tts.openai_speech_timeout" type="number" min="0" max="600" step="1" value={data.timeout} onChange={(e) => up("timeout", e.target.value)} />
      <div className="text-xs text-base-content/60">{t("tts.openai_speech_timeout_hint")}</div>
      <InputCode
        label="tts.openai_speech_extra"
        language="json"
        value={data.extra}
        placeholder='{"top_k": 15, "temperature": 1, "text_split_method": "cut5"}'
        onChange={(v) => up("extra", v ?? "")}
      />
      {extraInvalid && <div className="text-xs text-error">{t("tts.openai_speech_extra_invalid")}</div>}
      <div className="text-xs text-base-content/60">{t("tts.gpt_sovits_extra_hint")}</div>
    </>
  );
};
