import { STT_Backends, STT_State } from "@/core/services/stt/schema";
import { ServiceNetworkState } from "@/types";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { FC, useState, useEffect } from "react";
import { RiCharacterRecognitionFill, RiUserVoiceFill } from "react-icons/ri";
import { SiGooglechrome, SiMicrosoftedge } from "react-icons/si";
import { useSnapshot } from "valtio";
import { azureLanguages, deepGramLangs, nativeLangs } from "../../services/stt/stt_data";
import { VOSK_MODELS_FALLBACK, type VoskCatalogModel } from "../../services/stt/services/vosk";
import ServiceButton from "../service-button";
import Inspector from "./components";
import { InputCheckbox, InputMappedGroupSelect, InputSelect, InputText, InputWebAudioInput } from "./components/input";
import NiceModal from "@ebay/nice-modal-react";
import { useTranslation } from 'react-i18next';

// Rust-based input device selector
interface AudioDevice {
  id: string;
  name: string;
}

const RustInputDeviceSelect: FC<{ label: string; value: string; onChange: (value: string) => void }> = ({ label, value, onChange }) => {
  const [devices, setDevices] = useState<AudioDevice[]>([]);

  useEffect(() => {
    invoke<AudioDevice[]>("plugin:audio|list_input_devices")
      .then(setDevices)
      .catch(e => console.error("Failed to load input devices:", e));
  }, []);

  return (
    <InputSelect
      label={label}
      value={value}
      onValueChange={onChange}
      options={[
        { label: "System Default", value: "" },
        ...devices.map(d => ({ label: d.name, value: d.id }))
      ]}
    />
  );
};

const Native: FC = () => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.native);
  const updateLanguage = (value: { group: string, option: string }) => {
    window.ApiServer.state.services.stt.data.native.language = value.option;
    window.ApiServer.state.services.stt.data.native.language_group = value.group;
  };
  return <>
    <Inspector.SubHeader>{t('stt.native_title')}</Inspector.SubHeader>
    <InputMappedGroupSelect
      labelGroup="common.field_language"
      labelOption="common.field_dialect"
      value={{ option: pr.language, group: pr.language_group }}
      onChange={updateLanguage}
      library={nativeLangs} />
  </>
}

const micPageUrl = () => {
  const p = String(window.Config.serverNetwork.port);
  return `http://127.0.0.1:${p}/mic.html?port=${encodeURIComponent(p)}`;
};

const Chrome: FC = () => {
  const { t } = useTranslation();
  const handleOpenChrome = () => {
    invoke("plugin:web|open_browser", {
      data: {
        browser: "chrome",
        url: micPageUrl()
      }
    });
  };

  return <>
    <Inspector.SubHeader>{t('stt.browser_title')}</Inspector.SubHeader>
    <button className="btn btn-sm btn-neutral gap-2" onClick={handleOpenChrome}><SiGooglechrome /> Open Chrome</button>
  </>
}

const Edge: FC = () => {
  const { t } = useTranslation();
  const handleOpenEdge = () => {
    invoke("plugin:web|open_browser", {
      data: {
        browser: "edge",
        url: micPageUrl()
      }
    });
  };

  return <>
    <Inspector.SubHeader>{t('stt.browser_title')}</Inspector.SubHeader>
    <button className="btn btn-sm btn-neutral gap-2" onClick={handleOpenEdge}><SiMicrosoftedge /> Open Edge</button>
  </>
}


const Azure: FC = () => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.azure);
  const up = <K extends keyof STT_State["azure"]>(key: K, v: STT_State["azure"][K]) => window.ApiServer.state.services.stt.data.azure[key] = v;

  const updateLanguage = (value: { group: string, option: string }) => {
    window.ApiServer.state.services.stt.data.azure.language = value.option;
    window.ApiServer.state.services.stt.data.azure.language_group = value.group;
  };

  const updateSecondaryLanguage = (value: { group: string, option: string }) => {
    window.ApiServer.state.services.stt.data.azure.secondary_language = value.option;
    window.ApiServer.state.services.stt.data.azure.secondary_language_group = value.group;
  };

  return <>
    <Inspector.SubHeader>{t('stt.azure_title')}</Inspector.SubHeader>
    <InputText label="stt.azure_key" type="password" value={pr.key} onChange={e => up("key", e.target.value)} />
    <InputText label="stt.azure_location" value={pr.location} onChange={e => up("location", e.target.value)} />

    <InputWebAudioInput value={pr.device} onChange={e => up("device", e)} label="common.field_input_device" />

    <div className=" divider"></div>
    <InputMappedGroupSelect
      labelGroup="common.field_language"
      labelOption="common.field_dialect"
      value={{ option: pr.language, group: pr.language_group }}
      onChange={updateLanguage}
      library={azureLanguages} />
    <InputCheckbox label="stt.azure_use_secondary_language" onChange={e => up("use_secondary_language", e)} value={pr.use_secondary_language} />
    <Inspector.Switchable visible={pr.use_secondary_language}>
      <InputMappedGroupSelect
        labelGroup="stt.azure_secondary_language"
        labelOption="common.field_dialect"
        value={{ option: pr.secondary_language, group: pr.secondary_language_group }}
        onChange={updateSecondaryLanguage}
        library={azureLanguages} />
    </Inspector.Switchable>

    <InputSelect
      label="stt.azure_profanity"
      options={[
        { label: t('stt.azure_profanity_masked'), value: 'masked' },
        { label: t('stt.azure_profanity_removed'), value: 'removed' },
        { label: t('stt.azure_profanity_raw'), value: 'raw' },
      ]}
      value={pr.profanity}
      onValueChange={e => up("profanity", e)}
    />
    <InputText type="number" step="1" label="stt.azure_silence_timeout" value={pr.silenceTimeout} onChange={e => up("silenceTimeout", e.target.value)} />
    <InputCheckbox label="stt.field_enable_interim_results" onChange={e => up("interim", e)} value={pr.interim} />
  </>
}

const Deepgram: FC = () => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.deepgram);
  const up = <K extends keyof STT_State["deepgram"]>(key: K, v: STT_State["deepgram"][K]) => window.ApiServer.state.services.stt.data.deepgram[key] = v;

  const updateLanguage = (value: { group: string, option: string }) => {
    window.ApiServer.state.services.stt.data.deepgram.language = value.option;
    window.ApiServer.state.services.stt.data.deepgram.language_group = value.group;
  };

  return <>
    <Inspector.SubHeader>{t('stt.deepgram_title')}</Inspector.SubHeader>
    <InputText label="stt.deepgram_key" type="password" value={pr.key} onChange={e => up("key", e.target.value)} />

    <RustInputDeviceSelect value={pr.device} onChange={e => up("device", e)} label="common.field_input_device" />
    <InputMappedGroupSelect
      labelGroup="common.field_language"
      labelOption="common.field_dialect"
      value={{ option: pr.language, group: pr.language_group }}
      onChange={updateLanguage}
      library={deepGramLangs} />

    <InputSelect options={[
      { label: 'Base', value: 'base' },
      { label: 'Enhanced', value: 'enhanced' },
    ]} label="stt.deepgram_quality" value={pr.tier} onValueChange={e => up("tier", e)} />

    <span className="text-base-content/60 text-xs">
      {t('stt.deepgram_quality_notice')}
      <br />
      <a className="link link-primary link-hover" target="_blank" rel="noopener noreferrer" href="https://developers.deepgram.com/documentation/features/language/#language-options">{t('stt.deepgram_quality_notice_link')}</a>
    </span>

    <InputCheckbox label="stt.field_enable_interim_results" onChange={e => up("interim", e)} value={pr.interim} />
    <InputCheckbox label="stt.deepgram_profanity" onChange={e => up("profanity", e)} value={pr.profanity} />
    <InputCheckbox label="stt.deepgram_punctuate" onChange={e => up("punctuate", e)} value={pr.punctuate} />
  </>
}

type WhisperDownloadUi =
  | null
  | { file: string; progress: number; phase: "loading" }
  | { file: string; progress: number; phase: "complete" };

const Whisper: FC = () => {
  const { t } = useTranslation();
  const [downloadProgress, setDownloadProgress] = useState<WhisperDownloadUi>(null);
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.whisper);
  const up = <K extends keyof STT_State["whisper"]>(key: K, v: STT_State["whisper"][K]) => window.ApiServer.state.services.stt.data.whisper[key] = v;

  useEffect(() => {
    let dismissTimer: ReturnType<typeof setTimeout> | undefined;
    const unsubs: UnlistenFn[] = [];

    listen("whisper:download_progress", (event) => {
      const payload = event.payload as { file: string; progress: number };
      setDownloadProgress({ file: payload.file, progress: payload.progress, phase: "loading" });
    }).then((u) => unsubs.push(u));

    listen("whisper:download_complete", (event) => {
      const file = (event.payload as { file?: string }).file ?? "";
      if (dismissTimer) clearTimeout(dismissTimer);
      setDownloadProgress({ file, progress: 100, phase: "complete" });
      dismissTimer = setTimeout(() => setDownloadProgress(null), 2000);
    }).then((u) => unsubs.push(u));

    listen("whisper:download_dismiss", () => {
      if (dismissTimer) clearTimeout(dismissTimer);
      setDownloadProgress(null);
    }).then((u) => unsubs.push(u));

    return () => {
      if (dismissTimer) clearTimeout(dismissTimer);
      unsubs.forEach((u) => u());
    };
  }, []);

  return <>
    <Inspector.SubHeader>OpenAI Whisper</Inspector.SubHeader>
    <div className="text-xs text-base-content/70 mb-2">
      Runs locally using whisper.cpp. First run will download the selected model.
    </div>

    <RustInputDeviceSelect
      label="Input Device"
      value={pr.device}
      onChange={e => up("device", e)}
    />

    <InputSelect
      label="Model"
      value={pr.model}
      options={[
        { value: "tiny.en", label: "Tiny (English) - 75MB, Fastest" },
        { value: "tiny", label: "Tiny (Multilingual) - 75MB" },
        { value: "base.en", label: "Base (English) - 142MB, Fast" },
        { value: "base", label: "Base (Multilingual) - 142MB" },
        { value: "small.en", label: "Small (English) - 466MB, Balanced" },
        { value: "small", label: "Small (Multilingual) - 466MB" },
        { value: "medium.en", label: "Medium (English) - 1.5GB, Accurate" },
        { value: "medium", label: "Medium (Multilingual) - 1.5GB" },
      ]}
      onValueChange={e => up("model", e)}
    />
    <Inspector.Description>
      Larger models are more accurate but slower. English-only models are faster for English speech.
    </Inspector.Description>

    <InputSelect
      label="Language"
      value={pr.language}
      options={[
        { value: "en", label: "English" },
        { value: "auto", label: "Auto-detect" },
        { value: "es", label: "Spanish" },
        { value: "fr", label: "French" },
        { value: "de", label: "German" },
        { value: "it", label: "Italian" },
        { value: "pt", label: "Portuguese" },
        { value: "ru", label: "Russian" },
        { value: "ja", label: "Japanese" },
        { value: "ko", label: "Korean" },
        { value: "zh", label: "Chinese" },
      ]}
      onValueChange={e => up("language", e)}
    />
    <Inspector.Description>
      For ".en" models, language is always English. For multilingual models, you can select the language.
    </Inspector.Description>

    {downloadProgress && (
      <div className="flex flex-col gap-1 mb-2">
        {downloadProgress.phase === "complete" ? (
          <div className="flex items-center justify-between text-xs text-success">
            <span>{t("stt.whisper_download_complete", { file: downloadProgress.file })}</span>
            <span aria-hidden>✓</span>
          </div>
        ) : (
          <>
            <div className="flex justify-between text-xs">
              <span>{t("stt.whisper_downloading", { file: downloadProgress.file })}</span>
              <span>
                {downloadProgress.progress > 0 ? `${downloadProgress.progress.toFixed(0)}%` : "…"}
              </span>
            </div>
            <progress
              className="progress progress-primary w-full"
              value={downloadProgress.progress > 0 ? downloadProgress.progress : undefined}
              max={100}
            />
          </>
        )}
      </div>
    )}


    <Inspector.SubHeader>Voice Activity Detection</Inspector.SubHeader>
    <InputCheckbox
      label="Enable VAD (Auto-transcribe on silence)"
      onChange={e => up("vadEnabled", e)}
      value={pr.vadEnabled}
    />
    <Inspector.Description>
      When enabled, transcription triggers automatically when you pause speaking. Disable for manual control.
    </Inspector.Description>

    <Inspector.Switchable visible={pr.vadEnabled}>
      <InputText
        type="number"
        step="1"
        label="Silence Threshold (dB)"
        value={pr.silenceThresholdDb}
        onChange={e => up("silenceThresholdDb", e.target.value)}
      />
      <Inspector.Description>
        RMS threshold in dB (-60 to -20). Lower = more sensitive. Default: -40dB
      </Inspector.Description>

      <InputText
        type="number"
        step="100"
        label="Silence Duration (ms)"
        value={pr.silenceDurationMs}
        onChange={e => up("silenceDurationMs", e.target.value)}
      />
      <Inspector.Description>
        How long to wait in silence before transcribing (500-3000ms). Default: 1500ms
      </Inspector.Description>

      <InputText
        type="number"
        step="100"
        label="Minimum Chunk Duration (ms)"
        value={pr.minChunkDurationMs}
        onChange={e => up("minChunkDurationMs", e.target.value)}
      />
      <Inspector.Description>
        Minimum audio length before VAD can trigger (500-2000ms). Default: 1000ms
      </Inspector.Description>
    </Inspector.Switchable>
  </>
}

const Moonshine: FC = () => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.moonshine);
  const up = <K extends keyof STT_State["moonshine"]>(key: K, v: STT_State["moonshine"][K]) =>
    window.ApiServer.state.services.stt.data.moonshine[key] = v;

  return <>
    <Inspector.SubHeader>{t("stt.moonshine_title")}</Inspector.SubHeader>
    <div className="text-xs text-base-content/70 mb-2">
      {t("stt.moonshine_desc")}
    </div>
    <InputText
      label="stt.moonshine_endpoint"
      value={pr.endpoint}
      onChange={e => up("endpoint", e.target.value)}
    />
    <RustInputDeviceSelect
      label="common.field_input_device"
      value={pr.device}
      onChange={e => up("device", e)}
    />
    <InputText
      label="stt.moonshine_language"
      value={pr.language}
      onChange={e => up("language", e.target.value)}
    />
    <Inspector.Description>{t("stt.moonshine_language_desc")}</Inspector.Description>
  </>;
};

const OpenAIAudio: FC = () => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.openai_audio);
  const up = <K extends keyof STT_State["openai_audio"]>(key: K, v: STT_State["openai_audio"][K]) =>
    window.ApiServer.state.services.stt.data.openai_audio[key] = v;

  return <>
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
    <RustInputDeviceSelect
      label="common.field_input_device"
      value={pr.device}
      onChange={e => up("device", e)}
    />
    <InputText
      label="stt.openai_audio_language"
      value={pr.language}
      onChange={e => up("language", e.target.value)}
    />
    <Inspector.Description>{t("stt.openai_audio_language_desc")}</Inspector.Description>
  </>;
};

const Vosk: FC = () => {
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.vosk);
  const up = <K extends keyof STT_State["vosk"]>(key: K, v: STT_State["vosk"][K]) => window.ApiServer.state.services.stt.data.vosk[key] = v;

  const [voskCatalog, setVoskCatalog] = useState<VoskCatalogModel[]>(VOSK_MODELS_FALLBACK);

  useEffect(() => {
    invoke<VoskCatalogModel[]>("plugin:vosk-stt|get_vosk_models")
      .then(setVoskCatalog)
      .catch(() => { /* keep fallback */ });
  }, []);

  const modelOptions = voskCatalog.map(m => ({
    label: `${m.name} (${m.size})`,
    value: m.id
  }));

  return <>
    <Inspector.SubHeader>Vosk</Inspector.SubHeader>
    <div className="text-xs text-base-content/70 mb-2">
      Models download once into app data as a zip, then load locally (WASM). Rust captures audio; no repeat CDN fetch for the same model.
    </div>

    <RustInputDeviceSelect
      label="Input Device"
      value={pr.device}
      onChange={e => up("device", e)}
    />

    <InputSelect
      label="Language Model"
      options={modelOptions}
      value={pr.model}
      onValueChange={e => up("model", e)}
    />
    <Inspector.Description>
      Models are downloaded once and cached. Larger models = better accuracy but slower.
    </Inspector.Description>

    <InputText
      label="Custom Model URL (optional)"
      placeholder="https://example.com/model.zip"
      value={pr.modelUrl}
      onChange={e => up("modelUrl", e.target.value)}
    />
    <Inspector.Description>
      Full URL to a .zip Vosk model. When set, it overrides the catalog URL for download; the file is still saved under app data as the selected model id.
    </Inspector.Description>
  </>
}



const Inspector_STT: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.stt);
  const state = useSnapshot(window.ApiServer.stt.serviceState);

  const handleStart = (v: boolean) => window.ApiServer.state.services.stt.showActionButton = v;
  const up = <K extends keyof STT_State>(key: K, v: STT_State[K]) => window.ApiServer.patchService("stt", s => s.data[key] = v);

  const handleShowReplacements = () => {
    NiceModal.show('stt-replacements');
  }

  return <Inspector.Body>
    <Inspector.Header><RiUserVoiceFill /> {t('stt.title')}</Inspector.Header>
    <Inspector.Content>
      <Inspector.SubHeader>Speech Provider</Inspector.SubHeader>
      <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
        <InputSelect options={[
          { label: "Native", value: STT_Backends.native },
          { label: "Chrome", value: STT_Backends.chrome },
          { label: "Edge", value: STT_Backends.edge },
          { label: "Azure", value: STT_Backends.azure },
          { label: "Deepgram", value: STT_Backends.deepgram },
          { label: "Whisper", value: STT_Backends.whisper },
          { label: "Vosk", value: STT_Backends.vosk },
          { label: t("stt.provider_moonshine"), value: STT_Backends.moonshine },
          { label: t("stt.provider_openai_audio"), value: STT_Backends.openai_audio },
        ]} label="common.field_service" value={data.data.backend} onValueChange={e => up("backend", e as STT_Backends)} />

        {data.data.backend !== STT_Backends.chrome && data.data.backend !== STT_Backends.edge && (
          <div className="mt-2 mb-4">
            <InputCheckbox label='stt.field_uwu_filter' onChange={e => up("uwu", e)} value={data.data.uwu} />
            <Inspector.Description>{t('stt.field_uwu_filter_desc')}</Inspector.Description>
          </div>
        )}

        {data.data.backend === STT_Backends.chrome && <Chrome />}
        {data.data.backend === STT_Backends.edge && <Edge />}
        {data.data.backend === STT_Backends.azure && <Azure />}
        {data.data.backend === STT_Backends.deepgram && <Deepgram />}
        {data.data.backend === STT_Backends.whisper && <Whisper />}
        {data.data.backend === STT_Backends.vosk && <Vosk />}
        {data.data.backend === STT_Backends.moonshine && <Moonshine />}
        {data.data.backend === STT_Backends.openai_audio && <OpenAIAudio />}
        {data.data.backend === STT_Backends.native && <Native />}
      </Inspector.Deactivatable>

      {data.data.backend !== STT_Backends.chrome && data.data.backend !== STT_Backends.edge && (
        <>
          <div className="pt-4 flex flex-col">
            <ServiceButton status={state.status} onStart={() => window.ApiServer.stt.start()} onStop={() => window.ApiServer.stt.stop()} />
          </div>


        </>
      )}




      <div className="contents">
        <div className="h-2" />
        <InputCheckbox label="common.field_action_bar" onChange={handleStart} value={data.showActionButton} />
        <InputCheckbox label="common.field_auto_start" value={data.data.autoStart} onChange={e => up("autoStart", e)} />
        <InputCheckbox label="stt.field_stop_with_stream" value={data.data.stopWithStream} onChange={e => up("stopWithStream", e)} />
        <div className="h-2" />
        <span className="link link-accent link-hover font-semibold flex items-center gap-2 text-sm justify-end" onClick={handleShowReplacements}><RiCharacterRecognitionFill />{t('common.btn_edit_replacements')}</span>
      </div>

    </Inspector.Content>
  </Inspector.Body>
}

export default Inspector_STT;
