import { OpenAiAudioPreset, STT_Backends, STT_State } from "@/core/services/stt/schema";
import { ServiceNetworkState } from "@/types";
import { invoke, isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { FC, ReactNode, useState, useEffect, useMemo } from "react";
import { RiCharacterRecognitionFill, RiUserVoiceFill } from "react-icons/ri";
import { SiGooglechrome, SiMicrosoftedge } from "react-icons/si";
import { useSnapshot } from "valtio";
import { azureLanguages, deepGramLangs, nativeLangs } from "../../services/stt/stt_data";
import { VOSK_MODELS_FALLBACK, type VoskCatalogModel } from "../../services/stt/services/vosk";
import { GROQ_KEYS_URL, GROQ_MODELS } from "../../services/stt/sttOptions";
import {
  findWhisperModel,
  formatModelSize,
  WHISPER_LANGUAGES,
  WHISPER_MODEL_OPTIONS,
  type WhisperModelFamily,
  type WhisperModelOption,
} from "../../services/stt/whisperModels";
import ServiceButton from "../service-button";
import Inspector from "./components";
import { InputCheckbox, InputMappedGroupSelect, InputRustAudioInput, InputSelect, InputText, InputWebAudioInput } from "./components/input";
import NiceModal from "@ebay/nice-modal-react";
import { useTranslation } from 'react-i18next';
import { toast } from "react-toastify";
import { localPageUrl } from "@/utils/localPages";
import "../stt-replacements-modal";

const detailsClass =
  "rounded-lg border border-base-content/10 [&_summary]:cursor-pointer [&_summary]:list-none [&_summary::-webkit-details-marker]:hidden";

/** Collapsible group for settings most users never need. */
const Advanced: FC<{ title: string; children: ReactNode }> = ({ title, children }) => (
  <details className={detailsClass}>
    <summary className="px-3 py-2 text-sm font-medium">{title}</summary>
    <div className="px-3 pb-3 pt-2 flex flex-col gap-2 border-t border-base-content/10">{children}</div>
  </details>
);

/** Language name in the UI language (`Intl.DisplayNames`), falling back to the code. */
function useLanguageName() {
  const { i18n } = useTranslation();
  return useMemo(() => {
    let names: Intl.DisplayNames | null = null;
    try {
      names = new Intl.DisplayNames([(i18n.language || "en").replace("_", "-")], { type: "language" });
    } catch {
      names = null;
    }
    return (code: string) => {
      try {
        return names?.of(code) ?? code;
      } catch {
        return code;
      }
    };
  }, [i18n.language]);
}

const Native: FC = () => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.native);
  const updateLanguage = (value: { group: string, option: string }) => {
    window.ApiServer.state.services.stt.data.native.language = value.option;
    window.ApiServer.state.services.stt.data.native.language_group = value.group;
  };
  return <>
    <Inspector.SubHeader>{t('stt.native_title')}</Inspector.SubHeader>
    {isTauri() && <Inspector.Description>{t("stt.native_desktop_hint")}</Inspector.Description>}
    <InputMappedGroupSelect
      labelGroup="common.field_language"
      labelOption="common.field_dialect"
      value={{ option: pr.language, group: pr.language_group }}
      onChange={updateLanguage}
      library={nativeLangs} />
  </>
}

/** mic.html opens in an external Chrome/Edge on 127.0.0.1 (a secure context for getUserMedia). */
const micPageUrl = () => localPageUrl("mic.html");

const ExternalBrowser: FC<{ browser: "chrome" | "edge" }> = ({ browser }) => {
  const { t } = useTranslation();
  const handleOpen = () => {
    void invoke("plugin:web|open_browser", {
      data: {
        browser,
        url: micPageUrl()
      }
    }).catch((e) => {
      const msg = e instanceof Error ? e.message : String(e);
      console.error("open_browser:", e);
      toast.error(`${t("integrations.oauth_open_browser_failed")} ${msg}`);
    });
  };

  return <>
    <Inspector.SubHeader>{t('stt.browser_title')}</Inspector.SubHeader>
    <button className="btn btn-sm btn-neutral gap-2" onClick={handleOpen}>
      {browser === "chrome" ? <><SiGooglechrome /> {t("stt.browser_open_chrome")}</> : <><SiMicrosoftedge /> {t("stt.browser_open_edge")}</>}
    </button>
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
    <InputText type="number" step="50" min="100" max="5000" label="stt.azure_segmentation_silence" value={pr.segmentationSilenceMs} onChange={e => up("segmentationSilenceMs", e.target.value)} />
    <Inspector.Description>{t("stt.azure_segmentation_silence_desc")}</Inspector.Description>
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

    <InputWebAudioInput value={pr.device} onChange={e => up("device", e)} label="common.field_input_device" />
    <InputMappedGroupSelect
      labelGroup="common.field_language"
      labelOption="common.field_dialect"
      value={{ option: pr.language, group: pr.language_group }}
      onChange={updateLanguage}
      library={deepGramLangs} />

    <InputSelect options={[
      { label: 'Nova-3', value: 'nova-3' },
      { label: 'Nova-2', value: 'nova-2' },
    ]} label="stt.deepgram_quality" value={pr.model || "nova-3"} onValueChange={e => up("model", e)} />

    <span className="text-base-content/60 text-xs">
      {t('stt.deepgram_quality_notice')}
      <br />
      <a className="link link-primary link-hover" target="_blank" rel="noopener noreferrer" href="https://developers.deepgram.com/docs/models-languages-overview">{t('stt.deepgram_quality_notice_link')}</a>
    </span>

    <InputCheckbox label="stt.field_enable_interim_results" onChange={e => up("interim", e)} value={pr.interim} />
    <InputCheckbox label="stt.deepgram_profanity" onChange={e => up("profanity", e)} value={pr.profanity} />
    <InputCheckbox label="stt.deepgram_punctuate" onChange={e => up("punctuate", e)} value={pr.punctuate} />
  </>
}

/** Silero VAD settings shared by Whisper, Moonshine and the HTTP / Groq engines. */
const VoiceDetection: FC = () => {
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

type WhisperDownloadUi =
  | null
  | { file: string; progress: number; phase: "loading" }
  | { file: string; progress: number; phase: "complete" };

const WHISPER_FAMILY_ORDER: WhisperModelFamily[] = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3", "anime-whisper"];

const Whisper: FC = () => {
  const { t } = useTranslation();
  const languageName = useLanguageName();
  const [downloadProgress, setDownloadProgress] = useState<WhisperDownloadUi>(null);
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.whisper);
  const sttState = useSnapshot(window.ApiServer.stt.serviceState);
  const up = <K extends keyof STT_State["whisper"]>(key: K, v: STT_State["whisper"][K]) => window.ApiServer.state.services.stt.data.whisper[key] = v;

  useEffect(() => {
    let dismissTimer: ReturnType<typeof setTimeout> | undefined;
    const unsubs: UnlistenFn[] = [];
    let disposed = false;
    const keep = (u: UnlistenFn) => (disposed ? u() : unsubs.push(u));

    listen("whisper:download_progress", (event) => {
      const payload = event.payload as { file: string; progress: number };
      setDownloadProgress({ file: payload.file, progress: payload.progress, phase: "loading" });
    }).then(keep);

    listen("whisper:download_complete", (event) => {
      const file = (event.payload as { file?: string }).file ?? "";
      if (dismissTimer) clearTimeout(dismissTimer);
      setDownloadProgress({ file, progress: 100, phase: "complete" });
      dismissTimer = setTimeout(() => setDownloadProgress(null), 2000);
    }).then(keep);

    listen("whisper:download_dismiss", () => {
      if (dismissTimer) clearTimeout(dismissTimer);
      setDownloadProgress(null);
    }).then(keep);

    return () => {
      disposed = true;
      if (dismissTimer) clearTimeout(dismissTimer);
      unsubs.forEach((u) => u());
    };
  }, []);

  const familyLabel = (family: WhisperModelFamily) => t(`stt.whisper_family_${family.replace(/-/g, "_")}`);
  const modelLabel = (o: WhisperModelOption) => {
    const parts = [familyLabel(o.family)];
    if (o.family === "anime-whisper") parts.push(t("stt.whisper_byo"));
    else parts.push(o.englishOnly ? t("stt.whisper_english_only") : t("stt.whisper_multilingual"));
    if (o.quant !== "f16") parts.push(o.quant.startsWith("q8") ? "Q8" : "Q5");
    return `${parts.join(" · ")} (${formatModelSize(o.sizeMb)})`;
  };
  const modelGroups = WHISPER_FAMILY_ORDER.map(family => ({
    label: familyLabel(family),
    options: WHISPER_MODEL_OPTIONS.filter(o => o.family === family).map(o => ({ value: o.id, label: modelLabel(o) })),
  }));
  const selected = findWhisperModel(pr.model);
  // Keep a stored id the list does not offer (older alias, hand-edited settings) selectable instead of blank.
  const modelOptions = selected
    ? modelGroups
    : [{ value: pr.model, label: pr.model }, ...modelGroups];

  const languageOptions = [
    { value: "auto", label: t("stt.whisper_language_auto") },
    ...WHISPER_LANGUAGES.map(code => ({ value: code, label: languageName(code) })),
  ];
  if (pr.language && pr.language !== "auto" && !WHISPER_LANGUAGES.includes(pr.language))
    languageOptions.push({ value: pr.language, label: languageName(pr.language) });

  const pickFile = async () => {
    const path = await open({
      filters: [{ name: "Whisper GGML", extensions: ["bin"] }],
      multiple: false,
    });
    return path === null || Array.isArray(path) ? null : path;
  };

  return <>
    <Inspector.SubHeader>{t("stt.whisper_title")}</Inspector.SubHeader>
    <Inspector.Description>{t("stt.whisper_desc")}</Inspector.Description>

    <InputRustAudioInput
      label="common.field_input_device"
      value={pr.device}
      onChange={e => up("device", e)}
      hint="stt.rust_input_device_hint"
    />

    <InputSelect
      label="stt.whisper_model"
      value={pr.model}
      options={modelOptions}
      onValueChange={e => up("model", e)}
    />
    <Inspector.Description>
      {selected?.bringYourOwn ? t("stt.whisper_anime_hint") : t("stt.whisper_model_desc")}
    </Inspector.Description>

    {!selected?.language && (
      <>
        <InputSelect
          label="common.field_language"
          value={selected?.englishOnly ? "en" : pr.language}
          disabled={!!selected?.englishOnly}
          options={languageOptions}
          onValueChange={e => up("language", e)}
        />
        <Inspector.Description>{t("stt.whisper_language_desc")}</Inspector.Description>
      </>
    )}

    <InputCheckbox
      label={t("stt.whisper_use_local")}
      onChange={(e) => up("useLocalModel", e)}
      value={pr.useLocalModel}
    />
    {pr.useLocalModel && (
      <div className="flex flex-col gap-2 mb-2">
        <div className="flex gap-2 items-center">
          <input
            type="text"
            className="input input-bordered input-sm flex-1 min-w-0 font-mono text-xs"
            value={pr.localModelPath}
            onChange={(e) => up("localModelPath", e.target.value)}
            placeholder={t("stt.whisper_local_path")}
            aria-label={t("stt.whisper_local_path")}
          />
          {isTauri() && (
            <button
              type="button"
              className="btn btn-sm btn-outline shrink-0"
              onClick={async () => {
                const path = await pickFile();
                if (path) up("localModelPath", path);
              }}
            >
              {t("stt.whisper_browse_bin")}
            </button>
          )}
        </div>
        {isTauri() && (
          <button
            type="button"
            className="btn btn-sm btn-neutral w-fit"
            onClick={async () => {
              const path = await pickFile();
              if (!path) return;
              try {
                await invoke("plugin:whisper|import_whisper_model", {
                  model: window.ApiServer.state.services.stt.data.whisper.model,
                  sourcePath: path,
                });
                toast.success(t("stt.whisper_import_ok"));
              } catch (e) {
                toast.error(String(e));
              }
            }}
          >
            {t("stt.whisper_import_cached")}
          </button>
        )}
        <Inspector.Description>{t("stt.whisper_local_hint")}</Inspector.Description>
      </div>
    )}

    {/* Import progress; while starting, the status line under the Start button shows the download. */}
    {downloadProgress && sttState.status !== ServiceNetworkState.connecting && (
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

    <Advanced title={t("stt.whisper_accuracy_title")}>
      <InputCheckbox label="stt.whisper_hallucination_filter" value={pr.hallucinationFilter} onChange={e => up("hallucinationFilter", e)} />
      <Inspector.Description>{t("stt.whisper_hallucination_filter_desc")}</Inspector.Description>
      <InputText type="number" step="0.05" min="0.05" max="1" label="stt.whisper_no_speech_threshold" value={pr.noSpeechThreshold} onChange={e => up("noSpeechThreshold", e.target.value)} />
      <Inspector.Description>{t("stt.whisper_no_speech_threshold_desc")}</Inspector.Description>
      {!selected?.bringYourOwn && <>
        <InputText label="stt.whisper_hotwords" value={pr.hotwords} placeholder={t("stt.whisper_hotwords_placeholder")} onChange={e => up("hotwords", e.target.value)} />
        <Inspector.Description>{t("stt.whisper_hotwords_desc")}</Inspector.Description>
        <InputText label="stt.whisper_initial_prompt" value={pr.initialPrompt} onChange={e => up("initialPrompt", e.target.value)} />
        <Inspector.Description>{t("stt.whisper_initial_prompt_desc")}</Inspector.Description>
      </>}
      <InputText type="number" step="1" min="0" max="16" label="stt.whisper_threads" value={pr.threads} onChange={e => up("threads", e.target.value)} />
      <Inspector.Description>{t("stt.whisper_threads_desc")}</Inspector.Description>
      <InputText type="number" step="1" min="1" max="10" label="stt.whisper_max_backlog" value={pr.maxBacklog} onChange={e => up("maxBacklog", e.target.value)} />
      <Inspector.Description>{t("stt.whisper_max_backlog_desc")}</Inspector.Description>
    </Advanced>
    <VoiceDetection />
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
    <InputRustAudioInput
      label="common.field_input_device"
      value={pr.device}
      onChange={e => up("device", e)}
      hint="stt.rust_input_device_hint"
    />
    <InputText
      label="stt.moonshine_language"
      value={pr.language}
      onChange={e => up("language", e.target.value)}
    />
    <Inspector.Description>{t("stt.moonshine_language_desc")}</Inspector.Description>
    <VoiceDetection />
  </>;
};

const OpenAIAudio: FC = () => {
  const { t } = useTranslation();
  const pr = useSnapshot(window.ApiServer.state.services.stt.data.openai_audio);
  const up = <K extends keyof STT_State["openai_audio"]>(key: K, v: STT_State["openai_audio"][K]) =>
    window.ApiServer.state.services.stt.data.openai_audio[key] = v;
  const groq = pr.preset === OpenAiAudioPreset.groq;

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
      <Inspector.Description>{t("stt.groq_limits")}</Inspector.Description>
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

const Vosk: FC = () => {
  const { t } = useTranslation();
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
    <Inspector.SubHeader>{t("stt.vosk_title")}</Inspector.SubHeader>
    <Inspector.Description>{t("stt.vosk_desc")}</Inspector.Description>

    <InputRustAudioInput
      label="common.field_input_device"
      value={pr.device}
      onChange={e => up("device", e)}
      hint="stt.rust_input_device_hint"
    />

    <InputSelect
      label="stt.vosk_model"
      options={modelOptions}
      value={pr.model}
      onValueChange={e => up("model", e)}
    />
    <Inspector.Description>{t("stt.vosk_model_desc")}</Inspector.Description>

    <InputText
      label="stt.vosk_model_url"
      placeholder="https://example.com/model.zip"
      value={pr.modelUrl}
      onChange={e => up("modelUrl", e.target.value)}
    />
    <Inspector.Description>{t("stt.vosk_model_url_desc")}</Inspector.Description>

    <InputCheckbox
      label={t("stt.vosk_use_local")}
      onChange={(e) => up("useLocalZip", e)}
      value={pr.useLocalZip}
    />
    {pr.useLocalZip && (
      <div className="flex flex-col gap-2 mb-2">
        <div className="flex gap-2 items-center">
          <input
            type="text"
            className="input input-bordered input-sm flex-1 min-w-0 font-mono text-xs"
            value={pr.localZipPath}
            onChange={(e) => up("localZipPath", e.target.value)}
            placeholder={t("stt.vosk_local_path")}
            aria-label={t("stt.vosk_local_path")}
          />
          {isTauri() && (
            <button
              type="button"
              className="btn btn-sm btn-outline shrink-0"
              onClick={async () => {
                const path = await open({
                  filters: [{ name: "Vosk model", extensions: ["zip"] }],
                  multiple: false,
                });
                if (path === null || Array.isArray(path)) return;
                up("localZipPath", path);
              }}
            >
              {t("stt.vosk_browse_zip")}
            </button>
          )}
        </div>
        {isTauri() && (
          <button
            type="button"
            className="btn btn-sm btn-neutral w-fit"
            onClick={async () => {
              const path = await open({
                filters: [{ name: "Vosk model", extensions: ["zip"] }],
                multiple: false,
              });
              if (path === null || Array.isArray(path)) return;
              try {
                await invoke("plugin:vosk-stt|import_vosk_model_zip", {
                  modelId: window.ApiServer.state.services.stt.data.vosk.model,
                  sourcePath: path as string,
                });
                toast.success(t("stt.vosk_import_ok"));
              } catch (e) {
                toast.error(String(e));
              }
            }}
          >
            {t("stt.vosk_import_cached")}
          </button>
        )}
        <Inspector.Description>{t("stt.vosk_local_hint")}</Inspector.Description>
      </div>
    )}
  </>
}

/** Engine status (download / load progress, connecting, rate-limit waits): UI only, never a caption. */
const SttStatusLine: FC = () => {
  const s = useSnapshot(window.ApiServer.stt.serviceState);
  if (!s.statusText) return null;
  return <div role="status" aria-live="polite" className="flex flex-col gap-1 text-xs">
    <span className={s.statusLevel === "warning" ? "text-warning" : "text-base-content/70"}>{s.statusText}</span>
    {s.progress !== null && <progress className="progress progress-primary w-full" value={s.progress} max={100} />}
  </div>;
};

type SttShortcut = "start" | "muteMic";

/** Global shortcut field: records the next key combination (Esc cancels). */
const ShortcutField: FC<{ shortcut: SttShortcut; label: string }> = ({ shortcut, label }) => {
  const { t } = useTranslation();
  const { shortcuts } = useSnapshot(window.ApiServer.state);
  const kb = useSnapshot(window.ApiServer.keyboard.ui);
  const recording = kb.showRecorder && kb.currentTarget === shortcut;
  const value = shortcuts[shortcut];
  const hintKey = recording && kb.recorderHint ? `stt.shortcut_hint_${kb.recorderHint}` : "";

  return <div className="flex flex-col gap-1">
    <span className="font-medium text-base-content/80 text-xs">{t(label)}</span>
    <div className="flex items-center gap-2">
      <span
        className={`input input-sm input-bordered flex-1 min-w-0 flex items-center font-mono text-xs truncate ${recording ? "input-primary" : ""}`}
        aria-live={recording ? "polite" : undefined}
      >
        {recording ? (kb.currentValue || t("stt.shortcut_press_keys")) : (value || t("stt.shortcut_none"))}
      </span>
      {recording ? (
        <button type="button" className="btn btn-sm btn-ghost shrink-0" onClick={() => window.ApiServer.keyboard.cancelComboRecord()}>
          {t("stt.shortcut_cancel")}
        </button>
      ) : (
        <button type="button" className="btn btn-sm btn-neutral shrink-0" onClick={() => window.ApiServer.keyboard.startShortcutRecord(shortcut)}>
          {t("stt.shortcut_record")}
        </button>
      )}
      {!recording && value && (
        <button type="button" className="btn btn-sm btn-ghost shrink-0" onClick={() => window.ApiServer.keyboard.clearShortcut(shortcut)}>
          {t("stt.shortcut_clear")}
        </button>
      )}
    </div>
    {hintKey && <span className="text-xs text-warning">{t(hintKey)}</span>}
  </div>;
};

/** Provider list value for the Groq preset (stored as backend `openai_audio` + preset `groq`). */
const PROVIDER_GROQ = "groq";

const Inspector_STT: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.stt);
  const state = useSnapshot(window.ApiServer.stt.serviceState);

  const handleStart = (v: boolean) => window.ApiServer.state.services.stt.showActionButton = v;
  const up = <K extends keyof STT_State>(key: K, v: STT_State[K]) => window.ApiServer.patchService("stt", s => s.data[key] = v);

  const providerValue =
    data.data.backend === STT_Backends.openai_audio && data.data.openai_audio.preset === OpenAiAudioPreset.groq
      ? PROVIDER_GROQ
      : data.data.backend;
  const setProvider = (value: string) => window.ApiServer.patchService("stt", s => {
    if (value === PROVIDER_GROQ) {
      s.data.backend = STT_Backends.openai_audio;
      s.data.openai_audio.preset = OpenAiAudioPreset.groq;
    }
    else {
      s.data.backend = value as STT_Backends;
      if (value === STT_Backends.openai_audio)
        s.data.openai_audio.preset = OpenAiAudioPreset.custom;
    }
  });

  const handleShowReplacements = () => {
    NiceModal.show('stt-replacements');
  }

  const backend = data.data.backend;
  const external = backend === STT_Backends.chrome || backend === STT_Backends.edge;

  return <Inspector.Body>
    <Inspector.Header><RiUserVoiceFill /> {t('stt.title')}</Inspector.Header>
    <Inspector.Content>
      <Inspector.SubHeader>{t("stt.section_provider")}</Inspector.SubHeader>
      <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
        <InputSelect options={[
          { label: "Whisper", value: STT_Backends.whisper },
          { label: t("stt.provider_groq"), value: PROVIDER_GROQ },
          { label: "Vosk", value: STT_Backends.vosk },
          { label: "Chrome", value: STT_Backends.chrome },
          { label: "Edge", value: STT_Backends.edge },
          { label: "Azure", value: STT_Backends.azure },
          { label: "Deepgram", value: STT_Backends.deepgram },
          { label: t("stt.provider_moonshine"), value: STT_Backends.moonshine },
          { label: t("stt.provider_openai_audio"), value: STT_Backends.openai_audio },
          { label: t("stt.provider_native"), value: STT_Backends.native },
        ]} label="common.field_service" value={providerValue} onValueChange={setProvider} />

        {!external && (
          <div className="mt-2 mb-4">
            <InputCheckbox label='stt.field_uwu_filter' onChange={e => up("uwu", e)} value={data.data.uwu} />
            <Inspector.Description>{t('stt.field_uwu_filter_desc')}</Inspector.Description>
          </div>
        )}

        {backend === STT_Backends.chrome && <ExternalBrowser browser="chrome" />}
        {backend === STT_Backends.edge && <ExternalBrowser browser="edge" />}
        {backend === STT_Backends.azure && <Azure />}
        {backend === STT_Backends.deepgram && <Deepgram />}
        {backend === STT_Backends.whisper && <Whisper />}
        {backend === STT_Backends.vosk && <Vosk />}
        {backend === STT_Backends.moonshine && <Moonshine />}
        {backend === STT_Backends.openai_audio && <OpenAIAudio />}
        {(backend === STT_Backends.native || backend === STT_Backends.browser) && <Native />}
      </Inspector.Deactivatable>

      {!external && (
        <div className="pt-4 flex flex-col gap-2">
          <ServiceButton status={state.status} onStart={() => window.ApiServer.stt.start()} onStop={() => window.ApiServer.stt.stop()} />
          <SttStatusLine />
        </div>
      )}

      <div className="contents">
        <div className="h-2" />
        <InputCheckbox label="common.field_action_bar" onChange={handleStart} value={data.showActionButton} />
        <InputCheckbox label="common.field_auto_start" value={data.data.autoStart} onChange={e => up("autoStart", e)} />
        <InputCheckbox label="stt.field_stop_with_stream" value={data.data.stopWithStream} onChange={e => up("stopWithStream", e)} />
        <div className="h-2" />
        <span className="link link-accent link-hover font-semibold flex items-center gap-2 text-sm justify-end" onClick={handleShowReplacements}><RiCharacterRecognitionFill />{t('common.btn_edit_replacements')}</span>
      </div>

      {isTauri() && <>
        <Inspector.SubHeader>{t("stt.section_shortcuts")}</Inspector.SubHeader>
        <ShortcutField shortcut="start" label="stt.shortcut_start" />
        <ShortcutField shortcut="muteMic" label="stt.shortcut_mute" />
        <Inspector.Description>{t("stt.shortcut_desc")}</Inspector.Description>
      </>}

    </Inspector.Content>
  </Inspector.Body>
}

export default Inspector_STT;
