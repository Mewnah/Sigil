import type { STT_State } from "@/core/services/stt/schema";
import {
  findWhisperModel,
  formatModelSize,
  WHISPER_LANGUAGES,
  WHISPER_MODEL_OPTIONS,
  type WhisperModelFamily,
  type WhisperModelOption,
} from "@/core/services/stt/whisperModels";
import { ServiceNetworkState } from "@/types";
import { invoke, isTauri } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { FC, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputCheckbox, InputRustAudioInput, InputSelect, InputText } from "../components/input";
import { Advanced, useLanguageName } from "./common";
import { VoiceDetection } from "./VoiceDetection";

type WhisperDownloadUi =
  | null
  | { file: string; progress: number; phase: "loading" }
  | { file: string; progress: number; phase: "complete" };

const WHISPER_FAMILY_ORDER: WhisperModelFamily[] = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3", "anime-whisper"];

export const Whisper: FC = () => {
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
