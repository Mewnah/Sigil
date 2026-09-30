import { LOCAL_NEURAL_FOLDER, LOCAL_NEURAL_THREADS } from "@/core/services/tts/schema/localNeural";
import { TTS_Backends, type TTS_State } from "@/core/services/tts/schema";
import { ttsSinkTarget } from "@/core/services/tts/pipeline";
import {
  LOCAL_TTS_DEFAULT_MODEL,
  LocalTtsModelInfo,
  LocalTtsVoiceInfo,
  catalogIdOf,
  groupLocalModels,
  groupVoicesByLanguage,
  localNeuralSettings,
  localTtsState,
  previewLocalVoice,
  previewText,
  unloadLocalTtsIfIdle,
  voicesFor,
} from "@/core/services/tts/services/localNeural";
import { errorMessage } from "@/core/services/tts/types";
import type { RustModelEntryStatus } from "@/types";
import { ServiceNetworkState } from "@/types";
import { invoke, isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import i18n from "i18next";
import { FC, useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { RiFolderOpenLine, RiPlayLine } from "react-icons/ri";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { ModelRequirement, useModelsLoaded } from "../components/ModelRequirement";
import { InputRange, InputSelect, InputText } from "../components/input";
import { TtsOutputDevice } from "./Routing";

type LocalNeuralState = TTS_State["localNeural"];

/** Localized name of a language tag (`pt-BR` → "Brazilian Portuguese"), the tag itself when unknown. */
function languageName(tag: string | null, uiLanguage: string): string {
  if (!tag) return "";
  try {
    const names = new Intl.DisplayNames([uiLanguage.replace("_", "-"), "en"], { type: "language" });
    return names.of(tag) ?? tag;
  } catch {
    return tag;
  }
}

/**
 * The catalog models of `plugin:local-tts|list_local_tts_models` (also used by the NeuralOnnx panel), refreshed when a
 * `tts/` model or the sherpa runtime is installed or removed.
 */
export function useLocalModels(): { models: LocalTtsModelInfo[]; failed: boolean } {
  const tauri = isTauri();
  useModelsLoaded();
  const snap = useSnapshot(window.ApiServer.models.state);
  const states = snap.states as Record<string, RustModelEntryStatus | undefined>;
  // Changes only when a local voice (or the sherpa runtime) is installed or removed.
  const installedKey = useMemo(
    () =>
      Object.entries(states)
        .filter(([id, s]) => (id.startsWith("tts/") || id === "runtime/sherpa-onnx") && s?.state === "installed")
        .map(([id]) => id)
        .sort()
        .join(","),
    [states],
  );
  const [models, setModels] = useState<LocalTtsModelInfo[]>([]);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!tauri) return;
    let live = true;
    invoke<LocalTtsModelInfo[]>("plugin:local-tts|list_local_tts_models")
      .then((list) => {
        if (!live) return;
        setModels(Array.isArray(list) ? list : []);
        setFailed(false);
      })
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, [tauri, installedKey]);
  return { models, failed };
}

/**
 * Panel of the `localNeural` backend: in-process Kokoro (v1.0, v1.1-zh), Kitten and Piper voices on the processor
 * (Rust `local-tts` plugin). Model and voice pickers from `list_local_tts_models`, `<ModelRequirement>` for the
 * consented download (the model plus the sherpa-onnx runtime), a voice folder on this PC, a preview, speed, threads,
 * and the sink output device and volume. Owned by P2-H.
 */
export const LocalNeural: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.localNeural);
  const service = useSnapshot(window.ApiServer.tts.serviceState);
  const loaded = useSnapshot(localTtsState).loaded as LocalTtsModelInfo | null;
  const { models, failed } = useLocalModels();
  const [previewing, setPreviewing] = useState(false);
  const [sample, setSample] = useState("");
  const tauri = isTauri();

  const up = <K extends keyof LocalNeuralState>(key: K, v: LocalNeuralState[K]) =>
    window.ApiServer.patchService("tts", (s) => (s.data.localNeural[key] = v));

  const stopped = service.status === ServiceNetworkState.disconnected;
  const folderMode = data.model === LOCAL_NEURAL_FOLDER;
  const model = folderMode ? data.folder.trim() : data.model.trim() || LOCAL_TTS_DEFAULT_MODEL;
  const catalogId = folderMode ? null : catalogIdOf(model);
  const row = models.find((m) => m.id === model);
  const known = voicesFor(model, models, loaded);

  const modelOptions = useMemo(() => {
    const familyLabel: Record<string, string> = {
      kokoro: t("tts.local_neural_family_kokoro"),
      kitten: t("tts.local_neural_family_kitten"),
      piper: t("tts.local_neural_family_piper"),
    };
    const groups = groupLocalModels(models).map((g) => ({
      label: familyLabel[g.family] ?? g.family,
      options: g.models.map((m) => ({ label: m.installed ? `${m.title} ✓` : m.title, value: m.id })),
    }));
    const current = data.model.trim();
    // A saved model the list does not know (an older catalog id) stays selectable.
    const listed = !current || current === LOCAL_NEURAL_FOLDER || models.some((m) => m.id === current);
    const extra = listed ? [] : [{ label: current, value: current }];
    return [
      ...groups,
      ...extra,
      { label: t("tts.local_neural_other"), options: [{ label: t("tts.local_neural_model_folder"), value: LOCAL_NEURAL_FOLDER }] },
    ];
  }, [models, data.model, t]);

  const voiceOptions = useMemo(() => {
    if (!known) return null;
    const defaultRow = known.voices.find((v) => v.id === known.defaultVoice);
    const defaultLabel = t("tts.local_neural_voice_default", { name: defaultRow?.name ?? known.defaultVoice });
    const voiceLabel = (v: LocalTtsVoiceInfo) =>
      v.gender === "female"
        ? t("tts.local_neural_voice_female", { name: v.name })
        : v.gender === "male"
          ? t("tts.local_neural_voice_male", { name: v.name })
          : v.name;
    const groups = groupVoicesByLanguage(known.voices);
    const options: ({ label: string; value: string } | { label: string; options: { label: string; value: string }[] })[] = [
      { label: known.defaultVoice ? defaultLabel : t("tts.local_neural_voice_default_plain"), value: "" },
    ];
    if (groups.length === 1) {
      options.push(...groups[0].voices.map((v) => ({ label: voiceLabel(v), value: v.id })));
    } else {
      for (const g of groups) {
        options.push({
          label: languageName(g.language, i18n.language) || t("tts.local_neural_voice_other_language"),
          options: g.voices.map((v) => ({ label: voiceLabel(v), value: v.id })),
        });
      }
    }
    // A saved voice the model does not list stays selectable (Rust reports it if it is really unknown).
    const current = data.voice.trim();
    if (current && !known.voices.some((v) => v.id === current)) options.push({ label: current, value: current });
    return options;
  }, [known, data.voice, t]);

  const voiceLanguage = useMemo(() => {
    const id = data.voice.trim() || known?.defaultVoice || "";
    return known?.voices.find((v) => v.id === id)?.language ?? row?.languages[0] ?? null;
  }, [known, data.voice, row]);

  const browse = useCallback(async () => {
    try {
      const path = await open({ directory: true, multiple: false });
      if (typeof path === "string" && path) up("folder", path);
    } catch (e) {
      toast.error(errorMessage(e));
    }
  }, []);

  const preview = useCallback(async () => {
    const state = window.ApiServer.state.services.tts.data;
    const settings = localNeuralSettings(state.localNeural);
    if (!settings.model) return;
    setPreviewing(true);
    try {
      const text = sample.trim() || previewText(voiceLanguage);
      const target = ttsSinkTarget({ ...state, backend: TTS_Backends.localNeural }, window.ApiServer.state.audioOutputDevice ?? "");
      await previewLocalVoice(settings, text, target);
    } catch (e) {
      toast.error(t("tts.local_neural_preview_failed", { error: errorMessage(e) }));
    } finally {
      setPreviewing(false);
      unloadLocalTtsIfIdle();
    }
  }, [sample, voiceLanguage, t]);

  const installed = folderMode ? !!model : row?.installed ?? false;
  // Kokoro reads Japanese kanji with Chinese readings (no Japanese reading dictionary yet; Rust local_tts::kokoro).
  const japaneseKokoro = (row?.family ?? loaded?.family) === "kokoro" && (voiceLanguage ?? "").toLowerCase().startsWith("ja");

  return (
    <>
      <Inspector.SubHeader>{t("tts.local_neural_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("tts.local_neural_desc")}</Inspector.Description>
      {!tauri && <div className="text-xs text-base-content/60">{t("tts.local_neural_desktop_only")}</div>}
      {failed && <div className="text-xs text-error">{t("tts.local_neural_list_failed")}</div>}
      <Inspector.Deactivatable active={stopped}>
        <InputSelect
          label="tts.local_neural_model"
          value={data.model.trim() || LOCAL_TTS_DEFAULT_MODEL}
          options={modelOptions}
          onValueChange={(v) => {
            up("model", v);
            // Voice ids belong to a model.
            if (v !== data.model) up("voice", "");
          }}
        />
        {folderMode && (
          <div className="flex flex-col gap-1">
            <label className="font-medium text-base-content/80 text-xs" htmlFor="local-neural-folder">
              {t("tts.local_neural_folder")}
            </label>
            <div className="flex gap-2 items-center">
              <input
                id="local-neural-folder"
                type="text"
                className="input input-bordered input-sm flex-1 min-w-0 font-mono text-xs"
                value={data.folder}
                onChange={(e) => up("folder", e.target.value)}
              />
              {tauri && (
                <button type="button" className="btn btn-sm btn-outline shrink-0 gap-1" onClick={browse}>
                  <RiFolderOpenLine /> {t("tts.neural_onnx_browse_folder")}
                </button>
              )}
            </div>
            <div className="text-xs text-base-content/60">{t("tts.local_neural_folder_hint")}</div>
          </div>
        )}
        {catalogId && <ModelRequirement ids={[catalogId]} reason={t("tts.local_neural_requirement")} />}
        <InputSelect
          label="tts.local_neural_threads"
          value={data.threads}
          options={LOCAL_NEURAL_THREADS.map((v) => ({ label: v === "auto" ? t("tts.local_neural_threads_auto") : v, value: v }))}
          onValueChange={(v) => up("threads", v as LocalNeuralState["threads"])}
        />
      </Inspector.Deactivatable>
      {voiceOptions ? (
        <InputSelect label="tts.field_voice" value={data.voice.trim()} options={voiceOptions} onValueChange={(v) => up("voice", v)} />
      ) : (
        <InputText label="tts.field_voice" value={data.voice} placeholder={t("tts.local_neural_voice_placeholder")} onChange={(e) => up("voice", e.target.value)} />
      )}
      {japaneseKokoro && <div className="text-xs text-warning">{t("tts.local_neural_ja_note")}</div>}
      <InputRange label="tts.neural_onnx_speed" value={data.speed} min="0.5" max="2" step="0.05" onChange={(e) => up("speed", e.target.value)} />
      <div className="flex gap-2 items-center">
        <input
          type="text"
          className="input input-bordered input-sm flex-1 min-w-0 text-xs"
          aria-label={t("tts.local_neural_preview_text")}
          title={t("tts.local_neural_preview_text")}
          value={sample}
          placeholder={previewText(voiceLanguage)}
          onChange={(e) => setSample(e.target.value)}
        />
        <button
          type="button"
          className="btn btn-sm btn-outline shrink-0 gap-1"
          disabled={!tauri || previewing || !installed}
          onClick={preview}
          title={installed ? undefined : t("tts.local_neural_preview_needs_download")}
        >
          {previewing ? <span className="loading loading-spinner loading-xs" /> : <RiPlayLine />}
          {t("tts.local_neural_preview")}
        </button>
      </div>
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={(v) => up("device", v)} />
      <InputRange label="common.field_volume" value={data.volume} min="0" max="2" step="0.05" onChange={(e) => up("volume", e.target.value)} />
    </>
  );
};
