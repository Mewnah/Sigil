import { ttsSinkTarget } from "@/core/services/tts/pipeline";
import { TTS_Backends, type TTS_State } from "@/core/services/tts/schema";
import { POCKET_THREADS } from "@/core/services/tts/schema/pocket";
import { previewText } from "@/core/services/tts/services/localNeural";
import {
  POCKET_MODELS,
  POCKET_TEMPERATURE,
  pocketModel,
  pocketRequest,
  previewCloningVoice,
  unloadCloningIfIdle,
} from "@/core/services/tts/services/pocket";
import { errorMessage } from "@/core/services/tts/types";
import { ServiceNetworkState, type RustModelEntryStatus } from "@/types";
import { isTauri } from "@tauri-apps/api/core";
import { FC, useCallback, useState } from "react";
import { useTranslation } from "react-i18next";
import { RiPlayLine } from "react-icons/ri";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { ModelRequirement } from "../components/ModelRequirement";
import { ReferenceVoicePicker } from "../components/ReferenceVoiceRecorder";
import { InputRange, InputSelect } from "../components/input";
import { TtsOutputDevice } from "./Routing";

type PocketState = TTS_State["pocket"];

/** i18n key of a Pocket model's name. */
const MODEL_LABEL: Record<(typeof POCKET_MODELS)[number], string> = {
  "tts/pocket-tts-en-int8": "tts.pocket_model_int8",
  "tts/pocket-tts-en": "tts.pocket_model_full",
};

/** Whether catalog entry `id` is installed (models service state). */
export function useInstalled(id: string): boolean {
  const snap = useSnapshot(window.ApiServer.models.state);
  return (snap.states as Record<string, RustModelEntryStatus | undefined>)[id]?.state === "installed";
}

/**
 * A sample line and a preview button for a cloning voice (shared by the Pocket and Chatterbox panels). `placeholder`
 * is the line spoken when nothing is typed; `onPreview` gets the typed line (`""` = the placeholder), so the panel
 * can tag each with its own language.
 */
export const CloningPreview: FC<{ placeholder: string; installed: boolean; onPreview: (typed: string) => Promise<void> }> = ({
  placeholder,
  installed,
  onPreview,
}) => {
  const { t } = useTranslation();
  const [busy, setBusy] = useState(false);
  const [sample, setSample] = useState("");
  const tauri = isTauri();
  const run = async () => {
    setBusy(true);
    try {
      await onPreview(sample.trim());
    } catch (e) {
      toast.error(t("tts.cloning_preview_failed", { error: errorMessage(e) }));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="flex gap-2 items-center">
      <input
        type="text"
        className="input input-bordered input-sm flex-1 min-w-0 text-xs"
        aria-label={t("tts.cloning_preview_text")}
        title={t("tts.cloning_preview_text")}
        value={sample}
        placeholder={placeholder}
        onChange={(e) => setSample(e.target.value)}
      />
      <button
        type="button"
        className="btn btn-sm btn-outline shrink-0 gap-1"
        disabled={!tauri || busy || !installed}
        onClick={() => void run()}
        title={installed ? undefined : t("tts.cloning_preview_needs_download")}
      >
        {busy ? <span className="loading loading-spinner loading-xs" /> : <RiPlayLine />}
        {t("tts.cloning_preview")}
      </button>
    </div>
  );
};

/**
 * Panel of the `pocket` backend (P6-I): Kyutai Pocket TTS, voice cloning on the processor. Model (with its consented
 * download), threads, the voice-library voice to clone (or the built-in voice), variation, a preview, and the sink
 * output device and volume.
 */
export const Pocket: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.pocket);
  const service = useSnapshot(window.ApiServer.tts.serviceState);
  const tauri = isTauri();
  const model = pocketModel(data as PocketState);
  const installed = useInstalled(model);

  const up = <K extends keyof PocketState>(key: K, v: PocketState[K]) => window.ApiServer.patchService("tts", (s) => (s.data.pocket[key] = v));
  const stopped = service.status === ServiceNetworkState.disconnected;

  const preview = useCallback(async (typed: string) => {
    const state = window.ApiServer.state.services.tts.data;
    const target = ttsSinkTarget({ ...state, backend: TTS_Backends.pocket }, window.ApiServer.state.audioOutputDevice ?? "");
    try {
      // Pocket speaks English.
      await previewCloningVoice("pocket", pocketRequest(state.pocket, typed || previewText("en")), target);
    } finally {
      unloadCloningIfIdle("pocket", TTS_Backends.pocket);
    }
  }, []);

  return (
    <>
      <Inspector.SubHeader>{t("tts.pocket_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("tts.pocket_desc")}</Inspector.Description>
      {!tauri && <div className="text-xs text-base-content/60">{t("tts.cloning_desktop_only")}</div>}
      <Inspector.Deactivatable active={stopped}>
        <InputSelect
          label="tts.pocket_model"
          value={model}
          options={POCKET_MODELS.map((id) => ({ label: t(MODEL_LABEL[id]), value: id }))}
          onValueChange={(v) => up("model", v)}
        />
        <ModelRequirement ids={[model]} reason={t("tts.pocket_requirement")} />
        <InputSelect
          label="tts.cloning_threads"
          value={data.threads}
          options={POCKET_THREADS.map((v) => ({ label: v === "auto" ? t("tts.pocket_threads_auto") : v, value: v }))}
          onValueChange={(v) => up("threads", v as PocketState["threads"])}
        />
      </Inspector.Deactivatable>
      <ReferenceVoicePicker value={data.referenceVoiceId} onChange={(id) => up("referenceVoiceId", id)} label="tts.cloning_voice" />
      {!data.referenceVoiceId.trim() && <div className="text-xs text-base-content/60">{t("tts.pocket_builtin_voice")}</div>}
      <div className="text-xs text-base-content/60">{t("tts.pocket_english_only")}</div>
      <InputRange
        label="tts.cloning_temperature"
        value={data.temperature}
        min={String(POCKET_TEMPERATURE.min)}
        max={String(POCKET_TEMPERATURE.max)}
        step="0.05"
        onChange={(e) => up("temperature", e.target.value)}
      />
      <CloningPreview placeholder={previewText("en")} installed={installed} onPreview={preview} />
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={(v) => up("device", v)} />
      <InputRange label="common.field_volume" value={data.volume} min="0" max="2" step="0.05" onChange={(e) => up("volume", e.target.value)} />
    </>
  );
};
