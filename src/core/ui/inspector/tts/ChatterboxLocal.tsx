import { diagnosticsState } from "@/core/services/diagnostics/bridge";
import { epLabel } from "@/core/services/diagnostics/format";
import { ttsSinkTarget } from "@/core/services/tts/pipeline";
import { TTS_Backends, type TTS_State } from "@/core/services/tts/schema";
import { CHATTERBOX_LOCAL_DEVICES, CHATTERBOX_LOCAL_QUALITY, CHATTERBOX_LOCAL_THREADS } from "@/core/services/tts/schema/chatterboxLocal";
import {
  CHATTERBOX_LOCAL_LANGUAGES,
  CHATTERBOX_LOCAL_MODEL,
  CHATTERBOX_LOCAL_RANGES,
  chatterboxOptionalIds,
  chatterboxPreview,
  chatterboxPreviewText,
  chatterboxRequest,
} from "@/core/services/tts/services/chatterboxLocal";
import { previewCloningVoice, unloadCloningIfIdle } from "@/core/services/tts/services/pocket";
import { ServiceNetworkState, type RustEpOutcomeEntry } from "@/types";
import { isTauri } from "@tauri-apps/api/core";
import i18n from "i18next";
import { FC, useCallback, useMemo } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { ModelRequirement } from "../components/ModelRequirement";
import { ReferenceVoicePicker } from "../components/ReferenceVoiceRecorder";
import { InputRange, InputSelect } from "../components/input";
import { CloningPreview, useInstalled } from "./Pocket";
import { TtsOutputDevice } from "./Routing";

type ChatterboxState = TTS_State["chatterboxLocal"];

/** Diagnostics component of the language model (Rust `chatterbox_ort::model::COMPONENT_LM`). */
const COMPONENT_LM = "tts.chatterbox";

/** Localized name of a language tag ("ja" → "Japanese"), the tag itself when unknown. */
function languageName(tag: string, uiLanguage: string): string {
  try {
    return new Intl.DisplayNames([uiLanguage.replace("_", "-"), "en"], { type: "language" }).of(tag) ?? tag;
  } catch {
    return tag;
  }
}

/**
 * Panel of the `chatterboxLocal` backend (P6-I): Chatterbox Multilingual V3 on Sigil's ONNX Runtime. The model (and
 * the optional graphics-card decoder) with consented downloads, where it runs and what it actually runs on, threads,
 * the voice-library voice to clone (required), language, expressiveness, guidance, variation, quality, a preview,
 * and the sink output device and volume.
 */
export const ChatterboxLocal: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.chatterboxLocal) as ChatterboxState;
  const service = useSnapshot(window.ApiServer.tts.serviceState);
  const outcomes = useSnapshot(diagnosticsState).outcomes as Record<string, RustEpOutcomeEntry | undefined>;
  const tauri = isTauri();
  const installed = useInstalled(CHATTERBOX_LOCAL_MODEL);
  const r = CHATTERBOX_LOCAL_RANGES;

  const up = <K extends keyof ChatterboxState>(key: K, v: ChatterboxState[K]) =>
    window.ApiServer.patchService("tts", (s) => (s.data.chatterboxLocal[key] = v));
  const stopped = service.status === ServiceNetworkState.disconnected;
  const outcome = outcomes[COMPONENT_LM];

  const languageOptions = useMemo(
    () => [
      { label: t("tts.chatterbox_local_language_auto"), value: "auto" },
      ...CHATTERBOX_LOCAL_LANGUAGES.map((l) => ({ label: languageName(l, i18n.language), value: l })).sort((a, b) => a.label.localeCompare(b.label)),
    ],
    [t],
  );

  const preview = useCallback(async (typed: string) => {
    const state = window.ApiServer.state.services.tts.data;
    const c = state.chatterboxLocal;
    const target = ttsSinkTarget({ ...state, backend: TTS_Backends.chatterboxLocal }, window.ApiServer.state.audioOutputDevice ?? "");
    // The language tag goes with the line actually spoken (Chatterbox applies it literally).
    const { text, language } = chatterboxPreview(c.language, i18n.language, typed);
    try {
      await previewCloningVoice("chatterbox", { ...chatterboxRequest(c, text, null), language }, target);
    } finally {
      unloadCloningIfIdle("chatterbox", TTS_Backends.chatterboxLocal);
    }
  }, []);

  const previewPlaceholder = chatterboxPreviewText(data.language, i18n.language);
  const optional = chatterboxOptionalIds(data);

  return (
    <>
      <Inspector.SubHeader>{t("tts.chatterbox_local_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("tts.chatterbox_local_desc")}</Inspector.Description>
      {!tauri && <div className="text-xs text-base-content/60">{t("tts.cloning_desktop_only")}</div>}
      <Inspector.Deactivatable active={stopped}>
        <ModelRequirement ids={[CHATTERBOX_LOCAL_MODEL]} reason={t("tts.chatterbox_local_requirement")} />
        <InputSelect
          label="tts.chatterbox_local_device"
          value={data.ep}
          options={CHATTERBOX_LOCAL_DEVICES.map((v) => ({ label: t(`tts.chatterbox_local_device_${v}`), value: v }))}
          onValueChange={(v) => up("ep", v as ChatterboxState["ep"])}
        />
        {optional.length > 0 && <ModelRequirement ids={optional} reason={t("tts.chatterbox_local_gpu_requirement")} hideWhenReady />}
        <InputSelect
          label="tts.cloning_threads"
          value={data.threads}
          options={CHATTERBOX_LOCAL_THREADS.map((v) => ({ label: v === "auto" ? t("tts.chatterbox_local_threads_auto") : v, value: v }))}
          onValueChange={(v) => up("threads", v as ChatterboxState["threads"])}
        />
      </Inspector.Deactivatable>
      {outcome && (
        <div className={outcome.active !== outcome.requested ? "text-xs text-warning" : "text-xs text-base-content/60"}>
          {outcome.active !== outcome.requested && outcome.reason
            ? t("tts.chatterbox_local_fell_back", { ep: epLabel(outcome.active), reason: outcome.reason })
            : t("tts.chatterbox_local_running_on", { ep: epLabel(outcome.active) })}
        </div>
      )}
      {(data.ep === "cpu" || outcome?.active === "cpu") && <div className="text-xs text-base-content/60">{t("tts.chatterbox_local_cpu_note")}</div>}
      <ReferenceVoicePicker
        value={data.referenceVoiceId}
        onChange={(id) => up("referenceVoiceId", id)}
        label="tts.cloning_voice"
        allowNone={false}
      />
      <InputSelect label="tts.chatterbox_local_language" value={data.language || "auto"} options={languageOptions} onValueChange={(v) => up("language", v)} />
      <InputSelect
        label="tts.chatterbox_local_quality"
        value={data.quality}
        options={CHATTERBOX_LOCAL_QUALITY.map((v) => ({ label: t(`tts.chatterbox_local_quality_${v}`), value: v }))}
        onValueChange={(v) => up("quality", v as ChatterboxState["quality"])}
      />
      <InputRange
        label="tts.chatterbox_local_exaggeration"
        value={data.exaggeration}
        min={String(r.exaggeration.min)}
        max={String(r.exaggeration.max)}
        step="0.05"
        onChange={(e) => up("exaggeration", e.target.value)}
      />
      <InputRange
        label="tts.chatterbox_local_cfg"
        value={data.cfgWeight}
        min={String(r.cfgWeight.min)}
        max={String(r.cfgWeight.max)}
        step="0.05"
        onChange={(e) => up("cfgWeight", e.target.value)}
      />
      <div className="text-xs text-base-content/60">{t("tts.chatterbox_local_cfg_hint")}</div>
      <InputRange
        label="tts.cloning_temperature"
        value={data.temperature}
        min={String(r.temperature.min)}
        max={String(r.temperature.max)}
        step="0.05"
        onChange={(e) => up("temperature", e.target.value)}
      />
      <CloningPreview placeholder={previewPlaceholder} installed={installed && !!data.referenceVoiceId.trim()} onPreview={preview} />
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={(v) => up("device", v)} />
      <InputRange label="common.field_volume" value={data.volume} min="0" max="2" step="0.05" onChange={(e) => up("volume", e.target.value)} />
    </>
  );
};
