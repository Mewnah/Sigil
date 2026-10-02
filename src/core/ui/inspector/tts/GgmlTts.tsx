import { ttsSinkTarget } from "@/core/services/tts/pipeline";
import { TTS_Backends } from "@/core/services/tts/schema";
import {
  GGML_TTS_ENGINES,
  GGML_TTS_QUALITIES,
  GGML_TTS_VARIANTS,
  type GgmlTtsEngine,
  type GgmlTtsQuality,
  type GgmlTtsVariant,
  type TTS_GgmlTts_State,
} from "@/core/services/tts/schema/ggmlTts";
import {
  GGML_ENGINES,
  GGML_LOCAL_ONLY_IDS,
  GGML_STATE_EVENT,
  MAGPIE_SPEAKERS,
  ggmlEngine,
  ggmlGpu,
  ggmlHelperStatus,
  ggmlLanguageAfterChange,
  ggmlLanguageChoices,
  ggmlModel,
  ggmlSettingsProblem,
  ggmlUi,
  languageBase,
  planGgml,
  previewGgml,
  stopGgmlHelper,
} from "@/core/services/tts/services/ggmlTts";
import { errorMessage } from "@/core/services/tts/types";
import { ServiceNetworkState, type RustModelEntryStatus, type RustTtsHelperStatus } from "@/types";
import { isTauri } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import i18n from "i18next";
import { FC, useCallback, useEffect, useId, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { RiPlayLine } from "react-icons/ri";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { ModelRequirement, useModelsLoaded } from "../components/ModelRequirement";
import { ReferenceVoicePicker } from "../components/ReferenceVoiceRecorder";
import { InputCheckbox, InputRange, InputSelect, InputText } from "../components/input";
import { TtsOutputDevice } from "./Routing";

type GgmlState = TTS_GgmlTts_State;

/** Localized name of a language tag ("ja" -> "Japanese"), the tag itself when unknown. */
function languageName(tag: string, uiLanguage: string | undefined): string {
  try {
    const locales = uiLanguage ? [uiLanguage.replace("_", "-"), "en"] : ["en"];
    return new Intl.DisplayNames(locales, { type: "language" }).of(tag) ?? tag;
  } catch {
    return tag;
  }
}

/** The helper's state for `engine` (`tts_helper_status`, then `tts_helpers:state` events) while the panel is open. */
function useHelperStatus(engine: GgmlTtsEngine): RustTtsHelperStatus | null {
  const [status, setStatus] = useState<RustTtsHelperStatus | null>(null);
  useEffect(() => {
    if (!isTauri()) return;
    let live = true;
    let unlisten: (() => void) | null = null;
    setStatus(null);
    ggmlHelperStatus(engine)
      .then((s) => live && setStatus(s))
      .catch(() => {});
    listen<RustTtsHelperStatus>(GGML_STATE_EVENT, (e) => {
      if (live && e.payload?.engine === engine) setStatus(e.payload);
    })
      .then((u) => {
        if (live) unlisten = u;
        else u();
      })
      .catch(() => {});
    return () => {
      live = false;
      unlisten?.();
    };
  }, [engine]);
  return status;
}

const Hint: FC<{ tone?: "muted" | "warning" | "error"; children: React.ReactNode }> = ({ tone = "muted", children }) => (
  <div className={tone === "warning" ? "text-xs text-warning" : tone === "error" ? "text-xs text-error break-words" : "text-xs text-base-content/60"}>{children}</div>
);

/** Breeze voice descriptions to start from (English; Mandarin works as well). */
const BREEZE_EXAMPLES = [
  "A warm young woman, friendly and upbeat, speaking at a relaxed pace.",
  "A deep, calm male voice, like a late-night radio host.",
  "An energetic teenage boy who sounds about to laugh.",
] as const;

/**
 * Panel of the `ggmlTts` backend (P6-F): the experimental ggml voice engines (Magpie, Breeze TTS 2, OmniVoice, Fish S2
 * Pro). Engine, model and runtime pack, the downloads they need (and the local-only parts with how to get them), the
 * engine's voice settings, a test sentence, and the sink output.
 */
export const GgmlTts: FC = () => {
  const { t } = useTranslation();
  const tauri = isTauri();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.ggmlTts) as GgmlState;
  useModelsLoaded();
  const models = useSnapshot(window.ApiServer.models.state);
  const states = models.states as Record<string, RustModelEntryStatus | undefined>;
  const sys = useSnapshot(window.ApiServer.sysinfo.state);
  const ui = useSnapshot(ggmlUi);
  const service = useSnapshot(window.ApiServer.tts.serviceState);
  const engine = ggmlEngine(data.engine);
  const status = useHelperStatus(engine.id);
  const [previewing, setPreviewing] = useState(false);
  const [sample, setSample] = useState("");
  const instructionId = useId();

  const up = <K extends keyof GgmlState>(key: K, v: GgmlState[K]) =>
    window.ApiServer.patchService("tts", (s) => {
      s.data.ggmlTts[key] = v;
    });

  const changeEngine = (id: GgmlTtsEngine) =>
    window.ApiServer.patchService("tts", (s) => {
      const slice = s.data.ggmlTts;
      if (slice.engine === id) return;
      slice.engine = id;
      // Model values are per engine; a variant the new engine has no pack for goes back to automatic, and so does a
      // language it cannot speak.
      slice.model = "";
      if (slice.variant !== "auto" && !GGML_ENGINES[id].packs.some((p) => p.variant === slice.variant)) slice.variant = "auto";
      slice.language = ggmlLanguageAfterChange(slice);
    });

  // Magpie's checkpoints speak different languages: one the new checkpoint lacks goes back to automatic.
  const changeModel = (key: string) =>
    window.ApiServer.patchService("tts", (s) => {
      const slice = s.data.ggmlTts;
      slice.model = key;
      slice.language = ggmlLanguageAfterChange(slice);
    });

  // sys.info: the GPU facts come from the probe (recomputed when it lands).
  const gpu = useMemo(() => ggmlGpu(), [sys.info]);
  const plan = useMemo(
    () => planGgml(data, { installed: (id) => states[id]?.state === "installed", gpu }),
    [data.engine, data.model, data.variant, states, gpu],
  );
  const model = ggmlModel(engine, data.model);
  const problem = ggmlSettingsProblem(data);
  const ready = !plan.refused && plan.ids.every((id) => states[id]?.state === "installed");
  const localMissing = plan.ids.filter((id) => GGML_LOCAL_ONLY_IDS.has(id) && states[id]?.state !== "installed");
  const cloning = engine.cloning && !!data.referenceVoiceId.trim();

  const engineOptions = GGML_TTS_ENGINES.map((id) => ({ label: t(`tts.ggml_engine_${id}`), value: id }));
  const modelOptions = engine.models.map((m) => ({ label: t(`tts.ggml_model_${engine.id}_${m.key}`), value: m.key }));
  const variantOptions = GGML_TTS_VARIANTS.filter((v) => v === "auto" || engine.packs.some((p) => p.variant === v)).map((v) => ({
    label: v === "auto" ? t("tts.ggml_variant_auto", { variant: t(`tts.ggml_variant_${plan.pack.variant}`) }) : t(`tts.ggml_variant_${v}`),
    value: v,
  }));
  const languages = ggmlLanguageChoices(data);
  // The setting as it is, even when it is not in the list (a language OmniVoice speaks beyond the common ones, or one
  // the Magpie checkpoint lacks, which the settings problem below then names), so choosing another value always works.
  const languageValue = languageBase(data.language) ?? "auto";
  const languageOptions = useMemo(() => {
    const codes = [...(languages ?? [])];
    if (languageValue !== "auto" && !codes.includes(languageValue)) codes.push(languageValue);
    return [
      { label: t("tts.ggml_language_auto"), value: "auto" },
      ...codes.map((code) => ({ label: languageName(code, i18n.language), value: code })),
    ];
  }, [t, languages?.join(","), languageValue]);
  const speakerValue = MAGPIE_SPEAKERS.find((v) => v.toLowerCase() === data.voice.trim().toLowerCase()) ?? MAGPIE_SPEAKERS[0];

  const statusLine = (() => {
    if (!status) return null;
    switch (status.state) {
      case "starting":
        return <Hint>{t("tts.ggml_status_starting", { model: status.model ?? "" })}</Hint>;
      case "ready":
        return <Hint>{t("tts.ggml_status_ready", { model: status.model ?? "" })}</Hint>;
      case "failed":
        return <Hint tone="error">{t("tts.ggml_status_failed", { error: status.error ?? "" })}</Hint>;
      default:
        return null;
    }
  })();

  const preview = useCallback(async () => {
    const state = window.ApiServer.state.services.tts.data;
    setPreviewing(true);
    try {
      const text = sample.trim() || t(`tts.ggml_preview_default_${state.ggmlTts.engine}`);
      const target = ttsSinkTarget({ ...state, backend: TTS_Backends.ggmlTts }, window.ApiServer.state.audioOutputDevice ?? "");
      await previewGgml(state.ggmlTts, text, target);
    } catch (e) {
      toast.error(t("tts.ggml_preview_failed", { error: errorMessage(e) }));
    } finally {
      setPreviewing(false);
    }
  }, [sample, t]);

  const stopHelper = useCallback(() => {
    void stopGgmlHelper(engine.id, (e) => toast.error(errorMessage(e)));
  }, [engine.id]);

  return (
    <>
      <Inspector.SubHeader>{t("tts.ggml_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("tts.ggml_desc")}</Inspector.Description>
      {!tauri && <Hint>{t("tts.ggml_desktop_only")}</Hint>}

      <InputSelect label="tts.ggml_engine" value={engine.id} options={engineOptions} onValueChange={(v) => changeEngine(v as GgmlTtsEngine)} />
      <Hint>{t(`tts.ggml_engine_${engine.id}_desc`)}</Hint>
      {statusLine}
      {ui.last && ui.last.engine === engine.id && (
        <Hint>{t("tts.ggml_last_sentence", { first: ui.last.firstAudioMs, seconds: ui.last.audioSec })}</Hint>
      )}

      <InputSelect label="tts.ggml_model" value={model.key} options={modelOptions} onValueChange={changeModel} />
      <InputSelect label="tts.ggml_variant" value={data.variant} options={variantOptions} onValueChange={(v) => up("variant", v as GgmlTtsVariant)} />
      {plan.refused && <Hint tone="warning">{plan.refused}</Hint>}
      {!plan.refused && plan.pack.variant === "cpu" && <Hint tone="warning">{t("tts.ggml_cpu_slow")}</Hint>}
      <ModelRequirement ids={plan.ids} reason={t("tts.ggml_requirement", { engine: engine.name })} />
      {localMissing.length > 0 && (
        <Hint>{t(engine.id === "s2" ? "tts.ggml_s2_pack_hint" : model.key === "v2607" ? "tts.ggml_magpie_v2607_hint" : "tts.ggml_magpie_tokenizer_hint")}</Hint>
      )}

      {engine.id === "magpie" && (
        <InputSelect
          label="tts.field_voice"
          value={speakerValue}
          options={MAGPIE_SPEAKERS.map((v) => ({ label: v, value: v }))}
          onValueChange={(v) => up("voice", v)}
        />
      )}

      {engine.instructions && (
        <div className="flex flex-col gap-1">
          <label className="font-medium text-base-content/80 text-xs" htmlFor={instructionId}>
            {t(cloning ? "tts.ggml_breeze_direction" : "tts.ggml_breeze_description")}
          </label>
          <textarea
            id={instructionId}
            className="textarea textarea-bordered textarea-sm w-full text-xs leading-snug"
            rows={3}
            value={data.instruction}
            placeholder={cloning ? t("tts.ggml_breeze_direction_placeholder") : BREEZE_EXAMPLES[0]}
            onChange={(e) => up("instruction", e.target.value)}
          />
          {!cloning && (
            <div className="flex flex-wrap gap-1">
              {BREEZE_EXAMPLES.map((example, i) => (
                <button key={example} type="button" className="btn btn-xs btn-ghost" title={example} onClick={() => up("instruction", example)}>
                  {t("tts.ggml_example", { n: i + 1 })}
                </button>
              ))}
            </div>
          )}
          <Hint>{t("tts.ggml_breeze_hint")}</Hint>
        </div>
      )}

      {engine.cloning && (
        <>
          <ReferenceVoicePicker
            label="tts.ggml_clone_voice"
            value={data.referenceVoiceId}
            allowNone
            needsTranscript
            onChange={(id) => up("referenceVoiceId", id)}
          />
          <Hint>{t(cloning ? "tts.ggml_clone_hint" : `tts.ggml_no_clone_${engine.id}`)}</Hint>
        </>
      )}

      {engine.id === "omnivoice" && cloning && (
        <>
          <InputCheckbox label="tts.ggml_omnivoice_consent" value={data.cloneConsent} onChange={(v) => up("cloneConsent", v)} />
          <InputCheckbox label="tts.ggml_omnivoice_disclose" value={data.discloseAi} onChange={(v) => up("discloseAi", v)} />
          <Hint>{t(data.discloseAi ? "tts.ggml_omnivoice_disclose_on" : "tts.ggml_omnivoice_disclose_off")}</Hint>
        </>
      )}

      {languages && <InputSelect label="tts.ggml_language" value={languageValue} options={languageOptions} onValueChange={(v) => up("language", v)} />}
      {engine.id === "magpie" && model.key === "v2602" && <Hint>{t("tts.ggml_magpie_languages_hint")}</Hint>}

      {engine.id === "breeze" && (
        <InputRange label="tts.ggml_breeze_cfg" value={data.cfgScale} min="1" max="3" step="0.1" onChange={(e) => up("cfgScale", e.target.value)} />
      )}
      {engine.id === "omnivoice" && (
        <InputSelect
          label="tts.ggml_quality"
          value={data.quality}
          options={GGML_TTS_QUALITIES.map((q) => ({ label: t(`tts.ggml_quality_${q}`), value: q }))}
          onValueChange={(v) => up("quality", v as GgmlTtsQuality)}
        />
      )}
      {(engine.id === "breeze" || engine.id === "omnivoice") && (
        <InputText
          label="tts.ggml_seed"
          value={data.seed}
          placeholder={t("tts.ggml_seed_placeholder")}
          onChange={(e) => up("seed", e.target.value.replace(/[^0-9]/g, "").slice(0, 9))}
        />
      )}
      {engine.id === "s2" && (
        <>
          <InputCheckbox label="tts.ggml_s2_low_vram" value={data.lowVram} onChange={(v) => up("lowVram", v)} />
          <Hint>{t("tts.ggml_s2_hint")}</Hint>
        </>
      )}
      {problem && <Hint tone="warning">{problem}</Hint>}

      <div className="flex gap-2 items-center">
        <input
          type="text"
          className="input input-bordered input-sm flex-1 min-w-0 text-xs"
          aria-label={t("tts.ggml_preview_text")}
          title={t("tts.ggml_preview_text")}
          value={sample}
          placeholder={t(`tts.ggml_preview_default_${engine.id}`)}
          onChange={(e) => setSample(e.target.value)}
        />
        <button
          type="button"
          className="btn btn-sm btn-outline shrink-0 gap-1"
          disabled={!tauri || previewing || !ready || !!problem}
          onClick={preview}
          title={ready ? undefined : t("tts.ggml_preview_needs_download")}
        >
          {previewing ? <span className="loading loading-spinner loading-xs" /> : <RiPlayLine />}
          {t("tts.ggml_preview")}
        </button>
      </div>
      {tauri && status?.state === "ready" && service.status === ServiceNetworkState.disconnected && (
        <button type="button" className="btn btn-xs btn-ghost self-start" onClick={stopHelper}>
          {t("tts.ggml_unload")}
        </button>
      )}

      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={(v) => up("device", v)} />
      <InputRange label="common.field_volume" value={data.volume} min="0" max="2" step="0.05" onChange={(e) => up("volume", e.target.value)} />
    </>
  );
};
