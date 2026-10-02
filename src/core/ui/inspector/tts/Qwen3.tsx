import { ttsSinkTarget } from "@/core/services/tts/pipeline";
import { TTS_Backends } from "@/core/services/tts/schema";
import {
  QWEN3_TTS_MODELS,
  QWEN3_TTS_MODES,
  QWEN3_TTS_VARIANTS,
  type Qwen3TtsMode,
  type Qwen3TtsModel,
  type Qwen3TtsVariant,
  type TTS_Qwen3_State,
} from "@/core/services/tts/schema/qwen3";
import {
  QWEN3_DEFAULT_SPEAKER,
  QWEN3_DESIGN_EXAMPLES,
  QWEN3_LANGUAGES,
  QWEN3_SPEAKERS,
  QWEN3_STATE_EVENT,
  QWEN3_STYLE_PRESETS,
  planQwen3,
  previewQwen3,
  qwen3CudaUsable,
  qwen3HelperStatus,
  qwen3SettingsProblem,
  qwen3SizeOf,
  qwen3Ui,
  type Qwen3Recommendation,
} from "@/core/services/tts/services/qwen3";
import { errorMessage } from "@/core/services/tts/types";
import { ServiceNetworkState, type RustModelEntryStatus, type RustTtsHelperStatus } from "@/types";
import { invoke, isTauri } from "@tauri-apps/api/core";
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
import { InputRange, InputSelect, InputText } from "../components/input";
import { TtsOutputDevice } from "./Routing";

type Qwen3State = TTS_Qwen3_State;

/** Localized name of a language tag ("ja" -> "Japanese"), the tag itself when unknown. */
function languageName(tag: string, uiLanguage: string | undefined): string {
  try {
    const locales = uiLanguage ? [uiLanguage.replace("_", "-"), "en"] : ["en"];
    return new Intl.DisplayNames(locales, { type: "language" }).of(tag) ?? tag;
  } catch {
    return tag;
  }
}

/** The helper's state (`tts_helper_status`, then `tts_helpers:state` events) while the panel is open. */
function useHelperStatus(): RustTtsHelperStatus | null {
  const [status, setStatus] = useState<RustTtsHelperStatus | null>(null);
  useEffect(() => {
    if (!isTauri()) return;
    let live = true;
    let unlisten: (() => void) | null = null;
    qwen3HelperStatus()
      .then((s) => live && setStatus(s))
      .catch(() => {});
    listen<RustTtsHelperStatus>(QWEN3_STATE_EVENT, (e) => {
      if (live && e.payload?.engine === "qwen3") setStatus(e.payload);
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
  }, []);
  return status;
}

/** Recommendation reason codes with their own sentence (`resources::policy::Reason`). */
const REASONS = new Set(["fits", "vc_running", "shared_gpu", "small_gpu"]);

const Hint: FC<{ tone?: "muted" | "warning" | "error"; children: React.ReactNode }> = ({ tone = "muted", children }) => (
  <div className={tone === "warning" ? "text-xs text-warning" : tone === "error" ? "text-xs text-error break-words" : "text-xs text-base-content/60"}>{children}</div>
);

/**
 * Panel of the `qwen3` backend (P6-E): Qwen3-TTS through Sigil's qwentts.cpp helper. Mode (built-in voices with style
 * instructions, a cloned library voice, or a designed voice), size (automatic: the resource governor) and runtime pack,
 * the downloads the choice needs, the voice for the mode, language, seed, a test sentence, and the sink output.
 */
export const Qwen3: FC = () => {
  const { t } = useTranslation();
  const tauri = isTauri();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.qwen3) as Qwen3State;
  useModelsLoaded();
  const models = useSnapshot(window.ApiServer.models.state);
  const states = models.states as Record<string, RustModelEntryStatus | undefined>;
  const recs = useSnapshot(window.ApiServer.resources.state).recommendations;
  const sys = useSnapshot(window.ApiServer.sysinfo.state);
  const ui = useSnapshot(qwen3Ui);
  const service = useSnapshot(window.ApiServer.tts.serviceState);
  const status = useHelperStatus();
  const [previewing, setPreviewing] = useState(false);
  const [sample, setSample] = useState("");
  const instructionId = useId();
  const descriptionId = useId();

  const up = <K extends keyof Qwen3State>(key: K, v: Qwen3State[K]) =>
    window.ApiServer.patchService("tts", (s) => {
      s.data.qwen3[key] = v;
    });

  const rec = recs.qwen3_tts as Qwen3Recommendation | undefined;
  // sys.info: the plan reads CUDA eligibility through qwen3CudaUsable (recomputed when the probe lands).
  const cudaUsable = useMemo(() => qwen3CudaUsable(), [sys.info]);
  const plan = useMemo(
    () => planQwen3(data, { recommendation: rec, installed: (id) => states[id]?.state === "installed", cudaUsable }),
    [data.model, data.variant, data.mode, rec, states, cudaUsable],
  );
  const problem = qwen3SettingsProblem(data);
  const ready = plan.ids.every((id) => states[id]?.state === "installed");
  // The server packs are built locally and imported (never downloaded); the models download normally.
  const packMissing = states[plan.runtime]?.state !== "installed";

  const sizeLabel = (size: string) => (size === "0.6b" ? "0.6B" : "1.7B");
  const modeOptions = QWEN3_TTS_MODES.map((m) => ({ label: t(`tts.qwen3_mode_${m}`), value: m }));
  const modelOptions = QWEN3_TTS_MODELS.map((m) => ({
    label: m === "auto" ? t("tts.qwen3_model_auto", { size: sizeLabel(plan.size) }) : `Qwen3-TTS ${sizeLabel(m)}`,
    value: m,
  }));
  const variantOptions = QWEN3_TTS_VARIANTS.map((v) => ({
    label: v === "auto" ? t("tts.qwen3_variant_auto", { variant: t(`tts.qwen3_variant_${plan.variant}`) }) : t(`tts.qwen3_variant_${v}`),
    value: v,
  }));
  const languageOptions = useMemo(
    () => [
      { label: t("tts.qwen3_language_auto"), value: "auto" },
      ...QWEN3_LANGUAGES.map(([code]) => ({ label: languageName(code, i18n.language), value: code })),
    ],
    [t],
  );
  const speakerOptions: { label: string; value: string }[] = QWEN3_SPEAKERS.map((s) => ({
    label: t("tts.qwen3_speaker_option", { name: s.name, description: t(`tts.qwen3_speaker_${s.id}`), language: languageName(s.lang, i18n.language) }),
    value: s.id,
  }));
  const currentSpeaker = data.voice.trim() || QWEN3_DEFAULT_SPEAKER;
  if (!speakerOptions.some((o) => o.value === currentSpeaker)) speakerOptions.push({ label: currentSpeaker, value: currentSpeaker });
  const languageValue = (() => {
    const l = data.language.trim().toLowerCase();
    if (!l || l === "auto") return "auto";
    const base = l.split(/[-_]/)[0];
    return QWEN3_LANGUAGES.some(([code]) => code === base) ? base : "auto";
  })();

  const reason = (() => {
    if (data.model !== "auto" || data.mode === "design" || !rec) return null;
    const code = rec.reason_code ?? "";
    if (rec.device === "cpu") return t("tts.qwen3_reason_cpu");
    return REASONS.has(code) ? t(`tts.qwen3_reason_${code}`, { size: sizeLabel(plan.size) }) : null;
  })();

  const runningSize = qwen3SizeOf(status?.model);
  const statusLine = (() => {
    if (!status) return null;
    switch (status.state) {
      case "starting":
        return <Hint>{t("tts.qwen3_status_starting", { model: status.model ?? "" })}</Hint>;
      case "ready":
        return <Hint>{t("tts.qwen3_status_ready", { model: status.model ?? "", size: runningSize ? sizeLabel(runningSize) : "" })}</Hint>;
      case "failed":
        return <Hint tone="error">{t("tts.qwen3_status_failed", { error: status.error ?? "" })}</Hint>;
      default:
        return null;
    }
  })();

  const preview = useCallback(async () => {
    const state = window.ApiServer.state.services.tts.data;
    setPreviewing(true);
    try {
      const text = sample.trim() || t("tts.qwen3_preview_default");
      const target = ttsSinkTarget({ ...state, backend: TTS_Backends.qwen3 }, window.ApiServer.state.audioOutputDevice ?? "");
      await previewQwen3(state.qwen3, text, target);
    } catch (e) {
      toast.error(t("tts.qwen3_preview_failed", { error: errorMessage(e) }));
    } finally {
      setPreviewing(false);
    }
  }, [sample, t]);

  const stopHelper = useCallback(() => {
    void invoke("plugin:tts-helpers|tts_helper_stop", { engine: "qwen3" }).catch((e) => toast.error(errorMessage(e)));
  }, []);

  return (
    <>
      <Inspector.SubHeader>{t("tts.qwen3_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("tts.qwen3_desc")}</Inspector.Description>
      {!tauri && <Hint>{t("tts.qwen3_desktop_only")}</Hint>}
      {statusLine}
      {ui.last && (
        <Hint>
          {t("tts.qwen3_last_sentence", { first: ui.last.firstAudioMs, seconds: ui.last.audioSec })}
        </Hint>
      )}

      <InputSelect label="tts.qwen3_mode" value={data.mode} options={modeOptions} onValueChange={(v) => up("mode", v as Qwen3TtsMode)} />
      <Inspector.Deactivatable active={data.mode !== "design"}>
        <InputSelect label="tts.qwen3_model" value={data.model} options={modelOptions} onValueChange={(v) => up("model", v as Qwen3TtsModel)} />
      </Inspector.Deactivatable>
      {data.mode === "design" && <Hint>{t("tts.qwen3_design_large_only")}</Hint>}
      {reason && <Hint>{reason}</Hint>}
      {ui.swapBlocked && data.model === "auto" && (
        <Hint tone="warning">{t("tts.qwen3_swap_blocked", { size: sizeLabel(qwen3SizeOf(ui.swapBlocked) ?? "0.6b") })}</Hint>
      )}
      <InputSelect label="tts.qwen3_variant" value={data.variant} options={variantOptions} onValueChange={(v) => up("variant", v as Qwen3TtsVariant)} />
      {plan.variant === "cpu" && <Hint tone="warning">{t("tts.qwen3_cpu_slow")}</Hint>}
      <ModelRequirement ids={plan.ids} reason={t("tts.qwen3_requirement")} />
      {packMissing && <Hint>{t("tts.qwen3_pack_hint")}</Hint>}

      {data.mode === "preset" && (
        <>
          <InputSelect label="tts.field_voice" value={currentSpeaker} options={speakerOptions} onValueChange={(v) => up("voice", v)} />
          <div className="flex flex-col gap-1">
            <label className="font-medium text-base-content/80 text-xs" htmlFor={instructionId}>
              {t("tts.qwen3_instruction")}
            </label>
            <input
              id={instructionId}
              type="text"
              className="input input-bordered input-sm w-full text-xs"
              value={data.instruction}
              placeholder={t("tts.qwen3_instruction_placeholder")}
              onChange={(e) => up("instruction", e.target.value)}
            />
            <div className="flex flex-wrap gap-1">
              {QWEN3_STYLE_PRESETS.map((p) => (
                <button
                  key={p.key}
                  type="button"
                  className={data.instruction.trim() === p.instruction ? "btn btn-xs btn-primary" : "btn btn-xs btn-ghost"}
                  onClick={() => up("instruction", data.instruction.trim() === p.instruction ? "" : p.instruction)}
                >
                  {t(`tts.qwen3_style_${p.key}`)}
                </button>
              ))}
            </div>
            {!plan.instructions && <Hint tone="warning">{t("tts.qwen3_instruction_needs_large")}</Hint>}
          </div>
        </>
      )}

      {data.mode === "clone" && (
        <>
          <ReferenceVoicePicker
            label="tts.qwen3_clone_voice"
            value={data.referenceVoiceId}
            allowNone={false}
            needsTranscript
            onChange={(id) => up("referenceVoiceId", id)}
          />
          <Hint>{t("tts.qwen3_clone_hint")}</Hint>
        </>
      )}

      {data.mode === "design" && (
        <div className="flex flex-col gap-1">
          <label className="font-medium text-base-content/80 text-xs" htmlFor={descriptionId}>
            {t("tts.qwen3_design_description")}
          </label>
          <textarea
            id={descriptionId}
            className="textarea textarea-bordered textarea-sm w-full text-xs leading-snug"
            rows={3}
            value={data.voiceDescription}
            placeholder={QWEN3_DESIGN_EXAMPLES[0]}
            onChange={(e) => up("voiceDescription", e.target.value)}
          />
          <div className="flex flex-wrap gap-1">
            {QWEN3_DESIGN_EXAMPLES.map((example, i) => (
              <button key={example} type="button" className="btn btn-xs btn-ghost" title={example} onClick={() => up("voiceDescription", example)}>
                {t("tts.qwen3_design_example", { n: i + 1 })}
              </button>
            ))}
          </div>
          <Hint>{t("tts.qwen3_design_hint")}</Hint>
        </div>
      )}

      <InputSelect label="tts.qwen3_language" value={languageValue} options={languageOptions} onValueChange={(v) => up("language", v)} />
      <InputText
        label="tts.qwen3_seed"
        value={data.seed}
        placeholder={data.mode === "design" ? t("tts.qwen3_seed_design_placeholder") : t("tts.qwen3_seed_placeholder")}
        onChange={(e) => up("seed", e.target.value.replace(/[^0-9]/g, "").slice(0, 15))}
      />
      {problem && <Hint tone="warning">{problem}</Hint>}

      <div className="flex gap-2 items-center">
        <input
          type="text"
          className="input input-bordered input-sm flex-1 min-w-0 text-xs"
          aria-label={t("tts.qwen3_preview_text")}
          title={t("tts.qwen3_preview_text")}
          value={sample}
          placeholder={t("tts.qwen3_preview_default")}
          onChange={(e) => setSample(e.target.value)}
        />
        <button
          type="button"
          className="btn btn-sm btn-outline shrink-0 gap-1"
          disabled={!tauri || previewing || !ready || !!problem}
          onClick={preview}
          title={ready ? undefined : t("tts.qwen3_preview_needs_download")}
        >
          {previewing ? <span className="loading loading-spinner loading-xs" /> : <RiPlayLine />}
          {t("tts.qwen3_preview")}
        </button>
      </div>
      {tauri && status?.state === "ready" && service.status === ServiceNetworkState.disconnected && (
        <button type="button" className="btn btn-xs btn-ghost self-start" onClick={stopHelper}>
          {t("tts.qwen3_unload")}
        </button>
      )}

      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={(v) => up("device", v)} />
      <InputRange label="common.field_volume" value={data.volume} min="0" max="2" step="0.05" onChange={(e) => up("volume", e.target.value)} />
    </>
  );
};
