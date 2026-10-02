import { parseTtsFailure, ttsSinkTarget } from "@/core/services/tts/pipeline";
import { TTS_Backends, type TTS_State } from "@/core/services/tts/schema";
import {
  EDGE_DEFAULT_VOICE,
  FALLBACK_MODEL,
  edgeVoiceLabel,
  edgeVoiceLanguage,
  edgeVoiceLocale,
  edgeVoicesState,
  groupEdgeVoices,
  isEdgeVoiceName,
  loadEdgeVoices,
  previewEdgeVoice,
  type CloudFallbackSettings,
} from "@/core/services/tts/services/edge";
import { previewText } from "@/core/services/tts/services/localNeural";
import { ServiceNetworkState, type RustEdgeTtsVoice } from "@/types";
import { isTauri } from "@tauri-apps/api/core";
import i18n from "i18next";
import { FC, ReactNode, useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { RiPlayLine, RiRefreshLine } from "react-icons/ri";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { ModelRequirement } from "../components/ModelRequirement";
import { InputCheckbox, InputRange, InputSelect, InputText } from "../components/input";
import { TtsOutputDevice } from "./Routing";

type EdgeState = TTS_State["edge"];

export const Hint: FC<{ tone?: "muted" | "warning" | "error"; children: ReactNode }> = ({ tone = "muted", children }) => (
  <div className={tone === "warning" ? "text-xs text-warning" : tone === "error" ? "text-xs text-error break-words" : "text-xs text-base-content/60"}>
    {children}
  </div>
);

/**
 * The "if the service fails" section of the cloud panels (Edge, Gemini): the local Kokoro voice as fallback (with its
 * consented download), and how long a silent service is waited for before Kokoro takes over.
 */
export const CloudFallbackSection: FC<{
  value: CloudFallbackSettings;
  stopped: boolean;
  onChange: <K extends keyof CloudFallbackSettings>(key: K, v: CloudFallbackSettings[K]) => void;
}> = ({ value, stopped, onChange }) => {
  const { t } = useTranslation();
  return (
    <>
      <Inspector.SubHeader>{t("tts.cloud_fallback_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("tts.cloud_fallback_desc")}</Inspector.Description>
      <Inspector.Deactivatable active={stopped}>
        <InputCheckbox label="tts.cloud_fallback_enable" value={value.fallbackToLocal} onChange={(v) => onChange("fallbackToLocal", v)} />
      </Inspector.Deactivatable>
      {value.fallbackToLocal && (
        <>
          <InputRange
            label="tts.cloud_fallback_after"
            value={value.fallbackAfter}
            min="1"
            max="10"
            step="0.5"
            onChange={(e) => onChange("fallbackAfter", e.target.value)}
          />
          <Hint>{t("tts.cloud_fallback_after_hint")}</Hint>
          <ModelRequirement ids={[FALLBACK_MODEL]} reason={t("tts.cloud_fallback_requirement")} hideWhenReady />
        </>
      )}
    </>
  );
};

/** A test-sentence field with a play button (the cloud panels' "Test voice"). */
export const CloudPreview: FC<{
  placeholder: string;
  disabled: boolean;
  title?: string;
  onPreview: (text: string) => Promise<void>;
}> = ({ placeholder, disabled, title, onPreview }) => {
  const { t } = useTranslation();
  const [sample, setSample] = useState("");
  const [busy, setBusy] = useState(false);
  const run = async () => {
    setBusy(true);
    try {
      await onPreview(sample.trim() || placeholder);
    } catch (e) {
      // Readable text only: never the cloud plugins' internal failure tag.
      toast.error(t("tts.cloud_preview_failed", { error: parseTtsFailure(e).message }));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="flex gap-2 items-center">
      <input
        type="text"
        className="input input-bordered input-sm flex-1 min-w-0 text-xs"
        aria-label={t("tts.cloud_preview_text")}
        title={t("tts.cloud_preview_text")}
        value={sample}
        placeholder={placeholder}
        onChange={(e) => setSample(e.target.value)}
      />
      <button type="button" className="btn btn-sm btn-outline shrink-0 gap-1" disabled={disabled || busy} title={title} onClick={() => void run()}>
        {busy ? <span className="loading loading-spinner loading-xs" /> : <RiPlayLine />}
        {t("tts.cloud_preview")}
      </button>
    </div>
  );
};

/** The locale of a voice: from the list, else from its short name (`ja-JP-NanamiNeural` → `ja-JP`). */
function localeOf(voice: string, voices: readonly RustEdgeTtsVoice[]): string {
  const row = voices.find((v) => v.shortName === voice);
  if (row?.locale) return row.locale;
  return edgeVoiceLocale({ shortName: voice, locale: "" });
}

/**
 * Panel of the `edge` backend (P6-H): Microsoft Edge "Read aloud" voices, free and without a key. Language and voice
 * pickers from the service's voice list (a voice name can be typed while the list is unavailable), rate and pitch, a
 * test sentence, the local fallback voice, and the sink output. Everything applies to the next sentence.
 */
export const Edge: FC = () => {
  const { t } = useTranslation();
  const tauri = isTauri();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.edge) as EdgeState;
  const service = useSnapshot(window.ApiServer.tts.serviceState);
  const list = useSnapshot(edgeVoicesState);
  const voices = list.voices as RustEdgeTtsVoice[];

  const up = <K extends keyof EdgeState>(key: K, v: EdgeState[K]) =>
    window.ApiServer.patchService("tts", (s) => {
      s.data.edge[key] = v;
    });

  useEffect(() => {
    if (tauri) void loadEdgeVoices();
  }, [tauri]);

  const stopped = service.status === ServiceNetworkState.disconnected;
  const voice = data.voice.trim() || EDGE_DEFAULT_VOICE;
  const validVoice = isEdgeVoiceName(voice);
  const groups = useMemo(() => groupEdgeVoices(voices, i18n.language), [voices]);
  const locale = localeOf(voice, voices);

  const localeOptions = useMemo(() => {
    const options = groups.map((g) => ({ label: g.label, value: g.locale }));
    if (locale && !options.some((o) => o.value === locale)) options.push({ label: locale, value: locale });
    return options;
  }, [groups, locale]);

  const voiceOptions = useMemo(() => {
    const options = (groups.find((g) => g.locale === locale)?.voices ?? []).map((v) => ({ label: edgeVoiceLabel(v), value: v.shortName }));
    // The voice is the select's value even when the list does not have it (typed, or the list changed).
    if (!options.some((o) => o.value === voice)) options.push({ label: voice, value: voice });
    return options;
  }, [groups, locale, voice, t]);

  const chooseLocale = (next: string) => {
    if (next === locale) return;
    const group = groups.find((g) => g.locale === next);
    if (!group?.voices.length) return;
    // Keep the voice's gender where the language has one.
    const gender = voices.find((v) => v.shortName === voice)?.gender;
    const pick = group.voices.find((v) => gender && v.gender === gender) ?? group.voices[0];
    up("voice", pick.shortName);
  };

  const preview = useCallback(async (text: string) => {
    const state = window.ApiServer.state.services.tts.data;
    const target = ttsSinkTarget({ ...state, backend: TTS_Backends.edge }, window.ApiServer.state.audioOutputDevice ?? "");
    await previewEdgeVoice(state.edge, text, target);
  }, []);

  return (
    <>
      <Inspector.SubHeader>{t("tts.edge_title")}</Inspector.SubHeader>
      <Inspector.Description>{t("tts.edge_desc")}</Inspector.Description>
      {!tauri && <Hint>{t("tts.cloud_desktop_only")}</Hint>}

      {voices.length > 0 ? (
        <>
          <InputSelect label="tts.edge_language" value={locale} options={localeOptions} onValueChange={chooseLocale} />
          <InputSelect label="tts.field_voice" value={voice} options={voiceOptions} onValueChange={(v) => up("voice", v)} />
        </>
      ) : (
        <InputText
          label="tts.field_voice"
          value={data.voice}
          placeholder={EDGE_DEFAULT_VOICE}
          dir="ltr"
          onChange={(e) => up("voice", e.target.value)}
        />
      )}
      {list.loading && <Hint>{t("tts.edge_voices_loading")}</Hint>}
      {!list.loading && list.error && (
        <div className="flex gap-2 items-center">
          <div className="flex-1 min-w-0">
            <Hint tone="warning">{t("tts.edge_voices_failed", { error: list.error })}</Hint>
          </div>
          <button type="button" className="btn btn-xs btn-ghost gap-1 shrink-0" disabled={!tauri} onClick={() => void loadEdgeVoices(true)}>
            <RiRefreshLine /> {t("tts.edge_voices_retry")}
          </button>
        </div>
      )}
      {!validVoice && <Hint tone="error">{t("tts.edge_voice_invalid", { voice })}</Hint>}

      <InputRange label="tts.edge_rate" value={data.rate} min="0.5" max="2" step="0.05" onChange={(e) => up("rate", e.target.value)} />
      <InputRange label="tts.edge_pitch" value={data.pitchHz} min="-50" max="50" step="1" onChange={(e) => up("pitchHz", e.target.value)} />
      <CloudPreview
        placeholder={previewText(edgeVoiceLanguage(voice))}
        disabled={!tauri || !validVoice}
        onPreview={preview}
      />
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={(v) => up("device", v)} />
      <InputRange label="common.field_volume" value={data.volume} min="0" max="2" step="0.05" onChange={(e) => up("volume", e.target.value)} />

      <CloudFallbackSection value={data} stopped={stopped} onChange={(key, v) => up(key, v as EdgeState[typeof key])} />
    </>
  );
};
