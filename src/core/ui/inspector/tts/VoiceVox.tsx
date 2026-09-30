import { finiteOr } from "@/core/services/tts/num";
import type { TTS_State } from "@/core/services/tts/schema";
import { fetchVoicevoxSpeakers } from "@/core/services/tts/services/voicevox";
import { errorMessage } from "@/core/services/tts/types";
import {
  VOICEVOX_PRESET_HOSTS,
  VOICEVOX_PRESET_WEBSITES,
  VOICEVOX_RANGES,
  VoicevoxHostPreset,
  applyVoicevoxPreset,
} from "@/core/services/tts/voicevoxPresets";
import { ServiceNetworkState } from "@/types";
import classNames from "classnames";
import { FC, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputRange, InputSelect, InputText } from "../components/input";
import { TtsOutputDevice } from "./Routing";

interface VoicevoxSpeakerOption {
  label: string;
  value: string;
}

export const VoiceVox: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.voicevox);
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  const handleUpdate = <K extends keyof TTS_State["voicevox"]>(key: K, v: TTS_State["voicevox"][K]) =>
    window.ApiServer.patchService("tts", s => s.data.voicevox[key] = v);

  const [speakers, setSpeakers] = useState<VoicevoxSpeakerOption[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /** Only the latest speaker request may update the list (the URL can change mid-request). */
  const requestRef = useRef(0);

  const isAivis = data.hostPreset === "aivisspeech";
  const website = VOICEVOX_PRESET_WEBSITES[isAivis ? "aivisspeech" : "voicevox"];
  const pitchIsZero = finiteOr(data.pitchScale, 0) === 0;

  const selectPreset = (preset: VoicevoxHostPreset) => {
    const patch = applyVoicevoxPreset(preset);
    window.ApiServer.patchService("tts", s => { Object.assign(s.data.voicevox, patch); });
  };

  const loadSpeakers = async () => {
    const request = ++requestRef.current;
    const host = window.ApiServer.state.services.tts.data.voicevox.host;
    setLoading(true);
    setError(null);
    try {
      const speakerData = await fetchVoicevoxSpeakers(host);
      if (request !== requestRef.current) return;
      const options: VoicevoxSpeakerOption[] = speakerData.flatMap(speaker =>
        (speaker.styles ?? []).map(style => ({
          label: `${speaker.name} (${style.name})`,
          value: String(style.id),
        })));
      setSpeakers(options);
      if (options.length === 0) {
        setError(t("tts.voicevox_no_speakers"));
        return;
      }
      // Style ids differ per engine (a VOICEVOX id means nothing to AivisSpeech): keep the choice
      // when the engine has it, else pick its first style.
      const current = window.ApiServer.state.services.tts.data.voicevox.speaker;
      if (!options.some(o => o.value === current)) handleUpdate("speaker", options[0].value);
    } catch (err) {
      if (request !== requestRef.current) return;
      setSpeakers([]);
      setError(errorMessage(err) || t("tts.voicevox_no_speakers"));
    } finally {
      if (request === requestRef.current) setLoading(false);
    }
  };

  useEffect(() => {
    // Debounced, so typing a custom URL does not fire one request per keystroke.
    const timer = setTimeout(() => void loadSpeakers(), 300);
    return () => clearTimeout(timer);
  }, [data.host]);

  return <>
    <Inspector.SubHeader>{t("tts.backend_voicevox")}</Inspector.SubHeader>
    <Inspector.Description>
      <span className="text-base-content/60 text-xs">
        {t("tts.voicevox_desc")}{" "}
        <a className="link link-primary link-hover" target="_blank" rel="noopener noreferrer" href={website}>
          {website.replace(/^https:\/\//, "").replace(/\/$/, "")}
        </a>
      </span>
    </Inspector.Description>

    <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
      <InputSelect
        label="tts.voicevox_preset"
        value={data.hostPreset}
        onValueChange={e => selectPreset(e as VoicevoxHostPreset)}
        options={[
          { label: t("tts.voicevox_preset_voicevox"), value: "voicevox" },
          { label: t("tts.voicevox_preset_aivisspeech"), value: "aivisspeech" },
          { label: t("tts.voicevox_preset_custom"), value: "custom" },
        ]}
      />
      {data.hostPreset === "custom" ? (
        <InputText
          label="tts.voicevox_host"
          placeholder={VOICEVOX_PRESET_HOSTS.voicevox}
          value={data.host}
          onChange={e => handleUpdate("host", e.target.value)}
        />
      ) : (
        <div className="text-xs text-base-content/60 mb-1">{t("tts.voicevox_preset_host_note", { host: data.host })}</div>
      )}

      {error && (
        <div className="text-error text-xs py-1">{error}</div>
      )}

      <div className="flex items-center gap-2">
        <div className="flex-1 min-w-0">
          <InputSelect
            label="tts.voicevox_speaker"
            value={data.speaker}
            onValueChange={e => handleUpdate("speaker", e)}
            options={speakers}
          />
        </div>
        <button
          type="button"
          className={classNames("btn btn-sm btn-primary", { loading })}
          onClick={() => void loadSpeakers()}
          disabled={loading}
        >
          {t("tts.http_refresh_list")}
        </button>
      </div>

      <TtsOutputDevice
        label="common.field_output_device"
        value={data.device}
        onChange={e => handleUpdate("device", e)}
      />
    </Inspector.Deactivatable>

    <InputRange
      value={data.speedScale}
      onChange={e => handleUpdate("speedScale", e.target.value)}
      label="tts.voicevox_speed"
      step={String(VOICEVOX_RANGES.speedScale.step)}
      min={String(VOICEVOX_RANGES.speedScale.min)}
      max={String(VOICEVOX_RANGES.speedScale.max)}
    />
    <div className="flex items-center gap-2">
      <div className="flex-1 min-w-0">
        <InputRange
          value={data.pitchScale}
          onChange={e => handleUpdate("pitchScale", e.target.value)}
          label="tts.voicevox_pitch"
          step={String(VOICEVOX_RANGES.pitchScale.step)}
          min={String(VOICEVOX_RANGES.pitchScale.min)}
          max={String(VOICEVOX_RANGES.pitchScale.max)}
        />
      </div>
      <button
        type="button"
        className="btn btn-xs btn-ghost shrink-0"
        onClick={() => handleUpdate("pitchScale", "0")}
        disabled={pitchIsZero}
      >
        {t("tts.voicevox_reset_pitch")}
      </button>
    </div>
    <InputRange
      value={data.intonationScale}
      onChange={e => handleUpdate("intonationScale", e.target.value)}
      label="tts.voicevox_intonation"
      step={String(VOICEVOX_RANGES.intonationScale.step)}
      min={String(VOICEVOX_RANGES.intonationScale.min)}
      max={String(VOICEVOX_RANGES.intonationScale.max)}
    />
    <InputRange
      value={data.volumeScale}
      onChange={e => handleUpdate("volumeScale", e.target.value)}
      label="common.field_volume"
      step={String(VOICEVOX_RANGES.volumeScale.step)}
      min={String(VOICEVOX_RANGES.volumeScale.min)}
      max={String(VOICEVOX_RANGES.volumeScale.max)}
    />
    {isAivis && (
      <div className={classNames("text-xs mb-1", pitchIsZero ? "text-base-content/60" : "text-warning")}>
        {t("tts.voicevox_aivis_hint")}
      </div>
    )}
  </>
}
