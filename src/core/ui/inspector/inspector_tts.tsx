import { TTS_Backends, TTS_State } from "@/core/services/tts/schema";
import { ServiceNetworkState } from "@/types";
import { devLog } from "@/utils/devLog";
import NiceModal from "@ebay/nice-modal-react";
import { invoke, isTauri } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { FC, useEffect, useRef, useState } from "react";
import { RiCharacterRecognitionFill, RiChatVoiceFill, RiInformationLine } from "react-icons/ri";
import { proxy, useSnapshot } from "valtio";
import { azureVoices } from "../../services/tts/tts_data";
import { TTS_FishSpeechService } from "../../services/tts/services/fishspeech";
import { TTS_KokoroService } from "../../services/tts/services/kokoro";
import { TTS_MeloService } from "../../services/tts/services/melo";
import { fetchVoicevoxSpeakers } from "../../services/tts/services/voicevox";
import { normalizeAzureVoiceName } from "../../services/tts/ssml";
import { errorMessage } from "../../services/tts/types";
import { finiteOr } from "../../services/tts/num";
import {
  VOICEVOX_PRESET_HOSTS,
  VOICEVOX_PRESET_WEBSITES,
  VOICEVOX_RANGES,
  VoicevoxHostPreset,
  applyVoicevoxPreset,
} from "../../services/tts/voicevoxPresets";
import Modal from "../Modal";
import ServiceButton from "../service-button";
import Inspector from "./components";
import { InputCheckbox, InputMapObject, InputMappedGroupSelect, InputNativeAudioOutput, InputRange, InputSelect, InputSelectOption, InputText, InputTextSource } from "./components/input";
import classNames from "classnames";
import { useTranslation } from "react-i18next";
import { toast } from "react-toastify";
import Tooltip from "../dropdown/Tooltip";

type WindowsToken = {
  id: string;
  label: string;
}
type WindowsConfig = { devices: WindowsToken[], voices: WindowsToken[] };
const Windows: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.windows);
  const handleUpdate = <K extends keyof TTS_State["windows"]>(key: K, v: TTS_State["windows"][K]) => window.ApiServer.patchService("tts", s => s.data.windows[key] = v);

  const [config, setConfig] = useState<WindowsConfig>();

  useEffect(() => {
    invoke<WindowsConfig>("plugin:windows-tts|get_voices").then(setConfig);
  }, []);

  return <>
    <Inspector.SubHeader>{t('tts.windows_title')}</Inspector.SubHeader>
    <InputSelect
      value={data.device}
      onValueChange={e => handleUpdate("device", e)}
      options={config?.devices.map(d => ({ label: d.label, value: d.id })) || []}
      label="common.field_output_device" />
    {/* <InputNativeAudioOutput label="Audio Output" value={pr.device} onChange={e => handleUpdate("device", e)} /> */}
    <InputSelect
      value={data.voice}
      onValueChange={e => handleUpdate("voice", e)}
      options={config?.voices.map(d => ({ ...d, value: d.id })) || []}
      label="tts.field_voice" />
    <InputRange value={data.volume} onChange={e => handleUpdate("volume", e.target.value)} label="common.field_volume" step="0.05" min="0" max="1" />
    <InputRange value={data.rate} onChange={e => handleUpdate("rate", e.target.value)} label="common.field_rate" step="0.05" min="0.1" max="5" />
  </>
}

const Native: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.native);
  const handleUpdate = <K extends keyof TTS_State["native"]>(key: K, v: TTS_State["native"][K]) => window.ApiServer.patchService("tts", s => s.data.native[key] = v);
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  const [voices, setVoices] = useState<{ label: string, value: string }[]>([]);

  useEffect(() => {
    let voices = window.speechSynthesis.getVoices();

    if (voices.length) {
      setVoices(voices.map(v => ({ label: v.name, value: v.voiceURI })));
    }
    else {
      window.speechSynthesis.onvoiceschanged = () => {
        let options = window.speechSynthesis.getVoices().map(v => ({ label: v.name, value: v.voiceURI }))
        setVoices(options);
      }
    }

  }, []);

  return <>
    <Inspector.SubHeader>{t('tts.native_title')}</Inspector.SubHeader>
    <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
      <InputSelect
        value={data.voice}
        onValueChange={e => handleUpdate("voice", e)}
        options={voices}
        label="tts.field_voice" />
    </Inspector.Deactivatable>

    <InputRange value={data.pitch} onChange={e => handleUpdate("pitch", e.target.value)} label="common.field_pitch" step="0.1" min="0" max="2" />
    <InputRange value={data.rate} onChange={e => handleUpdate("rate", e.target.value)} label="common.field_rate" step="0.1" min="0.1" max="10" />
    <InputRange value={data.volume} onChange={e => handleUpdate("volume", e.target.value)} label="common.field_volume" step="0.05" min="0" max="1" />
  </>
}

const Azure: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.azure);
  const handleUpdate = <K extends keyof TTS_State["azure"]>(key: K, v: TTS_State["azure"][K]) =>
    window.ApiServer.patchService("tts", s => s.data.azure[key] = v);
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  const [voiceStyles, setVoiceStyles] = useState<{ label: string, value: string }[]>([]);
  const [voiceRoles, setVoiceRoles] = useState<{ label: string, value: string }[]>([]);

  const updateVoice = (value: { group: string, option: string }) => {
    window.ApiServer.patchService("tts", s => {
      s.data.azure.language = value.group;
      s.data.azure.voice = value.option;
      // Styles and roles differ per voice: clear them.
      s.data.azure.voiceStyle = "";
      s.data.azure.voiceRole = "";
    });
  };

  useEffect(() => {
    // find voice
    // set styles and roles
    if (!data.language || !data.voice)
      return;

    const voiceName = normalizeAzureVoiceName(data.voice);
    const voice = azureVoices[data.language]?.find(voice => voice[0] === voiceName);
    const voiceParams = voice?.[2];
    setVoiceStyles(voiceParams?.styles ? [{ label: "None", value: "" }, ...voiceParams.styles.map(style => ({ label: style, value: style }))] : []);
    setVoiceRoles(voiceParams?.roles ? [{ label: "None", value: "" }, ...voiceParams.roles.map(roles => ({ label: roles, value: roles }))] : []);

  }, [data.language, data.voice]);

  return <>
    <Inspector.SubHeader>{t('tts.azure_title')}</Inspector.SubHeader>
    <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
      <InputNativeAudioOutput label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
      <InputText type="password" label="tts.azure_key" value={data.key} onChange={e => handleUpdate("key", e.target.value)} />
      <InputText type="password" label="tts.azure_location" value={data.location} onChange={e => handleUpdate("location", e.target.value)} />
    </Inspector.Deactivatable>

    <InputMappedGroupSelect
      labelGroup="common.field_language"
      labelOption="common.field_dialect"
      value={{ group: data.language, option: data.voice }}
      onChange={updateVoice}
      library={azureVoices} />

    {voiceStyles.length > 0 && <InputSelect value={data.voiceStyle} options={voiceStyles} label="tts.azure_voice_style" onValueChange={e => handleUpdate("voiceStyle", e)} />}
    {voiceRoles.length > 0 && <InputSelect value={data.voiceRole} options={voiceRoles} label="tts.azure_voice_role" onValueChange={e => handleUpdate("voiceRole", e)} />}
    <InputSelect options={[
      { label: "silent", value: "silent" },
      { label: "x-soft", value: "x-soft" },
      { label: "soft", value: "soft" },
      { label: "medium", value: "medium" },
      { label: "loud", value: "loud" },
      { label: "x-loud", value: "x-loud" },
      { label: "default", value: "default" }
    ]} label="tts.azure_voice_volume" value={data.voiceVolume} onValueChange={e => handleUpdate("voiceVolume", e)} />
    <InputSelect options={[
      { label: "x-slow", value: "x-slow" },
      { label: "slow", value: "slow" },
      { label: "medium", value: "medium" },
      { label: "fast", value: "fast" },
      { label: "x-fast", value: "x-fast" },
      { label: "default", value: "default" },
    ]} label="tts.azure_voice_rate" value={data.voiceRate} onValueChange={e => handleUpdate("voiceRate", e)} />
    <InputSelect options={[
      { label: "x-low", value: "x-low" },
      { label: "low", value: "low" },
      { label: "medium", value: "medium" },
      { label: "high", value: "high" },
      { label: "x-high", value: "x-high" },
      { label: "default", value: "default" },
    ]} label="tts.azure_voice_pitch" value={data.voicePitch} onValueChange={e => handleUpdate("voicePitch", e)} />
    <InputSelect options={[
      { label: "x-low", value: "x-low" },
      { label: "low", value: "low" },
      { label: "medium", value: "medium" },
      { label: "high", value: "high" },
      { label: "x-high", value: "x-high" },
      { label: "default", value: "default" },
    ]} label="tts.azure_voice_range" value={data.voiceRange} onValueChange={e => handleUpdate("voiceRange", e)} />
    <InputRange value={data.volume} onChange={e => handleUpdate("volume", e.target.value)} label="common.field_volume" step="0.05" min="0" max="1" />
    {/* <InputRange value={data.rate} onChange={e => handleUpdate("rate", e.target.value)} label={`Rate (${data.rate})`} step="0.05" min="0.1" max="5" /> */}
  </>
}

interface VoicevoxSpeakerOption {
  label: string;
  value: string;
}

const VoiceVox: FC = () => {
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

      <InputNativeAudioOutput
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

const uberduckVoices = proxy<{ value: InputSelectOption[] }>({
  value: []
});

const UberDuck: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.uberduck);
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  const handleUpdate = <K extends keyof TTS_State["uberduck"]>(key: K, v: TTS_State["uberduck"][K]) => window.ApiServer.patchService("tts", s => s.data.uberduck[key] = v);
  const { value: voices } = useSnapshot(uberduckVoices);
  const [loadingVoices, setLoadingVoices] = useState(false);

  const loadVoices = () => {
    if (!data.api_key)
      return;
    setLoadingVoices(true);
    invoke("plugin:uberduck-tts|get_voices", {
      auth: {
        api_key: data.api_key,
        secret_key: data.secret_key,
      }
    }).then(res => {
      if (Array.isArray(res))
        uberduckVoices.value = res.map(v => ({ value: v.voicemodel_uuid, label: v.display_name }));
    }).catch(err => {
      // A wrong or expired API key is rejected here first (HTTP 401).
      toast.error(t("tts.uberduck_voices_failed", { error: errorMessage(err) }));
    }).finally(() => setLoadingVoices(false));
  }

  useEffect(() => {
    if (!voices.length)
      loadVoices();
  }, []);
  return <>
    <Inspector.SubHeader>{t('tts.uberduck_title')}</Inspector.SubHeader>
    <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
      <InputNativeAudioOutput label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
      <InputText label="tts.uberduck_api_key" type="password" value={data.api_key} onChange={e => handleUpdate("api_key", e.target.value)} />
      <Inspector.Description>
        <span className="text-base-content/60 text-xs mt-2">
          {t('tts.uberduck_voice_desc_key')}<br />
          <a className="link link-primary link-hover" target="_blank" rel="noopener noreferrer" href="https://app.uberduck.ai/speak#mode=tts-reference&voice=zwf">{t('tts.uberduck_voice_desc_link')}</a>
        </span>
      </Inspector.Description>
      <InputSelect
        value={data.voice}
        onValueChange={e => handleUpdate("voice", e)}
        options={voices as InputSelectOption[]}
        label="tts.field_voice" />
      <button onClick={loadVoices} className={classNames("btn btn-xs disabled", { loading: loadingVoices, "btn-disabled": !data.api_key })}>{loadingVoices ? t('tts.uberduck_voice_status_updating') : t('tts.uberduck_voice_status_reload')}</button>
      <InputRange label="common.field_volume" value={data.volume} onChange={e => handleUpdate("volume", e.target.value)} step="0.05" min="0" max="1" />
    </Inspector.Deactivatable>
  </>
}

const Kokoro: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.kokoro);
  const handleUpdate = <K extends keyof TTS_State["kokoro"]>(key: K, v: TTS_State["kokoro"][K]) =>
    window.ApiServer.patchService("tts", s => s.data.kokoro[key] = v);
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  const [voiceOpts, setVoiceOpts] = useState<InputSelectOption[]>([]);
  const [loading, setLoading] = useState(false);

  const refreshVoices = async () => {
    setLoading(true);
    try {
      const list = await TTS_KokoroService.getVoices(data.endpoint);
      setVoiceOpts(list.map((x) => ({ label: x.name, value: x.name })));
    } catch {
      toast.error(t("tts.http_list_fetch_failed"));
    } finally {
      setLoading(false);
    }
  };

  return <>
    <Inspector.SubHeader>{t("tts.kokoro_title")}</Inspector.SubHeader>
    <div className="text-xs text-base-content/70 mb-2">{t("tts.kokoro_desc")}</div>
    <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
      <InputText label="tts.kokoro_endpoint" value={data.endpoint} onChange={e => handleUpdate("endpoint", e.target.value)} />
      <div className="flex flex-wrap gap-2 items-end mb-1">
        <div className="flex-1 min-w-[12rem]">
          <InputSelect
            label="tts.field_voice"
            value={data.voice}
            onValueChange={e => handleUpdate("voice", e)}
            options={voiceOpts.length ? voiceOpts : [{ label: data.voice || "—", value: data.voice }]}
          />
        </div>
        <button type="button" className={classNames("btn btn-sm btn-primary mb-1", { loading })} onClick={() => void refreshVoices()} disabled={loading}>
          {t("tts.http_refresh_list")}
        </button>
      </div>
      <InputRange value={data.speed} onChange={e => handleUpdate("speed", e.target.value)} label="tts.kokoro_speed" step="0.05" min="0.5" max="2" />
      <div className="text-xs text-base-content/60 mb-1">{t("tts.neural_web_audio_output_note")}</div>
      <InputNativeAudioOutput label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
    </Inspector.Deactivatable>
  </>;
};

const Melo: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.melo);
  const handleUpdate = <K extends keyof TTS_State["melo"]>(key: K, v: TTS_State["melo"][K]) =>
    window.ApiServer.patchService("tts", s => s.data.melo[key] = v);
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  const [speakerOpts, setSpeakerOpts] = useState<InputSelectOption[]>([]);
  const [loading, setLoading] = useState(false);

  const refreshSpeakers = async () => {
    setLoading(true);
    try {
      const list = await TTS_MeloService.getSpeakers(data.endpoint);
      setSpeakerOpts(
        list.map((s) => ({
          label: `${s.name} (${s.language})`,
          value: s.id,
        })),
      );
    } catch {
      toast.error(t("tts.http_list_fetch_failed"));
    } finally {
      setLoading(false);
    }
  };

  return <>
    <Inspector.SubHeader>{t("tts.melo_title")}</Inspector.SubHeader>
    <div className="text-xs text-base-content/70 mb-2">{t("tts.melo_desc")}</div>
    <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
      <InputText label="tts.melo_endpoint" value={data.endpoint} onChange={e => handleUpdate("endpoint", e.target.value)} />
      <div className="flex flex-wrap gap-2 items-end mb-1">
        <div className="flex-1 min-w-[12rem]">
          <InputSelect
            label="tts.melo_speaker"
            value={data.speaker}
            onValueChange={e => handleUpdate("speaker", e)}
            options={speakerOpts.length ? speakerOpts : [{ label: data.speaker || "—", value: data.speaker }]}
          />
        </div>
        <button type="button" className={classNames("btn btn-sm btn-primary mb-1", { loading })} onClick={() => void refreshSpeakers()} disabled={loading}>
          {t("tts.http_refresh_list")}
        </button>
      </div>
      <InputRange value={data.speed} onChange={e => handleUpdate("speed", e.target.value)} label="tts.melo_speed" step="0.05" min="0.5" max="2" />
      <div className="text-xs text-base-content/60 mb-1">{t("tts.neural_web_audio_output_note")}</div>
      <InputNativeAudioOutput label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
    </Inspector.Deactivatable>
  </>;
};

const Chatterbox: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.chatterbox);
  const handleUpdate = <K extends keyof TTS_State["chatterbox"]>(key: K, v: TTS_State["chatterbox"][K]) =>
    window.ApiServer.patchService("tts", s => s.data.chatterbox[key] = v);
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  return <>
    <Inspector.SubHeader>{t("tts.chatterbox_title")}</Inspector.SubHeader>
    <div className="text-xs text-base-content/70 mb-2">{t("tts.chatterbox_desc")}</div>
    <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
      <InputText label="tts.chatterbox_endpoint" value={data.endpoint} onChange={e => handleUpdate("endpoint", e.target.value)} />
      <InputText label="tts.field_voice" value={data.voice} onChange={e => handleUpdate("voice", e.target.value)} />
      <InputRange value={data.speed} onChange={e => handleUpdate("speed", e.target.value)} label="tts.chatterbox_speed" step="0.05" min="0.5" max="2" />
      <InputRange value={data.exaggeration} onChange={e => handleUpdate("exaggeration", e.target.value)} label="tts.chatterbox_exaggeration" step="0.05" min="0" max="1" />
      <div className="text-xs text-base-content/60 mb-1">{t("tts.neural_web_audio_output_note")}</div>
      <InputNativeAudioOutput label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
    </Inspector.Deactivatable>
  </>;
};

const FishSpeech: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.fishSpeech);
  const handleUpdate = <K extends keyof TTS_State["fishSpeech"]>(key: K, v: TTS_State["fishSpeech"][K]) =>
    window.ApiServer.patchService("tts", s => s.data.fishSpeech[key] = v);
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  const [voiceOpts, setVoiceOpts] = useState<InputSelectOption[]>([]);
  const [loading, setLoading] = useState(false);

  const refreshVoices = async () => {
    setLoading(true);
    try {
      const list = await TTS_FishSpeechService.getVoices(data.endpoint, data.apiKey);
      setVoiceOpts(list.map((x) => ({ label: x.name || x.id, value: x.id })));
    } catch {
      toast.error(t("tts.http_list_fetch_failed"));
    } finally {
      setLoading(false);
    }
  };

  return <>
    <Inspector.SubHeader>{t("tts.fish_title")}</Inspector.SubHeader>
    <div className="text-xs text-base-content/70 mb-2">{t("tts.fish_desc")}</div>
    <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
      <InputText label="tts.fish_endpoint" value={data.endpoint} onChange={e => handleUpdate("endpoint", e.target.value)} />
      <InputText label="tts.fish_api_key" type="password" value={data.apiKey} onChange={e => handleUpdate("apiKey", e.target.value)} />
      <InputText label="tts.fish_model" placeholder="s2.1-pro-free" value={data.model} onChange={e => handleUpdate("model", e.target.value)} />
      <div className="text-xs text-base-content/60 mb-1">{t("tts.fish_hosted_hint")}</div>
      <div className="flex flex-wrap gap-2 items-end mb-1">
        <div className="flex-1 min-w-[12rem]">
          <InputSelect
            label="tts.fish_reference_id"
            value={data.referenceId}
            onValueChange={e => handleUpdate("referenceId", e)}
            options={voiceOpts.length ? voiceOpts : [{ label: data.referenceId || "—", value: data.referenceId }]}
          />
        </div>
        <button type="button" className={classNames("btn btn-sm btn-primary mb-1", { loading })} onClick={() => void refreshVoices()} disabled={loading}>
          {t("tts.http_refresh_list")}
        </button>
      </div>
      <div className="text-xs text-base-content/60 mb-1">{t("tts.neural_web_audio_output_note")}</div>
      <InputNativeAudioOutput label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
    </Inspector.Deactivatable>
  </>;
};

const WordsReplacementModal: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts);
  const up = <K extends keyof TTS_State>(key: K, v: TTS_State[K]) => window.ApiServer.patchService("tts", s => s.data[key] = v);

  return <Modal.Body width={420}>
    <Modal.Header>{t('word_replacements.title')}</Modal.Header>
    <Modal.Content>
      <div className="p-4">
        <InputCheckbox label="word_replacements.field_ignore_case" value={data.data.replaceWordsIgnoreCase} onChange={v => up("replaceWordsIgnoreCase", v)} />
        <InputMapObject keyPlaceholder={t('word_replacements.label_dictionary_key')} valuePlaceholder={t('word_replacements.label_dictionary_value')} addLabel="Add word" value={{ ...data.data.replaceWords }} onChange={e => up("replaceWords", e)} label="" />
      </div>
    </Modal.Content>
  </Modal.Body>
}
NiceModal.register('tts-replacements', (props) => <Modal.Base {...props}><WordsReplacementModal /></Modal.Base>);

const NeuralOnnx: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.neuralOnnx);
  const handleUpdate = <K extends keyof TTS_State["neuralOnnx"]>(key: K, v: TTS_State["neuralOnnx"][K]) =>
    window.ApiServer.patchService("tts", s => s.data.neuralOnnx[key] = v);
  const state = useSnapshot(window.ApiServer.tts.serviceState);

  const syncRustModelSource = async (source: "hub" | "local", localDir: string) => {
    try {
      await invoke("plugin:neural-onnx-tts|neural_onnx_tts_set_model_source", {
        source,
        localDir: source === "local" && localDir.trim() ? localDir.trim() : null,
      });
    } catch (err) {
      devLog("[ONNX TTS] set_model_source failed:", err);
    }
  };

  useEffect(() => {
    const ep = window.ApiServer.state.services.tts.data.neuralOnnx.executionProvider;
    void invoke("plugin:neural-onnx-tts|neural_onnx_tts_set_execution_hint", { hint: ep }).catch((err) => {
      devLog("[Local ONNX speech] execution hint on mount failed:", err);
    });
  }, []);

  useEffect(() => {
    const nx = window.ApiServer.state.services.tts.data.neuralOnnx;
    void syncRustModelSource(nx.modelSource, nx.localModelDir);
  }, [data.modelSource, data.localModelDir]);

  useEffect(() => {
    void invoke("plugin:neural-onnx-tts|neural_onnx_tts_set_local_package_format", {
      format: data.localPackageFormat,
    }).catch((err) => devLog("[ONNX TTS] set_local_package_format failed:", err));
  }, [data.localPackageFormat]);

  return <>
    <Inspector.SubHeader>{t("tts.neural_onnx_title")}</Inspector.SubHeader>
    <div className="text-xs text-base-content/70 mb-2">{t("tts.neural_onnx_desc")}</div>
    <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
      <InputSelect
        label="tts.neural_onnx_model_source"
        value={data.modelSource}
        onValueChange={async (v) => {
          const source = v as "hub" | "local";
          handleUpdate("modelSource", source);
          const dir = window.ApiServer.state.services.tts.data.neuralOnnx.localModelDir;
          await syncRustModelSource(source, dir);
        }}
        options={[
          { label: t("tts.neural_onnx_source_hub"), value: "hub" },
          { label: t("tts.neural_onnx_source_local"), value: "local" },
        ]}
      />
      {data.modelSource === "hub" && (
        <InputText label="tts.neural_onnx_hub" value={data.hubId} onChange={e => handleUpdate("hubId", e.target.value)} />
      )}
      {data.modelSource === "local" && (
        <>
          <InputSelect
            label="tts.neural_onnx_local_format"
            value={data.localPackageFormat}
            onValueChange={async (v) => {
              handleUpdate("localPackageFormat", v as TTS_State["neuralOnnx"]["localPackageFormat"]);
              try {
                await invoke("plugin:neural-onnx-tts|neural_onnx_tts_set_local_package_format", { format: v });
              } catch (err) {
                devLog("[ONNX TTS] set_local_package_format failed:", err);
              }
            }}
            options={[
              { label: t("tts.neural_onnx_local_format_speech_package_v1"), value: "speech_package_v1" },
            ]}
          />
          <p className="text-xs text-base-content/50 mb-1">{t("tts.neural_onnx_local_package_doc")}</p>
          <div className="flex flex-col gap-1">
            <label className="label py-0">
              <span className="label-text">{t("tts.neural_onnx_local_dir")}</span>
            </label>
            <div className="flex gap-2 items-center">
              <input
                type="text"
                className="input input-bordered input-sm flex-1 min-w-0 font-mono text-xs"
                value={data.localModelDir}
                onChange={(e) => handleUpdate("localModelDir", e.target.value)}
                onBlur={() => {
                  void syncRustModelSource("local", data.localModelDir);
                }}
              />
              {isTauri() ? (
                <button
                  type="button"
                  className="btn btn-sm btn-outline shrink-0"
                  onClick={async () => {
                    const path = await open({ directory: true, multiple: false });
                    if (path === null || Array.isArray(path)) return;
                    handleUpdate("localModelDir", path);
                    try {
                      await invoke("plugin:neural-onnx-tts|neural_onnx_tts_set_model_source", {
                        source: "local",
                        localDir: path,
                      });
                    } catch (err) {
                      devLog("[ONNX TTS] set_model_source after browse failed:", err);
                      toast.error(String(err));
                    }
                  }}
                >
                  {t("tts.neural_onnx_browse_folder")}
                </button>
              ) : (
                <span className="text-xs text-base-content/50 shrink-0 max-w-[10rem]">
                  {t("tts.neural_onnx_browse_unavailable")}
                </span>
              )}
            </div>
          </div>
          <p className="text-xs text-base-content/60 mb-1">{t("tts.neural_onnx_local_hint")}</p>
        </>
      )}
      <InputText label="tts.neural_onnx_voice" value={data.voice} onChange={e => handleUpdate("voice", e.target.value)} />
      <InputRange value={data.speed} onChange={e => handleUpdate("speed", e.target.value)} label="tts.neural_onnx_speed" step="0.05" min="0.5" max="2" />
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-2">
        <InputSelect
          label="tts.neural_onnx_ep"
          value={data.executionProvider}
          onValueChange={async (e) => {
            handleUpdate("executionProvider", e as TTS_State["neuralOnnx"]["executionProvider"]);
            try {
              await invoke("plugin:neural-onnx-tts|neural_onnx_tts_set_execution_hint", { hint: e });
            } catch (err) {
              devLog("[Local ONNX speech] execution hint failed:", err);
            }
          }}
          options={[
            { label: t("tts.ep_cpu_plain"), value: "cpu" },
            { label: t("tts.ep_cuda_plain"), value: "cuda" },
            { label: t("tts.ep_dml_plain"), value: "dml" },
          ]}
        />
        <Tooltip content={t("tts.neural_onnx_ep_help")}>
          <button
            type="button"
            className="btn btn-ghost btn-xs btn-circle mb-0.5 shrink-0"
            aria-label={t("voice_changer.info_learn_more")}
          >
            <RiInformationLine className="text-lg opacity-70" />
          </button>
        </Tooltip>
      </div>
      <div className="text-xs text-base-content/60 -mt-1 mb-1">{t("tts.neural_web_audio_output_note")}</div>
      <InputNativeAudioOutput label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
      <InputCheckbox label="tts.neural_route_through_vc" value={data.routeThroughVoiceChanger} onChange={v => handleUpdate("routeThroughVoiceChanger", v)} />
    </Inspector.Deactivatable>
  </>;
};

const NeuralSidecar: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts.data.neuralSidecar);
  const handleUpdate = <K extends keyof TTS_State["neuralSidecar"]>(key: K, v: TTS_State["neuralSidecar"][K]) =>
    window.ApiServer.patchService("tts", s => s.data.neuralSidecar[key] = v);
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  return <>
    <Inspector.SubHeader>{t("tts.neural_sidecar_title")}</Inspector.SubHeader>
    <div className="text-xs text-base-content/70 mb-2">{t("tts.neural_sidecar_desc")}</div>
    <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
      <InputText label="tts.neural_sidecar_endpoint" value={data.endpoint} onChange={e => handleUpdate("endpoint", e.target.value)} />
      <InputText label="tts.field_voice" value={data.voice} onChange={e => handleUpdate("voice", e.target.value)} />
      <div className="text-xs text-base-content/60 mb-1">{t("tts.neural_web_audio_output_note")}</div>
      <InputNativeAudioOutput label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
      <InputCheckbox label="tts.neural_route_through_vc" value={data.routeThroughVoiceChanger} onChange={v => handleUpdate("routeThroughVoiceChanger", v)} />
    </Inspector.Deactivatable>
  </>;
};

const Inspector_TTS: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts);
  const up = <K extends keyof TTS_State>(key: K, v: TTS_State[K]) => window.ApiServer.patchService("tts", s => s.data[key] = v);
  const handleStart = (v: boolean) => window.ApiServer.state.services.tts.showActionButton = v;
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  const handleShowReplacements = () => {
    NiceModal.show('tts-replacements');
  }

  return <Inspector.Body>
    <Inspector.Header><RiChatVoiceFill /> {t('tts.title')}</Inspector.Header>
    <Inspector.Content>
      <Inspector.SubHeader>{t("tts.provider_subheader")}</Inspector.SubHeader>
      <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
        <InputSelect label="common.field_service" value={data.data.backend} options={[
          { label: t("tts.backend_native"), value: TTS_Backends.native },
          { label: t("tts.backend_windows"), value: TTS_Backends.windows },
          { label: t("tts.backend_azure"), value: TTS_Backends.azure },
          { label: t("tts.backend_uberduck"), value: TTS_Backends.uberduck },
          { label: t("tts.backend_voicevox"), value: TTS_Backends.voicevox },
          { label: t("tts.backend_kokoro"), value: TTS_Backends.kokoro },
          { label: t("tts.backend_melo"), value: TTS_Backends.melo },
          { label: t("tts.backend_chatterbox"), value: TTS_Backends.chatterbox },
          { label: t("tts.backend_fish_speech"), value: TTS_Backends.fishSpeech },
          { label: t("tts.backend_local_onnx_speech"), value: TTS_Backends.neuralOnnx },
          { label: t("tts.backend_http_speech_server"), value: TTS_Backends.neuralSidecar },
        ]} onValueChange={e => up("backend", e as TTS_Backends)} />
        <InputTextSource label="common.field_text_source" value={data.data.source} onChange={e => up("source", e)} />
        <InputCheckbox label="common.field_use_keyboard_input" value={data.data.inputField} onChange={e => up("inputField", e)} />
      </Inspector.Deactivatable>
      {data.data.backend === TTS_Backends.windows && <Windows />}
      {data.data.backend === TTS_Backends.azure && <Azure />}
      {data.data.backend === TTS_Backends.native && <Native />}
      {data.data.backend === TTS_Backends.uberduck && <UberDuck />}
      {data.data.backend === TTS_Backends.voicevox && <VoiceVox />}
      {data.data.backend === TTS_Backends.kokoro && <Kokoro />}
      {data.data.backend === TTS_Backends.melo && <Melo />}
      {data.data.backend === TTS_Backends.chatterbox && <Chatterbox />}
      {data.data.backend === TTS_Backends.fishSpeech && <FishSpeech />}
      {data.data.backend === TTS_Backends.neuralOnnx && <NeuralOnnx />}
      {data.data.backend === TTS_Backends.neuralSidecar && <NeuralSidecar />}

      <ServiceButton
        status={state.status}
        onStart={() => window.ApiServer.tts.start()}
        onStop={() => window.ApiServer.tts.stop()}
        onPending={() => window.ApiServer.tts.stop()} />



      <div className="contents">
        <div className="h-2" />
        <InputCheckbox label="common.field_action_bar" value={data.showActionButton} onChange={handleStart} />
        <InputCheckbox label="common.field_auto_start" value={data.data.autoStart} onChange={e => up("autoStart", e)} />
        <InputCheckbox label="tts.field_stop_with_stream" value={data.data.stopWithStream} onChange={e => up("stopWithStream", e)} />
        <div className="h-2" />
        <span className="link link-accent link-hover font-semibold flex items-center gap-2 text-sm justify-end" onClick={handleShowReplacements}><RiCharacterRecognitionFill /> {t('common.btn_edit_replacements')}</span>
      </div>

    </Inspector.Content>
  </Inspector.Body>
}

export default Inspector_TTS;
