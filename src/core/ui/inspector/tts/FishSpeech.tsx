import type { TTS_State } from "@/core/services/tts/schema";
import { TTS_FishSpeechService } from "@/core/services/tts/services/fishspeech";
import { ServiceNetworkState } from "@/types";
import classNames from "classnames";
import { FC, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputRange, InputSelect, InputSelectOption, InputText } from "../components/input";
import { TtsOutputDevice } from "./Routing";

export const FishSpeech: FC = () => {
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
      {/* One-click preset for Fish Audio's hosted API (S2.1 Pro free tier; P6-H request). The key stays the user's. */}
      <button
        type="button"
        className="btn btn-xs btn-ghost mb-1 self-start"
        onClick={() => {
          handleUpdate("endpoint", "https://api.fish.audio");
          handleUpdate("model", "s2.1-pro-free");
        }}
      >
        {t("tts.fish_hosted_preset")}
      </button>
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
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
      <InputRange label="common.field_volume" value={data.volume} onChange={e => handleUpdate("volume", e.target.value)} step="0.05" min="0" max="2" />
    </Inspector.Deactivatable>
  </>;
};
