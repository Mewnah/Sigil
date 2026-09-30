import type { TTS_State } from "@/core/services/tts/schema";
import { TTS_KokoroService } from "@/core/services/tts/services/kokoro";
import { ServiceNetworkState } from "@/types";
import classNames from "classnames";
import { FC, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputRange, InputSelect, InputSelectOption, InputText } from "../components/input";
import { TtsOutputDevice } from "./Routing";

export const Kokoro: FC = () => {
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
    <div className="text-xs text-base-content/60 mb-2">{t("tts.http_legacy_openai_hint")}</div>
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
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
      <InputRange label="common.field_volume" value={data.volume} onChange={e => handleUpdate("volume", e.target.value)} step="0.05" min="0" max="2" />
    </Inspector.Deactivatable>
  </>;
};
