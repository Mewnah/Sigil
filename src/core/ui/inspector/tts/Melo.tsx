import type { TTS_State } from "@/core/services/tts/schema";
import { TTS_MeloService } from "@/core/services/tts/services/melo";
import { ServiceNetworkState } from "@/types";
import classNames from "classnames";
import { FC, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Inspector from "../components";
import { InputRange, InputSelect, InputSelectOption, InputText } from "../components/input";
import { TtsOutputDevice } from "./Routing";

export const Melo: FC = () => {
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
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
      <InputRange label="common.field_volume" value={data.volume} onChange={e => handleUpdate("volume", e.target.value)} step="0.05" min="0" max="2" />
    </Inspector.Deactivatable>
  </>;
};
