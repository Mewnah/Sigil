import type { TTS_State } from "@/core/services/tts/schema";
import { errorMessage } from "@/core/services/tts/types";
import { ServiceNetworkState } from "@/types";
import { invoke } from "@tauri-apps/api/core";
import classNames from "classnames";
import { FC, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "react-toastify";
import { proxy, useSnapshot } from "valtio";
import Inspector from "../components";
import { InputRange, InputSelect, InputSelectOption, InputText } from "../components/input";
import { TtsOutputDevice } from "./Routing";

const uberduckVoices = proxy<{ value: InputSelectOption[] }>({
  value: []
});

export const UberDuck: FC = () => {
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
      <TtsOutputDevice label="common.field_output_device" value={data.device} onChange={e => handleUpdate("device", e)} />
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
