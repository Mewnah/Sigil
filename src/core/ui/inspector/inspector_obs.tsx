import { useGetState } from "@/client";
import { OBS_State } from "@/core/services/obs/schema";
import { ObsCaptionOutputMode } from "@/core/services/obs/types";
import classNames from "classnames";
import { FC, useState } from "react";
import { RiFileCopyLine } from "react-icons/ri";
import { SiObsstudio } from "react-icons/si";
import { toast } from "react-toastify";
import { useSnapshot } from "valtio";
import Dropdown, { useDropdown } from "../dropdown/Dropdown";
import ServiceButton from "../service-button";
import Inspector from "./components";
import { InputCheckbox, InputMapObject, InputNetworkStatus, InputSelect, InputText, InputTextSource } from "./components/input";
import { useTranslation } from "react-i18next";

const detailsClass =
  "rounded-lg border border-base-content/10 mt-2 [&_summary]:cursor-pointer [&_summary]:list-none [&_summary::-webkit-details-marker]:hidden";

const ObsSetupDropdown: FC = () => {
  const { t } = useTranslation();
  const dropdown = useDropdown();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [name, setName] = useState("Sigil browser source");
  const [port, setPort] = useState("4455");
  const [password, setPassword] = useState("");

  const handleSetup = async () => {
    setError("");
    if (!port || !name)
      return;
    setLoading(true);
    const resp = await window.ApiServer.obs.setupObsScene({ port, password, name });
    if (!resp) {
      toast.success(t("toasts.obs_updated"));
      dropdown.close();
    }
    else
      setError(resp)
    setLoading(false);
  }

  return <div className="menu bg-base-100 p-4 w-72 rounded-box flex flex-col space-y-2">
    <InputText value={name} onChange={e => setName(e.target.value)} label="Source name" />
    <InputText value={port} onChange={e => setPort(e.target.value)} label="Port" />
    <InputText type="password" autoComplete="false" value={password} onChange={e => setPassword(e.target.value)} label="Password" />
    <span className="text-xs opacity-50">New source will be created in the currently active scene</span>
    {error && <span className="text-xs text-error">{error}</span>}
    <button onClick={handleSetup} className={classNames("btn btn-sm btn-primary", { loading })}>Confirm</button>
  </div>
}

const Inspector_OBS: FC = () => {
  const { t } = useTranslation();
  const canvas = useGetState(state => state.canvas);
  const data = useSnapshot(window.ApiServer.state.services.obs.data);
  const wsState = useSnapshot(window.ApiServer.obs.wsState);
  const up = <K extends keyof OBS_State>(key: K, v: OBS_State[K]) => window.ApiServer.patchService("obs", s => s.data[key] = v);

  const handleStartWs = () => {
    window.ApiServer.obs.wsConnect();
  }

  const copyObsBrowserSourceLink = () => {
    navigator.clipboard.writeText(window.ApiServer.obs.getObsBrowserSourceLink());
    toast.success(t("obs.toast_copied_obs_link"));
  };

  const handleStopWs = () => window.ApiServer.obs.wsDisconnect();
  const handleCancelWs = () => window.ApiServer.obs.wsCancel();

  return <Inspector.Body>
    <Inspector.Header><SiObsstudio /> {t("obs.title")}</Inspector.Header>
    <Inspector.Content>
      <Inspector.SubHeader>{t('obs.section_browser_source')}</Inspector.SubHeader>
      <p className="text-xs text-base-content/70 leading-snug mb-2">{t("obs.quick_start")}</p>

      <button type="button" onClick={copyObsBrowserSourceLink} className="btn btn-sm btn-primary gap-2 w-full sm:w-auto">
        <RiFileCopyLine /> {t("obs.btn_copy_browser_source_link")}
      </button>

      <div className="mt-3 space-y-2">
        <InputCheckbox value={data.browserCaptionsEnable} onChange={e => up("browserCaptionsEnable", e)} label="obs.field_enable_browser_captions_feed" />
        <InputTextSource value={data.browserSource} onChange={e => up("browserSource", e)} label="obs.field_browser_caption_source" />
      </div>

      <details className={detailsClass}>
        <summary className="px-3 py-2 text-sm font-medium">{t("obs.advanced_appearance")}</summary>
        <div className="px-3 pb-3 pt-0 space-y-2 border-t border-base-content/10">
          <p className="text-xs text-base-content/55 pt-2">{t("obs.browser_source_desc", { width: canvas.w, height: canvas.h })}</p>
          <InputCheckbox value={data.browserInputField} onChange={e => up("browserInputField", e)} label="common.field_use_keyboard_input" />
          <InputCheckbox value={data.browserInterim} onChange={e => up("browserInterim", e)} label="common.field_interim_results" />
          <InputSelect
            label="obs.field_caption_output_mode"
            value={data.browserOutputMode}
            onValueChange={e => up("browserOutputMode", e as ObsCaptionOutputMode)}
            options={[
              { value: ObsCaptionOutputMode.styled, label: t("obs.output_mode_styled") },
              { value: ObsCaptionOutputMode.plain, label: t("obs.output_mode_plain") },
            ]}
          />
          <InputText label="obs.field_browser_font_size" type="number" value={data.browserFontSizePx} onChange={e => up("browserFontSizePx", e.target.value)} />
          <InputText label="obs.field_browser_max_lines" type="number" value={data.browserMaxLines} onChange={e => up("browserMaxLines", e.target.value)} />
        </div>
      </details>

      <details className={detailsClass}>
        <summary className="px-3 py-2 text-sm font-medium">{t("obs.advanced_ws_native")}</summary>
        <div className="px-3 pb-3 pt-0 space-y-2 border-t border-base-content/10">
          <p className="text-xs text-base-content/55 pt-2">{t("obs.websocket_desc")}</p>
          <InputNetworkStatus label="common.field_connection_status" value={wsState.status} />
          <InputText label="common.field_connection_port" type="number" value={data.wsPort} onChange={e => up("wsPort", e.target.value)} />
          <InputText label="common.field_password" type="password" value={data.wsPassword} onChange={e => up("wsPassword", e.target.value)} />
          <ServiceButton
            showError
            errorLabel="Error - Try Again"
            stopLabel="common.btn_disconnect"
            startLabel="common.btn_connect"
            onError={handleStartWs}
            status={wsState.status}
            onStart={handleStartWs}
            onPending={handleCancelWs}
            onStop={handleStopWs} />
          <InputCheckbox label="obs.field_auto_connect" value={data.wsAutoStart} onChange={e => up("wsAutoStart", e)} />
          <div className="mt-4">
            <Inspector.SubHeader>{t('obs.section_native_captions')}</Inspector.SubHeader>
          </div>
          <p className="text-xs text-base-content/55">{t("obs.native_captions_explain")}</p>
          <InputCheckbox value={data.captionsEnable} onChange={e => up("captionsEnable", e)} label="common.field_enable" />
          <InputTextSource value={data.source} onChange={e => up("source", e)} label="common.field_text_source" />
          <InputCheckbox value={data.inputField} onChange={e => up("inputField", e)} label="common.field_use_keyboard_input" />
          <InputCheckbox value={data.interim} onChange={e => up("interim", e)} label="common.field_interim_results" />
        </div>
      </details>

      <details className={detailsClass}>
        <summary className="px-3 py-2 text-sm font-medium">{t('obs.section_scenes')}</summary>
        <div className="px-3 pb-3 pt-0 space-y-2 border-t border-base-content/10">
          <p className="text-xs text-base-content/55 pt-2">{t('obs.section_scenes_desc')}</p>
          <InputCheckbox value={data.scenesEnable} onChange={e => up("scenesEnable", e)} label="common.field_enable" />
          <InputText value={data.scenesFallback} onChange={e => up("scenesFallback", e.target.value)} label="obs.field_fallback_scene" />
          <InputMapObject keyPlaceholder="OBS name" valuePlaceholder="Scene" addLabel="Add word" value={{ ...data.scenesMap }} onChange={e => up("scenesMap", e)} label="obs.field_map_scenes" />
        </div>
      </details>

      <Dropdown className="btn btn-sm gap-2 mt-3" targetOffset={24} placement="right" content={<ObsSetupDropdown />}>
        <SiObsstudio /> {t("obs.btn_setup_browser_source")}
      </Dropdown>

      <details className={detailsClass}>
        <summary className="px-3 py-2 text-sm font-medium text-base-content/60">{t("obs.troubleshooting")}</summary>
        <div className="px-3 pb-3 pt-0 space-y-2 border-t border-base-content/10 text-xs text-base-content/55 whitespace-pre-line">
          <p className="pt-2">{t("obs.browser_source_tips")}</p>
          <p>{t("obs.checklist_browser_mirror")}</p>
        </div>
      </details>
    </Inspector.Content>
  </Inspector.Body>
}

export default Inspector_OBS;
