import { FC, ReactNode, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { RiAlertFill, RiMessage2Fill } from "react-icons/ri";
import { useSnapshot } from "valtio";
import { TextEventSource } from "@/types";
import { STT_Backends } from "../../services/stt/schema";
import { CHATBOX_MAX_INTERVAL_MS, CHATBOX_MIN_INTERVAL_MS } from "../../services/vrc/chatboxPager";
import { DEFAULT_LISTEN_PORT, isLoopbackHost, listenOptionsFor } from "../../services/vrc/receiver";
import { VRC_Backends, VRC_State } from "../../services/vrc/schema";
import Inspector from "./components";
import { InputCheckbox, InputSelect, InputText, InputTextSource } from "./components/input";
import { OverlaySection } from "./vrc/OverlaySection";

const targetOptions = [
  { label: "Chatbox", value: VRC_Backends.textbox },
  { label: "KillFrenzy Avatar Text", value: VRC_Backends.killfrenzy },
]

/** How often the open inspector refreshes the receiver status ("receiving from VRChat"). */
const STATUS_REFRESH_MS = 2_000;
/** VRChat counts as "sending" when something arrived this recently. */
const RECEIVING_WINDOW_MS = 15_000;

const patchVrc = (fn: (data: VRC_State) => void) => window.ApiServer.patchService("vrc", (s) => fn(s.data));

const TextboxInspector: FC = () => {
  const { t } = useTranslation();
  const state = useSnapshot(window.ApiServer.state.services.vrc.data.textbox);
  const source = useSnapshot(window.ApiServer.state.services.vrc.data).source;
  const stt = useSnapshot(window.ApiServer.state.services.stt.data);
  const translationTurns = useSnapshot(window.ApiServer.state.services.translation.data.partials).turnEndCommit;
  const up = <K extends keyof VRC_State["textbox"]>(key: K, v: VRC_State["textbox"][K]) => window.ApiServer.patchService("vrc", s => s.data.textbox[key] = v);
  const smartTurnOff = stt.backend !== STT_Backends.local || !stt.local.smartTurn;
  return <>
    <Inspector.SubHeader>{t('vrc.textbox_title')}</Inspector.SubHeader>
    <InputCheckbox label="vrc.textbox_show_indicator" value={state.indicator} onChange={e => up("indicator", e)} />
    <InputCheckbox label="vrc.textbox_play_notification_sound" value={state.playNotificationSound} onChange={e => up("playNotificationSound", e)} />
    <InputCheckbox label="vrc.textbox_paginate" value={state.paginate} onChange={e => up("paginate", e)} />
    <Inspector.Description>{t("vrc.textbox_paginate_desc")}</Inspector.Description>
    <InputText type="number" label="vrc.textbox_interval" min={CHATBOX_MIN_INTERVAL_MS} max={CHATBOX_MAX_INTERVAL_MS} step={100} value={state.pageIntervalMs} onChange={e => up("pageIntervalMs", e.target.value)} />
    <Inspector.Description>{t("vrc.textbox_interval_desc")}</Inspector.Description>
    <InputCheckbox label="vrc.textbox_commit_turn_end" value={state.commitOnTurnEnd} onChange={e => up("commitOnTurnEnd", e)} />
    <Inspector.Description>{t("vrc.textbox_commit_turn_end_desc")}</Inspector.Description>
    {state.commitOnTurnEnd && <>
      <InputText type="number" label="vrc.textbox_turn_hold" min={500} max={10000} step={100} value={state.turnHoldMs} onChange={e => up("turnHoldMs", e.target.value)} />
      {source === TextEventSource.translation && translationTurns
        ? <Inspector.Description>{t("vrc.textbox_commit_turn_end_translation")}</Inspector.Description>
        : smartTurnOff && <Inspector.Description><span className="text-warning">{t("vrc.textbox_commit_turn_end_needs_smart_turn")}</span></Inspector.Description>}
    </>}
    <button className="btn btn-sm" onClick={() => window.ApiServer.vrc.sendTest("Test test test!")}>{t('vrc.btn_send_test')}</button>
  </>
}

const KillFrenzyInspector: FC = () => {
  const { t } = useTranslation();
  const state = useSnapshot(window.ApiServer.state.services.vrc.data.killfrenzy);
  const up = <K extends keyof VRC_State["killfrenzy"]>(key: K, v: VRC_State["killfrenzy"][K]) => window.ApiServer.patchService("vrc", s => s.data.killfrenzy[key] = v);
  return <>
    <Inspector.SubHeader>{t('vrc.kfi_title')}</Inspector.SubHeader>
    {/* <InputCheckbox label="Show indicator" value={state.indicator} onChange={e => up("indicator", e)} /> */}
    <InputCheckbox label="vrc.kfi_split_large" value={state.splitSentences} onChange={e => up("splitSentences", e)} />
    <InputText type="number" label="vrc.kfi_hide_after" min={0} max={120000} value={state.visibleTimer} onChange={e => up("visibleTimer", e.target.value)} />
    <InputText type="number" label="vrc.kfi_sync_points" min={1} max={4} value={state.syncPoints} onChange={e => up("syncPoints", e.target.value)} />
    <InputText type="number" label="vrc.kfi_sync_delay" min={0} max={1000} value={state.syncDelay} onChange={e => up("syncDelay", e.target.value)} />

    <Inspector.SubHeader>OSC Keys</Inspector.SubHeader>
    <InputText label="vrc.kfi_visibility_key" value={state.KAT_Visible} onChange={e => up("KAT_Visible", e.target.value)} />
    <InputText label="vrc.kfi_pointer_key" value={state.KAT_Pointer} onChange={e => up("KAT_Pointer", e.target.value)} />
    <InputText label="vrc.kfi_chat_sync_key" value={state.KAT_CharSync} onChange={e => up("KAT_CharSync", e.target.value)} />
  </>
}

const StatusLine: FC<{ tone?: "error" | "warning" | "success"; children: ReactNode }> = ({ tone, children }) => (
  <Inspector.Description>
    <span className={tone === "error" ? "text-error" : tone === "warning" ? "text-warning" : tone === "success" ? "text-success" : undefined}>
      {children}
    </span>
  </Inspector.Description>
);

/** Live state of the OSC receiver and VRChat's microphone. */
const ReceiverStatus: FC = () => {
  const { t } = useTranslation();
  const state = useSnapshot(window.ApiServer.state.services.vrc.data);
  const svc = useSnapshot(window.ApiServer.vrc.serviceState);
  const wanted = listenOptionsFor(state) !== null;
  useEffect(() => {
    if (!wanted) return;
    void window.ApiServer.vrc.refreshReceiverStatus();
    const timer = setInterval(() => void window.ApiServer.vrc.refreshReceiverStatus(), STATUS_REFRESH_MS);
    return () => clearInterval(timer);
  }, [wanted]);

  if (!state.enable) return <StatusLine tone="warning">{t("vrc.receiver_needs_enable")}</StatusLine>;
  const r = svc.receiver;
  if (!r || r.mode === "off") return <StatusLine>{t("vrc.receiver_starting")}</StatusLine>;
  const receiving = r.packets > 0 && r.lastPacketAgoMs !== null && r.lastPacketAgoMs < RECEIVING_WINDOW_MS;
  return <>
    {r.error && <StatusLine tone="error">{r.error}</StatusLine>}
    {r.running && r.port !== null && <StatusLine>
      {t(r.mode === "oscQuery" ? "vrc.receiver_listening_oscquery" : "vrc.receiver_listening_port", { port: r.port })}
    </StatusLine>}
    {r.warning && <StatusLine tone="warning">{r.warning}</StatusLine>}
    {r.running && r.mode === "oscQuery" && <StatusLine>{t(r.vrchatFound ? "vrc.receiver_vrchat_found" : "vrc.receiver_vrchat_waiting")}</StatusLine>}
    {r.running && (receiving
      ? <StatusLine tone="success">{t("vrc.receiver_receiving")}</StatusLine>
      : <StatusLine>{t("vrc.receiver_no_messages")}</StatusLine>)}
    {svc.vrchatMuted !== null && <StatusLine tone={svc.vrchatMuted ? "warning" : undefined}>
      {t(svc.vrchatMuted ? (svc.mutedByVrchat ? "vrc.receiver_mic_muted_sync" : "vrc.receiver_mic_muted") : "vrc.receiver_mic_on")}
    </StatusLine>}
  </>;
};

/** P6-C: VRChat's microphone mute → speech-to-text mute. */
const MuteSyncSection: FC = () => {
  const { t } = useTranslation();
  const state = useSnapshot(window.ApiServer.state.services.vrc.data);
  // OSCQuery finds VRChat on this PC only; for another PC the fixed port is what VRChat is told to use.
  const localHost = isLoopbackHost(state.oscHost);
  return <>
    <Inspector.SubHeader>{t("vrc.mute_title")}</Inspector.SubHeader>
    <InputCheckbox label="vrc.mute_sync" value={state.muteSync} onChange={e => patchVrc(d => d.muteSync = e)} />
    <Inspector.Description>{t("vrc.mute_sync_desc")}</Inspector.Description>
    {state.muteSync && <>
      {localHost && <>
        <InputCheckbox label="vrc.osc_query" value={state.oscQuery} onChange={e => patchVrc(d => d.oscQuery = e)} />
        <Inspector.Description>{t("vrc.osc_query_desc")}</Inspector.Description>
      </>}
      {(!state.oscQuery || !localHost) && <>
        <InputText type="number" label="vrc.osc_listen_port" min={1} max={65535} placeholder={String(DEFAULT_LISTEN_PORT)} value={state.oscListenPort} onChange={e => patchVrc(d => d.oscListenPort = e.target.value)} />
        {!localHost && <Inspector.Description>{t("vrc.osc_listen_port_remote_desc")}</Inspector.Description>}
      </>}
      <ReceiverStatus />
    </>}
  </>;
};

/** P6-C: SenseVoice emotion / sound tags → avatar parameters and chatbox emoji. */
const TagsSection: FC = () => {
  const { t } = useTranslation();
  const tags = useSnapshot(window.ApiServer.state.services.vrc.data.tags);
  const up = <K extends keyof VRC_State["tags"]>(key: K, v: VRC_State["tags"][K]) => window.ApiServer.patchService("vrc", s => s.data.tags[key] = v);
  return <>
    <Inspector.SubHeader>{t("vrc.tags_title")}</Inspector.SubHeader>
    <InputCheckbox label="vrc.tags_enable" value={tags.enable} onChange={e => up("enable", e)} />
    <Inspector.Description>{t("vrc.tags_desc")}</Inspector.Description>
    {tags.enable && <>
      <InputText label="vrc.tags_emotion_parameter" value={tags.emotionParameter} onChange={e => up("emotionParameter", e.target.value)} />
      <Inspector.Description>{t("vrc.tags_emotion_values")}</Inspector.Description>
      <InputText label="vrc.tags_event_parameter" value={tags.eventParameter} onChange={e => up("eventParameter", e.target.value)} />
      <Inspector.Description>{t("vrc.tags_event_values")}</Inspector.Description>
      <InputText type="number" label="vrc.tags_reset" min={0} max={60000} step={500} value={tags.resetMs} onChange={e => up("resetMs", e.target.value)} />
      <Inspector.Description>{t("vrc.tags_reset_desc")}</Inspector.Description>
      <InputCheckbox label="vrc.tags_chatbox" value={tags.chatbox} onChange={e => up("chatbox", e)} />
    </>}
  </>;
};

const Inspector_VRC: FC = () => {
  const { t } = useTranslation();
  const state = useSnapshot(window.ApiServer.state.services.vrc.data);
  const up = <K extends keyof VRC_State>(key: K, v: VRC_State[K]) => window.ApiServer.patchService("vrc", s => s.data[key] = v);

  return <Inspector.Body>
    <Inspector.Header><RiMessage2Fill /> {t('vrc.title')}</Inspector.Header>
    <Inspector.Content>
      <Inspector.SubHeader>VRChat Integration</Inspector.SubHeader>
      <div className="p-2 border-2 border-primary rounded-lg text-xs flex space-x-2 items-center text-primary">
        <RiAlertFill className="text-primary" size={24} />
        <div className="flex flex-col gap-1">
          <span className="font-bold">{t('vrc.notice_title')}</span>
          <span className="font-medium">{t('vrc.notice_desc')}</span>
          <span className="font-medium opacity-80">{t("vrc.notice_desktop")}</span>
        </div>
      </div>
      <InputCheckbox label="common.field_enable" value={state.enable} onChange={e => up("enable", e)} />
      <Inspector.SubHeader>{t("vrc.osc_connection")}</Inspector.SubHeader>
      <InputText label="vrc.field_osc_host" value={state.oscHost} onChange={e => up("oscHost", e.target.value)} />
      <InputText type="number" label="vrc.field_osc_port" min={1} max={65535} value={state.oscPort} onChange={e => up("oscPort", e.target.value)} />
      <InputTextSource label="common.field_text_source" value={state.source} onChange={e => up("source", e)} />
      <InputCheckbox label="common.field_use_keyboard_input" value={state.inputField} onChange={e => up("inputField", e)} />

      <InputSelect options={targetOptions} label="vrc.field_target" value={state.target} onValueChange={e => up("target", e as VRC_Backends)} />
      {state.target === VRC_Backends.killfrenzy && <KillFrenzyInspector />}
      {state.target === VRC_Backends.textbox && <TextboxInspector />}
      <MuteSyncSection />
      <TagsSection />
      {/* P6-B: caption window and SteamVR overlay (scaffolded by P6-S0; renders nothing until implemented). */}
      <OverlaySection />
    </Inspector.Content>
  </Inspector.Body>
}

export default Inspector_VRC;
