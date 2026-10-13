import { OpenAiAudioPreset, STT_Backends, type STT_State } from "@/core/services/stt/schema";
import { noSignalMessage } from "@/core/services/stt/statusMessages";
import { ServiceNetworkState } from "@/types";
import NiceModal from "@ebay/nice-modal-react";
import { isTauri } from "@tauri-apps/api/core";
import { FC, Suspense, lazy } from "react";
import { useTranslation } from 'react-i18next';
import { RiCharacterRecognitionFill, RiUserVoiceFill } from "react-icons/ri";
import { useSnapshot } from "valtio";
import ServiceButton from "../../service-button";
import Inspector from "../components";
import { InputCheckbox, InputSelect } from "../components/input";
import { Azure } from "./Azure";
import { Deepgram } from "./Deepgram";
import { ExternalBrowser } from "./ExternalMic";
import { Local } from "./Local";
import { Native } from "./Native";
import { OpenAIAudio } from "./OpenAIAudio";
import { Vosk } from "./Vosk";
import { Whisper } from "./Whisper";
import { GeminiLive } from "./GeminiLive";
import { LanguageOfferPanel } from "./LanguageOffer";
import { PROVIDER_GROQ, sttEngineGroups, sttProviderValue } from "./engineList";
import { SttTabPanel, SttTabs, selectSttView, sttViewForSection } from "./SttTabs";
import { useAppUIStore } from "../../store";
import { Services } from "@/services-registry";
import { ReadingsSection } from "@/core/services/readings/ReadingsSection";
import "../../stt-replacements-modal";

// Rewrite is the second tab of this inspector; it loads when first opened.
const Rewrite = lazy(() => import("./Rewrite"));

// STT inspector (P2-S0 split of inspector_stt.tsx): this file and Local / Whisper are P2-E's; Vosk, OpenAIAudio, Azure,
// Deepgram and ExternalMic are P2-F's; common, VoiceDetection and Native are shared. Two tabs: Speech and Rewrite.
// The built-in engine (`local`) leads the grouped engine list (`engineList.ts`): it picks and runs the speech models itself.

/**
 * Engine status (download / load progress, connecting, rate-limit waits) and the silent-microphone notice ("Sigil hears
 * nothing from <mic>"): UI only, never a caption. The notice is a state of its own, so no other status line replaces it.
 */
const SttStatusLine: FC = () => {
  const { t } = useTranslation();
  const s = useSnapshot(window.ApiServer.stt.serviceState);
  if (!s.statusText && !s.silentInput) return null;
  return <div role="status" aria-live="polite" className="flex flex-col gap-1 text-xs">
    {s.statusText && <span className={s.statusLevel === "warning" ? "text-warning" : "text-base-content/70"}>{s.statusText}</span>}
    {s.statusText && s.progress !== null && <progress className="progress progress-primary w-full" value={s.progress} max={100} />}
    {s.silentInput && <span className="text-warning">{noSignalMessage(s.silentInput.device, t)}</span>}
  </div>;
};

type SttShortcut = "start" | "muteMic";

/** Global shortcut field: records the next key combination (Esc cancels). */
const ShortcutField: FC<{ shortcut: SttShortcut; label: string }> = ({ shortcut, label }) => {
  const { t } = useTranslation();
  const { shortcuts } = useSnapshot(window.ApiServer.state);
  const kb = useSnapshot(window.ApiServer.keyboard.ui);
  const recording = kb.showRecorder && kb.currentTarget === shortcut;
  const value = shortcuts[shortcut];
  const hintKey = recording && kb.recorderHint ? `stt.shortcut_hint_${kb.recorderHint}` : "";

  return <div className="flex flex-col gap-1">
    <span className="font-medium text-base-content/80 text-xs">{t(label)}</span>
    <div className="flex items-center gap-2">
      <span
        className={`input input-sm input-bordered flex-1 min-w-0 flex items-center font-mono text-xs truncate ${recording ? "input-primary" : ""}`}
        aria-live={recording ? "polite" : undefined}
      >
        {recording ? (kb.currentValue || t("stt.shortcut_press_keys")) : (value || t("stt.shortcut_none"))}
      </span>
      {recording ? (
        <button type="button" className="btn btn-sm btn-ghost shrink-0" onClick={() => window.ApiServer.keyboard.cancelComboRecord()}>
          {t("stt.shortcut_cancel")}
        </button>
      ) : (
        <button type="button" className="btn btn-sm btn-neutral shrink-0" onClick={() => window.ApiServer.keyboard.startShortcutRecord(shortcut)}>
          {t("stt.shortcut_record")}
        </button>
      )}
      {!recording && value && (
        <button type="button" className="btn btn-sm btn-ghost shrink-0" onClick={() => window.ApiServer.keyboard.clearShortcut(shortcut)}>
          {t("stt.shortcut_clear")}
        </button>
      )}
    </div>
    {hintKey && <span className="text-xs text-warning">{t(hintKey)}</span>}
  </div>;
};

const Inspector_STT: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.stt);
  const state = useSnapshot(window.ApiServer.stt.serviceState);

  // The store names the tab: Rewrite opens from the footer card as { tab: "stt", value: "rewrite" } (resolveInspectorTab), and
  // the tabs below move it in place (selectSttView never toggles the panel), so the status-bar card, the left menu
  // and the toasts always act on the tab that is showing.
  const view = useAppUIStore(s => sttViewForSection(s.sidebar.tab?.tab === Services.stt ? s.sidebar.tab.value : undefined));

  const handleStart = (v: boolean) => window.ApiServer.state.services.stt.showActionButton = v;
  const up = <K extends keyof STT_State>(key: K, v: STT_State[K]) => window.ApiServer.patchService("stt", s => s.data[key] = v);

  const providerValue = sttProviderValue(data.data.backend, data.data.openai_audio.preset);
  const setProvider = (value: string) => window.ApiServer.patchService("stt", s => {
    if (value === PROVIDER_GROQ) {
      s.data.backend = STT_Backends.openai_audio;
      s.data.openai_audio.preset = OpenAiAudioPreset.groq;
    }
    else {
      s.data.backend = value as STT_Backends;
      if (value === STT_Backends.openai_audio)
        s.data.openai_audio.preset = OpenAiAudioPreset.custom;
    }
  });

  const handleShowReplacements = () => {
    NiceModal.show('stt-replacements');
  }

  const backend = data.data.backend;
  const external = backend === STT_Backends.chrome || backend === STT_Backends.edge;

  return <Inspector.Body>
    <Inspector.Header><RiUserVoiceFill /> {t('stt.title')}</Inspector.Header>
    <SttTabs view={view} onChange={selectSttView} />

    <SttTabPanel view={view}>
      {view === "rewrite" && <Inspector.Content>
        <Suspense fallback={null}>
          <Rewrite />
        </Suspense>
      </Inspector.Content>}

      {view === "speech" && <Inspector.Content>
        <Inspector.SubHeader>{t("stt.section_provider")}</Inspector.SubHeader>
        <Inspector.Deactivatable active={state.status === ServiceNetworkState.disconnected}>
          <InputSelect
            options={sttEngineGroups(t, { desktop: isTauri(), current: providerValue })}
            label="common.field_service"
            value={providerValue}
            onValueChange={setProvider}
          />

          {!external && (
            <div className="mt-2 mb-4">
              <InputCheckbox label='stt.field_uwu_filter' onChange={e => up("uwu", e)} value={data.data.uwu} />
              <Inspector.Description>{t('stt.field_uwu_filter_desc')}</Inspector.Description>
            </div>
          )}

          {backend === STT_Backends.chrome && <ExternalBrowser browser="chrome" />}
          {backend === STT_Backends.edge && <ExternalBrowser browser="edge" />}
          {backend === STT_Backends.azure && <Azure />}
          {backend === STT_Backends.deepgram && <Deepgram />}
          {backend === STT_Backends.whisper && <Whisper />}
          {backend === STT_Backends.vosk && <Vosk />}
          {backend === STT_Backends.openai_audio && <OpenAIAudio />}
          {backend === STT_Backends.local && <Local />}
          {backend === STT_Backends.geminiLive && <GeminiLive />}
          {(backend === STT_Backends.native || backend === STT_Backends.browser) && <Native />}
        </Inspector.Deactivatable>

        {/* Outside the part that greys out while STT runs: "Lock to <language>" works with captions running. */}
        <LanguageOfferPanel />

        {!external && (
          <div className="pt-4 flex flex-col gap-2">
            <ServiceButton status={state.status} onStart={() => window.ApiServer.stt.start()} onStop={() => window.ApiServer.stt.stop()} />
            <SttStatusLine />
          </div>
        )}

        <div className="contents">
          <div className="h-2" />
          <InputCheckbox label="common.field_action_bar" onChange={handleStart} value={data.showActionButton} />
          <InputCheckbox label="common.field_auto_start" value={data.data.autoStart} onChange={e => up("autoStart", e)} />
          <InputCheckbox label="stt.field_stop_with_stream" value={data.data.stopWithStream} onChange={e => up("stopWithStream", e)} />
          <div className="h-2" />
          <span className="link link-accent link-hover font-semibold flex items-center gap-2 text-sm justify-end" onClick={handleShowReplacements}><RiCharacterRecognitionFill />{t('common.btn_edit_replacements')}</span>
        </div>

        {isTauri() && <>
          <Inspector.SubHeader>{t("stt.section_shortcuts")}</Inspector.SubHeader>
          <ShortcutField shortcut="start" label="stt.shortcut_start" />
          <ShortcutField shortcut="muteMic" label="stt.shortcut_mute" />
          <Inspector.Description>{t("stt.shortcut_desc")}</Inspector.Description>
        </>}

        {/* P6-L: Japanese / Chinese readings of captions (scaffolded by P6-S0; renders nothing until implemented). */}
        <ReadingsSection />

      </Inspector.Content>}
    </SttTabPanel>
  </Inspector.Body>
}

export default Inspector_STT;
