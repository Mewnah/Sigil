import { STT_Backends } from "@/core/services/stt/schema";
import { TTS_Backends, type TTS_State } from "@/core/services/tts/schema";
import { ServiceNetworkState, TextEventSource } from "@/types";
import NiceModal from "@ebay/nice-modal-react";
import { FC } from "react";
import { useTranslation } from "react-i18next";
import { RiCharacterRecognitionFill, RiChatVoiceFill } from "react-icons/ri";
import { useSnapshot } from "valtio";
import ServiceButton from "../../service-button";
import Inspector from "../components";
import { InputCheckbox, InputSelect, InputText, InputTextSource } from "../components/input";
import { Azure } from "./Azure";
import { Chatterbox } from "./Chatterbox";
import { FishSpeech } from "./FishSpeech";
import { Kokoro } from "./Kokoro";
import { LocalNeural } from "./LocalNeural";
import { Melo } from "./Melo";
import { Native } from "./Native";
import { NeuralOnnx } from "./NeuralOnnx";
import { NeuralSidecar } from "./NeuralSidecar";
import { OpenAISpeech } from "./OpenAISpeech";
import "./ReplacementsModal";
import { Routing } from "./Routing";
import { UberDuck } from "./UberDuck";
import { VoiceVox } from "./VoiceVox";
import { Windows } from "./Windows";
// P6 backend panels (P6-S0 stubs): Qwen3 P6-E, Edge + Gemini P6-H, Sbv2 + GptSovits P6-G, Pocket + ChatterboxLocal
// P6-I, GgmlTts P6-F.
import { ChatterboxLocal } from "./ChatterboxLocal";
import { Edge } from "./Edge";
import { Gemini } from "./Gemini";
import { GgmlTts } from "./GgmlTts";
import { GptSovits } from "./GptSovits";
import { Pocket } from "./Pocket";
import { Qwen3 } from "./Qwen3";
import { Sbv2 } from "./Sbv2";

// TTS inspector (P2-S0 split of inspector_tts.tsx): this file, Routing, Azure, VoiceVox, UberDuck, Native and Windows
// are P2-G's; NeuralOnnx and LocalNeural are P2-H's; OpenAISpeech, Kokoro, Chatterbox, Melo, FishSpeech and
// NeuralSidecar are P2-I's; ReplacementsModal is shared.

const Inspector_TTS: FC = () => {
  const { t } = useTranslation();
  const data = useSnapshot(window.ApiServer.state.services.tts);
  const up = <K extends keyof TTS_State>(key: K, v: TTS_State[K]) => window.ApiServer.patchService("tts", s => s.data[key] = v);
  const handleStart = (v: boolean) => window.ApiServer.state.services.tts.showActionButton = v;
  const state = useSnapshot(window.ApiServer.tts.serviceState);
  const stt = useSnapshot(window.ApiServer.state.services.stt.data);
  const translationTurns = useSnapshot(window.ApiServer.state.services.translation.data.partials).turnEndCommit;
  const smartTurnOff = stt.backend !== STT_Backends.local || !stt.local.smartTurn;
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
          { label: t("tts.backend_local_neural"), value: TTS_Backends.localNeural },
          { label: t("tts.backend_pocket"), value: TTS_Backends.pocket },
          { label: t("tts.backend_chatterbox_local"), value: TTS_Backends.chatterboxLocal },
          { label: t("tts.backend_qwen3"), value: TTS_Backends.qwen3 },
          { label: t("tts.backend_sbv2"), value: TTS_Backends.sbv2 },
          { label: t("tts.backend_ggml_tts"), value: TTS_Backends.ggmlTts },
          { label: t("tts.backend_edge"), value: TTS_Backends.edge },
          { label: t("tts.backend_gemini"), value: TTS_Backends.gemini },
          { label: t("tts.backend_openai_speech"), value: TTS_Backends.openaiSpeech },
          { label: t("tts.backend_gpt_sovits"), value: TTS_Backends.gptSovits },
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
      {/* Smart Turn commit (P6-C gate, wired by P6-I0): read on every sentence, so it can change while running. */}
      <InputCheckbox label="tts.commit_turn_end" value={data.data.commitOnTurnEnd} onChange={e => up("commitOnTurnEnd", e)} />
      <Inspector.Description>{t("tts.commit_turn_end_desc")}</Inspector.Description>
      {data.data.commitOnTurnEnd && <>
        <InputText type="number" label="tts.turn_hold" min={500} max={10000} step={100} value={data.data.turnHoldMs} onChange={e => up("turnHoldMs", e.target.value)} />
        {data.data.source === TextEventSource.translation && translationTurns
          ? <Inspector.Description>{t("tts.commit_turn_end_translation")}</Inspector.Description>
          : smartTurnOff && <Inspector.Description><span className="text-warning">{t("tts.commit_turn_end_needs_smart_turn")}</span></Inspector.Description>}
      </>}
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
      {data.data.backend === TTS_Backends.localNeural && <LocalNeural />}
      {data.data.backend === TTS_Backends.openaiSpeech && <OpenAISpeech />}
      {data.data.backend === TTS_Backends.qwen3 && <Qwen3 />}
      {data.data.backend === TTS_Backends.edge && <Edge />}
      {data.data.backend === TTS_Backends.gemini && <Gemini />}
      {data.data.backend === TTS_Backends.sbv2 && <Sbv2 />}
      {data.data.backend === TTS_Backends.gptSovits && <GptSovits />}
      {data.data.backend === TTS_Backends.pocket && <Pocket />}
      {data.data.backend === TTS_Backends.chatterboxLocal && <ChatterboxLocal />}
      {data.data.backend === TTS_Backends.ggmlTts && <GgmlTts />}
      <Routing />

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
