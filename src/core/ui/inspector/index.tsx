import { AnimatePresence } from "framer-motion";
import { FC } from "react";
import { ErrorBoundary } from "react-error-boundary";
import { toast } from "react-toastify";
import SimpleBar from "simplebar-react";
import { Services } from "@/core";
import { useGetState } from "@/client";
import Inspector_STT from "./inspector_stt";
import Inspector_Translation from "./inspector_translation";
import Inspector_TTS from "./inspector_tts";
import Inspector_VRC from "./inspector_vrc";
import Inspector_Transform from "./inspector_transform";
import Inspector_ElementImage from "./inspector_image";
import Inspector_ElementText from "./inspector_text";
import Inspector_Files from "./inspector_files";
import Inspector_ElementAudioViz from "./inspector_audioviz";
import { ElementType } from "@/client/elements/schema";
import { InspectorTabPath } from "@/types";
import Inspector_Scenes from "./inspector_scenes";
import Inspector_Settings from "./inspector_settings";
import Inspector_Twitch from "./inspector_twitch";
import Inspector_Discord from "./inspector_discord";
import Inspector_Kick from "./inspector_kick";
import Inspector_OBS from "./inspector_obs";
import { RiCloseLine } from "react-icons/ri";

import Inspector_Project from "./inspector_project";
import { Inspector_VoiceChanger } from "./inspector_voice_changer";

// LEFT PANEL INSPECTOR (Navigation Driven)
const Inspector: FC<{ path?: InspectorTabPath }> = ({ path }) => {
  const handleCopyError = (err: string) => {
    navigator.clipboard.writeText(err);
    toast.success("Copied!");
  }
  return <div className="w-[22rem] h-full flex-none bg-base-100 rounded-t-box flex flex-col overflow-hidden">
    <div className="flex-grow relative overflow-hidden">
      <ErrorBoundary fallbackRender={({ error, resetErrorBoundary }) => (
        <div className="w-full h-full flex flex-col items-center justify-center p-4 space-y-2">
          {/* Error UI ... */}
          <div className="text-error">Inspector Error</div>
        </div>
      )}>
        <AnimatePresence initial={false} mode="wait">
          {path?.tab === Services.stt && <Inspector_STT key="stt" />}
          {path?.tab === Services.tts && <Inspector_TTS key="tts" />}
          {path?.tab === Services.translation && <Inspector_Translation key="translation" />}
          {path?.tab === "obs" && <Inspector_OBS key="obs" />}
          {path?.tab === Services.vrc && <Inspector_VRC key="vrc" />}
          {path?.tab === Services.transform && <Inspector_Transform key="transform" />}
          {path?.tab === Services.twitch && <Inspector_Twitch key="twitch" />}
          {path?.tab === Services.kick && <Inspector_Kick key="kick" />}
          {path?.tab === Services.discord && <Inspector_Discord key="discord" />}
          {path?.tab === Services.voice_changer && <Inspector_VoiceChanger key="voice_changer" />}
          {path?.tab === "settings" && <Inspector_Settings key="settings" />}
          {path?.tab === "files" && <Inspector_Files key="files" />}

          {/* Default/Project Tab (Replaces Elements/Scenes/Studio) */}
          {(path?.tab === "project" || path?.tab === "scenes" || path?.tab === ElementType.text || path?.tab === ElementType.image) && <Inspector_Project key="project" />}

        </AnimatePresence>

      </ErrorBoundary>
    </div>
  </div>
}

// RIGHT PANEL INSPECTOR (Selection Driven)
export const PropertyInspector: FC<{ selectionId: string }> = ({ selectionId }) => {
  const element = useGetState(state => state.elements[selectionId]);
  if (!element) return null;

  return (
    <div className="w-[22rem] h-full flex-none bg-base-100 border-l border-base-content/10 flex flex-col overflow-hidden">
      <div className="flex-none p-4 border-b border-base-content/10 font-bold flex justify-between items-center z-10 bg-base-100">
        <div className="flex items-center gap-2">
          <button
            onClick={() => window.ApiServer.ui.sidebarState.selections = []}
            className="btn btn-sm btn-circle btn-ghost"
            title="Close Properties"
          >
            <RiCloseLine />
          </button>
          <span>Properties</span>
        </div>
        <span className="badge badge-sm badge-neutral font-mono opacity-50 capitalize">{element.type}</span>
      </div>
      <div className="flex-grow relative overflow-hidden">
        {element.type === ElementType.text && <Inspector_ElementText id={selectionId} />}
        {element.type === ElementType.image && <Inspector_ElementImage id={selectionId} />}
        {element.type === ElementType.audioViz && <Inspector_ElementAudioViz id={selectionId} />}
      </div>
    </div>
  );
};

export default Inspector;
