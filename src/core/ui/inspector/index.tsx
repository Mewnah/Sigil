import { AnimatePresence } from "framer-motion";
import { FC, lazy, Suspense } from "react";
import { ErrorBoundary, FallbackProps } from "react-error-boundary";
import { Services } from "@/services-registry";
import { useGetState } from "@/client";
import { ElementType } from "@/client/elements/schema";
import { InspectorTabPath } from "@/types";
import { RiCloseLine } from "react-icons/ri";
import { useAppUIStore } from "../store";

const Inspector_STT = lazy(() => import("./inspector_stt"));
const Inspector_Translation = lazy(() => import("./inspector_translation"));
const Inspector_TTS = lazy(() => import("./inspector_tts"));
const Inspector_VRC = lazy(() => import("./inspector_vrc"));
const Inspector_Transform = lazy(() => import("./inspector_transform"));
const Inspector_ElementImage = lazy(() => import("./inspector_image"));
const Inspector_ElementText = lazy(() => import("./inspector_text"));
const Inspector_Files = lazy(() => import("./inspector_files"));
const Inspector_Settings = lazy(() => import("./inspector_settings"));
const Inspector_Twitch = lazy(() => import("./inspector_twitch"));
const Inspector_Discord = lazy(() => import("./inspector_discord"));
const Inspector_Kick = lazy(() => import("./inspector_kick"));
const Inspector_OBS = lazy(() => import("./inspector_obs"));
const Inspector_Project = lazy(() => import("./inspector_project"));
const Inspector_Canvas = lazy(() => import("./inspector_canvas"));
const Inspector_VoiceChanger = lazy(() =>
  import("./inspector_voice_changer").then((m) => ({ default: m.Inspector_VoiceChanger }))
);

const inspectorFallback = (
  <div className="flex items-center justify-center h-full min-h-[8rem] text-sm text-base-content/50">
    Loading panel…
  </div>
);

const inspectorErrorFallback = ({ error, resetErrorBoundary }: FallbackProps) => (
  <div className="w-full h-full flex flex-col p-4 gap-3 bg-error/10 border border-error/30 rounded-lg overflow-auto">
    <div className="text-error font-semibold text-sm">Inspector error</div>
    <pre className="text-[10px] font-mono whitespace-pre-wrap opacity-90 flex-1">{error.message}</pre>
    <div className="flex flex-wrap gap-2">
      <button type="button" className="btn btn-sm btn-primary" onClick={resetErrorBoundary}>
        Try again
      </button>
      <button
        type="button"
        className="btn btn-sm btn-ghost"
        onClick={() => {
          window.ApiServer.closeSidebar();
          resetErrorBoundary();
        }}
      >
        Close panel
      </button>
    </div>
  </div>
);

const propertyInspectorErrorFallback = ({ error, resetErrorBoundary }: FallbackProps) => (
  <div className="w-full h-full flex flex-col p-4 gap-3 bg-error/10 border border-error/30 m-2 rounded-lg overflow-auto">
    <div className="text-error font-semibold text-sm">Properties error</div>
    <pre className="text-[10px] font-mono whitespace-pre-wrap opacity-90 flex-1">{error.message}</pre>
    <div className="flex flex-wrap gap-2">
      <button type="button" className="btn btn-sm btn-primary" onClick={resetErrorBoundary}>
        Try again
      </button>
      <button
        type="button"
        className="btn btn-sm btn-ghost"
        onClick={() => {
          useAppUIStore.getState().setSidebarSelections([]);
          resetErrorBoundary();
        }}
      >
        Clear selection
      </button>
    </div>
  </div>
);

// LEFT PANEL INSPECTOR (Navigation Driven)
const Inspector: FC<{ path?: InspectorTabPath }> = ({ path }) => {
  const canvasKey = path?.tab === "scenes" ? "scenes" : "canvas-elements";

  return <div className="w-[22rem] h-full flex-none bg-base-100 rounded-t-box flex flex-col overflow-hidden">
    <div className="flex-grow relative overflow-hidden">
      <ErrorBoundary FallbackComponent={inspectorErrorFallback}>
        <AnimatePresence initial={false} mode="wait">
          <Suspense fallback={inspectorFallback}>
            {path?.tab === Services.stt && <Inspector_STT key="stt" />}
            {path?.tab === Services.tts && <Inspector_TTS key="tts" />}
            {path?.tab === Services.translation && <Inspector_Translation key="translation" />}
            {path?.tab === Services.obs && <Inspector_OBS key="obs" />}
            {path?.tab === Services.vrc && <Inspector_VRC key="vrc" />}
            {path?.tab === Services.transform && <Inspector_Transform key="transform" />}
            {path?.tab === Services.twitch && <Inspector_Twitch key="twitch" />}
            {path?.tab === Services.kick && <Inspector_Kick key="kick" />}
            {path?.tab === Services.discord && <Inspector_Discord key="discord" />}
            {path?.tab === Services.voice_changer && <Inspector_VoiceChanger key="voice_changer" />}
            {path?.tab === "settings" && <Inspector_Settings key="settings" />}
            {path?.tab === "files" && <Inspector_Files key="files" />}
            {path?.tab === "project" && <Inspector_Project key="project" />}
            {(path?.tab === "scenes" ||
              path?.tab === "elements" ||
              path?.tab === ElementType.text ||
              path?.tab === ElementType.image) && (
              <Inspector_Canvas key={canvasKey} />
            )}
          </Suspense>
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
            onClick={() => useAppUIStore.getState().setSidebarSelections([])}
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
        <ErrorBoundary FallbackComponent={propertyInspectorErrorFallback} resetKeys={[selectionId]}>
          <Suspense fallback={inspectorFallback}>
            {element.type === ElementType.text && <Inspector_ElementText id={selectionId} />}
            {element.type === ElementType.image && <Inspector_ElementImage id={selectionId} />}
          </Suspense>
        </ErrorBoundary>
      </div>
    </div>
  );
};

export default Inspector;
