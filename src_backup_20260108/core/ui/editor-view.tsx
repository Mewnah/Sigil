import { FC, memo, useEffect, useState, MouseEvent as ReactMouseEvent } from "react";
import { useDebounce } from "react-use";
import { AnimatePresence, motion, Variants } from "framer-motion";
import { useSnapshot } from "valtio";
import { ElementEditorTransform } from "./element-transform";
import { useGetState, useUpdateState } from "@/client";
import classNames from "classnames";

// Transition helper
const getSceneVariants = (type: string = 'none', duration: number = 300): Variants => {
  const transition = { duration: duration / 1000, ease: "easeInOut" };

  switch (type) {
    case 'fade':
      return {
        initial: { opacity: 0 },
        animate: { opacity: 1, transition },
        exit: { opacity: 0, transition }
      };
    case 'blur':
      return {
        initial: { opacity: 0, filter: "blur(10px)" },
        animate: { opacity: 1, filter: "blur(0px)", transition },
        exit: { opacity: 0, filter: "blur(10px)", transition }
      };
    case 'slide':
      return {
        initial: { x: 20, opacity: 0 },
        animate: { x: 0, opacity: 1, transition },
        exit: { x: -20, opacity: 0, transition }
      };
    default:
      return {
        initial: { opacity: 1 },
        animate: { opacity: 1 },
        exit: { opacity: 1 }
      };
  }
};

const Canvas: FC<{ activeScene: string, transition?: { type: string, duration: number } }> = memo(({ activeScene, transition }) => {
  const canvas = useGetState(state => state.canvas);
  const ids = useGetState(state => state.elementsIds);
  const update = useUpdateState();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isCanvasSelected, setIsCanvasSelected] = useState(false);

  const [localDim, setLocalDim] = useState({ w: canvas?.w || 500, h: canvas?.h || 400 });
  const [resizing, setResizing] = useState<string | null>(null);

  useEffect(() => {
    if (canvas) setLocalDim({ w: canvas.w, h: canvas.h });
  }, [canvas?.w, canvas?.h]);

  useDebounce(() => {
    if (canvas && (canvas.w !== localDim.w || canvas.h !== localDim.h)) {
      update(s => { s.canvas.w = localDim.w; s.canvas.h = localDim.h });
    }
  }, 100, [localDim]);

  const { tab } = useSnapshot(window.ApiServer.ui.sidebarState);
  useEffect(() => {
    if (tab?.value) {
      setSelectedId(tab.value);
    }
    // If we're on a service page (like STT, TTS) that is NOT a studio tab, we might want to hide the canvas or dim it
    // But currently we always show it.
    // If we wanted to hide it for performance or focus:
    if (tab?.tab && tab.tab !== 'scenes' && tab.tab !== 'text' && tab.tab !== 'image') {
      // checks against ElementType "text" and "image"
      setIsCanvasSelected(false);
    } else if (tab?.tab === 'scenes') {
      setIsCanvasSelected(true);
    }
  }, [tab?.value, tab?.tab]);

  useEffect(() => {
    if (!resizing) return;
    const handleMove = (e: MouseEvent) => {
      setLocalDim(prev => {
        const next = { ...prev };
        if (resizing.includes('e')) next.w += e.movementX;
        if (resizing.includes('s')) next.h += e.movementY;
        return next;
      })
    }
    const handleUp = () => setResizing(null);
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);
    return () => { window.removeEventListener('mousemove', handleMove); window.removeEventListener('mouseup', handleUp); }
  }, [resizing]);

  const handleDragStart = (dir: string) => (e: ReactMouseEvent) => {
    e.stopPropagation();
    setResizing(dir);
  }

  const variants = getSceneVariants(transition?.type, transition?.duration);

  return (
    <motion.div
      key={activeScene}
      variants={variants}
      initial="initial"
      animate="animate"
      exit="exit"
      onMouseDown={() => { setSelectedId(null); setIsCanvasSelected(false); }}
      onDoubleClick={() => { setIsCanvasSelected(true); window.ApiServer.changeTab({ tab: "scenes" }); }}
      style={{ width: localDim.w, height: localDim.h }}
      className={classNames("relative bg-black rounded-lg border border-dashed border-primary/50 group shadow-2xl", {
        "border-primary": isCanvasSelected
      })}
    >
      {/* Canvas Resize Handles */}
      <div onMouseDown={handleDragStart("e")} className={classNames("absolute -right-2 top-0 bottom-0 w-4 cursor-e-resize flex items-center justify-center transition-opacity",
        isCanvasSelected ? "opacity-100" : "opacity-0 pointer-events-none")}>
        <div className="w-1 h-8 bg-base-content/20 rounded-full"></div>
      </div>
      <div onMouseDown={handleDragStart("s")} className={classNames("absolute -bottom-2 left-0 right-0 h-4 cursor-s-resize flex items-center justify-center transition-opacity",
        isCanvasSelected ? "opacity-100" : "opacity-0 pointer-events-none")}>
        <div className="h-1 w-8 bg-base-content/20 rounded-full"></div>
      </div>
      <div onMouseDown={handleDragStart("se")} className={classNames("absolute -bottom-2 -right-2 w-4 h-4 cursor-se-resize bg-primary rounded-full transition-opacity",
        isCanvasSelected ? "opacity-100" : "opacity-0 pointer-events-none")}></div>

      {ids?.map((elementId) => <ElementEditorTransform
        id={elementId}
        key={elementId}
        canvasSelected={selectedId === elementId || window.ApiServer.ui.sidebarState.selections.includes(elementId)}
        onSelect={(e) => {
          if (e.shiftKey) {
            const selections = window.ApiServer.ui.sidebarState.selections ? [...window.ApiServer.ui.sidebarState.selections] : [];
            const idx = selections.indexOf(elementId);
            if (idx === -1) selections.push(elementId);
            else selections.splice(idx, 1);
            window.ApiServer.ui.sidebarState.selections = selections;

            // If selected, make it the active tab too? Maybe not, keep focus on last clicked or don't change tab.
            if (selections.includes(elementId)) {
              setSelectedId(elementId); // Update local state for handles
            }
          } else {
            window.ApiServer.ui.sidebarState.selections = [elementId];
            setSelectedId(elementId);
          }
        }}
      />)}
    </motion.div>
  );
});

export const EditorViewport: FC = () => {
  const { activeScene } = useSnapshot(window.ApiClient.scenes.state);
  const transition = useGetState(state => state.canvas?.transition);

  return (
    <div className="w-full h-full relative flex items-center justify-center overflow-auto p-8">
      <AnimatePresence mode="wait">
        <Canvas activeScene={activeScene} transition={transition} />
      </AnimatePresence>
    </div>
  );
};

export default EditorViewport;
