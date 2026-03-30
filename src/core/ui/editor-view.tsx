import { FC, memo, useEffect, useRef, useState, MouseEvent as ReactMouseEvent } from "react";
import { useTranslation } from "react-i18next";
import { useDebounce } from "react-use";
import { AnimatePresence, motion, Variants } from "framer-motion";
import { useSnapshot } from "valtio";
import { useShallow } from "zustand/react/shallow";
import { useAppUIStore } from "./store";
import { ElementEditorTransform } from "./element-transform";
import { useGetState, useUpdateState } from "@/client";
import classNames from "classnames";
import { canvasToolbarState } from "./CanvasToolbar";
import { canvasDragGuideState, moveGuideHighlightPx } from "./canvas-drag-guide-state";

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
  const snapToGrid = useGetState(state => state.snapToGrid);
  const { zoom, showGrid, gridSize } = useSnapshot(canvasToolbarState);
  const dragGuide = useSnapshot(canvasDragGuideState);
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

  const { tab, selections } = useAppUIStore(
    useShallow((s) => ({ tab: s.sidebar.tab, selections: s.sidebar.selections }))
  );
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "ArrowUp" && e.key !== "ArrowDown" && e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.isContentEditable ||
          target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT")
      ) {
        return;
      }
      const sel = useAppUIStore.getState().sidebar.selections ?? [];
      if (sel.length === 0) return;

      const doc = window.ApiClient.document.fileBinder.get();
      const grid = canvasToolbarState.gridSize;
      const step = doc.snapToGrid ? (e.shiftKey ? grid * 5 : grid) : e.shiftKey ? 10 : 1;
      const dx = (e.key === "ArrowLeft" ? -1 : e.key === "ArrowRight" ? 1 : 0) * step;
      const dy = (e.key === "ArrowUp" ? -1 : e.key === "ArrowDown" ? 1 : 0) * step;
      if (dx === 0 && dy === 0) return;

      e.preventDefault();
      window.ApiClient.document.patch(state => {
        for (const id of sel) {
          const el = state.elements[id];
          const rect = el?.scenes[activeScene]?.rect;
          if (!rect) continue;
          rect.x = Math.round(rect.x + dx);
          rect.y = Math.round(rect.y + dy);
        }
      });
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeScene]);

  useEffect(() => {
    if (selections && selections.length > 0) {
      // Use the first selection as the primary one for handles
      setSelectedId(selections[0]);
    } else {
      setSelectedId(null);
    }
  }, [selections]);

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
      style={{ 
        width: localDim.w, 
        height: localDim.h,
        zoom: zoom / 100,
      }}
      className={classNames(
        "relative shrink-0 overflow-hidden bg-base-100 rounded-lg border border-dashed border-primary/50 group shadow-2xl transition-[zoom] duration-300",
        { "border-primary": isCanvasSelected }
      )}
      data-sigil-canvas-root
    >
      {/* Grid Overlay */}
      {showGrid && (
        <div
          className="absolute inset-0 pointer-events-none z-0 rounded-lg"
          style={{
            backgroundImage: `
              linear-gradient(to right, rgba(128,128,128,0.08) 1px, transparent 1px),
              linear-gradient(to bottom, rgba(128,128,128,0.08) 1px, transparent 1px)
            `,
            backgroundSize: `${gridSize}px ${gridSize}px`,
          }}
        />
      )}
      {snapToGrid && dragGuide.moveActive && dragGuide.dragRect && canvas && (() => {
        const hl = moveGuideHighlightPx(zoom);
        const ccx = canvas.w / 2;
        const ccy = canvas.h / 2;
        const ecx = dragGuide.dragRect.x + dragGuide.dragRect.w / 2;
        const ecy = dragGuide.dragRect.y + dragGuide.dragRect.h / 2;
        const emphasizeV = Math.abs(ecx - ccx) < hl;
        const emphasizeH = Math.abs(ecy - ccy) < hl;
        return (
          <div className="absolute inset-0 pointer-events-none z-[60] rounded-lg" aria-hidden>
            <div
              className={classNames(
                "absolute top-0 bottom-0 w-px -translate-x-1/2 transition-colors duration-75",
                emphasizeV ? "bg-primary" : "bg-primary/35"
              )}
              style={{ left: "50%" }}
            />
            <div
              className={classNames(
                "absolute left-0 right-0 h-px -translate-y-1/2 transition-colors duration-75",
                emphasizeH ? "bg-primary" : "bg-primary/35"
              )}
              style={{ top: "50%" }}
            />
          </div>
        );
      })()}
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
        canvasSelected={selectedId === elementId || selections.includes(elementId)}
        onSelect={(e) => {
          if (e.shiftKey) {
            const cur = useAppUIStore.getState().sidebar.selections ?? [];
            const next = [...cur];
            const idx = next.indexOf(elementId);
            if (idx === -1) next.push(elementId);
            else next.splice(idx, 1);
            useAppUIStore.getState().setSidebarSelections(next);

            if (next.includes(elementId)) {
              setSelectedId(elementId);
            }
          } else {
            useAppUIStore.getState().setSidebarSelections([elementId]);
            setSelectedId(elementId);
          }
        }}
      />)}
    </motion.div>
  );
});

export const EditorViewport: FC = () => {
  const { t } = useTranslation();
  const { activeScene } = useSnapshot(window.ApiClient.scenes.state);
  const transition = useGetState(state => state.transition);
  const viewportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      const overCanvas = !!(e.target as HTMLElement | null)?.closest?.("[data-sigil-canvas-root]");
      if (!e.ctrlKey && !e.metaKey && !overCanvas) return;
      e.preventDefault();
      const delta = e.deltaY > 0 ? -10 : 10;
      canvasToolbarState.zoom = Math.min(200, Math.max(25, canvasToolbarState.zoom + delta));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  return (
    <div
      ref={viewportRef}
      className="w-full h-full relative flex flex-col items-center overflow-auto p-8 py-6"
    >
      <div className="flex min-h-0 w-full flex-1 flex-col items-center justify-center">
        <div className="flex shrink-0 flex-col items-center gap-2">
          {/* Padding reserves space for resize handles (absolute -bottom-2 / -right-2) so they don’t paint over the hint */}
          <div className="shrink-0 pb-8">
            <AnimatePresence mode="wait">
              <Canvas activeScene={activeScene} transition={transition} />
            </AnimatePresence>
          </div>
          <div className="max-w-full overflow-x-auto">
            <p
              className="whitespace-nowrap select-none px-2 text-center text-xs leading-snug text-base-content/50 pointer-events-none"
              role="note"
            >
              {t("scenes.canvas_resize_hint")}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EditorViewport;
