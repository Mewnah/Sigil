import { useGetState, useUpdateState } from "@/client";
import { TransformRect } from "@/client/elements/schema";
import { ElementInstance } from "@/client/ui/element-instance";
import classNames from "classnames";
import { FC, memo, useEffect, useRef, useState, MouseEvent as ReactMouseEvent } from "react";
import { useDebounce } from "react-use";
import { useSnapshot } from "valtio";
import { useShallow } from "zustand/react/shallow";
import { useAppUIStore } from "./store";
import { canvasToolbarState } from "./CanvasToolbar";
import {
  canvasDragGuideState,
  moveDragSnapThreshold,
  resizeDragSnapThreshold,
} from "./canvas-drag-guide-state";

type TransformDirection = 'n' | 'e' | 's' | 'w' | 'nw' | 'ne' | 'se' | 'sw' | 'm';

function compareRect(a?: TransformRect, b?: TransformRect) {
  return !!a && !!b && a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;
}

/** Map viewport pointer to logical canvas coordinates (handles CSS zoom on the canvas root). */
function clientPointerToCanvas(
  clientX: number,
  clientY: number,
  canvasEl: HTMLElement,
  logicalW: number,
  logicalH: number
): { x: number; y: number } {
  const br = canvasEl.getBoundingClientRect();
  if (br.width <= 0 || br.height <= 0) return { x: 0, y: 0 };
  return {
    x: ((clientX - br.left) / br.width) * logicalW,
    y: ((clientY - br.top) / br.height) * logicalH,
  };
}

type DragComputeOpts = {
  direction: TransformDirection;
  movementX: number;
  movementY: number;
  pointerCanvas: { x: number; y: number } | null;
  /** Grab offset at start of this move (do not read live ref mid-compute). */
  grabOffset: { x: number; y: number };
  snapToGrid: boolean;
  canvas: { w: number; h: number };
  zoom: number;
  snapCandidates: TransformRect[];
};

/** One drag step: movement + snap + round. Pure — safe to call once per pointer event. */
function computeNextTransformRect(prev: TransformRect, o: DragComputeOpts): TransformRect {
  const rect = { ...prev };

  if (o.direction === "n") {
    rect.y += o.movementY;
    rect.h -= o.movementY;
  } else if (o.direction === "e") {
    rect.w += o.movementX;
  } else if (o.direction === "s") {
    rect.h += o.movementY;
  } else if (o.direction === "w") {
    rect.x += o.movementX;
    rect.w -= o.movementX;
  } else if (o.direction === "nw") {
    rect.y += o.movementY;
    rect.h -= o.movementY;
    rect.x += o.movementX;
    rect.w -= o.movementX;
  } else if (o.direction === "ne") {
    rect.y += o.movementY;
    rect.h -= o.movementY;
    rect.w += o.movementX;
  } else if (o.direction === "se") {
    rect.h += o.movementY;
    rect.w += o.movementX;
  } else if (o.direction === "sw") {
    rect.h += o.movementY;
    rect.x += o.movementX;
    rect.w -= o.movementX;
  } else if (o.direction === "m") {
    if (o.pointerCanvas) {
      rect.x = o.pointerCanvas.x - o.grabOffset.x;
      rect.y = o.pointerCanvas.y - o.grabOffset.y;
    } else {
      rect.x += o.movementX;
      rect.y += o.movementY;
    }
  }

  const { snapToGrid, canvas, snapCandidates } = o;
  if (snapToGrid && canvas) {
    const SNAP_M = moveDragSnapThreshold(o.zoom);
    const SNAP_R = resizeDragSnapThreshold(o.zoom);

    if (o.direction === "m") {
      if (Math.abs(rect.x) < SNAP_M) rect.x = 0;
      else if (Math.abs(rect.x + rect.w / 2 - canvas.w / 2) < SNAP_M) rect.x = canvas.w / 2 - rect.w / 2;
      else if (Math.abs(rect.x + rect.w - canvas.w) < SNAP_M) rect.x = canvas.w - rect.w;

      if (Math.abs(rect.y) < SNAP_M) rect.y = 0;
      else if (Math.abs(rect.y + rect.h / 2 - canvas.h / 2) < SNAP_M) rect.y = canvas.h / 2 - rect.h / 2;
      else if (Math.abs(rect.y + rect.h - canvas.h) < SNAP_M) rect.y = canvas.h - rect.h;

      for (const r2 of snapCandidates) {
        if (Math.abs(rect.x - r2.x) < SNAP_M) rect.x = r2.x;
        else if (Math.abs(rect.x - (r2.x + r2.w)) < SNAP_M) rect.x = r2.x + r2.w;
        else if (Math.abs(rect.x + rect.w - r2.x) < SNAP_M) rect.x = r2.x - rect.w;
        else if (Math.abs(rect.x + rect.w - (r2.x + r2.w)) < SNAP_M) rect.x = r2.x + r2.w - rect.w;
        else if (Math.abs(rect.x + rect.w / 2 - (r2.x + r2.w / 2)) < SNAP_M) rect.x = r2.x + r2.w / 2 - rect.w / 2;

        if (Math.abs(rect.y - r2.y) < SNAP_M) rect.y = r2.y;
        else if (Math.abs(rect.y - (r2.y + r2.h)) < SNAP_M) rect.y = r2.y + r2.h;
        else if (Math.abs(rect.y + rect.h - r2.y) < SNAP_M) rect.y = r2.y - rect.h;
        else if (Math.abs(rect.y + rect.h - (r2.y + r2.h)) < SNAP_M) rect.y = r2.y + r2.h - rect.h;
        else if (Math.abs(rect.y + rect.h / 2 - (r2.y + r2.h / 2)) < SNAP_M) rect.y = r2.y + r2.h / 2 - rect.h / 2;
      }
    } else {
      if (o.direction === "e" || o.direction === "se" || o.direction === "ne") {
        if (Math.abs(rect.x + rect.w - canvas.w) < SNAP_R) rect.w = canvas.w - rect.x;
      }
      if (o.direction === "s" || o.direction === "se" || o.direction === "sw") {
        if (Math.abs(rect.y + rect.h - canvas.h) < SNAP_R) rect.h = canvas.h - rect.y;
      }
      if (o.direction === "w" || o.direction === "sw" || o.direction === "nw") {
        if (Math.abs(rect.x) < SNAP_R) {
          rect.w += rect.x;
          rect.x = 0;
        }
      }
      if (o.direction === "n" || o.direction === "nw" || o.direction === "ne") {
        if (Math.abs(rect.y) < SNAP_R) {
          rect.h += rect.y;
          rect.y = 0;
        }
      }

      for (const r2 of snapCandidates) {
        if (o.direction === "e" || o.direction === "ne" || o.direction === "se") {
          if (Math.abs(rect.x + rect.w - r2.x) < SNAP_R) rect.w = r2.x - rect.x;
          else if (Math.abs(rect.x + rect.w - (r2.x + r2.w)) < SNAP_R) rect.w = r2.x + r2.w - rect.x;
        } else if (o.direction === "w" || o.direction === "nw" || o.direction === "sw") {
          let targetX: number | null = null;
          if (Math.abs(rect.x - r2.x) < SNAP_R) targetX = r2.x;
          else if (Math.abs(rect.x - (r2.x + r2.w)) < SNAP_R) targetX = r2.x + r2.w;
          if (targetX !== null) {
            const diff = targetX - rect.x;
            rect.x += diff;
            rect.w -= diff;
          }
        }

        if (o.direction === "s" || o.direction === "se" || o.direction === "sw") {
          if (Math.abs(rect.y + rect.h - r2.y) < SNAP_R) rect.h = r2.y - rect.y;
          else if (Math.abs(rect.y + rect.h - (r2.y + r2.h)) < SNAP_R) rect.h = r2.y + r2.h - rect.y;
        } else if (o.direction === "n" || o.direction === "ne" || o.direction === "nw") {
          let targetY: number | null = null;
          if (Math.abs(rect.y - r2.y) < SNAP_R) targetY = r2.y;
          else if (Math.abs(rect.y - (r2.y + r2.h)) < SNAP_R) targetY = r2.y + r2.h;
          if (targetY !== null) {
            const diff = targetY - rect.y;
            rect.y += diff;
            rect.h -= diff;
          }
        }
      }
    }
  }

  rect.y = Math.round(rect.y);
  rect.h = Math.round(rect.h);
  rect.x = Math.round(rect.x);
  rect.w = Math.round(rect.w);
  return rect;
}

export const ElementEditorTransform: FC<{ id: string, canvasSelected?: boolean, onSelect?: (e: ReactMouseEvent) => void }> = memo(({ id, canvasSelected, onSelect }) => {
  const { activeScene } = useSnapshot(window.ApiClient.scenes.state);
  const docRect = useGetState(state => state.elements[id]?.scenes[activeScene]?.rect);
  const [rect, setRect] = useState<TransformRect | undefined>(() =>
    window.ApiClient.document.fileBinder.get().elements[id]?.scenes[activeScene]?.rect
  );

  const update = useUpdateState();
  const snapToGrid = useGetState(state => state.snapToGrid);
  const canvas = useGetState(state => state.canvas);
  const canvasZoom = useSnapshot(canvasToolbarState).zoom;
  const zoomRef = useRef(canvasZoom);
  zoomRef.current = canvasZoom;

  useEffect(() => {
    const next = window.ApiClient.document.fileBinder.get().elements[id]?.scenes[activeScene]?.rect;
    if (next) setRect(next);
  }, [activeScene, docRect, id]);

  useDebounce(() => {
    if (rect === undefined) return;
    update(state => {
      const el = state.elements[id];
      const sceneRect = el?.scenes[activeScene]?.rect;
      if (!el || !sceneRect) return;
      if (!compareRect(rect, sceneRect)) {
        const oldRect = sceneRect;
        const deltaX = rect.x - oldRect.x;
        const deltaY = rect.y - oldRect.y;

        el.scenes[activeScene].rect = rect;

        // Apply to other selected elements
        const selections = useAppUIStore.getState().sidebar.selections;
        if (selections && selections.length > 1 && selections.includes(id)) {
          selections.forEach(otherId => {
            if (otherId === id) return;
            const otherEl = state.elements[otherId];
            if (otherEl && otherEl.scenes[activeScene]?.rect) {
              otherEl.scenes[activeScene].rect.x += deltaX;
              otherEl.scenes[activeScene].rect.y += deltaY;
            }
          });
        }
      }
    });
  }, 10, [rect]);

  const { tab, show } = useAppUIStore(
    useShallow((s) => ({ tab: s.sidebar.tab, show: s.sidebar.show }))
  );

  const selected = (show && tab?.value === id) || canvasSelected;

  const selectElement = () => {
    const state = window.ApiClient.document.fileBinder.get();
    const element = state.elements[id];
    if (!element) return;
    window.ApiServer.changeTab({ tab: element.type, value: id });
  }

  const [mouseDown, setMouseDown] = useState(false);
  const [transformDirection, setTransformDirection] = useState<TransformDirection>();
  const [snapCandidates, setSnapCandidates] = useState<TransformRect[]>([]);

  /** Canvas-space offset from element top-left to pointer — re-synced after each move+snap so grab stays under cursor. */
  const moveGrabOffsetRef = useRef({ x: 0, y: 0 });
  const canvasRef = useRef(canvas);
  canvasRef.current = canvas;
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  const snapToGridRef = useRef(snapToGrid);
  snapToGridRef.current = snapToGrid;
  const snapCandidatesRef = useRef(snapCandidates);
  snapCandidatesRef.current = snapCandidates;
  const transformDirectionRef = useRef(transformDirection);
  transformDirectionRef.current = transformDirection;
  const latestRectRef = useRef<TransformRect | undefined>(rect);
  latestRectRef.current = rect;
  const mouseDownRef = useRef(false);
  mouseDownRef.current = mouseDown;
  const endPointerDragRef = useRef<() => void>(() => {});
  endPointerDragRef.current = () => {
    canvasDragGuideState.moveActive = false;
    canvasDragGuideState.dragRect = null;
    setMouseDown(false);
    setSnapCandidates([]);
    setTransformDirection(undefined);
  };

  const handleMoveRef = useRef<(e: MouseEvent) => void>(() => {});
  handleMoveRef.current = (e: MouseEvent) => {
    if (!selectedRef.current) return;
    const dir = transformDirectionRef.current;
    if (!dir) return;
    const c = canvasRef.current;
    if (!c) return;

    const canvasEl = document.querySelector("[data-sigil-canvas-root]") as HTMLElement | null;
    const pointerCanvas =
      dir === "m" && canvasEl ? clientPointerToCanvas(e.clientX, e.clientY, canvasEl, c.w, c.h) : null;

    const grabAtStart = { ...moveGrabOffsetRef.current };
    const prev = latestRectRef.current;
    if (!prev) return;
    const next = computeNextTransformRect(prev, {
      direction: dir,
      movementX: e.movementX,
      movementY: e.movementY,
      pointerCanvas,
      grabOffset: grabAtStart,
      snapToGrid: snapToGridRef.current,
      canvas: c,
      zoom: zoomRef.current,
      snapCandidates: snapCandidatesRef.current,
    });

    if (dir === "m" && pointerCanvas) {
      moveGrabOffsetRef.current = {
        x: pointerCanvas.x - next.x,
        y: pointerCanvas.y - next.y,
      };
    }
    if (canvasDragGuideState.moveActive) {
      canvasDragGuideState.dragRect = { x: next.x, y: next.y, w: next.w, h: next.h };
    }

    latestRectRef.current = next;
    setRect(next);
  };

  useEffect(() => {
    if (!mouseDown) return;
    const fn = (e: MouseEvent) => handleMoveRef.current(e);
    window.addEventListener("mousemove", fn);
    return () => window.removeEventListener("mousemove", fn);
  }, [mouseDown]);

  useEffect(() => {
    const onUp = () => endPointerDragRef.current();
    window.addEventListener("mouseup", onUp);
    return () => window.removeEventListener("mouseup", onUp);
  }, []);

  const handleDragDown = (direction: TransformDirection) => {
    return (e: ReactMouseEvent<HTMLDivElement>) => {
      e.stopPropagation();
      onSelect?.(e);

      // Calculate snap candidates once
      const elements = window.ApiClient.document.fileBinder.get().elements;
      const candidates = Object.values(elements)
        .filter(el => el.id !== id && el.scenes[activeScene] && el.scenes[activeScene].rect)
        .map(el => el.scenes[activeScene].rect!);

      if (direction === "m") {
        const doc = window.ApiClient.document.fileBinder.get();
        const r = doc.elements[id]?.scenes[activeScene]?.rect;
        const cv = doc.canvas;
        const canvasEl = document.querySelector("[data-sigil-canvas-root]") as HTMLElement | null;
        if (r && cv && canvasEl) {
          const p = clientPointerToCanvas(e.clientX, e.clientY, canvasEl, cv.w, cv.h);
          moveGrabOffsetRef.current = { x: p.x - r.x, y: p.y - r.y };
          latestRectRef.current = { ...r };
        }
        if (snapToGrid && r) {
          canvasDragGuideState.moveActive = true;
          canvasDragGuideState.dragRect = { x: r.x, y: r.y, w: r.w, h: r.h };
        }
      }

      setSnapCandidates(candidates);
      setMouseDown(true);
      setTransformDirection(direction);
    }
  }

  if (!rect)
    return null;

  return <div
    onDoubleClick={(e) => { e.stopPropagation(); selectElement(); }}
    onMouseDown={handleDragDown('m')}
    className={classNames(
      "cursor-pointer group absolute min-h-0 min-w-0 overflow-hidden",
      { "transition-all duration-100": !selected, "z-50": selected }
    )}
    style={{
      width: rect?.w || 0,
      height: rect?.h || 0,
      left: rect?.x || 0,
      top: rect?.y || 0,
    }}>
    <ElementInstance id={id} />
    <div className={classNames("absolute inset-0 border-2 border-dashed opacity-0 border-secondary/50 transition-opacity",
      selected ? "opacity-100 border-primary cursor-move" : "group-hover:opacity-30 border border-dashed border-b-primary"
    )}>
      <div onMouseDown={handleDragDown("n")} className="cursor-n-resize absolute -top-1 w-full h-1"></div>
      <div onMouseDown={handleDragDown("e")} className="cursor-e-resize absolute -right-1 h-full w-1"></div>
      <div onMouseDown={handleDragDown("s")} className="cursor-s-resize absolute -bottom-1 w-full h-1"></div>
      <div onMouseDown={handleDragDown("w")} className="cursor-w-resize absolute -left-1 h-full w-1"></div>

      <div onMouseDown={handleDragDown("nw")} className="cursor-nw-resize absolute -top-1 -left-1 rounded-full bg-primary w-2 h-2"></div>
      <div onMouseDown={handleDragDown("ne")} className="cursor-ne-resize absolute -top-1 -right-1 rounded-full bg-primary w-2 h-2"></div>
      <div onMouseDown={handleDragDown("se")} className="cursor-se-resize absolute -bottom-1 -right-1 rounded-full bg-primary w-2 h-2"></div>
      <div onMouseDown={handleDragDown("sw")} className="cursor-sw-resize absolute -bottom-1 -left-1 rounded-full bg-primary w-2 h-2"></div>
    </div>
  </div>
});
