import { useGetState, useUpdateState } from "@/client";
import { TransformRect } from "@/client/elements/schema";
import { ElementInstance } from "@/client/ui/element-instance";
import classNames from "classnames";
import { FC, memo, useEffect, useState, MouseEvent as ReactMouseEvent } from "react";
import { useDebounce } from "react-use";
import { useSnapshot } from "valtio";

type TransformDirection = 'n' | 'e' | 's' | 'w' | 'nw' | 'ne' | 'se' | 'sw' | 'm';

function compareRect(a?: TransformRect, b?: TransformRect) {
  return !!a && !!b && a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;
}

export const ElementEditorTransform: FC<{ id: string, canvasSelected?: boolean, onSelect?: () => void }> = memo(({ id, canvasSelected, onSelect }) => {
  const { activeScene } = useSnapshot(window.ApiClient.scenes.state);
  const docRect = useGetState(state => state.elements[id].scenes[activeScene]?.rect);
  const [rect, setRect] = useState<TransformRect>(window.ApiClient.document.fileBinder.get().elements[id].scenes[activeScene]?.rect);

  const update = useUpdateState();
  const snapToGrid = useGetState(state => state.snapToGrid);
  const canvas = useGetState(state => state.canvas);

  useEffect(() => {
    setRect(window.ApiClient.document.fileBinder.get().elements[id].scenes[activeScene]?.rect);
  }, [activeScene, docRect]);

  useDebounce(() => {
    update(state => {
      if (!compareRect(rect, state.elements[id].scenes[activeScene].rect)) {
        state.elements[id].scenes[activeScene].rect = rect;
      }
    });
  }, 100, [rect]);

  const { tab, show } = useSnapshot(window.ApiServer.ui.sidebarState);

  const selected = (show && tab?.value === id) || canvasSelected;

  const selectElement = () => {
    const state = window.ApiClient.document.fileBinder.get();
    const element = state.elements[id]
    window.ApiServer.changeTab({ tab: element.type, value: id });
  }

  const [mouseDown, setMouseDown] = useState(false);
  const [transformDirection, setTransformDirection] = useState<TransformDirection>();
  const [snapCandidates, setSnapCandidates] = useState<TransformRect[]>([]);

  const handleMove = (e: MouseEvent) => {
    selected && setRect(oldRect => {
      const rect = { ...oldRect };

      // 1. Apply Movement
      if (transformDirection === 'n') {
        rect.y += e.movementY;
        rect.h -= e.movementY;
      }
      else if (transformDirection === 'e') {
        rect.w += e.movementX;
      }
      else if (transformDirection === 's') {
        rect.h += e.movementY;
      }
      else if (transformDirection === 'w') {
        rect.x += e.movementX;
        rect.w -= e.movementX;
      }
      else if (transformDirection === 'nw') {
        rect.y += e.movementY; rect.h -= e.movementY;
        rect.x += e.movementX; rect.w -= e.movementX;
      }
      else if (transformDirection === 'ne') {
        rect.y += e.movementY; rect.h -= e.movementY;
        rect.w += e.movementX;
      }
      else if (transformDirection === 'se') {
        rect.h += e.movementY; rect.w += e.movementX;
      }
      else if (transformDirection === 'sw') {
        rect.h += e.movementY;
        rect.x += e.movementX; rect.w -= e.movementX;
      }
      else if (transformDirection === 'm') {
        rect.x += e.movementX;
        rect.y += e.movementY;
      }

      // 2. Apply Snapping (if enabled)
      if (snapToGrid && canvas) {
        const SNAP = 2;

        // A. Snap to Canvas
        if (transformDirection === 'm') {
          // Snap X (Left, Center, Right)
          if (Math.abs(rect.x) < SNAP) rect.x = 0;
          else if (Math.abs(rect.x + rect.w / 2 - canvas.w / 2) < SNAP) rect.x = canvas.w / 2 - rect.w / 2;
          else if (Math.abs(rect.x + rect.w - canvas.w) < SNAP) rect.x = canvas.w - rect.w;

          // Snap Y (Top, Center, Bottom)
          if (Math.abs(rect.y) < SNAP) rect.y = 0;
          else if (Math.abs(rect.y + rect.h / 2 - canvas.h / 2) < SNAP) rect.y = canvas.h / 2 - rect.h / 2;
          else if (Math.abs(rect.y + rect.h - canvas.h) < SNAP) rect.y = canvas.h - rect.h;
        }
        // Snap to Canvas (Resize) - Simplified for clarity
        else if (transformDirection === 'e' || transformDirection === 'se' || transformDirection === 'ne') {
          if (Math.abs(rect.x + rect.w - canvas.w) < SNAP) rect.w = canvas.w - rect.x;
        }
        if (transformDirection === 's' || transformDirection === 'se' || transformDirection === 'sw') {
          if (Math.abs(rect.y + rect.h - canvas.h) < SNAP) rect.h = canvas.h - rect.y;
        }
        if (transformDirection === 'w' || transformDirection === 'sw' || transformDirection === 'nw') {
          if (Math.abs(rect.x) < SNAP) { rect.w += rect.x; rect.x = 0; }
        }
        if (transformDirection === 'n' || transformDirection === 'nw' || transformDirection === 'ne') {
          if (Math.abs(rect.y) < SNAP) { rect.h += rect.y; rect.y = 0; }
        }

        // B. Snap to Other Elements (using cached candidates)
        for (const r2 of snapCandidates) {
          // X Snapping
          // Move: Snap Left/Right/Center
          if (transformDirection === 'm') {
            // Left to Left
            if (Math.abs(rect.x - r2.x) < SNAP) rect.x = r2.x;
            // Left to Right
            else if (Math.abs(rect.x - (r2.x + r2.w)) < SNAP) rect.x = r2.x + r2.w;
            // Right to Left
            else if (Math.abs((rect.x + rect.w) - r2.x) < SNAP) rect.x = r2.x - rect.w;
            // Right to Right
            else if (Math.abs((rect.x + rect.w) - (r2.x + r2.w)) < SNAP) rect.x = (r2.x + r2.w) - rect.w;
            // Center to Center
            else if (Math.abs((rect.x + rect.w / 2) - (r2.x + r2.w / 2)) < SNAP) rect.x = (r2.x + r2.w / 2) - rect.w / 2;
          }
          // Resize East: Snap Right Edge
          else if ((transformDirection === 'e' || transformDirection === 'ne' || transformDirection === 'se')) {
            if (Math.abs((rect.x + rect.w) - r2.x) < SNAP) rect.w = r2.x - rect.x;
            else if (Math.abs((rect.x + rect.w) - (r2.x + r2.w)) < SNAP) rect.w = (r2.x + r2.w) - rect.x;
          }
          // Resize West: Snap Left Edge (Adjust X and W)
          else if ((transformDirection === 'w' || transformDirection === 'nw' || transformDirection === 'sw')) {
            let targetX = null;
            if (Math.abs(rect.x - r2.x) < SNAP) targetX = r2.x;
            else if (Math.abs(rect.x - (r2.x + r2.w)) < SNAP) targetX = r2.x + r2.w;

            if (targetX !== null) {
              const diff = targetX - rect.x;
              rect.x += diff;
              rect.w -= diff;
            }
          }

          // Y Snapping
          // Move: Snap Top/Bottom/Center
          if (transformDirection === 'm') {
            // Top to Top
            if (Math.abs(rect.y - r2.y) < SNAP) rect.y = r2.y;
            // Top to Bottom
            else if (Math.abs(rect.y - (r2.y + r2.h)) < SNAP) rect.y = r2.y + r2.h;
            // Bottom to Top
            else if (Math.abs((rect.y + rect.h) - r2.y) < SNAP) rect.y = r2.y - rect.h;
            // Bottom to Bottom
            else if (Math.abs((rect.y + rect.h) - (r2.y + r2.h)) < SNAP) rect.y = (r2.y + r2.h) - rect.h;
            // Center to Center
            else if (Math.abs((rect.y + rect.h / 2) - (r2.y + r2.h / 2)) < SNAP) rect.y = (r2.y + r2.h / 2) - rect.h / 2;
          }
          // Resize South: Snap Bottom Edge
          else if ((transformDirection === 's' || transformDirection === 'se' || transformDirection === 'sw')) {
            if (Math.abs((rect.y + rect.h) - r2.y) < SNAP) rect.h = r2.y - rect.y;
            else if (Math.abs((rect.y + rect.h) - (r2.y + r2.h)) < SNAP) rect.h = (r2.y + r2.h) - rect.y;
          }
          // Resize North: Snap Top Edge (Adjust Y and H)
          else if ((transformDirection === 'n' || transformDirection === 'ne' || transformDirection === 'nw')) {
            let targetY = null;
            if (Math.abs(rect.y - r2.y) < SNAP) targetY = r2.y;
            else if (Math.abs(rect.y - (r2.y + r2.h)) < SNAP) targetY = r2.y + r2.h;

            if (targetY !== null) {
              const diff = targetY - rect.y;
              rect.y += diff;
              rect.h -= diff;
            }
          }
        }
      }

      rect.y = Math.round(rect.y);
      rect.h = Math.round(rect.h);
      rect.x = Math.round(rect.x);
      rect.w = Math.round(rect.w);
      return rect;
    });
  };

  useEffect(() => {
    if (mouseDown)
      window.addEventListener("mousemove", handleMove);
    return () => window.removeEventListener("mousemove", handleMove);
  }, [mouseDown, transformDirection]);

  useEffect(() => {
    const removeDrag = () => { setMouseDown(false); setSnapCandidates([]); }
    window.addEventListener("mouseup", removeDrag);
    return () => window.removeEventListener("mousemove", removeDrag);
  }, []);

  const handleDragDown = (direction: TransformDirection) => {
    return (e: ReactMouseEvent<HTMLDivElement>) => {
      e.stopPropagation();
      onSelect?.();

      // Calculate snap candidates once
      const elements = window.ApiClient.document.fileBinder.get().elements;
      const candidates = Object.values(elements)
        .filter(el => el.id !== id && el.scenes[activeScene] && el.scenes[activeScene].rect)
        .map(el => el.scenes[activeScene].rect!);

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
    className={classNames("cursor-pointer group absolute inset-0", { "transition-all duration-100": !selected, "z-50": selected })}
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
