import { proxy } from "valtio";

export type DragGuideRect = { x: number; y: number; w: number; h: number };

/** Move-drag center guides on the canvas (when Snap to grid is on). */
export const canvasDragGuideState = proxy({
  moveActive: false,
  dragRect: null as null | DragGuideRect,
});

/** Move drag: 2px band — light alignment assist. */
export function moveDragSnapThreshold(_zoomPercent: number): number {
  return 2;
}

/** Resize handles: same band as move. */
export function resizeDragSnapThreshold(_zoomPercent: number): number {
  return 2;
}

/** Distance from canvas center at which center guides emphasize (preview band). */
export function moveGuideHighlightPx(zoomPercent: number): number {
  return moveDragSnapThreshold(zoomPercent) * 3;
}
