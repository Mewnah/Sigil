/**
 * Interface size for code that has to work in pixels (the mouse position, a floating-ui offset, a popover measured with
 * `getBoundingClientRect`). Everything the CSS draws is rem, so it follows the Interface size on its own; a number that
 * ends up as a pixel length does not. Write such a number as "px at Interface size 1" and pass it through `uiPx`.
 *
 * No imports on purpose: Tooltip, Dropdown and Drawer use this, and `uiScale.ts` pulls in the Tauri window API.
 */

/** px per rem at Interface size 1 (the browser default, and the `16px` in `html { font-size: calc(16px * var(--uiscale)) }`). */
export const BASE_REM_PX = 16;

/**
 * px per rem right now: the root element's computed font-size, 16 x Interface size. Read live, so it is also right in the
 * middle of the short font-size transition. 16 when there is no document (tests, SSR) or the value is not usable.
 */
export function remPx(): number {
  if (typeof document === "undefined" || typeof getComputedStyle !== "function") return BASE_REM_PX;
  const size = parseFloat(getComputedStyle(document.documentElement).fontSize);
  return Number.isFinite(size) && size > 0 ? size : BASE_REM_PX;
}

/** `px` as drawn at the current Interface size: `uiPx(8)` is 8 at size 1 and 12 at size 1.5. */
export function uiPx(px: number): number {
  return (px * remPx()) / BASE_REM_PX;
}

/**
 * A size prop of a component: a number is rem (`31.25` is the 500 px of Interface size 1, and grows with it), a string
 * is any CSS length (`"auto"`, `"min(90vw, 32rem)"`).
 */
export function cssLength(value: number | string): string {
  return typeof value === "number" ? `${value}rem` : value;
}

/** Screen pixels (a mouse position) to rem at the current Interface size. */
export function screenPxToRem(px: number): number {
  return px / remPx();
}
