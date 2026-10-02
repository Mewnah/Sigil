import type { Root } from "react-dom/client";
import { CaptionWindowView } from "./CaptionWindowView";

/**
 * Early route of the caption window (P6-B): `src/index.tsx` calls this for `/captions` before `AppConfiguration`,
 * `ApiShared`, `ApiClient` or `ApiServer` exist, so the window never boots the host (no settings file, no PeerJS, no
 * services). Outside the desktop app (a browser opening /captions on the LAN server) it renders nothing.
 */
export function mountCaptionWindow(root: Root): void {
  const html = document.documentElement;
  html.setAttribute("data-sigil-captions", "1");
  html.removeAttribute("data-theme");
  document.title = "Sigil captions";
  const tauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
  root.render(tauri ? <CaptionWindowView /> : null);
}
