import { invoke } from "@tauri-apps/api/core";

/**
 * The Google Fonts family list, fetched by the Rust `google-fonts` plugin (`src-tauri/src/services/google_fonts.rs`).
 *
 * The Developer API needs an API key. The key is compiled into the Rust side and added there, so it is not in
 * this bundle (which the embedded web server hands to every client) and never crosses IPC. Host only: the
 * Tauri command does not exist in a browser, and `src/client/**` callers guard on `window.Config.isServer()`.
 *
 * `T` is the shape of one family (`GoogleFont` in `src/client/services/files`). Rejects with a readable message
 * (for example when the build has no key).
 */
export async function fetchGoogleFontsList<T>(): Promise<T[]> {
  const list = await invoke<{ items?: T[] }>("plugin:google-fonts|google_fonts_list");
  if (!Array.isArray(list?.items)) throw new Error("The Google Fonts list was empty.");
  return list.items;
}
