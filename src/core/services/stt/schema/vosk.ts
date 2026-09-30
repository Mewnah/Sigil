import { zSafe } from "@/utils";
import { z } from "zod";

/** Vosk (WASM in the webview, fed by the capture hub) settings. */
export const STT_Vosk_Schema = z.object({
  device: zSafe(z.coerce.string(), ""),
  model: zSafe(z.coerce.string(), "vosk-model-small-en-us-0.15"),
  modelUrl: zSafe(z.coerce.string(), ""),
  /** When set, this zip is copied into app data instead of downloading from the catalog URL. */
  useLocalZip: zSafe(z.coerce.boolean(), false),
  localZipPath: zSafe(z.coerce.string(), ""),
}).default({});
