import { zSafe } from "@/utils";
import { z } from "zod";

/**
 * Legacy Moonshine HTTP server settings (`POST {endpoint}/transcribe`, `GET {endpoint}/health`): requests are sent by
 * the Rust speech pipeline's HTTP engine. The `local` backend runs Moonshine v2 in-process without a server.
 */
export const STT_Moonshine_Schema = z.object({
  device: zSafe(z.coerce.string(), ""),
  endpoint: zSafe(z.coerce.string(), "http://localhost:8090"),
  language: zSafe(z.coerce.string(), "en"),
}).default({});
