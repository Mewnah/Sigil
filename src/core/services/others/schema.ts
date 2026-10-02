import { zSafe } from "@/utils";
import { z } from "zod";

/** Engine tier of the other-player pipeline (Rust `speech::engine_api::Tier`). */
export const OTHERS_TIERS = ["auto", "cpu", "gpu"] as const;

/** Default capture source: VRChat's own sound (per-process loopback). */
export const OTHERS_DEFAULT_SOURCE = "process:VRChat.exe";

/**
 * Settings of the Others service (P6-S0; owned by P6-A): other players' speech, captured by loopback, transcribed by a
 * second speech pipeline (`loopback` key, `MuteGroup::Others`) and published on `others.stt`; optionally translated
 * in the reverse direction (their language into the user's) by a second translation pipeline instance on
 * `others.translation`. Never published on `text.*`, so it never reaches the chatbox, TTS or history. Off unless
 * started.
 *
 * - `source`: `process:VRChat.exe` (per-process loopback, Windows 10 build 20348+) falls back to the default output's
 *   loopback while the program is not running (`fallback`); `loopback:default`; or `loopback:<output name>`.
 * - `fallback` (P6-A): while the program is not running (or Windows cannot capture one program), capture what the
 *   default output plays instead; off = wait for the program.
 * - `language`: the other players' language (`auto` = spoken language ID); `tier` and `twoPass` as the `local` STT
 *   backend; `musicGate` (P6-A, P6-L's FireRedVAD gate): skip world music and singing.
 * - `translate` / `translateTo`: reverse translation ("" = the user's own language: the main translation's source,
 *   else the STT language, else the app language). The engine and its settings are the Translation tab's.
 */
export const Service_Others_Schema = z.object({
  autoStart: zSafe(z.coerce.boolean(), false),
  source: zSafe(z.coerce.string(), OTHERS_DEFAULT_SOURCE),
  language: zSafe(z.coerce.string(), "auto"),
  tier: zSafe(z.enum(OTHERS_TIERS), "auto"),
  /** Publish interim (partial) text too. */
  interim: zSafe(z.coerce.boolean(), true),
  translate: zSafe(z.coerce.boolean(), false),
  translateTo: zSafe(z.coerce.string(), ""),
  /** Capture the default output while the program cannot be captured (P6-A). */
  fallback: zSafe(z.coerce.boolean(), true),
  /** Re-decode each sentence with the final model when the router offers one (P6-A). */
  twoPass: zSafe(z.coerce.boolean(), true),
  /** Skip stretches that sound like music or singing (P6-L gate; needs its model; P6-A). */
  musicGate: zSafe(z.coerce.boolean(), false),
}).default({});

export type Others_State = z.infer<typeof Service_Others_Schema>;
