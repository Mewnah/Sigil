import { FC } from "react";
import { useTranslation } from "react-i18next";
import { listenOptionsFor } from "@/core/services/vrc/receiver";
import type { VRC_State } from "@/core/services/vrc/schema";
import Inspector from "../components";
import { effectiveOscHost, rustTreatsAsThisPc } from "./rustHost";

/**
 * Notes about the Windows Firewall question, next to the VRChat settings that can raise it. Each note describes a socket that
 * `osc.rs` / `osc_query.rs` really bind for the current settings (their module docs list them), so the predicates start from
 * the settings the receiver itself starts from (`listenOptionsFor`: the integration on and mute sync on) and from what the Rust
 * side makes of the VRChat address (`rustTreatsAsThisPc`), not from a second reading of the checkboxes.
 *
 * - "Find VRChat automatically (OSCQuery)": the mDNS service always binds UDP port 5353 on every network adapter, even though
 *   Sigil only talks to the VRChat on this PC (the library offers no loopback-only socket).
 * - A fixed listen port with VRChat on this PC: the receiver first checks that no other app holds the port on any address.
 * - A VRChat reached over the network: sending binds a wildcard UDP socket (on the first message), and with mute sync the
 *   receiver listens on every adapter, which has to be allowed for the mute to arrive.
 *
 * Whether the question shows up at all is Windows' decision (a rule for this Sigil.exe may exist already, and which binds raise
 * it has not been measured, see the `osc.rs` module doc), so the notes say "may". They never tell anyone to choose Cancel: that
 * makes Windows write a Block rule for Sigil, which silently breaks LAN access and mute sync later. They sit next to the
 * setting, not in a dialog.
 */

export type PromptNoteState = Pick<VRC_State, "enable" | "muteSync" | "oscQuery" | "oscListenPort" | "oscHost">;

export interface PromptNotes {
  /** OSCQuery is in use (VRChat on this PC): the mDNS service listens on UDP 5353 on every adapter. */
  mdns: boolean;
  /** A fixed listen port with VRChat on this PC: the port is checked on every address before the loopback bind. */
  fixedPort: boolean;
  /** VRChat is reached over the network: `send` binds a wildcard send socket only, `listen` also listens on every adapter. */
  remote: "send" | "listen" | null;
}

/** Which notes apply to these settings. Nothing is bound, so nothing shows, while the VRChat integration is off. */
export function promptNotesFor(state: PromptNoteState): PromptNotes {
  const none: PromptNotes = { mdns: false, fixedPort: false, remote: null };
  if (!state.enable) return none;
  const thisPc = rustTreatsAsThisPc(effectiveOscHost(state.oscHost));
  const receiver = listenOptionsFor(state);
  if (!thisPc) return { ...none, remote: receiver ? "listen" : "send" };
  if (!receiver) return none;
  return receiver.oscQuery ? { ...none, mdns: true } : { ...none, fixedPort: true };
}

/** Under the OSCQuery checkbox. */
export const MdnsPromptNote: FC<{ state: PromptNoteState }> = ({ state }) => {
  const { t } = useTranslation();
  if (!promptNotesFor(state).mdns) return null;
  return <Inspector.Description>{t("vrc.osc_query_firewall_note")}</Inspector.Description>;
};

/** Under the listen port field (the fallback the OSCQuery note names). */
export const FixedPortPromptNote: FC<{ state: PromptNoteState }> = ({ state }) => {
  const { t } = useTranslation();
  if (!promptNotesFor(state).fixedPort) return null;
  return <Inspector.Description>{t("vrc.osc_listen_port_firewall_note")}</Inspector.Description>;
};

/** Under the VRChat host field. */
export const RemoteHostPromptNote: FC<{ state: PromptNoteState }> = ({ state }) => {
  const { t } = useTranslation();
  const remote = promptNotesFor(state).remote;
  if (!remote) return null;
  return <Inspector.Description>{t(remote === "listen" ? "vrc.remote_host_firewall_note_listen" : "vrc.remote_host_firewall_note_send")}</Inspector.Description>;
};
