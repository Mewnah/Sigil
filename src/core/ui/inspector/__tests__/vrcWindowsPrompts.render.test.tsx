/**
 * Server-side render tests of the Windows Firewall notes in the VRChat inspector (W2): the OSCQuery setting (mute sync with
 * OSCQuery binds UDP 5353 on every adapter through mdns-sd), a fixed listen port (the port is checked on every address first),
 * and a VRChat address reached over the network (a wildcard send socket, and the mute listener on every adapter). The notes are
 * English strings in `public/i18n/en/translation.json`; the staged `scripts/i18n-staging/fix-W2-windows-prompts.json` (when it
 * exists) is read on top of it, with its `$remove` list applied, as the merge does. The tests compare with the product strings,
 * not with copies of them.
 */
import fs from "node:fs";
import path from "node:path";
import { renderToStaticMarkup } from "react-dom/server";
import type { ReactNode } from "react";
import { proxy } from "valtio";
import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import en from "../../../../../public/i18n/en/translation.json";

vi.mock("@tauri-apps/api/core", () => ({ isTauri: vi.fn(() => true), invoke: vi.fn(async () => undefined) }));
vi.mock("@tauri-apps/api/event", () => ({ listen: vi.fn(async () => () => {}) }));
vi.mock("@/client", () => ({ useGetState: (select: (state: unknown) => unknown) => select({ canvas: { w: 1920, h: 1080 }, author: "" }) }));
// The SteamVR / caption window section has its own state and tests; it is not what this file is about.
vi.mock("../vrc/OverlaySection", () => ({ OverlaySection: () => null }));

import { listenOptionsFor } from "@/core/services/vrc/receiver";
import { Service_VRC_Schema, type VRC_State } from "@/core/services/vrc/schema";
import Inspector_VRC from "../inspector_vrc";
import { FixedPortPromptNote, MdnsPromptNote, RemoteHostPromptNote, promptNotesFor, type PromptNoteState } from "../vrc/windowsPromptNotes";

const ROOT = path.resolve(__dirname, "../../../../..");

type Json = { [key: string]: string | string[] | Json };
function merge(base: Json, extra: Json): Json {
  const out: Json = { ...base };
  for (const [k, v] of Object.entries(extra)) {
    if (k.startsWith("$")) continue; // reserved: "$remove"
    const existing = out[k];
    out[k] = typeof v === "object" && !Array.isArray(v) && typeof existing === "object" && !Array.isArray(existing) ? merge(existing, v) : v;
  }
  return out;
}

function removeKey(tree: Json, dotted: string) {
  const parts = dotted.split(".");
  let node: Json | undefined = tree;
  for (const part of parts.slice(0, -1)) {
    const next: Json | string | string[] | undefined = node?.[part];
    node = typeof next === "object" && !Array.isArray(next) ? next : undefined;
  }
  if (node) delete node[parts[parts.length - 1]];
}

/** English as the app has it: the strings that are merged already, plus this workstream's staged ones (a no-op once merged). */
function english(): Json {
  const staged = path.join(ROOT, "scripts", "i18n-staging", "fix-W2-windows-prompts.json");
  if (!fs.existsSync(staged)) return en as unknown as Json;
  const extra = JSON.parse(fs.readFileSync(staged, "utf8")) as Json;
  const merged = merge(en as unknown as Json, extra);
  for (const dotted of (extra["$remove"] as string[] | undefined) ?? []) removeKey(merged, dotted);
  return merged;
}

const VRC = (english() as { vrc: Record<string, string> }).vrc;
const MDNS_NOTE = VRC.osc_query_firewall_note;
const PORT_NOTE = VRC.osc_listen_port_firewall_note;
const SEND_NOTE = VRC.remote_host_firewall_note_send;
const LISTEN_NOTE = VRC.remote_host_firewall_note_listen;
const ALL_NOTES = [MDNS_NOTE, PORT_NOTE, SEND_NOTE, LISTEN_NOTE];

function html(node: ReactNode): string {
  return renderToStaticMarkup(<>{node}</>).replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, "&");
}

/** Which of the four notes the page shows (by index into ALL_NOTES: mdns, fixed port, remote send, remote listen). */
function shown(out: string): string[] {
  return ["mdns", "port", "send", "listen"].filter((_, i) => out.includes(ALL_NOTES[i]));
}

/** The host-side globals the VRChat inspector reads, with the VRChat settings `patch` applied to the defaults. */
function mount(patch: Partial<VRC_State>) {
  const data = proxy({ ...Service_VRC_Schema.parse({}), enable: true, ...patch });
  vi.stubGlobal("window", {
    ApiServer: {
      state: {
        services: {
          vrc: { data },
          stt: { data: proxy({ backend: "native", local: { smartTurn: false } }) },
          translation: { data: { partials: proxy({ turnEndCommit: false }) } },
        },
      },
      vrc: { serviceState: proxy({ receiver: null, vrchatMuted: null, mutedByVrchat: false }), refreshReceiverStatus: vi.fn() },
      patchService: vi.fn(),
    },
  });
  return data;
}

beforeAll(async () => {
  await i18n.use(initReactI18next).init({ lng: "en", resources: { en: { translation: english() } }, interpolation: { escapeValue: false } });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("which notes show (pure helpers)", () => {
  const base: PromptNoteState = { enable: true, muteSync: true, oscQuery: true, oscListenPort: "9001", oscHost: "127.0.0.1" };
  const none = { mdns: false, fixedPort: false, remote: null };

  it("OSCQuery on this PC needs the integration on, mute sync on and the OSCQuery switch on", () => {
    expect(promptNotesFor(base)).toEqual({ ...none, mdns: true });
    expect(promptNotesFor({ ...base, oscHost: "" })).toEqual({ ...none, mdns: true }); // empty means the default 127.0.0.1
    expect(promptNotesFor({ ...base, oscHost: "localhost" })).toEqual({ ...none, mdns: true });
    expect(promptNotesFor({ ...base, enable: false })).toEqual(none); // the receiver (and the mDNS service) runs only while enabled
    expect(promptNotesFor({ ...base, muteSync: false })).toEqual(none); // nothing listens without mute sync
  });

  it("a fixed listen port on this PC is the port check, whatever the port", () => {
    expect(promptNotesFor({ ...base, oscQuery: false })).toEqual({ ...none, fixedPort: true });
    expect(promptNotesFor({ ...base, oscQuery: false, oscListenPort: "not a port" })).toEqual({ ...none, fixedPort: true });
    expect(promptNotesFor({ ...base, oscQuery: false, enable: false })).toEqual(none);
    expect(promptNotesFor({ ...base, oscQuery: false, muteSync: false })).toEqual(none);
  });

  it("a VRChat over the network: a send socket, plus the listener with mute sync, and nothing while the integration is off", () => {
    for (const oscHost of ["192.168.1.50", "gaming-pc", "gaming-pc.local", "10.0.0.2"]) {
      expect(promptNotesFor({ ...base, oscHost }), oscHost).toEqual({ ...none, remote: "listen" });
      expect(promptNotesFor({ ...base, oscHost, oscQuery: false }), oscHost).toEqual({ ...none, remote: "listen" });
      expect(promptNotesFor({ ...base, oscHost, muteSync: false }), oscHost).toEqual({ ...none, remote: "send" });
      expect(promptNotesFor({ ...base, oscHost, enable: false }), oscHost).toEqual(none);
    }
  });

  it("follows what the Rust side makes of the address (the settings' own loopback check is the same one)", () => {
    // Rust reads these as this PC (a loopback bind), and so does the check that shows the OSCQuery switch: OSCQuery, its note.
    for (const oscHost of ["0:0:0:0:0:0:0:1", "::ffff:7f00:1", "[::FFFF:7F00:1]"]) {
      expect(promptNotesFor({ ...base, oscHost }), oscHost).toEqual({ ...none, mdns: true });
      expect(promptNotesFor({ ...base, oscHost, oscQuery: false }), oscHost).toEqual({ ...none, fixedPort: true });
      expect(promptNotesFor({ ...base, oscHost, muteSync: false }), oscHost).toEqual(none);
    }
    // Rust does not (a name to look up): it binds every adapter, and the settings hide the OSCQuery switch for it too.
    expect(promptNotesFor({ ...base, oscHost: "127.999.0.1" })).toEqual({ ...none, remote: "listen" });
    expect(listenOptionsFor({ ...base, oscHost: "127.999.0.1" })?.oscQuery).toBe(false);
  });
});

describe("the notes in the VRChat inspector", () => {
  it("a default setup (VRChat on this PC, mute sync off) shows no note", () => {
    mount({});
    const out = html(<Inspector_VRC />);
    expect(out).toContain("Mute captions when I mute in VRChat");
    expect(shown(out)).toEqual([]);
  });

  it("mute sync with OSCQuery on this PC: the note sits under the OSCQuery description, before the receiver status", () => {
    mount({ muteSync: true, oscQuery: true });
    const out = html(<Inspector_VRC />);
    const description = out.indexOf("so it never clashes with other OSC apps that use port 9001");
    expect(description).toBeGreaterThan(-1);
    const note = out.indexOf(MDNS_NOTE);
    expect(note).toBeGreaterThan(description);
    expect(out.indexOf("Starting…")).toBeGreaterThan(note);
    expect(shown(out)).toEqual(["mdns"]);
  });

  it("with the integration off nothing is bound, so no note shows, however mute sync and the address are set", () => {
    mount({ enable: false, muteSync: true, oscQuery: true });
    expect(shown(html(<Inspector_VRC />))).toEqual([]);
    mount({ enable: false, muteSync: true, oscQuery: false });
    expect(shown(html(<Inspector_VRC />))).toEqual([]);
    mount({ enable: false, oscHost: "192.168.1.50", muteSync: true });
    expect(shown(html(<Inspector_VRC />))).toEqual([]);
    mount({ enable: false, oscHost: "192.168.1.50" });
    expect(shown(html(<Inspector_VRC />))).toEqual([]);
  });

  it("OSCQuery turned off: the fixed listen port it falls back to has its own note, under the port field", () => {
    mount({ muteSync: true, oscQuery: false });
    const out = html(<Inspector_VRC />);
    const field = out.indexOf("Listen port (VRChat sends to 9001)");
    expect(field).toBeGreaterThan(-1);
    const note = out.indexOf(PORT_NOTE);
    expect(note).toBeGreaterThan(field);
    expect(out.indexOf("Starting…")).toBeGreaterThan(note);
    expect(shown(out)).toEqual(["port"]);
  });

  it("the advice in the OSCQuery note is real: the switch it names is in the inspector, and turned off it gives the listen port field", () => {
    expect(MDNS_NOTE).toContain("turn this off and use a fixed listen port");
    mount({ muteSync: true, oscQuery: true });
    const on = html(<Inspector_VRC />);
    expect(on).toContain("Find VRChat automatically (OSCQuery)");
    expect(on).not.toContain("Listen port (VRChat sends to 9001)");
    mount({ muteSync: true, oscQuery: false });
    const off = html(<Inspector_VRC />);
    expect(off).toContain("Listen port (VRChat sends to 9001)");
    expect(off).toContain("Find VRChat automatically (OSCQuery)");
  });

  it("a VRChat over the network, mute sync off: only the send note, after the host field and before the port", () => {
    mount({ oscHost: "192.168.1.50" });
    const out = html(<Inspector_VRC />);
    const host = out.indexOf("Host (VRChat runs on this machine)");
    const note = out.indexOf(SEND_NOTE);
    const port = out.indexOf("Port (VRChat OSC receive port, often 9000)");
    expect(host).toBeGreaterThan(-1);
    expect(note).toBeGreaterThan(host);
    expect(port).toBeGreaterThan(note);
    expect(shown(out)).toEqual(["send"]);
  });

  it("a VRChat over the network with mute sync: the listener note replaces the send note (OSCQuery is hidden, the listen port is shown)", () => {
    mount({ oscHost: "gaming-pc", muteSync: true, oscQuery: true });
    const out = html(<Inspector_VRC />);
    expect(shown(out)).toEqual(["listen"]);
    expect(out).not.toContain("Find VRChat automatically (OSCQuery)");
    expect(out).toContain("VRChat runs on another PC: start it there with the launch option");
  });

  it("follows the address and the integration switch while the page is open", () => {
    const data = mount({});
    expect(shown(html(<Inspector_VRC />))).toEqual([]);
    data.oscHost = "192.168.1.50";
    expect(shown(html(<Inspector_VRC />))).toEqual(["send"]);
    data.muteSync = true;
    expect(shown(html(<Inspector_VRC />))).toEqual(["listen"]);
    data.enable = false;
    expect(shown(html(<Inspector_VRC />))).toEqual([]);
    data.enable = true;
    data.oscHost = "localhost";
    expect(shown(html(<Inspector_VRC />))).toEqual(["mdns"]);
    data.oscQuery = false;
    expect(shown(html(<Inspector_VRC />))).toEqual(["port"]);
  });

  it("an address the Rust side reads as this PC gets the notes of this PC, not 'over the network'", () => {
    mount({ oscHost: "0:0:0:0:0:0:0:1", muteSync: true, oscQuery: true });
    const out = html(<Inspector_VRC />);
    expect(out).toContain("Find VRChat automatically (OSCQuery)"); // the switch is shown, as for 127.0.0.1
    expect(shown(out)).toEqual(["mdns"]);
    mount({ oscHost: "::ffff:7f00:1" });
    expect(shown(html(<Inspector_VRC />))).toEqual([]);
  });

  it("the note components render nothing when they do not apply", () => {
    const base: PromptNoteState = { enable: true, muteSync: true, oscQuery: true, oscListenPort: "9001", oscHost: "127.0.0.1" };
    expect(html(<MdnsPromptNote state={{ ...base, muteSync: false }} />)).toBe("");
    expect(html(<MdnsPromptNote state={{ ...base, enable: false }} />)).toBe("");
    expect(html(<FixedPortPromptNote state={base} />)).toBe("");
    expect(html(<RemoteHostPromptNote state={base} />)).toBe("");
    expect(html(<MdnsPromptNote state={base} />)).toContain("UDP port 5353");
    expect(html(<FixedPortPromptNote state={{ ...base, oscQuery: false }} />)).toContain("no other app is using this port");
    expect(html(<RemoteHostPromptNote state={{ ...base, oscHost: "pc" }} />)).toContain("over your network");
  });
});

describe("the merged strings", () => {
  it("are in the English vrc section, without product-label wording, and never advise Cancel", () => {
    for (const key of ["osc_query_firewall_note", "osc_listen_port_firewall_note", "remote_host_firewall_note_send", "remote_host_firewall_note_listen"]) {
      expect(typeof VRC[key], key).toBe("string");
      expect(VRC[key], key).not.toMatch(/\bAI\b/);
      expect(VRC[key], key).toContain("Allow access");
      // Cancel makes Windows write a Block rule for Sigil: a note may say so, never ask for it.
      expect(VRC[key], key).not.toMatch(/\b(choose|click|press|select) Cancel\b/i);
    }
    expect(VRC.remote_host_firewall_note).toBeUndefined(); // replaced by the send and listen notes
  });

  it("the notes in the source use keys that exist", () => {
    const source = fs.readFileSync(path.join(ROOT, "src", "core", "ui", "inspector", "vrc", "windowsPromptNotes.tsx"), "utf8");
    const keys = [...source.matchAll(/"vrc\.([a-z_]+firewall_note[a-z_]*)"/g)].map((m) => m[1]);
    expect(keys.sort()).toEqual(["osc_listen_port_firewall_note", "osc_query_firewall_note", "remote_host_firewall_note_listen", "remote_host_firewall_note_send"]);
    for (const key of keys) expect(typeof VRC[key], key).toBe("string");
  });
});
