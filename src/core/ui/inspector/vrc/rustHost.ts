/**
 * Whether the Rust OSC plugin treats a VRChat address as this PC: a mirror of `literal_ip` + `is_loopback_host` in
 * `src-tauri/src/services/osc.rs`. That function decides what Sigil binds (the loopback, or every adapter), which is what the
 * Windows Firewall notes of the VRChat settings describe. The settings' own check that hides the OSCQuery switch
 * (`isLoopbackHost` in `services/vrc/receiver.ts`) delegates here, so the switch, the receiver options and the notes agree
 * with Rust on every spelling (`0:0:0:0:0:0:0:1`, `::ffff:7f00:1`, `127.999.0.1`). Both sides are checked against
 * `src-tauri/src/services/osc_host_cases.json`.
 */

/** The host a send or a listen uses: the setting trimmed, an empty one meaning the default 127.0.0.1 (as `sendOsc` and `listenOptionsFor` do). */
export function effectiveOscHost(raw: unknown): string {
  return String(raw ?? "").trim() || "127.0.0.1";
}

/** Rust's `Ipv4Addr` parser: four decimal parts of 0 to 255 without leading zeros. */
function parseV4(text: string): [number, number, number, number] | null {
  const parts = text.split(".");
  if (parts.length !== 4) return null;
  const out: number[] = [];
  for (const part of parts) {
    if (!/^(0|[1-9][0-9]{0,2})$/.test(part)) return null;
    const n = Number(part);
    if (n > 255) return null;
    out.push(n);
  }
  return out as [number, number, number, number];
}

/** One run of colon-separated 16-bit groups, the last of them possibly a dotted IPv4 address (two groups). */
function parseGroups(text: string, allowV4Tail: boolean): number[] | null {
  if (text === "") return [];
  const pieces = text.split(":");
  const out: number[] = [];
  for (let i = 0; i < pieces.length; i++) {
    const piece = pieces[i];
    if (allowV4Tail && i === pieces.length - 1 && piece.includes(".")) {
      const v4 = parseV4(piece);
      if (!v4) return null;
      out.push((v4[0] << 8) | v4[1], (v4[2] << 8) | v4[3]);
    } else {
      if (!/^[0-9a-fA-F]{1,4}$/.test(piece)) return null;
      out.push(parseInt(piece, 16));
    }
  }
  return out;
}

/** Rust's `Ipv6Addr` parser (no zone id): eight groups, one `::` for the missing zeros, an optional dotted IPv4 tail. */
function parseV6(text: string): number[] | null {
  const halves = text.split("::");
  if (halves.length > 2) return null;
  if (halves.length === 1) {
    const groups = parseGroups(text, true);
    return groups && groups.length === 8 ? groups : null;
  }
  const head = parseGroups(halves[0], false);
  const tail = parseGroups(halves[1], true);
  if (!head || !tail || head.length + tail.length > 7) return null;
  return [...head, ...new Array<number>(8 - head.length - tail.length).fill(0), ...tail];
}

/**
 * Whether Rust's `is_loopback_host` says `raw` is this PC: `localhost`, or an IP literal (brackets allowed) that is loopback
 * once an IPv4-mapped IPv6 address is read as the IPv4 one. An empty host is not (Rust refuses it; callers pass
 * {@link effectiveOscHost}), and neither is a name other than `localhost`.
 */
export function rustTreatsAsThisPc(raw: unknown): boolean {
  const host = String(raw ?? "").trim();
  if (host === "") return false;
  const bare = host.startsWith("[") && host.endsWith("]") ? host.slice(1, -1) : host;
  if (bare.toLowerCase() === "localhost") return true;
  const v4 = parseV4(bare);
  if (v4) return v4[0] === 127;
  const v6 = parseV6(bare);
  if (!v6) return false;
  const mapped = v6.slice(0, 5).every((g) => g === 0) && v6[5] === 0xffff;
  if (mapped) return (v6[6] >> 8) === 127;
  return v6.slice(0, 7).every((g) => g === 0) && v6[7] === 1;
}
