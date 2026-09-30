/**
 * Output device names for the TTS sink (P2-G). Pickers list cpal names (`plugin:audio|list_output_devices`);
 * older settings hold names from the SAPI / WinMM device list, which truncates names to 31 characters. This
 * mirrors the Rust sink's matching (`tts_sink::output::match_output_device`), so the picker can migrate a saved
 * name to the device it means, and both sides agree on which device plays.
 */

/** Empty or `"default"` (any case): the system default output. */
export function isDefaultOutputName(name: string | null | undefined): boolean {
  const t = (name ?? "").trim();
  return t === "" || t.toLowerCase() === "default";
}

/** Lower-cased, trimmed, without Windows' endpoint numbering (`N- ` at the start or after an opening parenthesis). */
export function deviceFamily(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/(^|\()\d+- /g, "$1");
}

function unique(names: readonly string[], pred: (name: string) => boolean): number | null {
  let hit: number | null = null;
  for (let i = 0; i < names.length; i++) {
    if (!pred(names[i])) continue;
    if (hit !== null) return null;
    hit = i;
  }
  return hit;
}

/**
 * Index of the device a saved name means among `names`: the exact (trimmed) name, else the same name ignoring
 * case, else the one device of the same family (Windows renumbering), else the one device the saved name is a
 * prefix of (a truncated SAPI / WinMM name, 12 characters or more). `null` when none or ambiguous.
 */
export function matchOutputDevice(saved: string, names: readonly string[]): number | null {
  const wanted = saved.trim();
  if (!wanted) return null;
  const exact = names.findIndex((n) => n.trim() === wanted);
  if (exact >= 0) return exact;
  const lower = wanted.toLowerCase();
  const sameCase = unique(names, (n) => n.trim().toLowerCase() === lower);
  if (sameCase !== null) return sameCase;
  const family = deviceFamily(wanted);
  const sameFamily = unique(names, (n) => deviceFamily(n) === family);
  if (sameFamily !== null) return sameFamily;
  if (Array.from(wanted).length >= 12) {
    const prefix = unique(names, (n) => n.trim().toLowerCase().startsWith(lower));
    if (prefix !== null) return prefix;
    return unique(names, (n) => deviceFamily(n).startsWith(family));
  }
  return null;
}

/**
 * The cpal name a saved output device should be stored as: unchanged when it is the default or already a listed
 * name, the matching listed name for an old SAPI / renumbered name, else unchanged (not connected now).
 */
export function migrateOutputDeviceName(saved: string, names: readonly string[]): string {
  if (isDefaultOutputName(saved) || names.some((n) => n === saved)) return saved;
  const i = matchOutputDevice(saved, names);
  return i === null ? saved : names[i];
}
