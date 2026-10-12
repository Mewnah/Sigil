import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { BackendSchema } from "../../../schema";
import { AtomicJsonFile, type JsonFileFs } from "../../diagnostics/atomicJsonFile";
import { applyFirstRunDecision, isFreshProfile, NEW_INSTALL_LAN_MODE, type FirstRunState } from "../firstRun";

/** Tiny in-memory filesystem: enough for `AtomicJsonFile.load`. */
class MemFs implements JsonFileFs {
  files = new Map<string, string>();
  unreadable = new Set<string>();
  async exists(path: string) {
    return this.files.has(path);
  }
  async readText(path: string) {
    if (this.unreadable.has(path)) throw new Error("locked by another program");
    const v = this.files.get(path);
    if (v === undefined) throw new Error(`ENOENT ${path}`);
    return v;
  }
  async writeText(path: string, text: string) {
    this.files.set(path, text);
  }
  async copy(from: string, to: string) {
    this.files.set(to, this.files.get(from) ?? "");
  }
  async rename(from: string, to: string) {
    this.files.set(to, this.files.get(from) ?? "");
    this.files.delete(from);
  }
  async mkdirp() {}
}

const PATH = "user/settings";

/** What `Service_State.init` does with a settings file, minus the proxy and the save. */
async function launch(fs: MemFs) {
  const loaded = await new AtomicJsonFile(fs, PATH, { now: () => new Date(0) }).load();
  const state = BackendSchema.parse(loaded.data ?? {});
  const changed = applyFirstRunDecision(state, isFreshProfile(loaded));
  return { state, changed, loaded };
}

describe("first run: who sees the setup wizard", () => {
  it("shows it when there is no settings file", async () => {
    const { state, changed, loaded } = await launch(new MemFs());
    expect(loaded.source).toBe("none");
    expect(state.onboardingComplete).toBe(false);
    expect(state.firstRunDecided).toBe(true);
    expect(changed).toBe(true); // recorded, so the caller saves the file
  });

  it("starts a new install on This PC only (the web server listens on 127.0.0.1, no firewall prompt)", async () => {
    const { state } = await launch(new MemFs());
    expect(state.lanAccess.mode).toBe("localhost");
    expect(state.lanAccess.token).toBe(""); // the host generates it on first start
    expect(state.lanAccess.legacyNoToken).toBe(false);
  });

  it("never shows it to a profile that already exists, even with onboardingComplete false (every old profile)", async () => {
    const fs = new MemFs();
    fs.files.set(PATH, JSON.stringify({ id: "me", uiLanguage: "de", onboardingComplete: false }));
    const { state, changed } = await launch(fs);
    expect(state.onboardingComplete).toBe(true);
    expect(state.firstRunDecided).toBe(true);
    expect(state.uiLanguage).toBe("de"); // nothing else is touched
    expect(changed).toBe(true);
  });

  it("never shows it to a profile without the flag either (written before the flag existed)", async () => {
    const fs = new MemFs();
    fs.files.set(PATH, "{}");
    const { state } = await launch(fs);
    expect(state.onboardingComplete).toBe(true);
  });

  it("treats a damaged settings file as an existing profile", async () => {
    const fs = new MemFs();
    fs.files.set(PATH, "{ this is not json");
    const { state, loaded } = await launch(fs);
    expect(loaded.source).toBe("none");
    expect(loaded.problems.length).toBeGreaterThan(0);
    expect(isFreshProfile(loaded)).toBe(false);
    expect(state.onboardingComplete).toBe(true);
  });

  it("treats an empty or unreadable settings file as an existing profile", async () => {
    const empty = new MemFs();
    empty.files.set(PATH, "");
    expect((await launch(empty)).state.onboardingComplete).toBe(true);

    const locked = new MemFs();
    locked.files.set(PATH, JSON.stringify({ id: "x" }));
    locked.unreadable.add(PATH);
    expect((await launch(locked)).state.onboardingComplete).toBe(true);
  });

  it("treats a profile recovered from its backup as an existing profile", async () => {
    const fs = new MemFs();
    fs.files.set(PATH, "garbage");
    fs.files.set(`${PATH}.bak`, JSON.stringify({ id: "me", onboardingComplete: false }));
    const { state, loaded } = await launch(fs);
    expect(loaded.source).toBe("backup");
    expect(state.onboardingComplete).toBe(true);
  });
});

describe("first run: who may connect (lanAccess.mode)", () => {
  it("keeps LAN access for a profile written before lanAccess existed (no key: the schema default)", async () => {
    for (const json of [JSON.stringify({ id: "me", uiLanguage: "de", onboardingComplete: false }), "{}", JSON.stringify({ lanAccess: {} })]) {
      const fs = new MemFs();
      fs.files.set(PATH, json);
      const { state, changed } = await launch(fs);
      expect(state.lanAccess.mode, json).toBe("lan");
      expect(changed).toBe(true); // the decision is recorded; the mode is not what changed it
    }
  });

  it("keeps whatever lanAccess an existing profile has", async () => {
    const lan = new MemFs();
    lan.files.set(PATH, JSON.stringify({ id: "me", lanAccess: { mode: "lan", token: "0123456789abcdefghjkmnpqrs", legacyNoToken: true } }));
    const kept = (await launch(lan)).state.lanAccess;
    expect(kept).toEqual({ mode: "lan", token: "0123456789abcdefghjkmnpqrs", legacyNoToken: true });

    const local = new MemFs();
    local.files.set(PATH, JSON.stringify({ id: "me", lanAccess: { mode: "localhost" } }));
    expect((await launch(local)).state.lanAccess.mode).toBe("localhost");
  });

  it("keeps LAN access for a damaged, empty or unreadable settings file (an existing profile, never a new install)", async () => {
    const damaged = new MemFs();
    damaged.files.set(PATH, "{ this is not json");
    expect((await launch(damaged)).state.lanAccess.mode).toBe("lan");

    const empty = new MemFs();
    empty.files.set(PATH, "");
    expect((await launch(empty)).state.lanAccess.mode).toBe("lan");

    const locked = new MemFs();
    locked.files.set(PATH, JSON.stringify({ id: "x" }));
    locked.unreadable.add(PATH);
    expect((await launch(locked)).state.lanAccess.mode).toBe("lan");
  });

  it("uses the mode of the backup when that is what was recovered", async () => {
    const fs = new MemFs();
    fs.files.set(PATH, "garbage");
    fs.files.set(`${PATH}.bak`, JSON.stringify({ id: "me", lanAccess: { mode: "localhost" } }));
    const { state, loaded } = await launch(fs);
    expect(loaded.source).toBe("backup");
    expect(state.lanAccess.mode).toBe("localhost");

    const older = new MemFs();
    older.files.set(PATH, "garbage");
    older.files.set(`${PATH}.bak`, JSON.stringify({ id: "me" }));
    expect((await launch(older)).state.lanAccess.mode).toBe("lan");
  });

  it("does not start a deleted settings file's old backup as LAN: a missing main file is a new install", async () => {
    // Rust's `startup_lan_access` makes the same call (the front end never resurrects a backup for a missing file).
    const fs = new MemFs();
    fs.files.set(`${PATH}.bak`, JSON.stringify({ id: "me", lanAccess: { mode: "lan" } }));
    const { state, loaded } = await launch(fs);
    expect(loaded.source).toBe("none");
    expect(state.lanAccess.mode).toBe("localhost");
    expect(state.onboardingComplete).toBe(false);
  });

  it("decides once: a later switch to LAN access is the user's and survives every launch", async () => {
    const first = await launch(new MemFs());
    expect(first.state.lanAccess.mode).toBe("localhost");
    // Settings → LAN access → "Other devices on my network", saved with the file.
    first.state.lanAccess.mode = "lan";
    const fs = new MemFs();
    fs.files.set(PATH, JSON.stringify(first.state));
    const second = await launch(fs);
    expect(second.state.lanAccess.mode).toBe("lan");
    expect(second.changed).toBe(false);
  });
});

/**
 * The table of "settings file state -> mode" that the Rust web plugin runs too (auth.rs `startup_lan_access`, which decides the
 * address it binds before the webview exists). Same file in both suites: a change on one side that the other does not follow fails
 * a test. `settings` / `backup` are null (no file), {json}, {raw} (exact text) or {unreadable}.
 */
type FileSpec = null | { json: unknown } | { raw: string } | { unreadable: true };
interface StartupCase {
  name: string;
  settings: FileSpec;
  backup: FileSpec;
  mode: "lan" | "localhost";
  newInstall: boolean;
}
const TABLE = path.resolve(__dirname, "../../../../../src-tauri/src/services/web/startup_lan_cases.json");
const startupCases: StartupCase[] = (JSON.parse(fs.readFileSync(TABLE, "utf8")) as { cases: StartupCase[] }).cases;

function seed(mem: MemFs, file: string, spec: FileSpec) {
  if (spec === null) return;
  if ("json" in spec) mem.files.set(file, JSON.stringify(spec.json));
  else if ("raw" in spec) mem.files.set(file, spec.raw);
  else {
    mem.files.set(file, "{}");
    mem.unreadable.add(file);
  }
}

describe("first run: the table shared with the Rust web plugin (startup_lan_cases.json)", () => {
  it("has rows for new installs and for profiles that exist", () => {
    expect(startupCases.length).toBeGreaterThanOrEqual(10);
    expect(startupCases.some(c => c.newInstall && c.mode === "localhost")).toBe(true);
    expect(startupCases.some(c => !c.newInstall && c.mode === "lan")).toBe(true);
    expect(startupCases.some(c => !c.newInstall && c.mode === "localhost")).toBe(true);
    // A new install is always This PC only, whatever the files say.
    for (const c of startupCases) if (c.newInstall) expect(c.mode, c.name).toBe("localhost");
  });

  it.each(startupCases.map(c => [c.name, c] as const))("%s", async (_name, c) => {
    const mem = new MemFs();
    seed(mem, PATH, c.settings);
    seed(mem, `${PATH}.bak`, c.backup);
    const { state, loaded } = await launch(mem);
    expect(state.lanAccess.mode).toBe(c.mode);
    expect(isFreshProfile(loaded)).toBe(c.newInstall);
    expect(state.onboardingComplete).toBe(!c.newInstall); // the wizard shows for a new install only
  });
});

describe("applyFirstRunDecision", () => {
  const undecided = (over: Partial<FirstRunState> = {}): FirstRunState => ({
    firstRunDecided: false,
    onboardingComplete: false,
    lanAccess: { mode: "lan" },
    ...over,
  });

  it("decides once: a second launch changes nothing", () => {
    const state = undecided();
    expect(applyFirstRunDecision(state, true)).toBe(true);
    expect(state).toEqual({ firstRunDecided: true, onboardingComplete: false, lanAccess: { mode: "localhost" } });
    // Quit in the middle of the wizard, start again: the wizard shows again.
    expect(applyFirstRunDecision(state, false)).toBe(false);
    expect(state).toEqual({ firstRunDecided: true, onboardingComplete: false, lanAccess: { mode: "localhost" } });
  });

  it("keeps a finished wizard finished", () => {
    const state = undecided({ firstRunDecided: true, onboardingComplete: true });
    expect(applyFirstRunDecision(state, true)).toBe(false);
    expect(state.onboardingComplete).toBe(true);
    expect(state.lanAccess.mode).toBe("lan"); // an already decided profile is never switched to This PC only
  });

  it("records the decision for an existing profile that already finished onboarding, and leaves its LAN mode", () => {
    const state = undecided({ onboardingComplete: true });
    expect(applyFirstRunDecision(state, false)).toBe(true);
    expect(state).toEqual({ firstRunDecided: true, onboardingComplete: true, lanAccess: { mode: "lan" } });
  });

  it("switches only a new install to This PC only", () => {
    const fresh = undecided();
    applyFirstRunDecision(fresh, true);
    expect(fresh.lanAccess.mode).toBe(NEW_INSTALL_LAN_MODE);
    expect(NEW_INSTALL_LAN_MODE).toBe("localhost");

    const existing = undecided();
    applyFirstRunDecision(existing, false);
    expect(existing.lanAccess.mode).toBe("lan");
    expect(existing.onboardingComplete).toBe(true);
  });

  it("round-trips through the schema: the flag survives a save", async () => {
    const first = await launch(new MemFs());
    const fs = new MemFs();
    fs.files.set(PATH, JSON.stringify(first.state));
    const second = await launch(fs);
    // Written by a fresh install and not finished: still the wizard, and nothing to save.
    expect(second.state.onboardingComplete).toBe(false);
    expect(second.state.firstRunDecided).toBe(true);
    expect(second.changed).toBe(false);
  });
});
