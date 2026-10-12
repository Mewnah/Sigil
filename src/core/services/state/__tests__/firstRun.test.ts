import { describe, expect, it } from "vitest";
import { BackendSchema } from "../../../schema";
import { AtomicJsonFile, type JsonFileFs } from "../../diagnostics/atomicJsonFile";
import { applyFirstRunDecision, isFreshProfile } from "../firstRun";

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

describe("applyFirstRunDecision", () => {
  it("decides once: a second launch changes nothing", () => {
    const state = { firstRunDecided: false, onboardingComplete: false };
    expect(applyFirstRunDecision(state, true)).toBe(true);
    expect(state).toEqual({ firstRunDecided: true, onboardingComplete: false });
    // Quit in the middle of the wizard, start again: the wizard shows again.
    expect(applyFirstRunDecision(state, false)).toBe(false);
    expect(state).toEqual({ firstRunDecided: true, onboardingComplete: false });
  });

  it("keeps a finished wizard finished", () => {
    const state = { firstRunDecided: true, onboardingComplete: true };
    expect(applyFirstRunDecision(state, true)).toBe(false);
    expect(state.onboardingComplete).toBe(true);
  });

  it("records the decision for an existing profile that already finished onboarding", () => {
    const state = { firstRunDecided: false, onboardingComplete: true };
    expect(applyFirstRunDecision(state, false)).toBe(true);
    expect(state).toEqual({ firstRunDecided: true, onboardingComplete: true });
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
