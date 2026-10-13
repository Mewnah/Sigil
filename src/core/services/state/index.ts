import { IServiceInterface } from "@/types";
import { BaseDirectory } from "@tauri-apps/api/path";
import { copyFile, exists, mkdir, readFile, rename, writeFile } from "@tauri-apps/plugin-fs";
import debounce from "lodash/debounce";
import { proxy, subscribe } from "valtio";
import { BackendState, BackendSchema } from "../../schema";
import { AtomicJsonFile, JsonFileFs } from "../diagnostics/atomicJsonFile";
import { pushSystemLog } from "../systemLog";
import { applyFirstRunDecision, isFreshProfile } from "./firstRun";
import { applyUiScaleAutoDecision } from "./uiScaleAuto";

/** Persisted settings, relative to the app-data directory (`%APPDATA%\com.sigil.app`). */
const SETTINGS_PATH = "user/settings";
/** Coalesce bursts of changes (typing, slider drags) into one write. */
const SAVE_DEBOUNCE_MS = 1000;

const baseDir = BaseDirectory.AppData;

/** `@tauri-apps/plugin-fs` adapter (binary read/write: the capabilities allow those commands). */
const appDataFs: JsonFileFs = {
  exists: (path) => exists(path, { baseDir }),
  readText: async (path) => new TextDecoder().decode(await readFile(path, { baseDir })),
  writeText: (path, text) => writeFile(path, new TextEncoder().encode(text), { baseDir }),
  copy: (from, to) => copyFile(from, to, { fromPathBaseDir: baseDir, toPathBaseDir: baseDir }),
  rename: (from, to) => rename(from, to, { oldPathBaseDir: baseDir, newPathBaseDir: baseDir }),
  mkdirp: (dir) => mkdir(dir, { baseDir, recursive: true }),
};

class Service_State implements IServiceInterface {
  state!: BackendState;

  /**
   * Crash-safe settings file: temp file + rename, `.bak` of the previous good version, and a
   * damaged file is kept as `settings.corrupt-<time>.bak` instead of being overwritten.
   */
  readonly #file = new AtomicJsonFile(appDataFs, SETTINGS_PATH);

  async init() {
    const loaded = await this.#load();
    let fresh = loaded.fresh;
    let parsed: BackendState;
    try {
      parsed = BackendSchema.parse(loaded.data ?? {});
    } catch (error) {
      // Valid JSON the schema cannot use: keep a copy before defaults replace it.
      const kept = await this.#file.preserveCurrent("the settings did not match the schema");
      pushSystemLog(
        "Settings",
        `Saved settings could not be used (${error instanceof Error ? error.message : String(error)}); defaults are loaded${kept ? ` and the old file was kept as ${kept}` : ""}.`,
        "error",
      );
      parsed = BackendSchema.parse({});
      loaded.source = "none";
      fresh = false; // there was a profile: an existing user, not a new install
    }
    // No settings file means a new install (the setup wizard); any profile that existed never sees it.
    const decided = applyFirstRunDecision(parsed, fresh);
    // Interface size Auto: a profile that picked S, L or XL keeps it, every other one (and a new install) uses Auto.
    const scaleDecided = applyUiScaleAutoDecision(parsed);
    this.state = proxy(parsed);
    // Write immediately unless the main file is the good one we just read and nothing changed: first run, the
    // settings were recovered from the backup / reset to defaults (replaces the damaged file, which was preserved
    // first), or the first-run or Interface size decision was just recorded.
    if (loaded.source !== "primary" || decided || scaleDecided) void this.#save();
    subscribe(this.state, () => this.#debouncedSave());
  }

  async #load(): Promise<{ data: Record<string, unknown> | undefined; source: "primary" | "backup" | "none"; fresh: boolean }> {
    try {
      const result = await this.#file.load();
      for (const problem of result.problems) {
        pushSystemLog("Settings", problem, result.source === "backup" ? "warning" : "error");
      }
      return { data: result.data, source: result.source, fresh: isFreshProfile(result) };
    } catch (error) {
      console.error("settings load failed", error);
      pushSystemLog("Settings", `Settings could not be loaded: ${error instanceof Error ? error.message : String(error)}`, "error");
      // The file could not be read at all: that is an existing profile, never a new install.
      return { data: undefined, source: "none", fresh: false };
    }
  }

  async #save(): Promise<void> {
    try {
      await this.#file.save(this.state);
    } catch (error) {
      console.error("settings save failed", error);
      pushSystemLog("Settings", error instanceof Error ? error.message : String(error), "error");
    }
  }

  #debouncedSave = debounce(() => void this.#save(), SAVE_DEBOUNCE_MS);
}
export default Service_State;
