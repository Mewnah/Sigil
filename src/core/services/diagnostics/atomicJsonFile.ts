/**
 * Crash-safe JSON settings file: temp file + rename, a `.bak` of the last good version, and damaged
 * files preserved (never overwritten).
 *
 * Save: write `<path>.tmp`, copy the current (known-good) `<path>` to `<path>.bak`, rename the temp
 * file over `<path>`. A crash at any point leaves either the old or the new complete file.
 *
 * Load: a `<path>` that cannot be read or parsed as a JSON object (truncated, empty, garbage) is
 * first copied to `<path>.corrupt-<timestamp>.bak`, then `<path>.bak` is tried, then defaults. Until
 * the damaged file has been copied away, saves are refused, so the user's data is never lost to a
 * default-settings write.
 *
 * The filesystem is injected ({@link JsonFileFs}) so the logic is unit-tested without Tauri; the
 * settings service passes a `@tauri-apps/plugin-fs` adapter. Saves are serialized and coalesced.
 */

export interface JsonFileFs {
  exists(path: string): Promise<boolean>;
  readText(path: string): Promise<string>;
  writeText(path: string, text: string): Promise<void>;
  copy(from: string, to: string): Promise<void>;
  /** Must replace `to` if it exists. */
  rename(from: string, to: string): Promise<void>;
  /** Create a directory and its parents (no error if it exists). */
  mkdirp(dir: string): Promise<void>;
}

export type JsonObject = Record<string, unknown>;

export interface LoadResult {
  /** The settings to use, or undefined for defaults. */
  data: JsonObject | undefined;
  /** Where `data` came from. */
  source: "primary" | "backup" | "none";
  /** Set when the main file was damaged: where its bytes were preserved. */
  preservedAs?: string;
  /** Problems worth telling the user about (English, for the system log). */
  problems: string[];
}

export interface AtomicJsonFileOptions {
  /** Clock for the preserved-file timestamp (tests). */
  now?: () => Date;
  /** Extra acceptance check for a parsed object. */
  validate?: (value: JsonObject) => boolean;
}

function describe(e: unknown): string {
  if (e instanceof Error) return e.message;
  return typeof e === "string" ? e : String(e);
}

/** Filename-safe timestamp: `2026-09-28T17-46-03-120Z`. */
export function fileStamp(d: Date): string {
  return d.toISOString().replace(/[:.]/g, "-");
}

export function parseJsonObject(text: string, validate?: (v: JsonObject) => boolean): JsonObject | undefined {
  const trimmed = text.trim();
  if (!trimmed) return undefined;
  try {
    const value: unknown = JSON.parse(trimmed);
    if (typeof value !== "object" || value === null || Array.isArray(value)) return undefined;
    const obj = value as JsonObject;
    return !validate || validate(obj) ? obj : undefined;
  } catch {
    return undefined;
  }
}

export class AtomicJsonFile {
  /** The on-disk main file is known-good (loaded fine, or written by us). */
  private primaryTrusted = false;
  /** A damaged main file still has to be copied away before it may be replaced. */
  private unpreserved: { reason: string } | null = null;
  private preservedAs: string | undefined;
  private pending: string | null = null;
  private chain: Promise<void> = Promise.resolve();

  constructor(
    private readonly fs: JsonFileFs,
    readonly path: string,
    private readonly options: AtomicJsonFileOptions = {},
  ) {}

  get tmpPath(): string {
    return `${this.path}.tmp`;
  }

  get backupPath(): string {
    return `${this.path}.bak`;
  }

  private get dir(): string {
    const i = Math.max(this.path.lastIndexOf("/"), this.path.lastIndexOf("\\"));
    return i > 0 ? this.path.slice(0, i) : "";
  }

  private now(): Date {
    return this.options.now ? this.options.now() : new Date();
  }

  /** Copy the (damaged) main file to a unique `.corrupt-<stamp>.bak` name. */
  private async preservePrimary(): Promise<string> {
    const target = `${this.path}.corrupt-${fileStamp(this.now())}.bak`;
    await this.fs.copy(this.path, target);
    return target;
  }

  /**
   * Keep a copy of the current main file before it is replaced, e.g. when the loaded data failed
   * schema validation. Saves are refused until the copy exists. Returns the copy's path.
   */
  async preserveCurrent(reason: string): Promise<string | undefined> {
    if (!(await this.fs.exists(this.path))) return undefined;
    this.primaryTrusted = false;
    try {
      this.preservedAs = await this.preservePrimary();
      this.unpreserved = null;
      return this.preservedAs;
    } catch {
      this.unpreserved = { reason };
      return undefined;
    }
  }

  async load(): Promise<LoadResult> {
    const problems: string[] = [];
    if (!(await this.fs.exists(this.path))) {
      // Fresh install (or a deliberate reset): never resurrect an old backup here.
      this.primaryTrusted = false;
      return { data: undefined, source: "none", problems };
    }
    let text: string | undefined;
    try {
      text = await this.fs.readText(this.path);
    } catch (e) {
      problems.push(`The settings file could not be read (${describe(e)}).`);
    }
    const parsed = text === undefined ? undefined : parseJsonObject(text, this.options.validate);
    if (parsed) {
      this.primaryTrusted = true;
      return { data: parsed, source: "primary", problems };
    }

    this.primaryTrusted = false;
    const damage = text === undefined ? "unreadable" : text.trim() ? "damaged (not valid JSON settings)" : "empty";
    try {
      this.preservedAs = await this.preservePrimary();
      problems.push(`The settings file was ${damage}; it was kept as ${this.preservedAs}.`);
    } catch (e) {
      this.unpreserved = { reason: `the settings file is ${damage}` };
      problems.push(
        `The settings file is ${damage} and could not be backed up (${describe(e)}); it will not be overwritten.`,
      );
    }

    if (await this.fs.exists(this.backupPath)) {
      try {
        const backup = parseJsonObject(await this.fs.readText(this.backupPath), this.options.validate);
        if (backup) {
          problems.push("Settings were restored from the last backup.");
          return { data: backup, source: "backup", preservedAs: this.preservedAs, problems };
        }
        problems.push("The settings backup is damaged too; defaults are used.");
      } catch (e) {
        problems.push(`The settings backup could not be read (${describe(e)}); defaults are used.`);
      }
    } else {
      problems.push("No settings backup exists; defaults are used.");
    }
    return { data: undefined, source: "none", preservedAs: this.preservedAs, problems };
  }

  /**
   * Save `value` as JSON. Calls are serialized; while one write runs, later values replace each
   * other and only the newest is written next. Rejects when this value could not be written.
   */
  save(value: unknown): Promise<void> {
    this.pending = JSON.stringify(value);
    const run = this.chain.then(() => this.flush());
    // Keep the chain alive after a failure; the caller still sees this call's rejection.
    this.chain = run.catch(() => undefined);
    return run;
  }

  private async flush(): Promise<void> {
    const text = this.pending;
    if (text === null) return; // Coalesced into an earlier flush.
    this.pending = null;
    try {
      const unpreserved = this.unpreserved;
      if (unpreserved) {
        // Retry the copy of the damaged file; never replace it without one.
        try {
          if (await this.fs.exists(this.path)) this.preservedAs = await this.preservePrimary();
          this.unpreserved = null;
        } catch (e) {
          throw new Error(`Settings were not saved: ${unpreserved.reason} and could not be backed up (${describe(e)}).`);
        }
      }
      if (this.dir) await this.fs.mkdirp(this.dir);
      await this.fs.writeText(this.tmpPath, text);
      if (this.primaryTrusted && (await this.fs.exists(this.path))) {
        try {
          await this.fs.copy(this.path, this.backupPath);
        } catch {
          // A missing .bak only weakens recovery; the atomic replace below still protects the file.
        }
      }
      await this.fs.rename(this.tmpPath, this.path);
      this.primaryTrusted = true;
    } catch (e) {
      // Keep the newest value for the next attempt unless a newer one arrived meanwhile.
      if (this.pending === null) this.pending = text;
      throw e instanceof Error ? e : new Error(describe(e));
    }
  }
}
