import { describe, expect, it } from "vitest";
import { AtomicJsonFile, JsonFileFs, fileStamp, parseJsonObject } from "../atomicJsonFile";

/** In-memory filesystem with failure injection and an operation log. */
class MemFs implements JsonFileFs {
  files = new Map<string, string>();
  dirs = new Set<string>();
  ops: string[] = [];
  failCopy = false;
  failWrite = false;
  failRead = new Set<string>();
  inFlight = 0;
  maxInFlight = 0;

  private async tick<T>(f: () => T): Promise<T> {
    this.inFlight++;
    this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
    await Promise.resolve();
    try {
      return f();
    } finally {
      this.inFlight--;
    }
  }

  exists(path: string) {
    return this.tick(() => this.files.has(path) || this.dirs.has(path));
  }
  readText(path: string) {
    return this.tick(() => {
      if (this.failRead.has(path)) throw new Error("locked by another program");
      const v = this.files.get(path);
      if (v === undefined) throw new Error(`ENOENT ${path}`);
      return v;
    });
  }
  writeText(path: string, text: string) {
    return this.tick(() => {
      this.ops.push(`write ${path}`);
      if (this.failWrite) throw new Error("disk full");
      this.files.set(path, text);
    });
  }
  copy(from: string, to: string) {
    return this.tick(() => {
      this.ops.push(`copy ${from} -> ${to}`);
      if (this.failCopy) throw new Error("access denied");
      const v = this.files.get(from);
      if (v === undefined) throw new Error(`ENOENT ${from}`);
      this.files.set(to, v);
    });
  }
  rename(from: string, to: string) {
    return this.tick(() => {
      this.ops.push(`rename ${from} -> ${to}`);
      const v = this.files.get(from);
      if (v === undefined) throw new Error(`ENOENT ${from}`);
      this.files.delete(from);
      this.files.set(to, v);
    });
  }
  mkdirp(dir: string) {
    return this.tick(() => {
      this.dirs.add(dir);
    });
  }
}

const PATH = "user/settings";
const fixedNow = () => new Date("2026-09-28T17:46:03.120Z");
const mk = (fs: MemFs) => new AtomicJsonFile(fs, PATH, { now: fixedNow });
const corruptName = `${PATH}.corrupt-${fileStamp(fixedNow())}.bak`;

describe("AtomicJsonFile", () => {
  it("fresh install: nothing to load, first save writes via temp file and leaves no .bak or .tmp", async () => {
    const fs = new MemFs();
    const f = mk(fs);
    const r = await f.load();
    expect(r).toEqual({ data: undefined, source: "none", problems: [] });
    await f.save({ a: 1 });
    expect(JSON.parse(fs.files.get(PATH)!)).toEqual({ a: 1 });
    expect(fs.files.has(`${PATH}.bak`)).toBe(false);
    expect(fs.files.has(`${PATH}.tmp`)).toBe(false);
    expect(fs.ops).toEqual([`write ${PATH}.tmp`, `rename ${PATH}.tmp -> ${PATH}`]);
  });

  it("normal save keeps the previous good file as .bak", async () => {
    const fs = new MemFs();
    fs.files.set(PATH, JSON.stringify({ v: 1 }));
    const f = mk(fs);
    expect((await f.load()).data).toEqual({ v: 1 });
    await f.save({ v: 2 });
    await f.save({ v: 3 });
    expect(JSON.parse(fs.files.get(PATH)!)).toEqual({ v: 3 });
    expect(JSON.parse(fs.files.get(`${PATH}.bak`)!)).toEqual({ v: 2 });
  });

  it("a truncated settings file is kept as .bak and the last backup is restored", async () => {
    const fs = new MemFs();
    const truncated = '{"clientTheme":"sigil-dark","stt":{"backend":"whi';
    fs.files.set(PATH, truncated);
    fs.files.set(`${PATH}.bak`, JSON.stringify({ clientTheme: "nord" }));
    const f = mk(fs);
    const r = await f.load();
    expect(r.source).toBe("backup");
    expect(r.data).toEqual({ clientTheme: "nord" });
    expect(r.preservedAs).toBe(corruptName);
    expect(corruptName.endsWith(".bak")).toBe(true);
    expect(fs.files.get(corruptName)).toBe(truncated);
    expect(r.problems.join(" ")).toMatch(/damaged/);

    // The next save replaces the damaged file but must not copy it over the good backup.
    await f.save({ clientTheme: "nord", x: 1 });
    expect(JSON.parse(fs.files.get(`${PATH}.bak`)!)).toEqual({ clientTheme: "nord" });
    expect(JSON.parse(fs.files.get(PATH)!)).toEqual({ clientTheme: "nord", x: 1 });
    expect(fs.files.get(corruptName)).toBe(truncated);
    // From now on the primary is trusted again and rotates into .bak.
    await f.save({ clientTheme: "nord", x: 2 });
    expect(JSON.parse(fs.files.get(`${PATH}.bak`)!)).toEqual({ clientTheme: "nord", x: 1 });
  });

  it.each([
    ["empty", ""],
    ["whitespace", "  \n"],
    ["array", "[1,2]"],
    ["null", "null"],
    ["string", '"x"'],
    ["garbage", "\u0000\u0000\u0000"],
  ])("treats a %s file as damaged", async (_name, content) => {
    const fs = new MemFs();
    fs.files.set(PATH, content);
    const r = await mk(fs).load();
    expect(r.data).toBeUndefined();
    expect(r.source).toBe("none");
    expect(fs.files.get(corruptName)).toBe(content);
  });

  it("falls back to defaults when the backup is damaged too", async () => {
    const fs = new MemFs();
    fs.files.set(PATH, "{");
    fs.files.set(`${PATH}.bak`, "{also broken");
    const r = await mk(fs).load();
    expect(r.data).toBeUndefined();
    expect(r.problems.join(" ")).toMatch(/backup is damaged/);
  });

  it("never overwrites a damaged file it could not back up, and recovers once it can", async () => {
    const fs = new MemFs();
    fs.files.set(PATH, '{"half":');
    fs.failCopy = true;
    const f = mk(fs);
    const r = await f.load();
    expect(r.problems.join(" ")).toMatch(/will not be overwritten/);
    await expect(f.save({ fresh: true })).rejects.toThrow(/not saved/);
    expect(fs.files.get(PATH)).toBe('{"half":');

    fs.failCopy = false;
    await f.save({ fresh: true });
    expect(fs.files.get(corruptName)).toBe('{"half":');
    expect(JSON.parse(fs.files.get(PATH)!)).toEqual({ fresh: true });
  });

  it("an unreadable (locked) file is preserved, not overwritten blindly", async () => {
    const fs = new MemFs();
    fs.files.set(PATH, JSON.stringify({ v: 1 }));
    fs.failRead.add(PATH);
    const r = await mk(fs).load();
    expect(r.problems[0]).toMatch(/could not be read/);
    expect(fs.files.get(corruptName)).toBe(JSON.stringify({ v: 1 }));
  });

  it("a failed temp write leaves the main file untouched and the value is retried later", async () => {
    const fs = new MemFs();
    fs.files.set(PATH, JSON.stringify({ v: 1 }));
    const f = mk(fs);
    await f.load();
    fs.failWrite = true;
    await expect(f.save({ v: 2 })).rejects.toThrow(/disk full/);
    expect(JSON.parse(fs.files.get(PATH)!)).toEqual({ v: 1 });
    fs.failWrite = false;
    await f.save({ v: 3 });
    expect(JSON.parse(fs.files.get(PATH)!)).toEqual({ v: 3 });
  });

  it("serializes and coalesces rapid saves", async () => {
    const fs = new MemFs();
    const f = mk(fs);
    await f.load();
    await Promise.all([f.save({ n: 1 }), f.save({ n: 2 }), f.save({ n: 3 })]);
    expect(JSON.parse(fs.files.get(PATH)!)).toEqual({ n: 3 });
    expect(fs.maxInFlight).toBe(1);
    expect(fs.ops.filter((o) => o.startsWith("write")).length).toBe(1);
  });

  it("preserveCurrent keeps a schema-rejected file before it is replaced", async () => {
    const fs = new MemFs();
    fs.files.set(PATH, JSON.stringify({ weird: true }));
    const f = mk(fs);
    await f.load();
    expect(await f.preserveCurrent("schema")).toBe(corruptName);
    await f.save({ ok: 1 });
    expect(fs.files.has(`${PATH}.bak`)).toBe(false); // the rejected file is not rotated into .bak
    expect(JSON.parse(fs.files.get(corruptName)!)).toEqual({ weird: true });
  });

  it("parseJsonObject honours the validator", () => {
    expect(parseJsonObject('{"a":1}', (o) => "a" in o)).toEqual({ a: 1 });
    expect(parseJsonObject('{"b":1}', (o) => "a" in o)).toBeUndefined();
  });
});
