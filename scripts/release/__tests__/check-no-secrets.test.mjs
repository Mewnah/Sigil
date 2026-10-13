import { execFileSync, spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import {
  MIN_VALUE_LENGTH,
  collectSecrets,
  listFiles,
  main,
  needlesFor,
  parseArgs,
  parseDotenv,
  scanFile,
  scanPaths,
} from "../check-no-secrets.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REAL_ROOT = path.join(HERE, "..", "..", "..");

// Fake values only: the test never reads the real .env, and no real secret appears in the repository.
const FAKE_KICK = "fake-kick-secret-AAAA1111bbbb2222";
const FAKE_FONTS = "fake-fonts-key-CCCC3333dddd4444";
const CLASSIFICATION = {
  public: ["SIGIL_KICK_CLIENT_ID"],
  secret: ["SIGIL_KICK_CLIENT_SECRET", "VITE_GOOGLE_FONTS_API_KEY"],
};

const tmpRoots = [];
function tmp() {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), "sigil-nosecrets-"));
  tmpRoots.push(d);
  return d;
}
afterEach(() => {
  while (tmpRoots.length) fs.rmSync(tmpRoots.pop(), { recursive: true, force: true });
});

/** A repository root with the classification file, a `dist/` and an optional `.env`. */
function fakeRepo({ env, dist = {} } = {}) {
  const root = tmp();
  fs.mkdirSync(path.join(root, "scripts", "release"), { recursive: true });
  fs.writeFileSync(path.join(root, "scripts", "release", "env-classification.json"), JSON.stringify(CLASSIFICATION));
  for (const [rel, content] of Object.entries(dist)) {
    const file = path.join(root, "dist", rel);
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, content);
  }
  if (!Object.keys(dist).length) fs.mkdirSync(path.join(root, "dist"), { recursive: true });
  if (env !== undefined) fs.writeFileSync(path.join(root, ".env"), env);
  return root;
}

/** Run the CLI in-process; returns the exit code and everything it printed. */
function run(argv, { env = {}, cwd } = {}) {
  let stdout = "";
  let stderr = "";
  const code = main(argv, { env, cwd, stdout: (s) => (stdout += s), stderr: (s) => (stderr += s) });
  return { code, stdout, stderr, all: stdout + stderr };
}

describe("parseDotenv", () => {
  it("reads the shapes Vite's dotenv accepts", () => {
    const pairs = parseDotenv(
      "\uFEFF# c\r\nA=plain\r\nexport B = spaced \nC=\"q # not comment\"\nD='single'\nE=val # tail\nF=\n1BAD=x\nnoequals\n",
    );
    const map = Object.fromEntries(pairs);
    expect(map).toMatchObject({ A: "plain", B: "spaced", C: "q # not comment", D: "single", E: "val", F: "" });
    expect(map["1BAD"]).toBeUndefined();
  });
});

describe("collectSecrets", () => {
  it("takes secret-named variables from files and the environment, by name pattern or classification", () => {
    const root = tmp();
    const file = path.join(root, ".env");
    fs.writeFileSync(
      file,
      [
        `SIGIL_KICK_CLIENT_SECRET=${FAKE_KICK}`,
        "SIGIL_KICK_CLIENT_ID=public-client-id-value-123", // public: not collected
        "SOME_API_KEY=fake-generic-api-key-12345", // matched by the name pattern alone
        "MY_PASSWORD=short", // too short to search
      ].join("\n"),
    );
    const { values, tooShort } = collectSecrets({
      envFiles: [file, path.join(root, "missing.env")],
      env: { VITE_GOOGLE_FONTS_API_KEY: FAKE_FONTS, PATH: "C:\\Windows", GH_TOKEN: "x".repeat(MIN_VALUE_LENGTH) },
      classification: CLASSIFICATION,
    });
    expect([...values.keys()].sort()).toEqual(["GH_TOKEN", "SIGIL_KICK_CLIENT_SECRET", "SOME_API_KEY", "VITE_GOOGLE_FONTS_API_KEY"]);
    expect(values.get("SIGIL_KICK_CLIENT_SECRET")).toEqual(new Set([FAKE_KICK]));
    expect(values.has("SIGIL_KICK_CLIENT_ID")).toBe(false);
    expect(tooShort).toEqual(["MY_PASSWORD"]);
  });

  it("keeps both values when a name differs between the file and the environment", () => {
    const root = tmp();
    const file = path.join(root, ".env");
    fs.writeFileSync(file, `SIGIL_KICK_CLIENT_SECRET=${FAKE_KICK}\n`);
    const { values } = collectSecrets({
      envFiles: [file],
      env: { SIGIL_KICK_CLIENT_SECRET: "fake-other-value-98765" },
      classification: CLASSIFICATION,
    });
    expect(values.get("SIGIL_KICK_CLIENT_SECRET")).toEqual(new Set([FAKE_KICK, "fake-other-value-98765"]));
  });
});

describe("needlesFor and scanFile", () => {
  it("searches UTF-8, UTF-16LE, JSON-escaped and URL-encoded forms, without duplicates", () => {
    expect(needlesFor("plainvalue123").map((n) => n.form)).toEqual(["utf8", "utf16le"]);
    const special = needlesFor('a"b/c d\\e-12345');
    expect(special.map((n) => n.form)).toEqual(["utf8", "utf16le", "json", "url"]);
  });

  it("finds a value in each form and none that is not there", () => {
    const dir = tmp();
    const needles = needlesFor(FAKE_KICK).map((n) => ({ ...n, name: "X" }));
    const utf8 = path.join(dir, "a.js");
    const utf16 = path.join(dir, "b.exe");
    const clean = path.join(dir, "c.js");
    fs.writeFileSync(utf8, `var k="${FAKE_KICK}";`);
    fs.writeFileSync(utf16, Buffer.concat([Buffer.from([0, 1, 2, 3]), Buffer.from(FAKE_KICK, "utf16le"), Buffer.from([9, 9])]));
    fs.writeFileSync(clean, `var k="${FAKE_KICK.slice(0, -1)}";`);
    expect(scanFile(utf8, needles).map((n) => n.form)).toEqual(["utf8"]);
    expect(scanFile(utf16, needles).map((n) => n.form)).toEqual(["utf16le"]);
    expect(scanFile(clean, needles)).toEqual([]);
    expect(scanFile(path.join(dir, "empty"), []).length).toBe(0);
  });

  it("finds a value that straddles a chunk boundary, at every offset", () => {
    const dir = tmp();
    const needles = needlesFor(FAKE_KICK).map((n) => ({ ...n, name: "X" }));
    const file = path.join(dir, "big.bin");
    for (let pad = 0; pad < 40; pad++) {
      fs.writeFileSync(file, Buffer.concat([Buffer.alloc(pad, 0x41), Buffer.from(FAKE_KICK), Buffer.alloc(50, 0x42)]));
      expect(scanFile(file, needles, 16).length, `offset ${pad}`).toBe(1);
    }
    // And an empty / short file is fine with a tiny chunk.
    fs.writeFileSync(file, "");
    expect(scanFile(file, needles, 16)).toEqual([]);
  });
});

describe("listFiles", () => {
  it("lists files recursively and a single file as itself", () => {
    const root = tmp();
    fs.mkdirSync(path.join(root, "a", "b"), { recursive: true });
    fs.writeFileSync(path.join(root, "a", "b", "x.txt"), "x");
    fs.writeFileSync(path.join(root, "y.txt"), "y");
    expect(listFiles(root).map((f) => path.relative(root, f)).sort()).toEqual([path.join("a", "b", "x.txt"), "y.txt"]);
    expect(listFiles(path.join(root, "y.txt"))).toEqual([path.join(root, "y.txt")]);
  });
});

describe("scanPaths", () => {
  it("reports the variable, the form and the file, nothing else", () => {
    const root = tmp();
    fs.mkdirSync(path.join(root, "nested"));
    fs.writeFileSync(path.join(root, "nested", "app.js"), `x("${FAKE_FONTS}")`);
    fs.writeFileSync(path.join(root, "ok.js"), "nothing here");
    const { files, leaks } = scanPaths([root], new Map([["VITE_GOOGLE_FONTS_API_KEY", new Set([FAKE_FONTS])], ["SIGIL_KICK_CLIENT_SECRET", new Set([FAKE_KICK])]]));
    expect(files).toBe(2);
    expect(leaks).toEqual([{ name: "VITE_GOOGLE_FONTS_API_KEY", form: "utf8", file: path.join(root, "nested", "app.js") }]);
  });
});

describe("main", () => {
  it("passes on a clean dist and says which variables it checked (names only)", () => {
    const root = fakeRepo({ env: `SIGIL_KICK_CLIENT_SECRET=${FAKE_KICK}\n`, dist: { "assets/index.js": "console.log('hello')" } });
    const r = run(["--root", root, "--no-process-env"]);
    expect(r.code).toBe(0);
    expect(r.stdout).toContain("SIGIL_KICK_CLIENT_SECRET");
    expect(r.stdout).toContain("OK: no secret value found");
    expect(r.all).not.toContain(FAKE_KICK);
  });

  it("fails on a leak in dist, naming only the variable and the file", () => {
    const root = fakeRepo({
      env: `SIGIL_KICK_CLIENT_SECRET=${FAKE_KICK}\nVITE_GOOGLE_FONTS_API_KEY=${FAKE_FONTS}\n`,
      dist: { "assets/index-abc.js": `fetch("x?key=${FAKE_FONTS}")`, "index.html": "<html></html>" },
    });
    const r = run(["--root", root, "--no-process-env"]);
    expect(r.code).toBe(1);
    expect(r.stdout).toContain(`LEAK: VITE_GOOGLE_FONTS_API_KEY (utf8) in ${path.join("dist", "assets", "index-abc.js")}`);
    expect(r.stderr).toContain("VITE_GOOGLE_FONTS_API_KEY");
    expect(r.all).not.toContain(FAKE_FONTS);
    expect(r.all).not.toContain(FAKE_KICK);
    expect(r.all).not.toContain("SIGIL_KICK_CLIENT_SECRET (utf8)");
  });

  it("scans extra paths (an exe, an extracted installer folder) and finds UTF-16 strings", () => {
    const root = fakeRepo({ env: `SIGIL_KICK_CLIENT_SECRET=${FAKE_KICK}\n`, dist: { "index.html": "ok" } });
    const extracted = tmp();
    fs.mkdirSync(path.join(extracted, "app"));
    fs.writeFileSync(path.join(extracted, "app", "Sigil.exe"), Buffer.concat([Buffer.from("MZ"), Buffer.from(FAKE_KICK, "utf16le")]));
    const clean = path.join(tmp(), "Sigil.exe");
    fs.writeFileSync(clean, Buffer.from("MZ plain exe with an obfuscated-only secret"));

    const ok = run(["--root", root, "--no-process-env", clean]);
    expect(ok.code).toBe(0);

    const bad = run(["--root", root, "--no-process-env", extracted]);
    expect(bad.code).toBe(1);
    expect(bad.stdout).toContain("LEAK: SIGIL_KICK_CLIENT_SECRET (utf16le)");
    expect(bad.all).not.toContain(FAKE_KICK);
  });

  it("reads secrets from the process environment unless told not to", () => {
    const root = fakeRepo({ dist: { "a.js": `var s="${FAKE_KICK}"` } });
    const withEnv = run(["--root", root], { env: { SIGIL_KICK_CLIENT_SECRET: FAKE_KICK } });
    expect(withEnv.code).toBe(1);
    expect(withEnv.all).not.toContain(FAKE_KICK);
    const without = run(["--root", root, "--no-process-env"], { env: { SIGIL_KICK_CLIENT_SECRET: FAKE_KICK } });
    expect(without.code).toBe(0);
    expect(without.stdout).toContain("nothing to compare");
  });

  it("ignores public names and values too short to search", () => {
    const root = fakeRepo({
      env: "SIGIL_KICK_CLIENT_ID=public-client-id-0123456789\nSOME_PASSWORD=short\n",
      dist: { "a.js": "var id='public-client-id-0123456789'; var p='short'" },
    });
    const r = run(["--root", root, "--no-process-env"]);
    expect(r.code).toBe(0);
    expect(r.stdout).toContain("SOME_PASSWORD");
    expect(r.stdout).toContain(`under ${MIN_VALUE_LENGTH} characters`);
  });

  it("--require-values turns 'nothing to compare' into a failure", () => {
    const root = fakeRepo({ dist: { "a.js": "x" } });
    expect(run(["--root", root, "--no-process-env"]).code).toBe(0);
    const r = run(["--root", root, "--no-process-env", "--require-values"]);
    expect(r.code).toBe(1);
    expect(r.stderr).toContain("--require-values");
  });

  it("exits 2 for a missing dist, a missing path, bad options and nothing to scan", () => {
    const root = fakeRepo({ env: `SIGIL_KICK_CLIENT_SECRET=${FAKE_KICK}\n` });
    fs.rmSync(path.join(root, "dist"), { recursive: true });
    const noDist = run(["--root", root, "--no-process-env"]);
    expect(noDist.code).toBe(2);
    expect(noDist.stderr).toContain("pnpm build");
    expect(run(["--root", root, "--no-dist", path.join(root, "nope.exe")]).code).toBe(2);
    expect(run(["--bogus"]).code).toBe(2);
    expect(run(["--root"]).code).toBe(2);
    expect(run(["--root", root, "--no-dist"]).code).toBe(2);
  });

  it("an extra --env-file adds values to look for", () => {
    const root = fakeRepo({ dist: { "a.js": `k="${FAKE_FONTS}"` } });
    const extra = path.join(tmp(), "extra.env");
    fs.writeFileSync(extra, `VITE_GOOGLE_FONTS_API_KEY=${FAKE_FONTS}\n`);
    const r = run(["--root", root, "--no-process-env", "--env-file", extra]);
    expect(r.code).toBe(1);
    expect(r.all).not.toContain(FAKE_FONTS);
  });

  it("parseArgs collects paths and flags", () => {
    expect(parseArgs(["a", "--no-dist", "b", "--require-values", "--env-file", "x.env"])).toMatchObject({
      paths: ["a", "b"],
      dist: false,
      requireValues: true,
      envFiles: ["x.env"],
    });
  });
});

describe("the CLI", () => {
  it("runs as a script and sets the exit code (clean repo copy, fake secret in the environment)", () => {
    const root = fakeRepo({ dist: { "a.js": "clean" } });
    const script = path.join(REAL_ROOT, "scripts", "release", "check-no-secrets.mjs");
    const env = { ...process.env, SIGIL_KICK_CLIENT_SECRET: FAKE_KICK };
    const out = execFileSync(process.execPath, [script, "--root", root], { env, encoding: "utf8" });
    expect(out).toContain("OK: no secret value found");

    fs.writeFileSync(path.join(root, "dist", "leak.js"), FAKE_KICK);
    const bad = spawnSync(process.execPath, [script, "--root", root], { env, encoding: "utf8" });
    expect(bad.status).toBe(1);
    expect(bad.stdout + bad.stderr).not.toContain(FAKE_KICK);
  });
});
