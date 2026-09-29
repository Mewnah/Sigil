import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  CUDA_DLL,
  DIRECTML_DLL,
  SHARED_DLL,
  candidateSourceDirs,
  check,
  findSourceDir,
  parseArgs,
  prepare,
  requiredDlls,
  sha256File,
  stage,
} from "./stage-native-dlls.mjs";

const tmpRoots = [];
function tmp() {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), "sigil-stage-"));
  tmpRoots.push(d);
  return d;
}
afterEach(() => {
  while (tmpRoots.length) fs.rmSync(tmpRoots.pop(), { recursive: true, force: true });
});

function fakeDll(dir, name, body = "payload") {
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, name), `MZ${body}-${name}`);
}

describe("stage-native-dlls", () => {
  it("parses modes and options", () => {
    expect(parseArgs([]).mode).toBe("stage");
    expect(parseArgs(["--prepare"]).mode).toBe("prepare");
    expect(parseArgs(["--profile", "debug", "--from", "x"])).toMatchObject({ profile: "debug", from: "x" });
    expect(() => parseArgs(["--profile", "fast"])).toThrow(/release or debug/);
    expect(() => parseArgs(["--bogus"])).toThrow(/unknown argument/);
  });

  it("looks in the explicit, triple and profile directories in that order", () => {
    const root = "/repo";
    const dirs = candidateSourceDirs({
      root,
      env: { TAURI_ENV_TARGET_TRIPLE: "x86_64-pc-windows-msvc", TAURI_ENV_DEBUG: "false" },
      targetDir: null,
      profile: null,
      from: "/custom",
    });
    expect(dirs).toEqual([
      path.resolve("/custom"),
      path.resolve(root, "src-tauri/target", "x86_64-pc-windows-msvc", "release"),
      path.resolve(root, "src-tauri/target", "release"),
    ]);
    const debug = candidateSourceDirs({ root, env: { TAURI_ENV_DEBUG: "true", CARGO_TARGET_DIR: "/t" }, targetDir: null, profile: null, from: null });
    expect(debug).toEqual([path.resolve("/t", "debug")]);
  });

  it("stages real copies, verifies hashes and detects tampering or extra files", () => {
    const root = tmp();
    const release = path.join(root, "target", "release");
    for (const n of requiredDlls()) fakeDll(release, n);
    const staging = path.join(root, "native-bundle");
    fakeDll(staging, "stale.dll"); // left over from an older build: must be cleared
    const src = findSourceDir([path.join(root, "nope"), release], requiredDlls());
    expect(src).toBe(release);
    const manifest = stage({ sourceDir: src, stagingDir: staging, names: requiredDlls() });
    expect(manifest.files.map((f) => f.name)).toEqual([SHARED_DLL, CUDA_DLL, DIRECTML_DLL]);
    expect(fs.readdirSync(staging).sort()).toEqual([...requiredDlls()].sort());
    expect(check(staging, manifest)).toEqual([]);
    expect(manifest.files[1].sha256).toBe(sha256File(path.join(release, CUDA_DLL)));

    fs.writeFileSync(path.join(staging, DIRECTML_DLL), "MZtampered");
    fakeDll(staging, "extra.dll");
    const problems = check(staging, manifest);
    expect(problems.some((p) => p.includes(DIRECTML_DLL))).toBe(true);
    expect(problems.some((p) => p.includes("extra.dll"))).toBe(true);
  });

  it("refuses files that are not DLLs", () => {
    const root = tmp();
    fs.mkdirSync(root, { recursive: true });
    fs.writeFileSync(path.join(root, DIRECTML_DLL), "<html>not a dll</html>");
    expect(() => stage({ sourceDir: root, stagingDir: path.join(root, "s"), names: [DIRECTML_DLL] })).toThrow(/MZ/);
  });

  it("prepare leaves an empty directory (valid for tauri-build at compile time)", () => {
    const d = path.join(tmp(), "native-bundle");
    fakeDll(d, CUDA_DLL);
    prepare(d);
    expect(fs.existsSync(d)).toBe(true);
    expect(fs.readdirSync(d)).toEqual([]);
  });
});
