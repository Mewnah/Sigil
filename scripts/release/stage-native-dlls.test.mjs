import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import {
  CUDA_DLL,
  DIRECTML_DLL,
  SHARED_DLL,
  VC_RUNTIME_DLLS,
  candidateSourceDirs,
  check,
  compareVersions,
  findSourceDir,
  findVcRuntime,
  isSystemDir,
  parseArgs,
  prepare,
  redistCrtDirs,
  requireVcRuntime,
  requiredDlls,
  sha256File,
  stage,
  stageFiles,
  vcRuntimeDlls,
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

/** A DLL with just enough PE structure for the staging checks: MZ, e_lfanew = 0x40, "PE\0\0", the machine type. */
function fakePe(dir, name, machine = 0x8664, body = "payload") {
  fs.mkdirSync(dir, { recursive: true });
  const head = Buffer.alloc(0x40 + 6);
  head.write("MZ", 0, "latin1");
  head.writeUInt32LE(0x40, 0x3c);
  head.write("PE\0\0", 0x40, "latin1");
  head.writeUInt16LE(machine, 0x44);
  fs.writeFileSync(path.join(dir, name), Buffer.concat([head, Buffer.from(`-${body}-${name}`)]));
}

/** A fake `<install>\VC\Redist\MSVC\<version>\x64\<toolset>\` with the four runtime DLLs (and an unrelated one). */
function fakeRedist(installPath, version, toolset = "Microsoft.VC143.CRT", names = VC_RUNTIME_DLLS) {
  const dir = path.join(installPath, "VC", "Redist", "MSVC", version, "x64", toolset);
  for (const n of names) fakePe(dir, n, 0x8664, `${version}-${toolset}`);
  fakePe(dir, "concrt140.dll");
  return dir;
}

describe("stage-native-dlls", () => {
  it("parses modes and options", () => {
    expect(parseArgs([]).mode).toBe("stage");
    expect(parseArgs(["--prepare"]).mode).toBe("prepare");
    expect(parseArgs(["--check-vc"]).mode).toBe("check-vc");
    expect(parseArgs(["--profile", "debug", "--from", "x", "--vc-redist", "y"])).toMatchObject({ profile: "debug", from: "x", vcRedist: "y" });
    expect(() => parseArgs(["--profile", "fast"])).toThrow(/release or debug/);
    expect(() => parseArgs(["--vc-redist"])).toThrow(/needs a value/);
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
    const redist = fakeRedist(path.join(root, "vs"), "14.50.35710");
    const staging = path.join(root, "native-bundle");
    fakeDll(staging, "stale.dll"); // left over from an older build: must be cleared
    const src = findSourceDir([path.join(root, "nope"), release], requiredDlls());
    expect(src).toBe(release);

    // The ONNX Runtime providers alone are what `stage` has always done; it is not the whole set any more.
    const providersOnly = stage({ sourceDir: src, stagingDir: staging, names: requiredDlls() });
    expect(providersOnly.files.map((f) => f.name)).toEqual([SHARED_DLL, CUDA_DLL, DIRECTML_DLL]);
    expect(check(staging, providersOnly).join("; ")).toMatch(/vcruntime140\.dll is not in the manifest/);

    const manifest = stageFiles({
      stagingDir: staging,
      files: [
        ...requiredDlls().map((name) => ({ name, source: path.join(release, name), group: "onnxruntime" })),
        ...VC_RUNTIME_DLLS.map((name) => ({ name, source: path.join(redist, name), group: "vc-runtime", x64: true })),
      ],
    });
    expect(manifest.files.map((f) => f.name)).toEqual([...requiredDlls(), ...VC_RUNTIME_DLLS]);
    expect(manifest.files.filter((f) => f.group === "vc-runtime")).toHaveLength(4);
    expect(fs.readdirSync(staging).sort()).toEqual([...requiredDlls(), ...VC_RUNTIME_DLLS].sort());
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

describe("the Visual C++ runtime (app-local)", () => {
  it("is the four load-time imports of expected-imports.txt", () => {
    expect(vcRuntimeDlls()).toEqual(["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll"]);
    const here = path.dirname(fileURLToPath(import.meta.url));
    const expected = fs.readFileSync(path.join(here, "expected-imports.txt"), "utf8").split(/\r?\n/).map((l) => l.trim().toLowerCase());
    for (const n of VC_RUNTIME_DLLS) expect(expected, n).toContain(n);
  });

  it("compares versions numerically", () => {
    expect(compareVersions("14.50.35710", "14.44.35112")).toBeGreaterThan(0);
    expect(compareVersions("14.9.1", "14.10.0")).toBeLessThan(0);
    expect(compareVersions("14.50", "14.50.0")).toBe(0);
  });

  it("takes the newest redistributable of every Visual Studio installation, skipping debug and x86 folders", () => {
    const root = tmp();
    const vs2022 = path.join(root, "2022", "BuildTools");
    const vs18 = path.join(root, "18", "Community");
    const old = fakeRedist(vs2022, "14.44.35112", "Microsoft.VC143.CRT");
    const mid = fakeRedist(vs2022, "14.43.34808", "Microsoft.VC143.CRT");
    const newest = fakeRedist(vs18, "14.50.35710", "Microsoft.VC145.CRT");
    // Not candidates: the vc_redist installers folder, an x86 CRT, a folder missing msvcp140_1.dll, debug_nonredist.
    fs.mkdirSync(path.join(vs18, "VC", "Redist", "MSVC", "v145"), { recursive: true });
    fakePe(path.join(vs18, "VC", "Redist", "MSVC", "14.51.0", "x86", "Microsoft.VC145.CRT"), "vcruntime140.dll", 0x14c);
    fakeRedist(vs18, "14.49.1", "Microsoft.VC145.CRT", ["vcruntime140.dll", "msvcp140.dll"]);
    fakePe(path.join(vs18, "VC", "Redist", "MSVC", "14.50.35710", "debug_nonredist", "x64", "Microsoft.VC145.DebugCRT"), "vcruntime140d.dll");

    const found = findVcRuntime({ env: { SystemRoot: path.join(root, "Windows") }, installPaths: [vs2022, vs18] });
    expect(found.dir).toBe(newest);
    expect(found.version).toBe("14.50.35710");
    expect(found.toolset).toBe("Microsoft.VC145.CRT");
    expect(found.tried).toHaveLength(2);
    expect(redistCrtDirs(path.join(vs2022, "VC", "Redist", "MSVC")).map((c) => c.dir)).toEqual([old, mid]);
    // Only the older installation present: that one.
    expect(findVcRuntime({ env: {}, installPaths: [vs2022] }).dir).toBe(old);
  });

  it("fails loudly, and says what to install, when no installation carries the runtime", () => {
    const root = tmp();
    expect(findVcRuntime({ env: {}, installPaths: [path.join(root, "empty")] }).dir).toBeNull();
    expect(() => requireVcRuntime({ env: {}, installPaths: [path.join(root, "empty")] })).toThrow(
      /vcruntime140\.dll.*not found[\s\S]*Desktop development with C\+\+[\s\S]*never taken from System32 and never downloaded/,
    );
    expect(() => requireVcRuntime({ env: {}, installPaths: [] })).toThrow(/no Visual Studio/);
  });

  it("never takes the DLLs from the Windows system folders", () => {
    const root = tmp();
    const windows = path.join(root, "Windows");
    const system32 = path.join(windows, "System32");
    for (const n of VC_RUNTIME_DLLS) fakePe(system32, n);
    const env = { SystemRoot: windows };
    expect(isSystemDir(system32, env)).toBe(true);
    expect(isSystemDir(path.join(windows, "SysWOW64"), env)).toBe(true);
    expect(isSystemDir(path.join(root, "vs"), env)).toBe(false);
    expect(() => findVcRuntime({ env, explicitDir: system32 })).toThrow(/system folder/);
    expect(() => findVcRuntime({ env: { ...env, SIGIL_VC_REDIST_DIR: system32 } })).toThrow(/system folder/);
    // A Visual Studio folder given explicitly is accepted, one that lacks a DLL is not.
    const redist = fakeRedist(path.join(root, "vs"), "14.50.35710");
    expect(findVcRuntime({ env, explicitDir: redist }).dir).toBe(path.resolve(redist));
    fs.rmSync(path.join(redist, "msvcp140_1.dll"));
    expect(() => findVcRuntime({ env, explicitDir: redist })).toThrow(/lacks msvcp140_1\.dll/);
  });

  it("stages only genuine x64 DLLs", () => {
    const root = tmp();
    const x86 = path.join(root, "x86");
    fakePe(x86, "vcruntime140.dll", 0x14c);
    const files = (dir) => [{ name: "vcruntime140.dll", source: path.join(dir, "vcruntime140.dll"), group: "vc-runtime", x64: true }];
    expect(() => stageFiles({ stagingDir: path.join(root, "s"), files: files(x86) })).toThrow(/not an x64 DLL/);
    fakeDll(x86, "vcruntime140.dll"); // MZ only, no PE header
    expect(() => stageFiles({ stagingDir: path.join(root, "s"), files: files(x86) })).toThrow(/no PE header/);
    const x64 = path.join(root, "x64");
    fakePe(x64, "vcruntime140.dll");
    expect(stageFiles({ stagingDir: path.join(root, "s"), files: files(x64) }).files[0].group).toBe("vc-runtime");
  });
});
