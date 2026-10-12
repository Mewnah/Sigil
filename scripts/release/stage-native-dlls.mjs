/**
 * Stage the native DLLs that must sit next to Sigil.exe in the Windows installers:
 *
 *  1. ONNX Runtime's GPU execution-provider DLLs. Sigil links ONNX Runtime statically, so ORT loads
 *     `onnxruntime_providers_cuda.dll` (with its bridge `onnxruntime_providers_shared.dll`) from the
 *     directory of Sigil.exe, and Sigil.exe load-time imports `DirectML.dll` (the application
 *     directory wins over System32). ort-sys copies all three next to
 *     `target/<profile>/Sigil.exe` during `cargo build`.
 *  2. The Visual C++ runtime, app-local: `vcruntime140.dll`, `vcruntime140_1.dll`, `msvcp140.dll`
 *     and `msvcp140_1.dll` are load-time imports of Sigil.exe (expected-imports.txt) that neither a
 *     clean Windows nor the installers carried: on such a PC Sigil.exe exited with 0xC0000135
 *     (STATUS_DLL_NOT_FOUND) before it showed anything (found by the 0.5.0 Windows Sandbox smoke).
 *     They are the redistributable files of the installed Visual Studio / Build Tools,
 *     `VC\Redist\MSVC\<version>\x64\Microsoft.VC14*.CRT\`, which Microsoft licenses for app-local
 *     deployment (REDIST.TXT). They are found with vswhere (newest version of all installations),
 *     never taken from System32 and never downloaded; if they are not found the step fails.
 *
 * Everything is copied into `src-tauri/target/native-bundle/`, which the release config
 * (`scripts/release/tauri.release.conf.json`) maps to the install root.
 *
 * Two phases, because tauri-build copies bundle resources at *compile* time (before ort-sys has
 * produced the DLLs on a clean machine) and the bundler reads them *after* the build:
 *
 *   --prepare  (beforeBuildCommand)  create the staging directory and empty it. An empty directory
 *              is a valid resource for tauri-build, and stale DLLs from an older ORT can never be
 *              copied over ort-sys's fresh ones.
 *   --stage    (beforeBundleCommand, default) copy the DLLs (ORT providers from the cargo target
 *              directory, the VC runtime from the Visual Studio redistributable folder), verify them
 *              (PE header, x64 for the VC runtime, size, sha256) and write a manifest next to the
 *              directory (`src-tauri/target/native-bundle.json`, read by verify-bundle.ps1).
 *   --check    verify an existing staging directory against its manifest.
 *   --check-vc only look for the VC runtime and print where it is (exit 1 when it is missing); CI runs it
 *              before the long compile so a runner without the redistributable fails in seconds.
 *
 * Options: --target-dir <dir> (default $CARGO_TARGET_DIR or src-tauri/target), --profile
 * release|debug (default from TAURI_ENV_DEBUG), --from <dir> (explicit source directory of the
 * provider DLLs), --vc-redist <dir> (a Microsoft.VC14*.CRT folder instead of the vswhere search; also
 * $SIGIL_VC_REDIST_DIR; the Windows system folders are refused).
 *
 * Note: the WiX (MSI) bundler additionally adds every `*.dll` next to the built Sigil.exe that is not
 * already a resource, so the MSI also carries ort-sys's unused `onnxruntime_providers_tensorrt.dll`;
 * NSIS ships exactly the staged set. verify-bundle.ps1 reports such extras.
 */
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const STAGING_REL = path.join("src-tauri", "target", "native-bundle");
export const MANIFEST_REL = path.join("src-tauri", "target", "native-bundle.json");
export const SHARED_DLL = "onnxruntime_providers_shared.dll";
export const CUDA_DLL = "onnxruntime_providers_cuda.dll";
export const DIRECTML_DLL = "DirectML.dll";
/** Load-time imports of Sigil.exe from the Visual C++ runtime (keep equal to expected-imports.txt and verify-bundle.ps1). */
export const VC_RUNTIME_DLLS = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll"];
const MACHINE_X64 = 0x8664;

export function repoRoot() {
  return path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
}

export function requiredDlls() {
  return [SHARED_DLL, CUDA_DLL, DIRECTML_DLL];
}

export function vcRuntimeDlls() {
  return [...VC_RUNTIME_DLLS];
}

export function parseArgs(argv) {
  const opts = { mode: "stage", targetDir: null, profile: null, from: null, vcRedist: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const value = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`${a} needs a value`);
      return v;
    };
    if (a === "--prepare") opts.mode = "prepare";
    else if (a === "--stage") opts.mode = "stage";
    else if (a === "--check") opts.mode = "check";
    else if (a === "--check-vc") opts.mode = "check-vc";
    else if (a === "--target-dir") opts.targetDir = value();
    else if (a === "--profile") opts.profile = value();
    else if (a === "--from") opts.from = value();
    else if (a === "--vc-redist") opts.vcRedist = value();
    else throw new Error(`unknown argument ${a}`);
  }
  if (opts.profile && !["release", "debug"].includes(opts.profile)) {
    throw new Error(`--profile must be release or debug, got ${opts.profile}`);
  }
  return opts;
}

/** Directories that may hold the DLLs, most specific first. */
export function candidateSourceDirs({ root, env, targetDir, profile, from }) {
  const dirs = [];
  if (from) dirs.push(path.resolve(from));
  const target = path.resolve(root, targetDir || env.CARGO_TARGET_DIR || path.join("src-tauri", "target"));
  const prof = profile || (String(env.TAURI_ENV_DEBUG).toLowerCase() === "true" ? "debug" : "release");
  if (env.TAURI_ENV_TARGET_TRIPLE) dirs.push(path.join(target, env.TAURI_ENV_TARGET_TRIPLE, prof));
  dirs.push(path.join(target, prof));
  return [...new Set(dirs)];
}

export function findSourceDir(candidates, names) {
  return candidates.find((d) => names.every((n) => fs.existsSync(path.join(d, n)))) ?? null;
}

// --- the Visual C++ runtime -------------------------------------------------------------------------

/** Numeric comparison of dotted versions ("14.50.35710" > "14.44.35112"); negative, zero or positive. */
export function compareVersions(a, b) {
  const pa = String(a).split(".").map((n) => Number.parseInt(n, 10) || 0);
  const pb = String(b).split(".").map((n) => Number.parseInt(n, 10) || 0);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const d = (pa[i] ?? 0) - (pb[i] ?? 0);
    if (d !== 0) return d;
  }
  return 0;
}

/** The Windows system folders: the runtime DLLs found there are the machine's own copies, never redistributed from them. */
export function isSystemDir(dir, env = process.env) {
  const win = env.SystemRoot || env.windir || "C:\\Windows";
  const norm = (p) => path.resolve(p).toLowerCase();
  const d = norm(dir);
  return ["System32", "SysWOW64", "Sysnative"].some((s) => {
    const sys = norm(path.join(win, s));
    return d === sys || d.startsWith(sys + path.sep);
  });
}

/**
 * The x64 `Microsoft.VC14*.CRT` folders under one `VC\Redist\MSVC` directory that hold all `names`,
 * newest first (version folder, then toolset: VC145 before VC143).
 */
export function redistCrtDirs(msvcRedistRoot, names = VC_RUNTIME_DLLS) {
  if (!fs.existsSync(msvcRedistRoot)) return [];
  const found = [];
  for (const version of fs.readdirSync(msvcRedistRoot)) {
    if (!/^\d+(\.\d+)+$/.test(version)) continue; // skips "v145" (the vc_redist installers) and the like
    const x64 = path.join(msvcRedistRoot, version, "x64");
    if (!fs.existsSync(x64)) continue;
    for (const crt of fs.readdirSync(x64)) {
      if (!/^Microsoft\.VC\d+\.CRT$/i.test(crt)) continue;
      const dir = path.join(x64, crt);
      if (names.every((n) => fs.existsSync(path.join(dir, n)))) found.push({ dir, version, toolset: crt });
    }
  }
  const toolsetNumber = (crt) => Number.parseInt(crt.replace(/\D/g, ""), 10) || 0;
  return found.sort((a, b) => compareVersions(b.version, a.version) || toolsetNumber(b.toolset) - toolsetNumber(a.toolset));
}

/** Installation paths of every Visual Studio / Build Tools instance (vswhere ships with the Visual Studio installer). */
export function vswhereInstallPaths(env = process.env, run = execFileSync) {
  const programFiles = env["ProgramFiles(x86)"] || env.ProgramFiles;
  if (!programFiles) return [];
  const exe = path.join(programFiles, "Microsoft Visual Studio", "Installer", "vswhere.exe");
  if (!fs.existsSync(exe)) return [];
  try {
    const out = run(exe, ["-all", "-products", "*", "-property", "installationPath"], { encoding: "utf8", windowsHide: true });
    return String(out)
      .split(/\r?\n/)
      .map((s) => s.trim())
      .filter(Boolean);
  } catch {
    return [];
  }
}

/**
 * Find the folder with the four VC runtime DLLs: an explicit folder (refused when it is a Windows system folder), else
 * the newest `Microsoft.VC14*.CRT` of all Visual Studio installations. Returns `{ dir, version, toolset, tried }` or
 * `{ dir: null, tried }`.
 */
export function findVcRuntime({ env = process.env, explicitDir = null, installPaths = null, names = VC_RUNTIME_DLLS } = {}) {
  const tried = [];
  const explicit = explicitDir || env.SIGIL_VC_REDIST_DIR;
  if (explicit) {
    const dir = path.resolve(explicit);
    tried.push(dir);
    if (isSystemDir(dir, env)) {
      throw new Error(`${dir} is a Windows system folder: the VC runtime is never redistributed from System32, point --vc-redist at a Visual Studio Microsoft.VC14*.CRT folder`);
    }
    const missing = names.filter((n) => !fs.existsSync(path.join(dir, n)));
    if (missing.length) throw new Error(`${dir} lacks ${missing.join(", ")}`);
    return { dir, version: "explicit", toolset: path.basename(dir), tried };
  }
  const roots = new Set();
  for (const install of installPaths ?? vswhereInstallPaths(env)) roots.add(path.join(install, "VC", "Redist", "MSVC"));
  if (env.VCToolsRedistDir) roots.add(path.dirname(path.resolve(env.VCToolsRedistDir))); // a developer prompt's own toolset
  const all = [];
  for (const root of roots) {
    tried.push(root);
    all.push(...redistCrtDirs(root, names));
  }
  const best = all
    .filter((c) => !isSystemDir(c.dir, env))
    .sort((a, b) => compareVersions(b.version, a.version))[0];
  return best ? { ...best, tried } : { dir: null, tried };
}

/** `findVcRuntime`, or an error that says what to install. */
export function requireVcRuntime(options) {
  const found = findVcRuntime(options);
  if (found.dir) return found;
  const where = found.tried.length ? found.tried.map((t) => `\n  ${t}`).join("") : "\n  (no Visual Studio / Build Tools installation found by vswhere)";
  throw new Error(
    `the Visual C++ runtime DLLs (${VC_RUNTIME_DLLS.join(", ")}) were not found in a Visual Studio redistributable folder; searched:${where}\n` +
      "Install the Visual Studio Build Tools workload \"Desktop development with C++\" (it carries VC\\Redist\\MSVC\\<version>\\x64\\Microsoft.VC14*.CRT), " +
      "or pass --vc-redist <that folder> (or set SIGIL_VC_REDIST_DIR). They are never taken from System32 and never downloaded, " +
      "and without them Sigil.exe does not start on a clean Windows (0xC0000135).",
  );
}

// --- staging ----------------------------------------------------------------------------------------

export function sha256File(file) {
  const hash = createHash("sha256");
  const fd = fs.openSync(file, "r");
  try {
    const buf = Buffer.allocUnsafe(1 << 20);
    let n;
    while ((n = fs.readSync(fd, buf, 0, buf.length, null)) > 0) hash.update(buf.subarray(0, n));
  } finally {
    fs.closeSync(fd);
  }
  return hash.digest("hex");
}

/** MZ header; with `x64`, also the PE signature and the AMD64 machine type (an x86 CRT folder must not slip in). */
function assertPe(file, { x64 = false } = {}) {
  const fd = fs.openSync(file, "r");
  try {
    const head = Buffer.alloc(64);
    const n = fs.readSync(fd, head, 0, 64, 0);
    if (n < 2 || head.toString("latin1", 0, 2) !== "MZ") throw new Error(`${file} is not a Windows DLL (no MZ header)`);
    if (!x64) return;
    const peOffset = n >= 64 ? head.readUInt32LE(0x3c) : 0;
    const pe = Buffer.alloc(6);
    const m = peOffset > 0 ? fs.readSync(fd, pe, 0, 6, peOffset) : 0;
    if (m < 6 || pe.toString("latin1", 0, 4) !== "PE\0\0") throw new Error(`${file} has no PE header`);
    if (pe.readUInt16LE(4) !== MACHINE_X64) throw new Error(`${file} is not an x64 DLL`);
  } finally {
    fs.closeSync(fd);
  }
}

/** Remove every entry of `dir` (created if missing). */
export function prepare(stagingDir) {
  fs.mkdirSync(stagingDir, { recursive: true });
  for (const entry of fs.readdirSync(stagingDir)) {
    fs.rmSync(path.join(stagingDir, entry), { recursive: true, force: true });
  }
}

/**
 * Copy `files` (`{ name, source, group, x64? }`) into `stagingDir` as real files (ort-sys places symlinks into its
 * download cache; copying through them keeps the cache untouched) and return the manifest.
 */
export function stageFiles({ stagingDir, files }) {
  prepare(stagingDir);
  const entries = files.map(({ name, source, group = "onnxruntime", x64 = false }) => {
    const src = fs.realpathSync(source);
    assertPe(src, { x64 });
    const dest = path.join(stagingDir, name);
    fs.copyFileSync(src, dest);
    const size = fs.statSync(dest).size;
    if (size !== fs.statSync(src).size || size === 0) throw new Error(`copy of ${name} is incomplete`);
    const sha256 = sha256File(dest);
    if (sha256 !== sha256File(src)) throw new Error(`copy of ${name} does not match its source`);
    return { name, size, sha256, group, source: src };
  });
  return { schema: 1, staged_at: new Date().toISOString(), staging_dir: stagingDir, files: entries };
}

/** `stageFiles` for `names` found in one `sourceDir` (the ONNX Runtime providers). */
export function stage({ sourceDir, stagingDir, names }) {
  return stageFiles({ stagingDir, files: names.map((name) => ({ name, source: path.join(sourceDir, name) })) });
}

/** Verify `stagingDir` holds exactly the manifest's files with matching hashes. */
export function check(stagingDir, manifest) {
  const problems = [];
  const expected = new Set(manifest.files.map((f) => f.name.toLowerCase()));
  for (const f of manifest.files) {
    const p = path.join(stagingDir, f.name);
    if (!fs.existsSync(p)) problems.push(`${f.name} is missing`);
    else if (fs.statSync(p).size !== f.size) problems.push(`${f.name} has the wrong size`);
    else if (sha256File(p) !== f.sha256) problems.push(`${f.name} has the wrong sha256`);
  }
  for (const entry of fs.existsSync(stagingDir) ? fs.readdirSync(stagingDir) : []) {
    if (!expected.has(entry.toLowerCase())) problems.push(`unexpected file ${entry} would be installed`);
  }
  // Everything Sigil.exe needs at load time must be in the set: a manifest without the VC runtime is a build that would
  // not start on a clean Windows.
  for (const name of [...requiredDlls(), ...VC_RUNTIME_DLLS]) {
    if (!expected.has(name.toLowerCase())) problems.push(`${name} is not in the manifest`);
  }
  return problems;
}

function main() {
  const opts = parseArgs(process.argv.slice(2));
  const root = repoRoot();
  const stagingDir = path.join(root, STAGING_REL);
  const manifestPath = path.join(root, MANIFEST_REL);

  if (opts.mode === "prepare") {
    prepare(stagingDir);
    fs.rmSync(manifestPath, { force: true });
    console.log(`[stage-native-dlls] prepared empty ${path.relative(root, stagingDir)}`);
    return;
  }
  if (opts.mode === "check") {
    if (!fs.existsSync(manifestPath)) throw new Error(`${MANIFEST_REL} not found; run --stage first`);
    const problems = check(stagingDir, JSON.parse(fs.readFileSync(manifestPath, "utf8")));
    if (problems.length) throw new Error(problems.join("; "));
    console.log("[stage-native-dlls] staging directory matches its manifest");
    return;
  }
  if (opts.mode === "check-vc") {
    const vc = requireVcRuntime({ explicitDir: opts.vcRedist });
    console.log(`[stage-native-dlls] VC runtime ${vc.version} (${vc.toolset}): ${vc.dir}`);
    return;
  }

  const platform = process.env.TAURI_ENV_PLATFORM || process.platform;
  if (!/^win/i.test(platform)) {
    prepare(stagingDir);
    console.log(`[stage-native-dlls] ${platform}: no Windows DLLs to stage`);
    return;
  }
  const names = requiredDlls();
  const candidates = candidateSourceDirs({ root, env: process.env, ...opts });
  const sourceDir = findSourceDir(candidates, names);
  if (!sourceDir) {
    throw new Error(
      `ONNX Runtime provider DLLs (${names.join(", ")}) not found in:\n  ${candidates.join("\n  ")}\n` +
        "Build first (ort-sys copies them next to Sigil.exe), or pass --from <dir>.",
    );
  }
  const vc = requireVcRuntime({ explicitDir: opts.vcRedist });
  const manifest = stageFiles({
    stagingDir,
    files: [
      ...names.map((name) => ({ name, source: path.join(sourceDir, name), group: "onnxruntime" })),
      ...VC_RUNTIME_DLLS.map((name) => ({ name, source: path.join(vc.dir, name), group: "vc-runtime", x64: true })),
    ],
  });
  manifest.vc_redist = { dir: vc.dir, version: vc.version, toolset: vc.toolset };
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  for (const f of manifest.files) {
    console.log(`[stage-native-dlls] ${f.name}  ${(f.size / 1e6).toFixed(1)} MB  sha256 ${f.sha256.slice(0, 16)}…`);
  }
  console.log(
    `[stage-native-dlls] staged ${manifest.files.length} DLLs: ${names.length} ONNX Runtime from ${sourceDir}, ${VC_RUNTIME_DLLS.length} VC runtime ${vc.version} from ${vc.dir}`,
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    main();
  } catch (e) {
    console.error(`[stage-native-dlls] ${e instanceof Error ? e.message : e}`);
    process.exit(1);
  }
}
