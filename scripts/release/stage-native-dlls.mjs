/**
 * Stage ONNX Runtime's GPU execution-provider DLLs for the Windows installers.
 *
 * Sigil links ONNX Runtime statically, so ORT loads `onnxruntime_providers_cuda.dll` (with its
 * bridge `onnxruntime_providers_shared.dll`) from the directory of Sigil.exe, and Sigil.exe
 * load-time imports `DirectML.dll` (the application directory wins over System32). ort-sys copies
 * all three next to `target/<profile>/Sigil.exe` during `cargo build`; this script copies them
 * into `src-tauri/target/native-bundle/`, which the release config
 * (`scripts/release/tauri.release.conf.json`) maps to the install root.
 *
 * Two phases, because tauri-build copies bundle resources at *compile* time (before ort-sys has
 * produced the DLLs on a clean machine) and the bundler reads them *after* the build:
 *
 *   --prepare  (beforeBuildCommand)  create the staging directory and empty it. An empty directory
 *              is a valid resource for tauri-build, and stale DLLs from an older ORT can never be
 *              copied over ort-sys's fresh ones.
 *   --stage    (beforeBundleCommand, default) copy the DLLs from the cargo target directory,
 *              verify them (PE header, size, sha256) and write a manifest next to the directory
 *              (`src-tauri/target/native-bundle.json`, read by verify-bundle.ps1).
 *   --check    verify an existing staging directory against its manifest.
 *
 * Options: --target-dir <dir> (default $CARGO_TARGET_DIR or src-tauri/target), --profile
 * release|debug (default from TAURI_ENV_DEBUG), --from <dir> (explicit source directory).
 *
 * Note: the WiX (MSI) bundler additionally adds every `*.dll` next to the built Sigil.exe that is not
 * already a resource, so the MSI also carries ort-sys's unused `onnxruntime_providers_tensorrt.dll`;
 * NSIS ships exactly the staged set. verify-bundle.ps1 reports such extras.
 */
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const STAGING_REL = path.join("src-tauri", "target", "native-bundle");
export const MANIFEST_REL = path.join("src-tauri", "target", "native-bundle.json");
export const SHARED_DLL = "onnxruntime_providers_shared.dll";
export const CUDA_DLL = "onnxruntime_providers_cuda.dll";
export const DIRECTML_DLL = "DirectML.dll";

export function repoRoot() {
  return path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
}

export function requiredDlls() {
  return [SHARED_DLL, CUDA_DLL, DIRECTML_DLL];
}

export function parseArgs(argv) {
  const opts = { mode: "stage", targetDir: null, profile: null, from: null };
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
    else if (a === "--target-dir") opts.targetDir = value();
    else if (a === "--profile") opts.profile = value();
    else if (a === "--from") opts.from = value();
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

function assertPe(file) {
  const fd = fs.openSync(file, "r");
  try {
    const head = Buffer.alloc(2);
    fs.readSync(fd, head, 0, 2, 0);
    if (head.toString("latin1") !== "MZ") throw new Error(`${file} is not a Windows DLL (no MZ header)`);
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
 * Copy `names` from `sourceDir` into `stagingDir` as real files (ort-sys places symlinks into its
 * download cache; copying through them keeps the cache untouched) and return the manifest.
 */
export function stage({ sourceDir, stagingDir, names }) {
  prepare(stagingDir);
  const files = names.map((name) => {
    const src = fs.realpathSync(path.join(sourceDir, name));
    assertPe(src);
    const dest = path.join(stagingDir, name);
    fs.copyFileSync(src, dest);
    const size = fs.statSync(dest).size;
    if (size !== fs.statSync(src).size || size === 0) throw new Error(`copy of ${name} is incomplete`);
    const sha256 = sha256File(dest);
    if (sha256 !== sha256File(src)) throw new Error(`copy of ${name} does not match its source`);
    return { name, size, sha256, source: src };
  });
  return { schema: 1, staged_at: new Date().toISOString(), staging_dir: stagingDir, files };
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

  const platform = process.env.TAURI_ENV_PLATFORM || process.platform;
  if (!/^win/i.test(platform)) {
    prepare(stagingDir);
    console.log(`[stage-native-dlls] ${platform}: no Windows provider DLLs to stage`);
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
  const manifest = stage({ sourceDir, stagingDir, names });
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  for (const f of manifest.files) {
    console.log(`[stage-native-dlls] ${f.name}  ${(f.size / 1e6).toFixed(1)} MB  sha256 ${f.sha256.slice(0, 16)}…`);
  }
  console.log(`[stage-native-dlls] staged ${manifest.files.length} DLLs from ${sourceDir}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    main();
  } catch (e) {
    console.error(`[stage-native-dlls] ${e instanceof Error ? e.message : e}`);
    process.exit(1);
  }
}
