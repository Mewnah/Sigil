// publish-assets.ps1 is the only script that can upload anything, so its guards are tested for real:
// the plan runs offline, and every upload path runs against a stand-in `gh` placed first on PATH
// (the real GitHub CLI is never called, whatever is installed on the PC or the CI runner).
import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { makeWorld } from "../../models/__tests__/fixtures.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SCRIPT = path.join(HERE, "..", "publish-assets.ps1");
const text = fs.readFileSync(SCRIPT, "utf8");
const win = process.platform === "win32";
const hasPwsh = win && !spawnSync("pwsh", ["-NoProfile", "-Command", "1"], { encoding: "utf8" }).error;

const FAKE_GH = `param([Parameter(ValueFromRemainingArguments = $true)][string[]]$rest)
$dir = $env:FAKE_GH_DIR
$statePath = Join-Path $dir 'state.json'
Add-Content -LiteralPath (Join-Path $dir 'calls.log') -Value ($rest -join ' | ')
if (-not (Test-Path $statePath)) { '{"exists":false,"assets":[]}' | Set-Content $statePath }
$state = Get-Content $statePath -Raw | ConvertFrom-Json
function Save { $state | ConvertTo-Json -Depth 5 | Set-Content $statePath }
if ($rest[0] -eq 'auth' -and $rest[1] -eq 'status') { if ($env:FAKE_GH_LOGGED_OUT) { exit 1 }; exit 0 }
if ($rest[0] -eq 'release' -and $rest[1] -eq 'view') {
    if (-not $state.exists) { [Console]::Error.WriteLine('release not found'); exit 1 }
    @{ tagName = $rest[2]; assets = @($state.assets) } | ConvertTo-Json -Depth 5 -Compress
    exit 0
}
if ($rest[0] -eq 'release' -and $rest[1] -eq 'create') { $state.exists = $true; Save; exit 0 }
if ($rest[0] -eq 'release' -and $rest[1] -eq 'upload') {
    $name = Split-Path -Leaf $rest[3]
    $clobber = $rest -contains '--clobber'
    $have = @($state.assets) | Where-Object { $_.name -eq $name }
    if ($have -and -not $clobber) { exit 1 }
    $others = @(@($state.assets) | Where-Object { $_.name -ne $name })
    $state.assets = @($others + [pscustomobject]@{ name = $name; size = (Get-Item -LiteralPath $rest[3]).Length })
    Save
    exit 0
}
exit 2
`;

const cleanups = [];
afterEach(() => {
  while (cleanups.length) cleanups.pop()();
});

/** A catalog with only fragment a.json (four pinned files), a staging folder holding two of them, and the stand-in gh. */
function setup() {
  const w = makeWorld();
  fs.rmSync(path.join(w.catalogDir, "b.json"));
  for (const n of ["pack-1.zip", "up-model.gguf", "t-multi-b.json", "t-dep-model.onnx"]) fs.rmSync(path.join(w.staging, n), { force: true });
  const ghDir = fs.mkdtempSync(path.join(os.tmpdir(), "sigil-fakegh-"));
  fs.writeFileSync(path.join(ghDir, "gh-impl.ps1"), FAKE_GH);
  fs.writeFileSync(path.join(ghDir, "gh.cmd"), '@echo off\r\npwsh -NoProfile -File "%~dp0gh-impl.ps1" %*\r\nexit /b %ERRORLEVEL%\r\n');
  cleanups.push(
    () => w.cleanup(),
    () => fs.rmSync(ghDir, { recursive: true, force: true }),
  );
  const calls = () => (fs.existsSync(path.join(ghDir, "calls.log")) ? fs.readFileSync(path.join(ghDir, "calls.log"), "utf8").split(/\r?\n/).filter(Boolean) : []);
  const env = { ...process.env, PATH: `${ghDir}${path.delimiter}${process.env.PATH}`, FAKE_GH_DIR: ghDir };
  return { w, ghDir, calls, env };
}

const run = (args, env) => spawnSync("pwsh", ["-NoProfile", "-File", SCRIPT, ...args], { encoding: "utf8", env });

describe("publish-assets.ps1 source", () => {
  it("is ASCII only and never handles a token, a password or a credential store", () => {
    expect([...text].filter((c) => c.charCodeAt(0) > 127)).toEqual([]);
    expect(text).not.toMatch(/GH_TOKEN|GITHUB_TOKEN|Get-Credential|ConvertTo-SecureString|--with-token/i);
  });

  it.runIf(win)("parses without errors in Windows PowerShell 5.1 and PowerShell 7", () => {
    const cmd = `$e = $null; [System.Management.Automation.Language.Parser]::ParseFile('${SCRIPT}', [ref]$null, [ref]$e) | Out-Null; if ($e) { $e | ForEach-Object { $_.Message + ' @' + $_.Extent.StartLineNumber }; exit 1 }`;
    for (const shell of ["powershell.exe", "pwsh"]) {
      const r = spawnSync(shell, ["-NoProfile", "-Command", cmd], { encoding: "utf8" });
      if (r.error && /ENOENT/.test(String(r.error))) continue;
      expect(r.stdout + r.stderr, shell).toBe("");
      expect(r.status, shell).toBe(0);
    }
  });
});

describe.runIf(hasPwsh)("publish-assets.ps1 against a stand-in gh", () => {
  it("the stand-in is what `gh` resolves to (the real CLI is never reached)", () => {
    const { env, ghDir } = setup();
    const r = spawnSync("pwsh", ["-NoProfile", "-Command", "(Get-Command gh).Source"], { encoding: "utf8", env });
    expect(path.dirname(r.stdout.trim())).toBe(ghDir);
  });

  it("prints the plan and uploads nothing without -Execute", () => {
    const { w, calls, env } = setup();
    const r = run(["-Staging", w.staging, "-CatalogDir", w.catalogDir], env);
    expect(r.status).toBe(0);
    expect(r.stdout).toMatch(/to upload: 2 file\(s\)/);
    expect(r.stdout).toMatch(/t-solo-model\.onnx/);
    expect(r.stdout).toMatch(/Dry run: nothing was uploaded/);
    expect(calls()).toEqual([]);
  });

  it("-Execute creates the pre-release, uploads one file at a time, and a second run skips them", () => {
    const { w, calls, env } = setup();
    const first = run(["-Staging", w.staging, "-CatalogDir", w.catalogDir, "-Execute", "-DelaySeconds", "2"], env);
    expect(first.status, first.stdout + first.stderr).toBe(0);
    const log = calls();
    expect(log[0]).toBe("auth | status");
    expect(log.some((l) => l.startsWith("release | create | assets-v1 | --repo | Mewnah/Sigil-assets | --prerelease"))).toBe(true);
    const uploads = log.filter((l) => l.startsWith("release | upload"));
    expect(uploads).toHaveLength(2);
    expect(uploads.every((l) => !l.includes("--clobber"))).toBe(true);
    expect(first.stdout).toMatch(/uploaded 2, skipped 0, failed 0/);

    fs.writeFileSync(path.join(env.FAKE_GH_DIR, "calls.log"), "");
    const second = run(["-Staging", w.staging, "-CatalogDir", w.catalogDir, "-Execute", "-DelaySeconds", "2"], env);
    expect(second.status).toBe(0);
    expect(second.stdout).toMatch(/uploaded 0, skipped 2, failed 0/);
    expect(calls().filter((l) => l.startsWith("release | upload"))).toEqual([]);
  }, 60_000);

  it("an asset that exists with another size stops that file until -Replace, which uploads with --clobber", () => {
    const { w, calls, env } = setup();
    expect(run(["-Staging", w.staging, "-CatalogDir", w.catalogDir, "-Execute", "-DelaySeconds", "2"], env).status).toBe(0);
    const statePath = path.join(env.FAKE_GH_DIR, "state.json");
    const state = JSON.parse(fs.readFileSync(statePath, "utf8"));
    state.assets.find((a) => a.name === "t-solo-model.onnx").size = 1;
    fs.writeFileSync(statePath, JSON.stringify(state));
    fs.writeFileSync(path.join(env.FAKE_GH_DIR, "calls.log"), "");

    const stop = run(["-Staging", w.staging, "-CatalogDir", w.catalogDir, "-Execute", "-DelaySeconds", "2"], env);
    expect(stop.status).toBe(1);
    expect(stop.stdout).toMatch(/STOP\s+t-solo-model\.onnx: the release has 1 bytes/);
    expect(calls().filter((l) => l.startsWith("release | upload"))).toEqual([]);

    const replace = run(["-Staging", w.staging, "-CatalogDir", w.catalogDir, "-Execute", "-Replace", "-DelaySeconds", "2"], env);
    expect(replace.status, replace.stdout).toBe(0);
    const up = calls().filter((l) => l.startsWith("release | upload"));
    expect(up).toHaveLength(1);
    expect(up[0]).toMatch(/t-solo-model\.onnx.*--clobber/);
  }, 90_000);

  it("refuses to start while a staged file disagrees with its pin, before any gh call", () => {
    const { w, calls, env } = setup();
    fs.writeFileSync(path.join(w.staging, "t-solo-model.onnx"), "tampered bytes of another length");
    const r = run(["-Staging", w.staging, "-CatalogDir", w.catalogDir, "-Execute"], env);
    expect(r.status).toBe(1);
    expect(r.stdout).toMatch(/blocked/);
    expect(r.stdout).toMatch(/Nothing was uploaded/);
    expect(calls()).toEqual([]);
  });

  it("stops when gh is not logged in", () => {
    const { w, calls, env } = setup();
    const r = run(["-Staging", w.staging, "-CatalogDir", w.catalogDir, "-Execute"], { ...env, FAKE_GH_LOGGED_OUT: "1" });
    expect(r.status).toBe(1);
    expect(r.stdout).toMatch(/gh is not logged in/);
    expect(calls()).toEqual(["auth | status"]);
  });
});
