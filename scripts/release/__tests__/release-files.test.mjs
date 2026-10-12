import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const RELEASE = path.join(HERE, "..");
const ROOT = path.join(RELEASE, "..", "..");
const read = (...p) => fs.readFileSync(path.join(...p), "utf8").replace(/\r\n/g, "\n");

const SMOKE = read(RELEASE, "packaged-smoke-all.ps1");
const WORKFLOW = read(ROOT, ".github", "workflows", "release.yml");

/** Strip comments, here-strings and quoted strings so syntax checks only see code. */
function powershellCode(text) {
  return text
    .replace(/<#[\s\S]*?#>/g, " ")
    .replace(/@'\n[\s\S]*?\n'@/g, '""')
    .replace(/@"\n[\s\S]*?\n"@/g, '""')
    .replace(/'(?:[^']|'')*'/g, "''")
    .replace(/"(?:`.|[^"`])*"/g, '""')
    .replace(/#.*$/gm, "");
}

describe("packaged-smoke-all.ps1 runs in Windows PowerShell 5.1 (the Sandbox guest has nothing else)", () => {
  it("is ASCII only (5.1 reads a BOM-less file as ANSI)", () => {
    const bad = [...SMOKE].filter((c) => c.charCodeAt(0) > 127);
    expect(bad).toEqual([]);
  });

  it("uses no PowerShell 7 syntax", () => {
    const code = powershellCode(SMOKE);
    expect(code).not.toMatch(/&&|\|\|/); // pipeline chain operators
    expect(code).not.toMatch(/\?\?|\?\.|\?\[/); // null-coalescing / conditional access
    expect(code).not.toMatch(/\)\s*\?\s*[^\s]+\s*:\s*[^\s]/); // ternary
    expect(code).not.toMatch(/-AsHashtable|-Parallel|-ThrottleLimit|-SkipHttpErrorCheck|-AsByteStream/);
    // Join-Path takes two positional arguments in 5.1 (no -AdditionalChildPath): nest the calls instead.
    expect(code).not.toMatch(/-AdditionalChildPath/);
  });

  it.runIf(process.platform === "win32")("parses without errors in Windows PowerShell 5.1 and PowerShell 7", () => {
    for (const name of ["packaged-smoke-all.ps1", "exit-probe.ps1"]) {
      const script = path.join(RELEASE, name);
      const cmd = `$e = $null; [System.Management.Automation.Language.Parser]::ParseFile('${script}', [ref]$null, [ref]$e) | Out-Null; if ($e) { $e | ForEach-Object { $_.Message + ' @' + $_.Extent.StartLineNumber }; exit 1 }`;
      for (const shell of ["powershell.exe", "pwsh"]) {
        const r = spawnSync(shell, ["-NoProfile", "-Command", cmd], { encoding: "utf8" });
        if (r.error && /ENOENT/.test(String(r.error))) continue; // pwsh not installed
        expect(r.stdout + r.stderr, `${name} in ${shell}`).toBe("");
        expect(r.status, `${name} in ${shell}`).toBe(0);
      }
    }
  });

  it("exit-probe.ps1 is ASCII only and reuses the smoke script's window helper, refusing a build that would pop a window", () => {
    const probe = read(RELEASE, "exit-probe.ps1");
    expect([...probe].filter((c) => c.charCodeAt(0) > 127)).toEqual([]);
    expect(probe).toContain("packaged-smoke-all.ps1");
    expect(probe).toContain("SIGIL_START_MINIMIZED");
    expect(probe).toMatch(/CloseTitled\(\$app\.Id, 'Sigil'\)/);
  });
});

describe("packaged-smoke-all.ps1 contract", () => {
  it("offers the documented modes and switches", () => {
    for (const m of ["Extract", "Portable", "Layout", "Installed", "Sandbox"]) expect(SMOKE).toContain(`'${m}'`);
    for (const p of ["ExpectNoGpu", "Online", "OnlineModel", "PackStore", "SkipGui", "RequireGui", "NoDevTools", "AllowInstall", "Strict", "ResultsDir", "InstallerPath"]) {
      expect(SMOKE, p).toMatch(new RegExp(`\\$${p}\\b`));
    }
    expect(SMOKE).toContain("sigil-packaged-smoke/1");
  });

  it("covers the clean-machine smoke checklist of RELEASE_AND_ENGINEERING.md", () => {
    for (const id of ["layout", "sysinfo", "catalog", "ep-probe", "gui-server", "gui-page", "gui-models", "gui-idle", "gui-focus", "gui-log", "gui-exit"]) {
      expect(SMOKE, id).toContain(`'${id}'`);
    }
  });

  it("starts the Sandbox guest without a GPU, with the results folder writable and a shutdown at the end", () => {
    expect(SMOKE).toContain("<vGPU>Disable</vGPU>");
    expect(SMOKE).toMatch(/<SandboxFolder>C:\\smoke\\out<\/SandboxFolder><ReadOnly>false<\/ReadOnly>/);
    expect(SMOKE).toMatch(/<SandboxFolder>C:\\smoke\\in<\/SandboxFolder><ReadOnly>true<\/ReadOnly>/);
    expect(SMOKE).toContain("shutdown.exe /s");
  });

  it("never enables a Windows feature and refuses to install on a PC that is not disposable", () => {
    const code = powershellCode(SMOKE);
    expect(code).not.toMatch(/Enable-WindowsOptionalFeature|dism(\.exe)?\s|Add-WindowsCapability|Set-ItemProperty\s+-Path\s+'?HKLM/i);
    expect(SMOKE).toContain("WDAGUtilityAccount");
    expect(SMOKE).toContain("refusing to run the installer");
  });

  it("keeps the real profile out of the app under test", () => {
    for (const v of ["USERPROFILE", "APPDATA", "LOCALAPPDATA"]) expect(SMOKE).toMatch(new RegExp(`${v} = `));
    expect(SMOKE).toContain("--remote-debugging-port=");
    expect(SMOKE).toContain("'--port'");
  });
});

describe("packaged-smoke-all.ps1 never takes the foreground of the PC it runs on", () => {
  const MAIN_RS = read(ROOT, "src-tauri", "src", "main.rs");
  const GUARD_RS = read(ROOT, "src-tauri", "src", "exit_guard.rs");
  const helperSource = () => {
    const m = SMOKE.match(/Add-Type -TypeDefinition @'\n([\s\S]*?)\n'@/);
    expect(m, "the C# window helper").not.toBeNull();
    return m[1];
  };

  it("starts the app with SIGIL_START_MINIMIZED=1 unless -ShowWindow, an environment switch main.rs reads", () => {
    expect(SMOKE).toMatch(/\$guiEnv\['SIGIL_START_MINIMIZED'\] = \$\(if \(\$ShowWindow\) \{ '' \} else \{ '1' \}\)/);
    expect(SMOKE).toMatch(/\[switch\]\$ShowWindow/);
    expect(MAIN_RS).toContain('"SIGIL_START_MINIMIZED"');
  });

  it("does not start a build that does not know the switch on a PC someone may be using", () => {
    expect(SMOKE).toMatch(/elseif \(-not \$ShowWindow -and -not \(Test-Disposable\) -and -not \(Test-ExeKnowsStartMinimized \$exe\)\)/);
    expect(SMOKE).toContain("'SIGIL_START_MINIMIZED'");
  });

  it("samples the foreground window from the first instant: 100 ms, 15 s, never Sigil's", () => {
    expect(SMOKE).toMatch(/New-Object SmokeFocusSampler -ArgumentList \$app\.Id, 15000, 100/);
    expect(SMOKE).toMatch(/\$sampler\.Hits -gt 0/);
    // The sampler is created before the first check runs against the app.
    expect(SMOKE.indexOf("SmokeFocusSampler -ArgumentList")).toBeLessThan(SMOKE.indexOf("Invoke-Check 'gui-server'"));
    // The end state: minimized, and no window besides the main one (a dialog).
    expect(SMOKE).toContain("the main window is not minimized");
    expect(SMOKE).toContain("a dialog?");
  });

  it("keeps the C# helper inside what Windows PowerShell 5.1 compiles (C# 5)", () => {
    const cs = helperSource();
    expect(cs).not.toMatch(/\$"/); // string interpolation
    expect(cs).not.toMatch(/\bout var\b|\bis \w+ \w+\)|\bnameof\(|\?\.|\?\?/); // C# 6 and 7 syntax
    expect(cs).not.toMatch(/\)\s*=>\s*[^{(]*;\s*$/m); // expression-bodied members
  });

  it.runIf(process.platform === "win32")("compiles in Windows PowerShell 5.1 and finds a string in an exe", () => {
    const cs = helperSource();
    const tmp = fs.mkdtempSync(path.join(process.env.TEMP || ".", "smoke-helper-"));
    try {
      const csFile = path.join(tmp, "helper.cs");
      const probe = path.join(tmp, "probe.bin");
      // A needle that straddles the 4 MB read buffer boundary.
      const buf = Buffer.alloc(4 * 1024 * 1024 + 100, 1);
      Buffer.from("SIGIL_START_MINIMIZED").copy(buf, 4 * 1024 * 1024 - 8);
      fs.writeFileSync(probe, buf);
      fs.writeFileSync(csFile, cs);
      const script = [
        `Add-Type -TypeDefinition (Get-Content -Raw -LiteralPath '${csFile}')`,
        `if (-not [SmokeWindows]::FileContains('${probe}', 'SIGIL_START_MINIMIZED')) { exit 11 }`,
        `if ([SmokeWindows]::FileContains('${probe}', 'SIGIL_START_MINIMIZEX')) { exit 12 }`,
        `$s = New-Object SmokeFocusSampler -ArgumentList 0, 300, 50; $s.Start(); $s.Join(); if ($s.Samples -lt 3 -or $s.Hits -ne 0) { exit 13 }`,
        `if ([SmokeWindows]::CloseTitled(0, 'Sigil') -ne 0) { exit 14 }`,
        `if (@([SmokeWindows]::Describe($PID)).Count -lt 0) { exit 15 }`,
        `exit 0`,
      ].join("; ");
      const r = spawnSync("powershell.exe", ["-NoProfile", "-Command", script], { encoding: "utf8" });
      if (r.error && /ENOENT/.test(String(r.error))) return;
      expect(r.stdout + r.stderr).toBe("");
      expect(r.status).toBe(0);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it("closes the real window by title, never through Process.CloseMainWindow or every visible window", () => {
    // tao's thread event target ("Tao Thread Event Target", untitled, 16 x 16, a tool window) counts as a visible window;
    // WM_CLOSE to it destroys the event loop's message window and the process then lingers after the real window closes
    // (reproduced by scripts/release/exit-probe.ps1 on 2026-10-10). The close goes to the window titled "Sigil" only; CloseMainWindow is the
    // last resort when none is found.
    expect(SMOKE).toMatch(/\[SmokeWindows\]::CloseTitled\(\$app\.Id, 'Sigil'\)/);
    expect(SMOKE).not.toMatch(/\[SmokeWindows\]::Close\(/);
    const calls = SMOKE.split("\n").filter((l) => !l.trim().startsWith("#") && l.includes("$app.CloseMainWindow()"));
    expect(calls).toHaveLength(1); // the fallback when no window is titled Sigil
    expect(SMOKE.indexOf("CloseTitled($app.Id, 'Sigil')")).toBeLessThan(SMOKE.indexOf("$app.CloseMainWindow()"));
  });

  it("does not count the event loop's invisible helper windows as a dialog", () => {
    expect(SMOKE).toContain("-not $_.ToolWindow");
    expect(SMOKE).toMatch(/\$_\.Minimized -or \(\$_\.Width \* \$_\.Height\) -gt 0/);
  });

  it("keeps the script start time apart from the app's start time", () => {
    // The GUI section once reassigned $started, which made the result's start time and duration those of the app.
    const assigns = SMOKE.match(/^\s*\$started = /gm) ?? [];
    expect(assigns).toEqual([]);
    expect(SMOKE).toContain("$script:started = Get-Date");
    expect(SMOKE).toContain("$appStarted = Get-Date");
  });

  it("reads the exit guard's log marker from the Rust source of truth, and the app arms the guard on a close request", () => {
    const marker = GUARD_RS.match(/pub const LOG_MARKER: &str = "([^"]+)";/);
    expect(marker, "LOG_MARKER in exit_guard.rs").not.toBeNull();
    expect(SMOKE).toContain(marker[1]);
    expect(MAIN_RS).toMatch(/mod exit_guard;/);
    expect(MAIN_RS).toMatch(/CloseRequested \{ \.\. \} => exit_guard::arm\(\)/);
    expect(MAIN_RS).toMatch(/ExitRequested \{ \.\. \} => exit_guard::arm\(\)/);
    // The guard's grace is shorter than the time the smoke waits for the process to end.
    const grace = Number(GUARD_RS.match(/pub const GRACE: Duration = Duration::from_secs\((\d+)\);/)[1]);
    const wait = Number(SMOKE.match(/\[int\]\$ExitWaitSec = (\d+)/)[1]);
    expect(wait).toBeGreaterThan(grace + 5);
  });
});

/** Runs `body` with a scratch directory that is removed afterwards. */
function withTemp(body) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "release-test-"));
  try {
    return body(dir);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

const REGION = (SMOKE.match(/#region testable-helpers\n([\s\S]*?)\n#endregion testable-helpers/) ?? [])[1];
const SHELLS = ["powershell.exe", "pwsh"]; // Windows PowerShell 5.1 (the Sandbox guest) and PowerShell 7

/** Runs a PowerShell script that first defines the smoke script's helper functions; returns its output lines (null: no such shell). */
function runHelpers(shell, dir, body, env = process.env) {
  fs.writeFileSync(path.join(dir, "region.ps1"), REGION);
  const script = [`. ([scriptblock]::Create((Get-Content -Raw -LiteralPath '${path.join(dir, "region.ps1")}')))`, body].join("\n");
  const file = path.join(dir, "case.ps1");
  fs.writeFileSync(file, script);
  const r = spawnSync(shell, ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", file], { encoding: "utf8", env });
  if (r.error && /ENOENT/.test(String(r.error))) return null;
  expect(r.stderr, `${shell} stderr`).toBe("");
  expect(r.status, `${shell} exit code`).toBe(0);
  return r.stdout.split(/\r?\n/).filter(Boolean);
}

describe.runIf(process.platform === "win32")("the smoke script's helper functions (run for real in 5.1 and 7)", () => {
  it("has the helper region the tests (and exit-probe.ps1) extract", () => {
    expect(REGION, "#region testable-helpers").toBeTruthy();
    for (const f of ["Test-Disposable", "Test-PortFree", "Get-FreePort", "Get-FetchFailureKind", "Get-RequiredTier", "Get-BuildInputsNewerThan", "Get-Text", "Format-ExitCode", "Get-FrontendBootEvidence", "Test-Elevated"]) {
      expect(REGION, f).toContain(`function ${f}`);
    }
  });

  // A bare CI=true in a developer's shell made -Mode Installed run the real installer on the developer's PC.
  it("Test-Disposable: only a GitHub-hosted runner or Windows Sandbox is disposable, never a bare CI variable", () => {
    const cases = [
      { name: "bare-ci", env: { CI: "true" }, expect: false },
      { name: "bare-github-actions", env: { GITHUB_ACTIONS: "true" }, expect: false },
      { name: "self-hosted-runner", env: { GITHUB_ACTIONS: "true", RUNNER_ENVIRONMENT: "self-hosted", CI: "true" }, expect: false },
      { name: "github-hosted-runner", env: { GITHUB_ACTIONS: "true", RUNNER_ENVIRONMENT: "github-hosted", CI: "true" }, expect: true },
      { name: "runner-environment-alone", env: { RUNNER_ENVIRONMENT: "github-hosted" }, expect: false },
      { name: "windows-sandbox", env: {}, user: "WDAGUtilityAccount", expect: true },
      { name: "ordinary-user", env: {}, user: "someone", expect: false },
    ];
    for (const shell of SHELLS) {
      withTemp((dir) => {
        const env = { ...process.env };
        for (const k of ["CI", "GITHUB_ACTIONS", "RUNNER_ENVIRONMENT"]) delete env[k];
        fs.writeFileSync(path.join(dir, "cases.json"), JSON.stringify(cases));
        const out = runHelpers(
          shell,
          dir,
          [
            `$cases = Get-Content -Raw -LiteralPath '${path.join(dir, "cases.json")}' | ConvertFrom-Json`,
            "foreach ($c in $cases) {",
            "  foreach ($n in 'CI', 'GITHUB_ACTIONS', 'RUNNER_ENVIRONMENT') { Remove-Item \"Env:$n\" -ErrorAction SilentlyContinue }",
            "  $env:USERNAME = $(if ($c.user) { $c.user } else { 'someone' })",
            "  foreach ($p in $c.env.PSObject.Properties) { Set-Item -LiteralPath ('Env:' + $p.Name) -Value $p.Value }",
            "  $c.name + '=' + (Test-Disposable)",
            "}",
          ].join("\n"),
          env,
        );
        if (!out) return;
        expect(Object.fromEntries(out.map((l) => l.split("="))), shell).toEqual(Object.fromEntries(cases.map((c) => [c.name, c.expect ? "True" : "False"])));
      });
    }
    expect(SMOKE).not.toMatch(/\$env:CI\b/); // never trust the bare variable again
  });

  // Every non-zero exit of `fetch` was a skip, so a stale pin passed the release gate.
  it("Get-FetchFailureKind: an answering host with a wrong hash, size or status is a pin failure, an unreachable host is a network skip", () => {
    const cases = [
      { name: "hash", text: "fetch: could not download model.onnx (hf: the file does not match the catalog (sha256 1a2b3c4d5e6f expected, got sha256 9f8e7d6c5b4a))", kind: "pin" },
      { name: "short-body", text: "fetch: could not download model.onnx (hf: got 123 bytes, the catalog expects 456)", kind: "pin" },
      { name: "other-version", text: "fetch: could not download x.zip (github-release: the server has 5 bytes but the catalog expects 6 (a different version?))", kind: "pin" },
      { name: "404", text: 'fetch: could not download x.zip (github-release: HTTP 404 Not Found content-type=text/html body_preview="")', kind: "pin" },
      { name: "403", text: 'fetch: could not download x.zip (hf: HTTP 403 Forbidden content-type=text/html body_preview="")', kind: "pin" },
      { name: "410", text: "fetch: could not download x.zip (hf: HTTP 410 Gone)", kind: "pin" },
      { name: "503", text: "fetch: could not download x.zip (hf: HTTP 503 Service Unavailable content-type=text/html)", kind: "pin" },
      { name: "redirect", text: "fetch: could not download x.zip (hf: unexpected HTTP 301)", kind: "pin" },
      { name: "no-source", text: "fetch: could not download x.zip (x.zip has no download source)", kind: "pin" },
      { name: "mixed-hard-wins", text: "fetch: could not download x.zip (hf: cannot connect (dns error); hf-mirror: the file does not match the catalog (sha256 aaaaaaaaaaaa expected, got sha256 bbbbbbbbbbbb))", kind: "pin" },
      { name: "unknown", text: "fetch: something nobody has seen before", kind: "pin" },
      { name: "empty", text: "", kind: "pin" },
      { name: "timeout-with-mismatch", text: "fetch: could not download x.zip (hf: the file does not match the catalog (sha256 aaaaaaaaaaaa expected, got sha256 bbbbbbbbbbbb))", timedOut: true, kind: "pin" },
      { name: "dns", text: "fetch: could not download x.zip (hf: cannot connect (dns error: No such host is known. (os error 11001)))", kind: "network" },
      { name: "no-response", text: "fetch: could not download x.zip (hf: no response within 30 s; github-release: no response within 30 s)", kind: "network" },
      { name: "timeout-error", text: "fetch: could not download x.zip (hf: timed out (operation timed out))", kind: "network" },
      { name: "dropped", text: "fetch: could not download x.zip (hf: connection lost after 1.2 MB: connection reset by peer (gave up after 3 resumes))", kind: "network" },
      { name: "rate-limit", text: "fetch: could not download x.zip (hf: HTTP 429 (rate limited))", kind: "network" },
      { name: "request-timeout", text: "fetch: could not download x.zip (hf: HTTP 408 Request Timeout)", kind: "network" },
      { name: "process-timeout", text: "", timedOut: true, kind: "network" },
    ];
    for (const shell of SHELLS) {
      withTemp((dir) => {
        fs.writeFileSync(path.join(dir, "cases.json"), JSON.stringify(cases));
        const out = runHelpers(
          shell,
          dir,
          [
            `$cases = Get-Content -Raw -LiteralPath '${path.join(dir, "cases.json")}' | ConvertFrom-Json`,
            "foreach ($c in $cases) { $k = Get-FetchFailureKind ([string]$c.text) ([bool]$c.timedOut); $c.name + '=' + $k.Kind }",
          ].join("\n"),
        );
        if (!out) return;
        expect(Object.fromEntries(out.map((l) => l.split("="))), shell).toEqual(Object.fromEntries(cases.map((c) => [c.name, c.kind])));
      });
    }
  });

  // The 0.5.0 Windows Sandbox run: .Stderr.Trim() on an empty stderr threw in 5.1 and hid the exit code (0xC0000135).
  it("Get-Text never returns null (an empty file read with Get-Content -Raw is nothing in 5.1) and Format-ExitCode prints hex", () => {
    for (const shell of SHELLS) {
      withTemp((dir) => {
        const empty = path.join(dir, "empty.txt");
        fs.writeFileSync(empty, "");
        const out = runHelpers(
          shell,
          dir,
          [
            `$raw = if (Test-Path -LiteralPath '${empty}') { Get-Content -Raw -LiteralPath '${empty}' } else { '' }`,
            "$o = [pscustomobject]@{ Stderr = \"$raw\"; Bare = $raw }",
            "'stderr-trim=' + ($o.Stderr.Trim().Length)",
            "'get-text-null=' + ($null -eq (Get-Text $o.Bare))",
            "'get-text-of-null=' + ((Get-Text $null).Length)",
            "'get-text-keeps=' + (Get-Text ' abc ').Trim()",
            "'dll=' + (Format-ExitCode -1073741515)",
            "'dll-unsigned=' + (Format-ExitCode ([int]-1073741515))",
            "'zero=' + (Format-ExitCode 0)",
            "'one=' + (Format-ExitCode 1)",
            "'none=' + (Format-ExitCode $null)",
          ].join("\n"),
        );
        if (!out) return;
        const got = Object.fromEntries(out.map((l) => [l.slice(0, l.indexOf("=")), l.slice(l.indexOf("=") + 1)]));
        expect(got["stderr-trim"], shell).toBe("0");
        expect(got["get-text-null"], shell).toBe("False");
        expect(got["get-text-of-null"], shell).toBe("0");
        expect(got["get-text-keeps"], shell).toBe("abc");
        expect(got.dll, shell).toBe("-1073741515 (0xC0000135) STATUS_DLL_NOT_FOUND: a DLL that Sigil.exe imports is missing");
        expect(got["dll-unsigned"], shell).toContain("0xC0000135");
        expect(got.zero, shell).toBe("0 (0x00000000)");
        expect(got.one, shell).toBe("1 (0x00000001)");
        expect(got.none, shell).toBe("unknown");
      });
    }
    // No bare .Stderr.Trim() is left to throw on a null.
    const code = powershellCode(SMOKE);
    expect(code).not.toMatch(/\.Stderr\.Trim\(\)/);
    expect(code).not.toMatch(/\.Stdout\.Trim\(\)/);
  });

  // Windows Sandbox 26100 with WebView2 155 ignores the WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS override (Edge itself opens the port
  // from its command line), so gui-page cannot read the page there. The log decides instead: the 0.5.0 run on a PC without a
  // network stopped after the start-up probe (6 lines, no front-end line); the fixed build logged 19 lines including the page's own.
  it("Get-FrontendBootEvidence: a log with the page's own start-up line is a booted front end, one that stops after the probe is not", () => {
    const bad = [
      "2026-10-12T05:34:09.941267Z  INFO main sigil::diagnostics: ===== Sigil 0.5.0 started (pid 8116, release build) =====",
      "2026-10-12T05:34:09.946557Z  INFO tokio-runtime-worker sigil::web: local web server listening on 0.0.0.0:49670 (lan mode)",
      "2026-10-12T05:34:09.978840Z  INFO       sigil-ep-probe sigil::diagnostics: startup EP probe: recommended DirectML; compiled EPs [cpu, cuda, dml]",
    ];
    const good = [
      ...bad,
      "2026-10-12T06:15:36.081051Z  INFO tokio-runtime-worker sigil::web: no default route (offline?): advertising the first LAN address, else loopback",
      "2026-10-12T06:15:36.292904Z  INFO                 main sigil::frontend: [System] Monitoring STT, TTS, Translation, Rewrite, OBS WebSocket, and voice changer state.",
      "2026-10-12T06:15:36.293265Z  INFO                 main sigil::frontend: [STT] Disconnected",
    ];
    // Front-end lines that are not the start-up line (an early error) do not count as booted.
    const early = [...bad, "2026-10-12T06:15:36.300000Z ERROR                 main sigil::frontend: [Diagnostics] Diagnostics bridge unavailable: boom"];
    for (const shell of SHELLS) {
      withTemp((dir) => {
        const files = { bad, good, early };
        for (const [name, lines] of Object.entries(files)) fs.writeFileSync(path.join(dir, `${name}.log`), lines.join("\n") + "\n");
        const out = runHelpers(
          shell,
          dir,
          [
            ...Object.keys(files).map(
              (n) => `$e = Get-FrontendBootEvidence '${path.join(dir, `${n}.log`)}'; '${n}=' + $e.Booted + ',' + $e.FrontendLines + ',' + $e.Line.Length`,
            ),
            `$e = Get-FrontendBootEvidence '${path.join(dir, "missing.log")}'; 'missing=' + $e.Booted + ',' + $e.FrontendLines`,
            "$e = Get-FrontendBootEvidence ''; 'empty-path=' + $e.Booted",
            "'elevated-is-bool=' + ((Test-Elevated) -is [bool])",
          ].join("\n"),
        );
        if (!out) return;
        const got = Object.fromEntries(out.map((l) => [l.slice(0, l.indexOf("=")), l.slice(l.indexOf("=") + 1)]));
        expect(got.bad, shell).toBe("False,0,0");
        expect(got.good.startsWith("True,2,"), `${shell} ${got.good}`).toBe(true);
        expect(Number(got.good.split(",")[2]), shell).toBeGreaterThan(20);
        expect(got.early, shell).toBe("False,1,0");
        expect(got.missing, shell).toBe("False,0");
        expect(got["empty-path"], shell).toBe("False");
        expect(got["elevated-is-bool"], shell).toBe("True");
      });
    }
  });

  it("Get-RequiredTier and Add-Check: a skip inside a required tier is a failure", () => {
    const cases = [
      { id: "gui", gui: true, expect: "-RequireGui" },
      { id: "gui-page", gui: true, expect: "-RequireGui" },
      { id: "gui-focus", gui: true, expect: "-RequireGui" },
      { id: "gui-focus", gui: true, show: true, expect: "" }, // -ShowWindow skips the focus check on purpose
      { id: "gui-page", gui: true, show: true, expect: "-RequireGui" },
      { id: "online-fetch", gui: true, expect: "" },
      { id: "online-fetch", online: true, expect: "-RequireOnline" },
      { id: "online-sensevoice", online: true, expect: "-RequireOnline" },
      { id: "gui-page", online: true, expect: "" },
      { id: "layout", gui: true, online: true, expect: "" },
      { id: "online-fetch", expect: "" },
      { id: "gui", expect: "" },
    ];
    // Add-Check itself, with the reporting stubbed out.
    const addCheck = SMOKE.match(/function Add-Check\([\s\S]*?\n}\n/);
    expect(addCheck, "function Add-Check").not.toBeNull();
    for (const shell of SHELLS) {
      withTemp((dir) => {
        fs.writeFileSync(path.join(dir, "cases.json"), JSON.stringify(cases));
        const out = runHelpers(
          shell,
          dir,
          [
            "function Write-Line([string]$Text, [string]$Color = '') { }",
            addCheck[0],
            "$script:checks = New-Object System.Collections.ArrayList",
            `$cases = Get-Content -Raw -LiteralPath '${path.join(dir, "cases.json")}' | ConvertFrom-Json`,
            "$n = 0",
            "foreach ($c in $cases) {",
            "  $RequireGui = [bool]$c.gui; $RequireOnline = [bool]$c.online; $ShowWindow = [bool]$c.show",
            "  $n++",
            "  'tier' + $n + '=' + (Get-RequiredTier $c.id)",
            "  Add-Check $c.id 'skip' 'because'",
            "  Add-Check $c.id 'pass' 'fine'",
            "}",
            "$i = 0",
            "foreach ($c in $script:checks) { $i++; 'status' + $i + '=' + $c.status + ':' + $c.detail }",
          ].join("\n"),
        );
        if (!out) return;
        const got = Object.fromEntries(out.map((l) => [l.slice(0, l.indexOf("=")), l.slice(l.indexOf("=") + 1)]));
        cases.forEach((c, i) => {
          expect(got[`tier${i + 1}`], `${shell} ${JSON.stringify(c)}`).toBe(c.expect);
          const skipped = got[`status${2 * i + 1}`];
          if (c.expect) expect(skipped, `${shell} ${JSON.stringify(c)}`).toBe(`fail:skipped, but ${c.expect} requires it: because`);
          else expect(skipped, `${shell} ${JSON.stringify(c)}`).toBe("skip:because");
          expect(got[`status${2 * i + 2}`]).toBe("pass:fine"); // a check that ran is never touched
        });
      });
    }
  });

  // A random port inside the OS's dynamic range was never checked, and the app's start-up check shows a modal dialog.
  it("Test-PortFree sees a listener, Get-FreePort never repeats a port it was told to exclude", () => {
    for (const shell of SHELLS) {
      withTemp((dir) => {
        const out = runHelpers(
          shell,
          dir,
          [
            "$l = New-Object System.Net.Sockets.TcpListener ([System.Net.IPAddress]::Loopback, 0)",
            "$l.Start()",
            "$busy = $l.LocalEndpoint.Port",
            "'busy-is-seen=' + (-not (Test-PortFree $busy))",
            "'free-port-is-not-the-busy-one=' + ((Get-FreePort) -ne $busy)",
            "$l.Stop()",
            "'released-is-free=' + (Test-PortFree $busy)",
            "$seen = @()",
            "for ($i = 0; $i -lt 8; $i++) { $seen += Get-FreePort -Exclude $seen }",
            "'distinct=' + (@($seen | Select-Object -Unique).Count -eq 8)",
            "'all-free=' + (@($seen | Where-Object { -not (Test-PortFree $_) }).Count -eq 0)",
          ].join("\n"),
        );
        if (!out) return;
        expect(out, shell).toEqual(["busy-is-seen=True", "free-port-is-not-the-busy-one=True", "released-is-free=True", "distinct=True", "all-free=True"]);
      });
    }
  });

  it("exit-probe.ps1 picks its port with the shared helpers and checks it again at the launch, never at random", () => {
    const probe = read(RELEASE, "exit-probe.ps1");
    expect(probe).not.toMatch(/Get-Random/);
    expect(probe).toContain("#region testable-helpers");
    expect(probe).toMatch(/Get-FreePort/);
    expect(probe.indexOf("Test-PortFree $candidate")).toBeGreaterThan(0);
    expect(probe.indexOf("Test-PortFree $candidate")).toBeLessThan(probe.indexOf("Start-Process -FilePath $Exe"));
    // The smoke script re-checks its port right before the launch as well, and its two ports differ.
    expect(SMOKE).toContain("if (-not (Test-PortFree $port)) { $port = Get-FreePort -Exclude @($cdpPort) }");
    expect(SMOKE).toContain("$cdpPort = Get-FreePort -Exclude @($port)");
    expect(SMOKE.indexOf("Test-PortFree $port")).toBeLessThan(SMOKE.indexOf("Start-Process -FilePath $exe -ArgumentList @('--port'"));
  });

  // A smoke on a 20:21 build vouched for the 21:00 tree.
  it("Get-BuildInputsNewerThan lists the shipped sources that are newer than the build, newest first", () => {
    for (const shell of SHELLS) {
      withTemp((dir) => {
        const root = path.join(dir, "repo");
        const base = 1_800_000_000;
        const files = {
          "src/a.ts": base + 10, // shipped, newer
          "src/__tests__/x.test.ts": base + 20, // a test
          "src/b.test.tsx": base + 25, // a test
          "src-tauri/src/m.rs": base + 30, // shipped, newer
          "src-tauri/src/tests.rs": base + 35, // a Rust test module
          "src-tauri/src/old.rs": base - 100, // older than the build
          "docs/x.md": base + 40, // not part of the app
          "package.json": base + 50, // shipped, newer
          "public/p.txt": base + 5, // shipped, newer
        };
        for (const [rel, t] of Object.entries(files)) {
          const full = path.join(root, ...rel.split("/"));
          fs.mkdirSync(path.dirname(full), { recursive: true });
          fs.writeFileSync(full, "x");
          fs.utimesSync(full, t, t);
        }
        const out = runHelpers(
          shell,
          dir,
          [
            `$root = '${root}'`,
            `$stamp = [DateTimeOffset]::FromUnixTimeSeconds(${base}).UtcDateTime`,
            "@(Get-BuildInputsNewerThan $root $stamp) | ForEach-Object { ($_.FullName.Substring($root.Length).TrimStart([char]92)).Replace([string][char]92, '/') }",
            `'none-newer=' + (@(Get-BuildInputsNewerThan $root ([DateTimeOffset]::FromUnixTimeSeconds(${base + 100}).UtcDateTime)).Count -eq 0)`,
          ].join("\n"),
        );
        if (!out) return;
        expect(out, shell).toEqual(["package.json", "src-tauri/src/m.rs", "src/a.ts", "public/p.txt", "none-newer=True"]);
      });
    }
  });
});

/** A stand-in for Sigil.exe: every command fails, and `fetch` prints the message the test asks for. Windows PowerShell 5.1 compiles it. */
function buildFakeSigil(dir) {
  const cs = [
    "using System;",
    "public static class FakeSigil {",
    "  public static int Main(string[] a) {",
    '    string joined = string.Join(" ", a);',
    '    string fakeExit = Environment.GetEnvironmentVariable("FAKE_EXIT");',
    "    if (fakeExit != null && fakeExit.Length > 0) return int.Parse(fakeExit);", // silent: a loader failure writes nothing
    '    bool sherpa = joined.IndexOf("sherpa", StringComparison.Ordinal) >= 0;',
    '    if (joined.IndexOf("fetch", StringComparison.Ordinal) >= 0 && joined.IndexOf("--list", StringComparison.Ordinal) < 0) {',
    '      Console.Error.WriteLine(Environment.GetEnvironmentVariable(sherpa ? "FAKE_STDERR_SHERPA" : "FAKE_STDERR_SMALL") ?? "fetch: boom");',
    "      return 1;",
    "    }",
    '    Console.Error.WriteLine("fake: " + joined);',
    "    return 1;",
    "  }",
    "}",
  ].join("\n");
  fs.writeFileSync(path.join(dir, "fake.cs"), cs);
  const exe = path.join(dir, "layout", "Sigil.exe");
  fs.mkdirSync(path.dirname(exe), { recursive: true });
  const r = spawnSync(
    "powershell.exe",
    ["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", `Add-Type -TypeDefinition (Get-Content -Raw -LiteralPath '${path.join(dir, "fake.cs")}') -OutputType ConsoleApplication -OutputAssembly '${exe}'`],
    { encoding: "utf8" },
  );
  expect(r.stderr + r.stdout).toBe("");
  expect(fs.existsSync(exe)).toBe(true);
  return path.dirname(exe);
}

/** A repository stand-in for -RepoRoot: an empty catalog and one source file, dated `ageSec` seconds from now (negative: past). */
function buildFakeRepo(dir, name, ageSec) {
  const root = path.join(dir, name);
  fs.mkdirSync(path.join(root, "src-tauri", "model-catalog"), { recursive: true });
  fs.mkdirSync(path.join(root, "src"), { recursive: true });
  const catalog = path.join(root, "src-tauri", "model-catalog", "x.json");
  fs.writeFileSync(catalog, JSON.stringify({ entries: [] }));
  const src = path.join(root, "src", "new.ts");
  fs.writeFileSync(src, "x");
  // Everything is older than the build, except the one source file, which is `ageSec` away from now.
  const past = Date.now() / 1000 - 7200;
  fs.utimesSync(catalog, past, past);
  const t = Date.now() / 1000 + ageSec;
  fs.utimesSync(src, t, t);
  return root;
}

describe.runIf(process.platform === "win32")("packaged-smoke-all.ps1 end to end against a fake Sigil.exe (Windows PowerShell 5.1)", () => {
  const NETWORK = "fetch: could not download model.bin (huggingface: cannot connect (dns error: No such host is known. (os error 11001)))";
  const MISMATCH = "fetch: could not download model.bin (huggingface: the file does not match the catalog (sha256 aaaaaaaaaaaa expected, got sha256 bbbbbbbbbbbb))";
  const NOT_FOUND = 'fetch: could not download sherpa.zip (github-release: HTTP 404 Not Found content-type=text/html body_preview="")';

  function runSmoke(dir, layout, repo, extra, env) {
    const out = path.join(dir, "out-" + Math.random().toString(36).slice(2, 8));
    const r = spawnSync(
      "powershell.exe",
      ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path.join(RELEASE, "packaged-smoke-all.ps1"), "-Mode", "Layout", "-LayoutDir", layout, "-RepoRoot", repo, "-ResultsDir", out, "-SkipGui", ...extra],
      { encoding: "utf8", env: { ...process.env, ...env }, timeout: 150_000 },
    );
    let result = null;
    const file = path.join(out, "smoke-result.json");
    if (fs.existsSync(file)) result = JSON.parse(fs.readFileSync(file, "utf8").replace(/^﻿/, ""));
    const status = (id) => result?.checks.find((c) => c.id === id);
    return { r, result, status };
  }

  it("skips only when no host can be reached, fails on a wrong hash or an HTTP error, and fails a skip that -RequireOnline asked for", () => {
    withTemp((dir) => {
      const layout = buildFakeSigil(dir);
      const repoOld = buildFakeRepo(dir, "repo-old", -3600);
      const repoNewer = buildFakeRepo(dir, "repo-newer", +3600);

      // No host reachable, nothing required: both downloads are skips, and the build is the tree.
      let run = runSmoke(dir, layout, repoOld, ["-Online"], { FAKE_STDERR_SMALL: NETWORK, FAKE_STDERR_SHERPA: NETWORK });
      expect(run.status("online-fetch")?.status, run.r.stdout + run.r.stderr).toBe("skip");
      expect(run.status("online-sherpa")?.status, run.r.stdout).toBe("skip");
      expect(run.status("online-fetch")?.detail).toContain("no host could be reached");
      expect(run.status("build-fresh")?.status, run.r.stdout).toBe("pass");

      // The same, but the release gate asked for the online tier (which implies -Online), and a source is newer than the build.
      run = runSmoke(dir, layout, repoNewer, ["-RequireOnline"], { FAKE_STDERR_SMALL: NETWORK, FAKE_STDERR_SHERPA: NETWORK });
      expect(run.status("online-fetch")?.status, run.r.stdout + run.r.stderr).toBe("fail");
      expect(run.status("online-fetch")?.detail).toContain("-RequireOnline requires it");
      expect(run.status("online-sherpa")?.status, run.r.stdout).toBe("fail");
      expect(run.r.status).toBe(1);
      expect(run.status("build-fresh")?.status, run.r.stdout).toBe("skip");
      expect(run.status("build-fresh")?.detail).toMatch(/1 source file\(s\) are newer than Sigil\.exe.*src[\\/]new\.ts/);

      // A host that answers with a hash the catalog does not know, or a 404, is a failure without any switch.
      run = runSmoke(dir, layout, repoOld, ["-Online"], { FAKE_STDERR_SMALL: MISMATCH, FAKE_STDERR_SHERPA: NOT_FOUND });
      expect(run.status("online-fetch")?.status, run.r.stdout + run.r.stderr).toBe("fail");
      expect(run.status("online-fetch")?.detail).toContain("does not match the catalog");
      expect(run.status("online-sherpa")?.status, run.r.stdout).toBe("fail");
      expect(run.status("online-sherpa")?.detail).toContain("HTTP 404");
    });
  }, 240_000);

  // Windows Sandbox, 0.5.0: Sigil.exe died with 0xC0000135 and printed nothing; the harness threw a null-reference on the empty
  // stderr instead of naming the exit code.
  it("reports an app that dies silently with its exit code in hex, not a null reference", () => {
    withTemp((dir) => {
      const layout = buildFakeSigil(dir);
      const repo = buildFakeRepo(dir, "repo", -3600);
      const run = runSmoke(dir, layout, repo, [], { FAKE_EXIT: "-1073741515" });
      for (const id of ["sysinfo", "catalog", "ep-probe"]) {
        const c = run.status(id);
        expect(c?.status, run.r.stdout + run.r.stderr).toBe("fail");
        expect(c?.detail, id).toContain("0xC0000135");
        expect(c?.detail, id).toContain("STATUS_DLL_NOT_FOUND");
        expect(c?.detail, id).not.toContain("null-valued");
      }
      // A plain folder only gets a note about the Visual C++ runtime; the ONNX Runtime providers are still required.
      expect(run.status("layout")?.status).toBe("fail");
      expect(run.status("layout")?.detail).toContain("not next to Sigil.exe");
    });
  }, 240_000);

  it("refuses contradictory switches with exit code 2 before it does anything", () => {
    withTemp((dir) => {
      const layout = path.join(dir, "empty");
      fs.mkdirSync(layout);
      const out = path.join(dir, "out");
      for (const args of [
        ["-RequireGui", "-SkipGui"],
        ["-RequireOnline", "-SkipHeadless"],
      ]) {
        const r = spawnSync(
          "powershell.exe",
          ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path.join(RELEASE, "packaged-smoke-all.ps1"), "-Mode", "Layout", "-LayoutDir", layout, "-ResultsDir", out, ...args],
          { encoding: "utf8", timeout: 60_000 },
        );
        expect(r.status, args.join(" ")).toBe(2);
      }
    });
  });
});

describe("release.yml", () => {
  const stepNames = [...WORKFLOW.matchAll(/^ {6}- (?:name|uses): (.+)$/gm)].map((m) => m[1].trim());
  const index = (needle) => {
    const i = stepNames.findIndex((n) => n.includes(needle));
    expect(i, `step "${needle}"`).toBeGreaterThanOrEqual(0);
    return i;
  };

  it("checks the catalog pins before it compiles, and verifies the bundle before the smoke and the release", () => {
    expect(index("Catalog pins")).toBeLessThan(index("compile"));
    expect(index("compile")).toBeLessThan(index("Verify bundle"));
    expect(index("Verify bundle")).toBeLessThan(index("Packaged smoke"));
    expect(index("Packaged smoke")).toBeLessThan(index("create release"));
    expect(WORKFLOW).toContain("node scripts/release/check-catalog-pins.mjs");
  });

  it("smoke-tests the installer on the hosted runner (no GPU) and keeps the result", () => {
    expect(WORKFLOW).toMatch(/packaged-smoke-all\.ps1 -Mode Installed -ExpectNoGpu/);
    expect(WORKFLOW).toContain("actions/upload-artifact@v4");
  });

  it("does not let the release gate pass on a skipped download or app check", () => {
    // A skip leaves the exit code at 0: without these switches a missing WebView2 runtime or an unreachable host would pass.
    const line = WORKFLOW.split("\n").find((l) => l.includes("packaged-smoke-all.ps1 -Mode"));
    expect(line, "the smoke step").toBeTruthy();
    expect(line).toContain("-RequireOnline");
    expect(line).toContain("-RequireGui");
    expect(line).not.toMatch(/-SkipGui|-SkipHeadless/);
  });

  it("only drafts the release: publishing stays with the user", () => {
    expect(WORKFLOW).toMatch(/draft: true/);
    expect(WORKFLOW).not.toMatch(/draft: false/);
  });

  it("is manually dispatched and still builds portable ggml", () => {
    expect(WORKFLOW).toMatch(/^on:\n {2}workflow_dispatch:/m);
    expect(WORKFLOW).toContain('GGML_NATIVE: "OFF"');
    expect(WORKFLOW).toContain('GGML_AVX2: "ON"');
  });
});

describe("scripts/release/README.md", () => {
  const README = read(RELEASE, "README.md");

  it("mentions every script of the directory", () => {
    const scripts = fs
      .readdirSync(RELEASE)
      .filter((n) => /\.(ps1|mjs|json|txt)$/.test(n) && !n.endsWith(".test.mjs"));
    expect(scripts.length).toBeGreaterThanOrEqual(6);
    for (const s of scripts) expect(README, s).toContain(s);
  });

  it("states the whisper-rs decision", () => {
    expect(README).toMatch(/## Why whisper-rs stays/);
  });
});

describe("the Visual C++ runtime is app-local and every place that lists it agrees", () => {
  const STAGE = read(RELEASE, "stage-native-dlls.mjs");
  const VERIFY = read(RELEASE, "verify-bundle.ps1");
  const IMPORTS = read(RELEASE, "expected-imports.txt");
  const names = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll"];

  it("is staged, verified in both installers, listed as expected imports and required by the smoke's layout check", () => {
    for (const n of names) {
      expect(STAGE, n).toContain(`"${n}"`);
      expect(VERIFY, n).toContain(`'${n}'`);
      expect(IMPORTS.toLowerCase(), n).toContain(n);
      expect(SMOKE, n).toContain(`'${n}'`);
    }
    // verify-bundle: staged copy compared, MSI and NSIS layouts both go through Test-InstalledLayout.
    expect(VERIFY).toContain("$bundledDlls");
    expect(VERIFY).toMatch(/Test-InstalledLayout "MSI/);
    expect(VERIFY).toMatch(/Test-InstalledLayout "NSIS/);
    // The overlay maps the staging directory (where the DLLs go) to the install root; the base config lists no generated file.
    const overlay = JSON.parse(read(RELEASE, "tauri.release.conf.json"));
    expect(overlay.bundle.resources["target/native-bundle"]).toBe(".");
    expect(overlay.build.beforeBundleCommand).toContain("stage-native-dlls.mjs --stage");
    const base = read(ROOT, "src-tauri", "tauri.conf.json");
    expect(base).not.toMatch(/native-bundle|vcruntime|msvcp/);
  });

  it("is staged by the release workflow's compile through the same overlay, found before the compile, and zipped into the portable build", () => {
    expect(WORKFLOW).toContain("--config scripts/release/tauri.release.conf.json");
    expect(WORKFLOW).toContain("stage-native-dlls.mjs --check-vc");
    expect(WORKFLOW.indexOf("run: node scripts/release/stage-native-dlls.mjs --check-vc")).toBeLessThan(WORKFLOW.indexOf("run: pnpm tauri build --config"));
    expect(WORKFLOW).toContain("src-tauri/target/native-bundle");
  });
});
