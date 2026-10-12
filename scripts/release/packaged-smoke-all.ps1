<#
.SYNOPSIS
  Packaged smoke test of a Sigil Windows release: the installed (or extracted) app starts on a PC with
  no GPU and no model or runtime pack, and every offline entry point works.

.DESCRIPTION
  Runs against the artifacts `pnpm tauri:release` produces and answers the release question "does a
  friend's clean PC start Sigil?". It needs nothing but Windows PowerShell 5.1 or PowerShell 7 and the
  installer, so the same file runs on the dev PC, in Windows Sandbox / a VM, and on a CI runner.

  Modes (-Mode)
    Extract    default. Unpacks the NSIS installer with 7-Zip into the results folder and runs it from there
               against a scratch Windows profile (USERPROFILE / APPDATA / LOCALAPPDATA redirected). Nothing is
               installed; the real %APPDATA%\com.sigil.app is never touched (the script checks that).
    Portable   the same with the portable zip (Sigil.exe plus the provider DLLs).
    Layout     the same with a folder that already holds Sigil.exe (-LayoutDir).
    Installed  runs the NSIS installer silently (per-user), tests the installed app, then uninstalls it. Only
               on a disposable machine: a GitHub-hosted runner (GITHUB_ACTIONS=true and RUNNER_ENVIRONMENT=
               github-hosted; a bare CI variable proves nothing), Windows Sandbox (user WDAGUtilityAccount).
               Elsewhere, a VM included, it refuses unless -AllowInstall is given, because installing writes
               shortcuts and an uninstall entry to the user profile.
    Sandbox    builds a Windows Sandbox configuration (virtual GPU off, so the guest has no GPU), runs this
               script inside it in -Mode Installed and reads back the result. Only when Windows Sandbox is
               already enabled on this PC (this script never enables Windows features); otherwise it prints
               the manual VM steps and exits with code 3.

  Checks (see the result file for the exact list)
    build-fresh   with the repository at hand: no source file that goes into the app is newer than the build.
                  A stale build is a skip (so -Strict fails it): the other checks would vouch for another tree.
    layout        Sigil.exe and the three ONNX Runtime provider DLLs sit together
    sysinfo       `--headless sysinfo` reports CPU, memory and GPUs; with -ExpectNoGpu there is no hardware GPU
    catalog       `--headless fetch --list` lists the whole embedded catalog and nothing is installed
    ep-probe      `--headless ep-probe` finishes, finds the provider DLLs next to the exe, no CUDA with -ExpectNoGpu
    gui-*         the app starts with a fresh profile: web server answers, the page mounts (no "Application
                  failed to start"), the host services boot, Models & Runtimes lists every entry as not
                  installed, the log has no errors, no download or pack appeared, closing the window ends
                  the process. The app is started with SIGIL_START_MINIMIZED=1 (minimized, never in front of
                  what the person at the PC is doing) and gui-focus proves it: the foreground window is
                  sampled every 100 ms for the first 15 s and is never Sigil's, the window ends minimized and
                  no dialog appears. -ShowWindow starts it the normal way instead.
    online-*      with -Online: a tiny catalog entry installs through the model store, and the sherpa-onnx
                  runtime pack (8 MB) installs and loads next to the static ONNX Runtime. Both verify the
                  catalog pins end to end: a hash or size that disagrees with the host fails the check.
    packs-*       with -PackStore <app data dir of a model store> (read only): the release exe decodes real
                  speech with the default models (Nemotron 560 ms on French, SenseVoice on English)

  A step that cannot run here (no WebView2 runtime, 7-Zip missing, no network) is reported as SKIP, not
  FAIL; -Strict turns every skip into a failure, -RequireGui only the gui-* skips, -RequireOnline only the
  online-* skips (and implies -Online). A download that fails because the host answered with the wrong thing
  (a sha256 or size that disagrees with the catalog, HTTP 4xx / 5xx) is always a FAIL: only a host that
  cannot be reached at all is a skip. The release workflow passes -RequireGui -RequireOnline, so a runner
  without WebView2 or a stale pin stops the release instead of passing silently. The result is written to
  <ResultsDir>\smoke-result.json (schema sigil-packaged-smoke/1).

  Exit code: 0 passed, 1 a check failed, 2 usage error, 3 Windows Sandbox is not available.

.EXAMPLE
  pwsh scripts/release/packaged-smoke-all.ps1                        # dev PC: newest NSIS installer, extracted
  pwsh scripts/release/packaged-smoke-all.ps1 -PackStore "$env:LOCALAPPDATA\sigil-dev-cache\appdata-p7"
  pwsh scripts/release/packaged-smoke-all.ps1 -Mode Sandbox          # clean Windows, no GPU, no packs
  pwsh scripts/release/packaged-smoke-all.ps1 -Mode Installed -ExpectNoGpu -Online -RequireOnline -RequireGui   # GitHub-hosted runner (a VM: add -AllowInstall)
#>
[CmdletBinding()]
param(
  [ValidateSet('Extract', 'Portable', 'Layout', 'Installed', 'Sandbox')][string]$Mode = 'Extract',
  [string]$InstallerPath,
  [string]$ZipPath,
  [string]$LayoutDir,
  [string]$ResultsDir,
  [string]$RepoRoot,
  # Profile of the app under test: Auto = seeded when the PC is not disposable, fresh otherwise.
  [ValidateSet('Auto', 'Seeded', 'Fresh')][string]$ProfileKind = 'Auto',
  [switch]$ExpectNoGpu,
  [switch]$Online,
  [switch]$OnlineModel,
  # A skipped online-* check fails instead (implies -Online): the release gate must not pass without proving the pins.
  [switch]$RequireOnline,
  [string]$PackStore,
  [switch]$SkipGui,
  [switch]$SkipHeadless,
  # A skipped gui-* check (no WebView2 runtime, no DevTools page, ...) fails instead.
  [switch]$RequireGui,
  # Start the app the normal way (window in front) instead of minimized without focus. For a manual run only.
  [switch]$ShowWindow,
  [switch]$AllowInstall,
  [switch]$KeepFiles,
  [switch]$Strict,
  [int]$GuiTimeoutSec = 120,
  # How long closing the main window may take to end the process (the app's own exit guard ends it after 20 s).
  [int]$ExitWaitSec = 35,
  [int]$SandboxTimeoutMin = 25,
  # Internal: the Windows Sandbox guest reports through this folder.
  [switch]$InSandbox
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if ($RequireGui -and $SkipGui) { Write-Host '-RequireGui cannot be combined with -SkipGui' -ForegroundColor Red; exit 2 }
if ($RequireOnline -and $SkipHeadless) { Write-Host '-RequireOnline cannot be combined with -SkipHeadless (the online checks are part of the headless tier)' -ForegroundColor Red; exit 2 }
if ($RequireOnline) { $Online = $true }
$script:started = Get-Date
$script:checks = New-Object System.Collections.ArrayList
$script:notes = New-Object System.Collections.ArrayList

if (-not $RepoRoot) {
  $candidate = Join-Path $PSScriptRoot '..\..'
  if (Test-Path -LiteralPath (Join-Path $candidate 'src-tauri\tauri.conf.json')) { $RepoRoot = (Resolve-Path $candidate).Path }
}
if (-not $ResultsDir) {
  $stamp = $script:started.ToString('yyyyMMdd-HHmmss')
  if ($RepoRoot) { $ResultsDir = Join-Path (Join-Path (Join-Path $RepoRoot 'bench-results') 'release-smoke') $stamp }
  else { $ResultsDir = Join-Path ([System.IO.Path]::GetTempPath()) ("sigil-smoke-" + $stamp) }
}
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
$ResultsDir = (Resolve-Path $ResultsDir).Path

# ------------------------------------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------------------------------------
function Write-Line([string]$Text, [string]$Color = '') {
  if ($Color) { Write-Host $Text -ForegroundColor $Color } else { Write-Host $Text }
}
function Add-Check([string]$Id, [string]$Status, [string]$Detail, [double]$Seconds = 0) {
  # A tier the caller requires (-RequireGui, -RequireOnline) cannot be skipped: a gate must not pass on a check it never ran.
  if ($Status -eq 'skip') {
    $tier = Get-RequiredTier $Id
    if ($tier) { $Status = 'fail'; $Detail = "skipped, but $tier requires it: $Detail" }
  }
  [void]$script:checks.Add([pscustomobject]@{ id = $Id; status = $Status; detail = $Detail; seconds = [math]::Round($Seconds, 1) })
  $color = switch ($Status) { 'pass' { 'Green' } 'fail' { 'Red' } 'skip' { 'Yellow' } default { 'Gray' } }
  $tag = $Status.ToUpper().PadRight(4)
  Write-Line ("[{0}] {1,-22} {2}" -f $tag, $Id, $Detail) $color
}
class SkipCheck : System.Exception { SkipCheck([string]$m) : base($m) {} }
function Skip-Check([string]$Reason) { throw ([SkipCheck]::new($Reason)) }

# Run one check: the block returns the detail text; throw to fail, Skip-Check to skip.
function Invoke-Check([string]$Id, [scriptblock]$Body) {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  try {
    $detail = & $Body
    Add-Check $Id 'pass' ([string]$detail) $sw.Elapsed.TotalSeconds
  } catch [SkipCheck] {
    Add-Check $Id 'skip' $_.Exception.Message $sw.Elapsed.TotalSeconds
  } catch {
    $m = $_.Exception.Message
    if ($_.Exception.GetType().Name -eq 'RuntimeException' -and $m -like 'SKIP:*') { Add-Check $Id 'skip' $m.Substring(5).Trim() $sw.Elapsed.TotalSeconds }
    else { Add-Check $Id 'fail' $m $sw.Elapsed.TotalSeconds }
  }
}
function Add-Note([string]$Text) { [void]$script:notes.Add($Text); Write-Line "       note: $Text" 'DarkGray' }

function Write-Result([int]$ExitCode, [hashtable]$Extra = @{}) {
  $failed = @($script:checks | Where-Object { $_.status -eq 'fail' }).Count
  $skipped = @($script:checks | Where-Object { $_.status -eq 'skip' }).Count
  $passed = @($script:checks | Where-Object { $_.status -eq 'pass' }).Count
  $result = [ordered]@{
    schema     = 'sigil-packaged-smoke/1'
    mode       = $Mode
    started    = $script:started.ToString('o')
    seconds    = [math]::Round(((Get-Date) - $script:started).TotalSeconds, 1)
    machine    = $script:machine
    exe        = $script:exeInfo
    counts     = [ordered]@{ pass = $passed; fail = $failed; skip = $skipped }
    exit_code  = $ExitCode
    checks     = @($script:checks)
    notes      = @($script:notes)
  }
  foreach ($k in $Extra.Keys) { $result[$k] = $Extra[$k] }
  ($result | ConvertTo-Json -Depth 8) | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $ResultsDir 'smoke-result.json')
}

# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------
# The block between the two region lines is run on its own by scripts/release/__tests__/release-files.test.mjs: keep it free
# of anything that needs the rest of the script, except the switches named in Get-RequiredTier.
#region testable-helpers
# Is this PC one that may be changed (a per-user install, an uninstall entry, a start menu shortcut)? A GitHub-hosted runner
# is a fresh VM per job and Windows Sandbox runs as WDAGUtilityAccount. A bare CI=true proves nothing (plenty of developer
# shells set it), and a self-hosted runner can be someone's own PC, so neither counts; -AllowInstall says it for a VM.
function Test-Disposable {
  if ($env:GITHUB_ACTIONS -eq 'true' -and $env:RUNNER_ENVIRONMENT -eq 'github-hosted') { return $true }
  if ($env:USERNAME -eq 'WDAGUtilityAccount') { return $true }
  return $false
}

# Can a listener bind this loopback port right now? (Sigil binds 127.0.0.1 with lanAccess.mode = localhost, which the
# seeded profile sets; its own start-up check shows a modal dialog when the bind fails, and a dialog is what this script
# must never cause. Only loopback is tried: a wildcard listener makes Windows Firewall ask the person at the PC.)
function Test-PortFree([int]$Port) {
  $l = $null
  try {
    $l = New-Object System.Net.Sockets.TcpListener ([System.Net.IPAddress]::Loopback, $Port)
    $l.Start()
    return $true
  } catch {
    return $false
  } finally {
    if ($l) { try { $l.Stop() } catch { } }
  }
}

# A loopback port nobody listens on (the OS picks it, so it is never in use and never inside a range other programs
# are known to use), different from the ones in -Exclude (a second call must not hand out the first call's port again).
function Get-FreePort([int[]]$Exclude = @()) {
  for ($i = 0; $i -lt 50; $i++) {
    $l = New-Object System.Net.Sockets.TcpListener ([System.Net.IPAddress]::Loopback, 0)
    $l.Start()
    $p = $l.LocalEndpoint.Port
    $l.Stop()
    if ($Exclude -contains $p) { continue }
    return $p
  }
  throw 'no free TCP port on 127.0.0.1'
}

# Why did `Sigil.exe --headless fetch` fail? kind 'network': no host could be reached (DNS, connect, timeout, rate
# limit): the only outcome a smoke may skip. kind 'pin': a host answered and what it served is wrong (a sha256 or size
# that disagrees with the catalog, HTTP 4xx / 5xx, a different version), or the message is not recognized. Both are
# failures: an unknown message must never turn a stale pin into a green run. The wording comes from
# src-tauri/src/services/model_store/download.rs ("could not download <file> (<source>: <reason>; ...)").
function Get-FetchFailureKind([string]$Text, [bool]$TimedOut = $false) {
  $t = [string]$Text
  $hard = @(
    'does not match the catalog', 'catalog expects', 'different version', 'sha256 [0-9a-f]{6}', 'unexpected HTTP',
    'HTTP (?:40[0-79]|41\d|42[0-8]|4[3-9]\d|5\d\d)(?!\d)', 'has no download source'
  )
  foreach ($p in $hard) {
    $m = [regex]::Match($t, $p, 'IgnoreCase')
    if ($m.Success) { return [pscustomobject]@{ Kind = 'pin'; Why = $m.Value } }
  }
  if ($TimedOut) { return [pscustomobject]@{ Kind = 'network'; Why = 'timed out' } }
  $soft = @(
    'cannot connect', 'no response within', 'timed out', 'connection lost', 'dns error', 'error sending request',
    'HTTP 429', 'HTTP 408', 'connection (?:was )?(?:reset|refused|closed|aborted)', 'network is unreachable', 'no such host'
  )
  foreach ($p in $soft) {
    $m = [regex]::Match($t, $p, 'IgnoreCase')
    if ($m.Success) { return [pscustomobject]@{ Kind = 'network'; Why = $m.Value } }
  }
  return [pscustomobject]@{ Kind = 'pin'; Why = 'the failure is not a recognized network error' }
}

# Which required tier (-RequireGui, -RequireOnline) a check id belongs to ('' when none): a skip there is a failure.
function Get-RequiredTier([string]$Id) {
  if ($RequireGui -and $Id -like 'gui*' -and -not ($ShowWindow -and $Id -eq 'gui-focus')) { return '-RequireGui' }
  if ($RequireOnline -and $Id -like 'online*') { return '-RequireOnline' }
  return ''
}

# The source files that go into the app and are newer than $Stamp (the build), newest first. Tests and docs that are not
# shipped do not count: they cannot make the packaged app differ from the tree.
function Get-BuildInputsNewerThan([string]$Root, [datetime]$Stamp) {
  $dirs = @('src', 'public', 'src-tauri\src', 'src-tauri\model-catalog', 'src-tauri\capabilities', 'src-tauri\plugin-commands', 'src-tauri\resources', 'scripts\neural-sidecar')
  $single = @('index.html', 'package.json', 'pnpm-lock.yaml', 'vite.config.ts', 'src-tauri\Cargo.toml', 'src-tauri\Cargo.lock', 'src-tauri\build.rs', 'src-tauri\tauri.conf.json')
  $found = New-Object System.Collections.ArrayList
  foreach ($d in $dirs) {
    $p = Join-Path $Root $d
    if (-not (Test-Path -LiteralPath $p)) { continue }
    foreach ($f in @(Get-ChildItem -LiteralPath $p -Recurse -File -ErrorAction SilentlyContinue)) {
      if ($f.LastWriteTimeUtc -le $Stamp) { continue }
      if ($f.FullName -match '[\\/]__tests__[\\/]' -or $f.Name -match '\.(test|spec)\.[jt]sx?$' -or $f.Name -eq 'tests.rs') { continue }
      [void]$found.Add($f)
    }
  }
  foreach ($s in $single) {
    $p = Join-Path $Root $s
    if (Test-Path -LiteralPath $p) { $f = Get-Item -LiteralPath $p; if ($f.LastWriteTimeUtc -gt $Stamp) { [void]$found.Add($f) } }
  }
  return @($found | Sort-Object LastWriteTimeUtc -Descending)
}
#endregion testable-helpers

# A `--headless fetch` result: return when it worked, skip when no host could be reached, fail when a host answered wrongly.
function Assert-FetchSucceeded($Result, [string]$What) {
  if (-not $Result.TimedOut -and $Result.ExitCode -eq 0) { return }
  $lines = @($Result.Stderr -split "`r?`n" | Where-Object { $_.Trim() })
  $line = @($lines | Where-Object { $_ -match '^\s*fetch:' } | Select-Object -Last 1)
  $text = if ($line.Count -gt 0) { [string]$line[0] } elseif ($lines.Count -gt 0) { [string]$lines[-1] } else { '' }
  if ($text.Length -gt 400) { $text = $text.Substring(0, 400) + '...' }
  $kind = Get-FetchFailureKind $Result.Stderr $Result.TimedOut
  if ($kind.Kind -eq 'network') {
    Skip-Check ("{0}: no host could be reached ({1}); exit {2}: {3}" -f $What, $kind.Why, $Result.ExitCode, $text)
  }
  throw ("{0} failed ({1}); exit {2}: {3}" -f $What, $kind.Why, $Result.ExitCode, $text)
}

function Get-WebView2Version {
  $id = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
  foreach ($root in @('HKLM:\SOFTWARE\WOW6432Node', 'HKLM:\SOFTWARE', 'HKCU:\Software')) {
    $key = "$root\Microsoft\EdgeUpdate\Clients\$id"
    $v = (Get-ItemProperty -LiteralPath $key -ErrorAction SilentlyContinue).pv
    if ($v -and $v -ne '0.0.0.0') { return [string]$v }
  }
  return $null
}

# Top-level windows of a process: list them, minimize them, ask them to close; the foreground-window sampler of the
# focus check; a byte search in a big file. C# 5 only (Windows PowerShell 5.1 compiles it).
function Initialize-WindowHelper {
  if ('SmokeWindows' -as [type]) { return }
  Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
public static class SmokeWindows {
  delegate bool EnumProc(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc p, IntPtr l);
  [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] static extern bool IsIconic(IntPtr h);
  [DllImport("user32.dll")] static extern bool IsHungAppWindow(IntPtr h);
  [DllImport("user32.dll")] static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l);
  [DllImport("user32.dll")] static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] static extern int GetWindowLong(IntPtr h, int index);
  [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr h, out RECT r);
  [StructLayout(LayoutKind.Sequential)] struct RECT { public int Left, Top, Right, Bottom; }
  [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetClassName(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetWindowText(IntPtr h, StringBuilder s, int n);

  public static int ForEachWindow(int pid, int action) {
    int n = 0;
    EnumWindows((h, l) => {
      uint p; GetWindowThreadProcessId(h, out p);
      if (p == (uint)pid && IsWindowVisible(h)) {
        if (action == 0) ShowWindow(h, 6); else PostMessage(h, 0x0010, IntPtr.Zero, IntPtr.Zero);
        n++;
      }
      return true;
    }, IntPtr.Zero);
    return n;
  }
  public static int Minimize(int pid) { return ForEachWindow(pid, 0); }

  // Ask the process's window with exactly this title to close (WM_CLOSE) and nothing else. Not Process.CloseMainWindow and
  // not "every visible window": Sigil's event loop (tao) owns an invisible tool window (class "Tao Thread Event Target", no
  // title, 16 x 16) that counts as visible and has no owner, so it is what CloseMainWindow() posts to once the real window is
  // minimized (a minimized window sits at the bottom of the z-order). WM_CLOSE there destroys the event loop's own message
  // window: the real window then closes, but the process lingers until the app's exit guard ends it (measured 2026-10-10).
  public static int CloseTitled(int pid, string title) {
    int n = 0;
    EnumWindows((h, l) => {
      uint p; GetWindowThreadProcessId(h, out p);
      if (p != (uint)pid) return true;
      StringBuilder t = new StringBuilder(256); GetWindowText(h, t, 256);
      if (t.ToString() == title && PostMessage(h, 0x0010, IntPtr.Zero, IntPtr.Zero)) n++;
      return true;
    }, IntPtr.Zero);
    return n;
  }

  // One line per top-level window of the process: "handle|visible|minimized|unresponsive|exstyle|width|height|class|title".
  public static string[] Describe(int pid) {
    List<string> rows = new List<string>();
    EnumWindows((h, l) => {
      uint p; GetWindowThreadProcessId(h, out p);
      if (p == (uint)pid) {
        StringBuilder cls = new StringBuilder(256); GetClassName(h, cls, 256);
        StringBuilder title = new StringBuilder(256); GetWindowText(h, title, 256);
        RECT r; GetWindowRect(h, out r);
        rows.Add(h.ToInt64() + "|" + (IsWindowVisible(h) ? 1 : 0) + "|" + (IsIconic(h) ? 1 : 0) + "|" + (IsHungAppWindow(h) ? 1 : 0)
          + "|" + GetWindowLong(h, -20) + "|" + (r.Right - r.Left) + "|" + (r.Bottom - r.Top) + "|" + cls + "|" + title);
      }
      return true;
    }, IntPtr.Zero);
    return rows.ToArray();
  }

  public static int ForegroundPid() {
    IntPtr h = GetForegroundWindow();
    if (h == IntPtr.Zero) return 0;
    uint p; GetWindowThreadProcessId(h, out p);
    return (int)p;
  }

  // True when the file holds the ASCII text (the exe of a build that knows an environment switch).
  public static bool FileContains(string path, string text) {
    byte[] needle = Encoding.ASCII.GetBytes(text);
    byte[] buf = new byte[4 * 1024 * 1024 + needle.Length];
    using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite)) {
      int keep = 0;
      while (true) {
        int n = fs.Read(buf, keep, buf.Length - keep);
        if (n <= 0) return false;
        int total = keep + n;
        for (int i = 0; i + needle.Length <= total; i++) {
          if (buf[i] != needle[0]) continue;
          int j = 1;
          while (j < needle.Length && buf[i + j] == needle[j]) j++;
          if (j == needle.Length) return true;
        }
        keep = Math.Min(needle.Length - 1, total);
        Buffer.BlockCopy(buf, total - keep, buf, 0, keep);
      }
    }
  }
}

// Samples the foreground window while the app starts: how often it belonged to the process under test.
public sealed class SmokeFocusSampler {
  readonly int pid; readonly int durationMs; readonly int intervalMs;
  Thread thread;
  public volatile int Samples;
  public volatile int Hits;
  public volatile int FirstHitMs = -1;
  [DllImport("user32.dll")] static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  public SmokeFocusSampler(int pid, int durationMs, int intervalMs) { this.pid = pid; this.durationMs = durationMs; this.intervalMs = intervalMs; }
  public void Start() { thread = new Thread(Run); thread.IsBackground = true; thread.Start(); }
  public void Join() { if (thread != null) thread.Join(); }
  void Run() {
    Stopwatch sw = Stopwatch.StartNew();
    while (sw.ElapsedMilliseconds < durationMs) {
      IntPtr h = GetForegroundWindow();
      if (h != IntPtr.Zero) {
        uint p; GetWindowThreadProcessId(h, out p);
        if (p == (uint)pid) { Hits++; if (FirstHitMs < 0) FirstHitMs = (int)sw.ElapsedMilliseconds; }
      }
      Samples++;
      Thread.Sleep(intervalMs);
    }
  }
}
'@
}

# The windows of a process as objects (see SmokeWindows.Describe).
function Get-AppWindows([int]$ProcessId) {
  Initialize-WindowHelper
  foreach ($row in [SmokeWindows]::Describe($ProcessId)) {
    $f = $row -split '\|', 9
    $ex = [int]$f[4]
    [pscustomobject]@{
      Handle = $f[0]; Visible = ($f[1] -eq '1'); Minimized = ($f[2] -eq '1'); Unresponsive = ($f[3] -eq '1')
      ToolWindow = (($ex -band 0x80) -ne 0); Width = [int]$f[5]; Height = [int]$f[6]; Class = $f[7]; Title = $f[8]
    }
  }
}

# Does this Sigil.exe know SIGIL_START_MINIMIZED (added after the P7 release candidate was first built)?
function Test-ExeKnowsStartMinimized([string]$ExePath) {
  Initialize-WindowHelper
  return [SmokeWindows]::FileContains($ExePath, 'SIGIL_START_MINIMIZED')
}

function Quote-Arg([string]$a) { if ($a -match '[\s"]') { return '"' + ($a -replace '"', '\"') + '"' } else { return $a } }

# Runs an exe to completion with captured output. $EnvVars are set for the child only.
function Invoke-Exe([string]$Exe, [string[]]$Arguments, [int]$TimeoutSec = 120, [hashtable]$EnvVars = @{}) {
  $outFile = Join-Path $ResultsDir ("proc-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
  $saved = @{}
  foreach ($k in $EnvVars.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k, 'Process'); [Environment]::SetEnvironmentVariable($k, $EnvVars[$k], 'Process') }
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $timedOut = $false
  try {
    $argLine = ($Arguments | ForEach-Object { Quote-Arg $_ }) -join ' '
    $p = Start-Process -FilePath $Exe -ArgumentList $argLine -RedirectStandardOutput "$outFile.out" -RedirectStandardError "$outFile.err" -PassThru -NoNewWindow
    $null = $p.Handle
    if (-not $p.WaitForExit($TimeoutSec * 1000)) { $timedOut = $true; try { $p.Kill() } catch {} }
    $p.WaitForExit()
    $code = if ($timedOut) { -1 } else { $p.ExitCode }
  } finally {
    foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process') }
  }
  $stdout = if (Test-Path -LiteralPath "$outFile.out") { Get-Content -Raw -LiteralPath "$outFile.out" } else { '' }
  $stderr = if (Test-Path -LiteralPath "$outFile.err") { Get-Content -Raw -LiteralPath "$outFile.err" } else { '' }
  Remove-Item -LiteralPath "$outFile.out", "$outFile.err" -Force -ErrorAction SilentlyContinue
  return [pscustomobject]@{ ExitCode = $code; Stdout = [string]$stdout; Stderr = [string]$stderr; Seconds = $sw.Elapsed.TotalSeconds; TimedOut = $timedOut }
}

function Read-JsonFile([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { throw "no report was written ($Path)" }
  return (Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json)
}

# One Chrome DevTools Protocol call (Runtime.evaluate) over a WebSocket; works in Windows PowerShell 5.1 and 7.
function Invoke-CdpEvaluate([string]$WsUrl, [string]$Expression, [int]$TimeoutSec = 30) {
  $ws = New-Object System.Net.WebSockets.ClientWebSocket
  $cts = New-Object System.Threading.CancellationTokenSource ([TimeSpan]::FromSeconds($TimeoutSec))
  try {
    $ws.ConnectAsync([Uri]$WsUrl, $cts.Token).GetAwaiter().GetResult()
    $msg = @{ id = 1; method = 'Runtime.evaluate'; params = @{ expression = $Expression; awaitPromise = $true; returnByValue = $true } } | ConvertTo-Json -Depth 6 -Compress
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($msg)
    $ws.SendAsync([System.ArraySegment[byte]]::new($bytes), [System.Net.WebSockets.WebSocketMessageType]::Text, $true, $cts.Token).GetAwaiter().GetResult()
    $buf = New-Object byte[] 65536
    while ($true) {
      $ms = New-Object System.IO.MemoryStream
      do {
        $r = $ws.ReceiveAsync([System.ArraySegment[byte]]::new($buf), $cts.Token).GetAwaiter().GetResult()
        if ($r.MessageType -eq [System.Net.WebSockets.WebSocketMessageType]::Close) { throw 'the page closed the DevTools connection' }
        $ms.Write($buf, 0, $r.Count)
      } while (-not $r.EndOfMessage)
      $o = [System.Text.Encoding]::UTF8.GetString($ms.ToArray()) | ConvertFrom-Json
      if ($o.PSObject.Properties.Name -contains 'id' -and $o.id -eq 1) {
        if ($o.PSObject.Properties.Name -contains 'error') { throw ("DevTools: " + $o.error.message) }
        $res = $o.result
        if ($res.PSObject.Properties.Name -contains 'exceptionDetails') { throw ("page script failed: " + $res.exceptionDetails.text + " " + $res.exceptionDetails.exception.description) }
        return $res.result.value
      }
    }
  } finally {
    try { if ($ws.State -eq 'Open') { $ws.CloseAsync([System.Net.WebSockets.WebSocketCloseStatus]::NormalClosure, 'done', [System.Threading.CancellationToken]::None).GetAwaiter().GetResult() } } catch {}
    $ws.Dispose()
    $cts.Dispose()
  }
}

# ------------------------------------------------------------------------------------------------
# Windows Sandbox
# ------------------------------------------------------------------------------------------------
function Test-SandboxAvailable {
  # Never enables anything: the feature is "already enabled" when its launcher and service exist.
  $exe = Join-Path $env:SystemRoot 'System32\WindowsSandbox.exe'
  if (-not (Test-Path -LiteralPath $exe)) { return $false }
  $svc = Get-Service -Name 'CmService' -ErrorAction SilentlyContinue
  return [bool]$svc
}

function Write-ManualVmSteps {
  Write-Line ''
  Write-Line 'Manual clean-machine smoke (Windows Sandbox is not enabled on this PC):' 'Yellow'
  Write-Line '  1. Use a Windows 10/11 x64 virtual machine (Hyper-V, VMware, VirtualBox) with no GPU passthrough and no Sigil data;'
  Write-Line '     a fresh snapshot is best. Copy Sigil_<version>_x64-setup.exe and this script into it.'
  Write-Line '  2. In an elevated or normal Windows PowerShell inside the VM:'
  Write-Line '       powershell -ExecutionPolicy Bypass -File .\packaged-smoke-all.ps1 -Mode Installed -AllowInstall -InstallerPath .\Sigil_<version>_x64-setup.exe -ExpectNoGpu -Online -ResultsDir C:\smoke-out'
  Write-Line '     (add -OnlineModel to also download SenseVoice, 163 MB, and decode English speech)'
  Write-Line '  3. Read C:\smoke-out\smoke-result.json; every check should be "pass" (a GUI check may "skip" without the WebView2 runtime).'
  Write-Line '  4. By hand, once: start Sigil, open Settings > GPU & diagnostics (processor / DirectML with the reason, Test GPU acceleration'
  Write-Line '     completes), Models & Runtimes (every entry "not installed"), STT backend "Built-in (runs on your PC)", English, Start:'
  Write-Line '     it asks before downloading and shows the size. See docs/RELEASE_AND_ENGINEERING.md, "Clean machine: no GPU, no packs".'
}

function Invoke-SandboxRun {
  if (-not (Test-SandboxAvailable)) {
    Add-Check 'sandbox' 'skip' 'Windows Sandbox is not enabled on this PC (this script never enables Windows features)'
    Write-ManualVmSteps
    Write-Result 3 @{ manual_steps = 'scripts/release/README.md#clean-machine-smoke' }
    exit 3
  }
  if (-not $InstallerPath -and $RepoRoot) {
    $dir = Join-Path $RepoRoot 'src-tauri\target\release\bundle\nsis'
    $newest = Get-ChildItem -LiteralPath $dir -Filter '*-setup.exe' -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($newest) { $InstallerPath = $newest.FullName }
  }
  if (-not $InstallerPath -or -not (Test-Path -LiteralPath $InstallerPath)) { throw 'no NSIS installer found: pass -InstallerPath' }
  $InstallerPath = (Resolve-Path $InstallerPath).Path
  $inDir = Join-Path $ResultsDir 'sandbox-in'
  $outDir = Join-Path $ResultsDir 'sandbox-out'
  New-Item -ItemType Directory -Force -Path $inDir, $outDir | Out-Null
  Copy-Item -LiteralPath $InstallerPath -Destination $inDir -Force
  Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $inDir 'packaged-smoke-all.ps1') -Force
  $setup = Split-Path -Leaf $InstallerPath
  $flags = '-Mode Installed -InstallerPath C:\smoke\in\' + $setup + ' -ResultsDir C:\smoke\out -ExpectNoGpu -InSandbox'
  if ($Online) { $flags += ' -Online' }
  if ($OnlineModel) { $flags += ' -OnlineModel' }
  if ($RequireOnline) { $flags += ' -RequireOnline' }
  if ($RequireGui) { $flags += ' -RequireGui' }
  if ($Strict) { $flags += ' -Strict' }
  $net = if ($Online -or $OnlineModel) { 'Default' } else { 'Disable' }
  $logon = 'cmd.exe /c "powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\smoke\in\packaged-smoke-all.ps1 ' + $flags + ' > C:\smoke\out\console.txt 2>&1 & shutdown.exe /s /t 5"'
  $xmlEsc = { param($s) [System.Security.SecurityElement]::Escape($s) }
  $wsb = @"
<Configuration>
  <vGPU>Disable</vGPU>
  <Networking>$net</Networking>
  <AudioInput>Disable</AudioInput>
  <VideoInput>Disable</VideoInput>
  <PrinterRedirection>Disable</PrinterRedirection>
  <ClipboardRedirection>Disable</ClipboardRedirection>
  <MemoryInMB>4096</MemoryInMB>
  <MappedFolders>
    <MappedFolder><HostFolder>$(& $xmlEsc $inDir)</HostFolder><SandboxFolder>C:\smoke\in</SandboxFolder><ReadOnly>true</ReadOnly></MappedFolder>
    <MappedFolder><HostFolder>$(& $xmlEsc $outDir)</HostFolder><SandboxFolder>C:\smoke\out</SandboxFolder><ReadOnly>false</ReadOnly></MappedFolder>
  </MappedFolders>
  <LogonCommand><Command>$(& $xmlEsc $logon)</Command></LogonCommand>
</Configuration>
"@
  $wsbPath = Join-Path $ResultsDir 'smoke.wsb'
  Set-Content -LiteralPath $wsbPath -Value $wsb -Encoding UTF8
  Add-Note "Windows Sandbox: virtual GPU off, networking $net, results in $outDir"
  Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\WindowsSandbox.exe') -ArgumentList ('"' + $wsbPath + '"') | Out-Null
  $resultFile = Join-Path $outDir 'smoke-result.json'
  $deadline = (Get-Date).AddMinutes($SandboxTimeoutMin)
  while (-not (Test-Path -LiteralPath $resultFile) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 5 }
  if (-not (Test-Path -LiteralPath $resultFile)) {
    Add-Check 'sandbox' 'fail' "no result after $SandboxTimeoutMin minutes (see $outDir\console.txt)"
    Write-Result 1
    exit 1
  }
  Start-Sleep -Seconds 2
  $guest = Get-Content -Raw -LiteralPath $resultFile | ConvertFrom-Json
  foreach ($c in $guest.checks) { Add-Check ("sandbox:" + $c.id) $c.status $c.detail $c.seconds }
  foreach ($n in $guest.notes) { Add-Note ("guest: " + $n) }
  $code = [int]$guest.exit_code
  Write-Result $code @{ guest = $guest }
  Write-Line ''
  if ($code -eq 0) { Write-Line "Sandbox smoke PASSED ($($guest.counts.pass) pass, $($guest.counts.skip) skip)" 'Green' } else { Write-Line "Sandbox smoke FAILED ($($guest.counts.fail) failed)" 'Red' }
  exit $code
}

# ------------------------------------------------------------------------------------------------
# Machine facts
# ------------------------------------------------------------------------------------------------
function Get-MachineFacts {
  $os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
  $cpu = Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue | Select-Object -First 1
  $gpus = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | ForEach-Object { $_.Name })
  return [ordered]@{
    os           = if ($os) { "$($os.Caption) $($os.Version)" } else { [Environment]::OSVersion.VersionString }
    cpu          = if ($cpu) { $cpu.Name.Trim() } else { '' }
    logical      = [Environment]::ProcessorCount
    ram_mb       = if ($os) { [int]($os.TotalVisibleMemorySize / 1024) } else { 0 }
    video        = $gpus
    powershell   = $PSVersionTable.PSVersion.ToString()
    webview2     = Get-WebView2Version
    disposable   = Test-Disposable
    in_sandbox   = [bool]$InSandbox
    user         = $env:USERNAME
  }
}

# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------
$script:machine = Get-MachineFacts
$script:exeInfo = $null
Write-Line ("Sigil packaged smoke ({0}) on {1}; results in {2}" -f $Mode, $script:machine.os, $ResultsDir) 'Cyan'

if ($Mode -eq 'Sandbox') { Invoke-SandboxRun }

$work = Join-Path $ResultsDir 'app'
$uninstall = $null
$installRoot = $null

# --- 1. Get a Sigil.exe to test --------------------------------------------------------------------
function Find-Newest([string]$Dir, [string]$Filter) {
  if (-not (Test-Path -LiteralPath $Dir)) { return $null }
  $f = Get-ChildItem -LiteralPath $Dir -Filter $Filter -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($f) { return $f.FullName }
  return $null
}

$exe = $null
try {
  switch ($Mode) {
    'Layout' {
      if (-not $LayoutDir) { throw '-Mode Layout needs -LayoutDir' }
      $exe = Join-Path (Resolve-Path $LayoutDir).Path 'Sigil.exe'
    }
    'Portable' {
      if (-not $ZipPath -and $RepoRoot) { $ZipPath = Find-Newest (Join-Path $RepoRoot 'src-tauri\target\release') 'Sigil_*_x64_portable.zip' }
      if (-not $ZipPath) { throw 'no portable zip found: pass -ZipPath' }
      New-Item -ItemType Directory -Force -Path $work | Out-Null
      Expand-Archive -LiteralPath $ZipPath -DestinationPath $work -Force
      $exe = Join-Path $work 'Sigil.exe'
    }
    default {
      if (-not $InstallerPath -and $RepoRoot) { $InstallerPath = Find-Newest (Join-Path $RepoRoot 'src-tauri\target\release\bundle\nsis') '*-setup.exe' }
      if (-not $InstallerPath) { throw 'no NSIS installer found: pass -InstallerPath' }
      $InstallerPath = (Resolve-Path $InstallerPath).Path
      Add-Note ("installer: {0} ({1} MB)" -f (Split-Path -Leaf $InstallerPath), [math]::Round((Get-Item -LiteralPath $InstallerPath).Length / 1MB, 1))
      if ($Mode -eq 'Extract') {
        $z = (Get-Command 7z.exe -ErrorAction SilentlyContinue)
        $sevenZip = if ($z) { $z.Source } elseif (Test-Path -LiteralPath (Join-Path $env:ProgramFiles '7-Zip\7z.exe')) { Join-Path $env:ProgramFiles '7-Zip\7z.exe' } else { $null }
        if (-not $sevenZip) { throw '7-Zip (7z.exe) is needed to unpack the NSIS installer; install it, or use -Mode Portable / Layout / Installed' }
        New-Item -ItemType Directory -Force -Path $work | Out-Null
        & $sevenZip x -y "-o$work" $InstallerPath | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "7-Zip could not unpack $InstallerPath" }
        $found = Get-ChildItem -LiteralPath $work -Recurse -Filter 'Sigil.exe' -File -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $found) { throw 'Sigil.exe is not in the installer' }
        $exe = $found.FullName
      } else {
        # Installed: only on a disposable machine.
        if (-not (Test-Disposable) -and -not $AllowInstall) {
          throw 'refusing to run the installer on this PC (it writes shortcuts and an uninstall entry to your profile). Use -Mode Extract, or -Mode Sandbox, or pass -AllowInstall on a disposable machine.'
        }
        $installRoot = Join-Path $ResultsDir 'installed'
        $p = Start-Process -FilePath $InstallerPath -ArgumentList @('/S', ('/D=' + $installRoot)) -PassThru -Wait
        if ($p.ExitCode -ne 0) { throw "the installer exited with code $($p.ExitCode)" }
        $exe = Join-Path $installRoot 'Sigil.exe'
        $uninstall = Join-Path $installRoot 'uninstall.exe'
      }
    }
  }
  if (-not (Test-Path -LiteralPath $exe)) { throw "Sigil.exe not found at $exe" }
  $exe = (Resolve-Path $exe).Path
  $exeDir = Split-Path -Parent $exe
  $vi = (Get-Item -LiteralPath $exe).VersionInfo
  $script:exeInfo = [ordered]@{ path = $exe; size_mb = [math]::Round((Get-Item -LiteralPath $exe).Length / 1MB, 1); version = $vi.ProductVersion; modified = (Get-Item -LiteralPath $exe).LastWriteTime.ToString('o') }
  Add-Check 'acquire' 'pass' ("{0}: Sigil.exe {1} MB, version {2}" -f $Mode, $script:exeInfo.size_mb, $script:exeInfo.version)
} catch {
  Add-Check 'acquire' 'fail' $_.Exception.Message
  Write-Result 1
  exit 1
}

# --- 1b. Is the app under test the tree in front of us? ------------------------------------------------
# Everything below proves things about the build, not about the sources. A source file that is newer than the build
# (a fix landed after `pnpm tauri:release`) means the run vouches for an older tree: say so instead of passing quietly.
if ($RepoRoot) {
  Invoke-Check 'build-fresh' {
    $built = $null
    foreach ($cand in @($InstallerPath, $ZipPath, $exe)) {
      if ($cand -and (Test-Path -LiteralPath $cand)) { $built = Get-Item -LiteralPath $cand; break }
    }
    if (-not $built) { Skip-Check 'no build artifact to date' }
    $newer = @(Get-BuildInputsNewerThan $RepoRoot $built.LastWriteTimeUtc)
    $when = $built.LastWriteTime.ToString('yyyy-MM-dd HH:mm')
    if ($newer.Count -gt 0) {
      $first = $newer[0]
      Skip-Check ("{0} source file(s) are newer than {1} (built {2}); newest: {3} ({4}). The checks below describe the older tree: rebuild with pnpm tauri:release" -f $newer.Count, $built.Name, $when, $first.FullName.Substring($RepoRoot.Length).TrimStart('\', '/'), $first.LastWriteTime.ToString('yyyy-MM-dd HH:mm'))
    }
    "no source file that goes into the app is newer than $($built.Name) (built $when)"
  }
}

# --- 2. The catalog this build embeds (only known when the repo is at hand) -------------------------
$expectedCatalog = $null
if ($RepoRoot) {
  $n = 0
  foreach ($f in Get-ChildItem -LiteralPath (Join-Path $RepoRoot 'src-tauri\model-catalog') -Filter '*.json' -File | Where-Object { -not $_.Name.StartsWith('_') }) {
    $doc = Get-Content -Raw -LiteralPath $f.FullName | ConvertFrom-Json
    if ($doc.PSObject.Properties.Name -contains 'entries') { $n += @($doc.entries).Count }
  }
  $expectedCatalog = $n
}

# --- 3. Scratch profile: nothing of the real user's data is used --------------------------------------
$guiProfile = Join-Path $ResultsDir 'profile'
$roaming = Join-Path $guiProfile 'AppData\Roaming'
$localApp = Join-Path $guiProfile 'AppData\Local'
$appData = Join-Path $roaming 'com.sigil.app'
New-Item -ItemType Directory -Force -Path $roaming, $localApp | Out-Null
$realSettings = Join-Path $env:APPDATA 'com.sigil.app\user\settings'
$realBefore = if (Test-Path -LiteralPath $realSettings) { (Get-Item -LiteralPath $realSettings).LastWriteTimeUtc.Ticks } else { 0 }
$profileEnv = @{ USERPROFILE = $guiProfile; APPDATA = $roaming; LOCALAPPDATA = $localApp; SIGIL_APP_DATA_DIR = ''; SIGIL_MODEL_CATALOG_OVERRIDE = ''; SIGIL_LOCAL_ASSETS = '' }
$headlessEnv = @{ APPDATA = $roaming; LOCALAPPDATA = $localApp; SIGIL_APP_DATA_DIR = ''; SIGIL_MODEL_CATALOG_OVERRIDE = ''; SIGIL_LOCAL_ASSETS = '' }

try {
  if (-not $SkipHeadless) {
    # --- layout
    Invoke-Check 'layout' {
      $need = @('onnxruntime_providers_shared.dll', 'onnxruntime_providers_cuda.dll', 'DirectML.dll')
      $missing = @($need | Where-Object { -not (Test-Path -LiteralPath (Join-Path $exeDir $_)) })
      if ($missing.Count -gt 0) { throw ("not next to Sigil.exe: " + ($missing -join ', ')) }
      $sizes = ($need | ForEach-Object {
        $len = (Get-Item -LiteralPath (Join-Path $exeDir $_)).Length
        if ($len -lt 1MB) { "{0} {1} KB" -f $_, [math]::Round($len / 1KB) } else { "{0} {1} MB" -f $_, [math]::Round($len / 1MB, 1) }
      }) -join '; '
      $sizes
    }

    # --- sysinfo
    $sysinfo = $null
    Invoke-Check 'sysinfo' {
      $out = Join-Path $ResultsDir 'sysinfo.json'
      $r = Invoke-Exe $exe @('--headless', 'sysinfo', '--out', $out) 120 $headlessEnv
      if ($r.TimedOut) { throw 'timed out after 120 s' }
      if ($r.ExitCode -ne 0) { throw ("exit code {0}: {1}" -f $r.ExitCode, $r.Stderr.Trim()) }
      $script:sysinfo = Read-JsonFile $out
      $s = $script:sysinfo.system
      if (-not $s -or $s.cpu.logical_cores -lt 1 -or $s.total_ram_mb -lt 512) { throw 'the report has no CPU / memory facts' }
      $hw = @($s.gpus | Where-Object { -not $_.software })
      $sw = @($s.gpus | Where-Object { $_.software })
      "{0}, {1} logical cores, {2} MB; {3} hardware GPU(s), {4} software adapter(s)" -f $s.cpu.brand, $s.cpu.logical_cores, $s.total_ram_mb, $hw.Count, $sw.Count
    }
    if ($ExpectNoGpu) {
      Invoke-Check 'no-gpu' {
        if (-not $script:sysinfo) { Skip-Check 'sysinfo did not run' }
        $hw = @($script:sysinfo.system.gpus | Where-Object { -not $_.software })
        if ($hw.Count -gt 0) { throw ("this machine has a hardware GPU: " + (($hw | ForEach-Object { $_.name }) -join ', ')) }
        'no hardware GPU, as expected'
      }
    }

    # --- catalog: everything not installed
    Invoke-Check 'catalog' {
      $r = Invoke-Exe $exe @('--headless', 'fetch', '--list') 60 $headlessEnv
      if ($r.ExitCode -ne 0) { throw ("exit code {0}: {1}" -f $r.ExitCode, $r.Stderr.Trim()) }
      $rows = @($r.Stdout -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -Skip 1)
      if ($rows.Count -lt 100) { throw "only $($rows.Count) catalog rows" }
      $installed = @($rows | Where-Object { $_ -notmatch '\snot_installed\s' })
      if ($installed.Count -gt 0) { throw ("entries already installed on a clean profile: " + (($installed | Select-Object -First 3) -join ' | ')) }
      if ($null -ne $expectedCatalog -and $rows.Count -ne $expectedCatalog) { throw "the build lists $($rows.Count) entries, the repository catalog has $expectedCatalog (stale build?)" }
      "$($rows.Count) entries, none installed" + $(if ($null -ne $expectedCatalog) { ' (equals the repository catalog)' } else { '' })
    }

    # --- ep probe
    Invoke-Check 'ep-probe' {
      $out = Join-Path $ResultsDir 'ep-probe.json'
      $r = Invoke-Exe $exe @('--headless', 'ep-probe', '--out', $out) 240 $headlessEnv
      if ($r.TimedOut) { throw 'timed out after 240 s' }
      if ($r.ExitCode -ne 0) { throw ("exit code {0}: {1}" -f $r.ExitCode, $r.Stderr.Trim()) }
      $p = Read-JsonFile $out
      foreach ($d in @('onnxruntime_providers_shared.dll', 'onnxruntime_providers_cuda.dll', 'DirectML.dll')) {
        $hit = @($p.provider_dlls | Where-Object { $_.name -ieq $d })
        if ($hit.Count -eq 0 -or $hit[0].source -ne 'exe_dir') { throw "$d was not found next to the exe" }
      }
      $okCuda = @($p.ep_checks | Where-Object { $_.ep -eq 'cuda' -and $_.ok })
      if ($ExpectNoGpu -and $okCuda.Count -gt 0) { throw 'CUDA works on a machine that should have no GPU' }
      if (-not $p.recommended_ep) { throw 'no recommended execution provider' }
      $eps = @($p.ep_checks | ForEach-Object { "{0}={1}" -f $_.ep, $(if ($_.ok) { 'ok' } else { 'no' }) }) -join ', '
      "recommended '{0}'; checks: {1}" -f $p.recommended_ep, $(if ($eps) { $eps } else { 'none (no adapter)' })
    }

    # --- with packs (read only)
    if ($PackStore) {
      $packCases = @(
        @{ id = 'packs-nemotron-fr'; model = 'stt/nemotron-3.5-streaming-560ms-int8'; wav = 'models\stt\nemotron-3.5-streaming-560ms-int8\2026-06-11\test_wavs\fr.wav'; expect = 'demandez' },
        @{ id = 'packs-sensevoice-en'; model = 'stt/sense-voice-small-int8'; wav = 'models\stt\sense-voice-small-int8\2024-07-17\test_wavs\en.wav'; expect = 'chieftain' }
      )
      foreach ($c in $packCases) {
        $case = $c
        Invoke-Check $case.id {
          $wav = Join-Path $PackStore $case.wav
          if (-not (Test-Path -LiteralPath $wav)) { Skip-Check "not in the pack store: $($case.wav)" }
          $out = Join-Path $ResultsDir ($case.id + '.json')
          $r = Invoke-Exe $exe @('--headless', 'engine-probe', 'sherpa', $case.model, $wav, '--out', $out) 240 @{ SIGIL_APP_DATA_DIR = $PackStore; APPDATA = $roaming; LOCALAPPDATA = $localApp }
          if ($r.ExitCode -ne 0) { throw ("exit code {0}: {1}" -f $r.ExitCode, ($r.Stderr -split "`r?`n" | Select-Object -Last 2) -join ' ') }
          $j = Read-JsonFile $out
          $text = [string]$j.output.text
          if ($text -notmatch $case.expect) { throw "the transcript has no '$($case.expect)': $text" }
          "sherpa-onnx $($j.runtime.version) next to ONNX Runtime $($j.sigil_onnxruntime_version): '$($text.Substring(0, [math]::Min(60, $text.Length)))'"
        }
      }
    }

    # --- online: the model store with a clean profile
    if ($Online -or $OnlineModel) {
      $storeDir = Join-Path $ResultsDir 'online-store'
      $storeEnv = @{ SIGIL_APP_DATA_DIR = $storeDir; APPDATA = $roaming; LOCALAPPDATA = $localApp }
      Invoke-Check 'online-fetch' {
        $r = Invoke-Exe $exe @('--headless', 'fetch', 'base/test-small') 180 $storeEnv
        Assert-FetchSucceeded $r 'fetch base/test-small'
        $l = Invoke-Exe $exe @('--headless', 'fetch', '--list') 60 $storeEnv
        $row = @($l.Stdout -split "`r?`n" | Where-Object { $_ -match '^base/test-small\s' })
        if ($row.Count -eq 0 -or $row[0] -notmatch '\sinstalled\s') { throw "base/test-small is not installed after the fetch: $($row -join ' ')" }
        'base/test-small installed through the model store (size and sha256 verified)'
      }
      Invoke-Check 'online-sherpa' {
        $r = Invoke-Exe $exe @('--headless', 'fetch', 'runtime/sherpa-onnx') 300 $storeEnv
        Assert-FetchSucceeded $r 'fetch runtime/sherpa-onnx'
        $out = Join-Path $ResultsDir 'online-sherpa.json'
        $p = Invoke-Exe $exe @('--headless', 'engine-probe', 'sherpa', '--out', $out) 120 $storeEnv
        if ($p.ExitCode -ne 0) { throw ("engine-probe exit {0}: {1}" -f $p.ExitCode, $p.Stderr.Trim()) }
        $j = Read-JsonFile $out
        if (-not $j.runtime.version) { throw 'the runtime reported no version' }
        "sherpa-onnx $($j.runtime.version) (ONNX Runtime $($j.runtime.onnxruntime_version)) loaded next to Sigil's ONNX Runtime $($j.sigil_onnxruntime_version) in $([math]::Round($j.runtime_load_ms)) ms"
      }
      if ($OnlineModel) {
        Invoke-Check 'online-sensevoice' {
          $r = Invoke-Exe $exe @('--headless', 'fetch', 'stt/sense-voice-small-int8') 1800 $storeEnv
          Assert-FetchSucceeded $r 'fetch stt/sense-voice-small-int8'
          $model = Get-ChildItem -LiteralPath (Join-Path $storeDir 'models\stt\sense-voice-small-int8') -Recurse -Filter 'en.wav' -File | Select-Object -First 1
          if (-not $model) { throw 'the installed model has no test_wavs\en.wav' }
          $out = Join-Path $ResultsDir 'online-sensevoice.json'
          $p = Invoke-Exe $exe @('--headless', 'engine-probe', 'sherpa', 'stt/sense-voice-small-int8', $model.FullName, '--out', $out) 240 $storeEnv
          if ($p.ExitCode -ne 0) { throw ("engine-probe exit {0}: {1}" -f $p.ExitCode, $p.Stderr.Trim()) }
          $text = [string](Read-JsonFile $out).output.text
          if ($text -notmatch 'chieftain') { throw "unexpected transcript: $text" }
          "downloaded SenseVoice (163 MB) and decoded: '$($text.Substring(0, [math]::Min(60, $text.Length)))'"
        }
      }
    }
  }

  # --- the app itself, fresh profile -----------------------------------------------------------------
  # The installer may have installed the WebView2 runtime (it ships a bootstrapper), so look again.
  $script:machine.webview2 = Get-WebView2Version
  if ($SkipGui) {
    Add-Check 'gui' 'skip' '-SkipGui'
  } elseif (-not $script:machine.webview2) {
    if ($RequireGui) { Add-Check 'gui' 'fail' 'the WebView2 runtime is not installed (required by -RequireGui)' }
    else { Add-Check 'gui' 'skip' 'the WebView2 runtime is not installed on this machine' }
  } elseif (-not $ShowWindow -and -not (Test-Disposable) -and -not (Test-ExeKnowsStartMinimized $exe)) {
    # A build without the switch would put its window in front of whoever is using this PC: do not start it.
    Add-Check 'gui' 'skip' 'this Sigil.exe predates SIGIL_START_MINIMIZED (a window would come to the front of whatever you are doing); rebuild it, or pass -ShowWindow to start it the normal way'
  } else {
    $seeded = switch ($ProfileKind) { 'Seeded' { $true } 'Fresh' { $false } default { -not (Test-Disposable) } }
    $port = Get-FreePort
    $cdpPort = Get-FreePort -Exclude @($port)
    $app = $null
    $appStarted = Get-Date
    $cdp = $null
    $sampler = $null
    $saved = @{}
    try {
      if ($seeded) {
        # Same seed the benchmark harnesses use: the one-time native offers count as answered (they are dialogs over the
        # user's desktop) and the server binds this PC only (no firewall prompt).
        New-Item -ItemType Directory -Force -Path (Join-Path $appData 'user') | Out-Null
        $seed = @{
          id = 'packaged-smoke'; uiLanguage = 'en'; onboardingComplete = $true
          lanAccess = @{ mode = 'localhost' }
          services = @{
            stt = @{ data = @{ backend = 'local'; autoStart = $false; desktopDefaultMigrated = $true; localOfferShown = $true } }
            tts = @{ data = @{ backend = 'localNeural'; desktopDefaultMigrated = $true; localOfferShown = $true } }
          }
        }
        ($seed | ConvertTo-Json -Depth 8) | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $appData 'user\settings')
      }
      $guiEnv = @{}
      foreach ($k in $profileEnv.Keys) { $guiEnv[$k] = $profileEnv[$k] }
      $guiEnv['WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS'] = "--remote-debugging-port=$cdpPort"
      # The window starts minimized and never takes the foreground (main.rs, SIGIL_START_MINIMIZED); gui-focus checks it.
      $guiEnv['SIGIL_START_MINIMIZED'] = $(if ($ShowWindow) { '' } else { '1' })
      foreach ($k in $guiEnv.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k, 'Process'); [Environment]::SetEnvironmentVariable($k, $(if ($guiEnv[$k] -eq '') { $null } else { $guiEnv[$k] }), 'Process') }
      # Sigil shows a modal "Port N is not available!" dialog when the bind fails: never launch into a taken port.
      if (-not (Test-PortFree $port)) { $port = Get-FreePort -Exclude @($cdpPort) }
      $app = Start-Process -FilePath $exe -ArgumentList @('--port', "$port") -PassThru
      foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process') }
      $saved = @{}
      if (-not $ShowWindow) {
        # From the first instant: which window is in front, every 100 ms for the first 15 s.
        Initialize-WindowHelper
        $sampler = New-Object SmokeFocusSampler -ArgumentList $app.Id, 15000, 100
        $sampler.Start()
      }
      try { $app.PriorityClass = 'BelowNormal' } catch {}
      Add-Note ("app started: pid {0}, port {1}, {2} profile, {3}" -f $app.Id, $port, $(if ($seeded) { 'seeded' } else { 'fresh' }), $(if ($ShowWindow) { 'window shown normally' } else { 'minimized without focus' }))

      Invoke-Check 'gui-server' {
        $deadline = (Get-Date).AddSeconds($GuiTimeoutSec)
        $up = $false
        while ((Get-Date) -lt $deadline -and -not $app.HasExited) {
          try { $c = New-Object System.Net.Sockets.TcpClient; $c.Connect('127.0.0.1', $port); $c.Close(); $up = $true; break } catch { Start-Sleep -Milliseconds 500 }
        }
        if ($app.HasExited) { throw "Sigil exited during start-up with code $($app.ExitCode)" }
        if (-not $up) { throw "the web server did not answer on port $port within $GuiTimeoutSec s" }
        $root = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/" -TimeoutSec 20
        if ($root.StatusCode -ne 200 -or $root.Content -notmatch 'id="root"') { throw "/ did not serve the app shell (HTTP $($root.StatusCode))" }
        $client = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/client" -TimeoutSec 20
        if ($client.StatusCode -ne 200) { throw "/client answered HTTP $($client.StatusCode)" }
        $i18n = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/i18n/en/translation.json" -TimeoutSec 20
        if ($i18n.StatusCode -ne 200) { throw "the English translation answered HTTP $($i18n.StatusCode)" }
        "embedded server on port $port serves /, /client and the translations"
      }

      Invoke-Check 'gui-page' {
        $target = $null
        $deadline = (Get-Date).AddSeconds($GuiTimeoutSec)
        while ((Get-Date) -lt $deadline -and -not $target) {
          try {
            $list = Invoke-RestMethod -Uri "http://127.0.0.1:$cdpPort/json" -TimeoutSec 5
            $target = @($list | Where-Object { $_.type -eq 'page' -and $_.webSocketDebuggerUrl -and $_.url -notlike 'devtools:*' }) | Select-Object -First 1
          } catch { }
          if (-not $target) { Start-Sleep -Seconds 1 }
        }
        if (-not $target) { throw 'the WebView never exposed a page over DevTools' }
        $script:cdpWs = $target.webSocketDebuggerUrl
        $probe = @'
(async () => {
  const r = { url: location.href, title: document.title };
  r.rootChildren = document.getElementById('root') ? document.getElementById('root').childElementCount : -1;
  r.text = (document.body.innerText || '').slice(0, 300);
  r.failed = (document.body.innerText || '').indexOf('Application failed to start') >= 0;
  r.hasApi = !!window.ApiServer;
  r.hasTauri = !!(window.__TAURI__ && window.__TAURI__.core);
  return r;
})()
'@
        $info = $null
        $deadline = (Get-Date).AddSeconds($GuiTimeoutSec)
        while ((Get-Date) -lt $deadline) {
          $info = Invoke-CdpEvaluate $script:cdpWs $probe 30
          if ($info.failed -or ($info.hasApi -and $info.rootChildren -gt 0)) { break }
          Start-Sleep -Seconds 2
        }
        if ($info.failed) { throw ("the app shows 'Application failed to start': " + ($info.text -replace '\s+', ' ')) }
        if (-not $info.hasTauri) { throw 'window.__TAURI__ is missing: the page is not running inside the desktop app' }
        if (-not $info.hasApi -or $info.rootChildren -lt 1) { throw "the host did not finish booting within $GuiTimeoutSec s (ApiServer=$($info.hasApi), #root children=$($info.rootChildren))" }
        "page '$($info.title)' mounted, host services booted, no error screen"
      }

      Invoke-Check 'gui-models' {
        if (-not $script:cdpWs) { Skip-Check 'no DevTools page' }
        $js = @'
(async () => {
  const rows = await window.__TAURI__.core.invoke('plugin:model-store|get_entry_states');
  const cat = await window.__TAURI__.core.invoke('plugin:model-store|get_catalog');
  const states = {};
  for (const r of rows) states[r.state] = (states[r.state] || 0) + 1;
  return { rows: rows.length, catalog: cat.length, states: states };
})()
'@
        $m = Invoke-CdpEvaluate $script:cdpWs $js 60
        $notInstalled = [int]$m.states.not_installed
        if ($m.rows -lt 100 -or $m.catalog -ne $m.rows) { throw "Models & Runtimes lists $($m.rows) states for $($m.catalog) catalog entries" }
        if ($notInstalled -ne $m.rows) { throw ("not every entry is 'not installed': " + ($m.states | ConvertTo-Json -Compress)) }
        if ($null -ne $expectedCatalog -and $m.catalog -ne $expectedCatalog) { throw "the app embeds $($m.catalog) entries, the repository catalog has $expectedCatalog" }
        "Models & Runtimes lists $($m.rows) entries, all not installed"
      }

      Invoke-Check 'gui-idle' {
        # Nothing may download or install by itself: no pack, no model, no partial file.
        Start-Sleep -Seconds 5
        $bad = @()
        foreach ($d in @('models', 'runtimes')) {
          $p = Join-Path $appData $d
          if (Test-Path -LiteralPath $p) { $bad += @(Get-ChildItem -LiteralPath $p -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName.Substring($appData.Length) }) }
        }
        if ($bad.Count -gt 0) { throw ("files appeared without a request: " + (($bad | Select-Object -First 3) -join ', ')) }
        $kids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$($app.Id)" -ErrorAction SilentlyContinue | Where-Object { $_.Name -notmatch '^(msedgewebview2|conhost|WerFault)' })
        if ($kids.Count -gt 0) { throw ("helper processes started by themselves: " + (($kids | ForEach-Object { $_.Name }) -join ', ')) }
        $cpu = if ($app.HasExited) { 0 } else { $app.Refresh(); $app.TotalProcessorTime.TotalSeconds }
        "no files in models/runtimes, no helper processes; app CPU time so far $([math]::Round($cpu, 1)) s"
      }

      Invoke-Check 'gui-focus' {
        # The user's rule for automated launches: no window pops up and nothing takes focus. The sampler has watched the
        # foreground window since the process was created; what is left is the end state.
        if ($ShowWindow) { Skip-Check '-ShowWindow: the window was allowed to come to the front' }
        $sampler.Join()
        if ($app.HasExited) { throw "Sigil exited before the focus check finished (exit code $($app.ExitCode))" }
        # Windows a person could see: not the event loop's invisible helpers (tao's thread event target counts as visible
        # but is 0 x 0, layered, transparent and a tool window).
        $wins = @(Get-AppWindows $app.Id | Where-Object { $_.Visible -and -not $_.ToolWindow -and ($_.Minimized -or ($_.Width * $_.Height) -gt 0) })
        $listing = ($wins | ForEach-Object { "'{0}' [{1}, {2}x{3}]{4}" -f $_.Title, $_.Class, $_.Width, $_.Height, $(if ($_.Minimized) { ' (minimized)' } else { '' }) }) -join ', '
        if ($sampler.Hits -gt 0) { throw ("Sigil's window was the foreground window in {0} of {1} samples (first after {2} ms); windows: {3}" -f $sampler.Hits, $sampler.Samples, $sampler.FirstHitMs, $listing) }
        if ($sampler.Samples -lt 50) { throw "only $($sampler.Samples) foreground samples were taken in 15 s" }
        if ($wins.Count -eq 0) { throw 'Sigil has no visible window (it should be shown minimized)' }
        $stray = @($wins | Where-Object { $_.Title -ne 'Sigil' })
        if ($stray.Count -gt 0) { throw ("a window other than Sigil's main window is showing (a dialog?): " + $listing) }
        $up = @($wins | Where-Object { -not $_.Minimized })
        if ($up.Count -gt 0) { throw ("the main window is not minimized: " + $listing) }
        "the foreground window was never Sigil's in $($sampler.Samples) samples over 15 s; the window is minimized; no dialog"
      }

      Invoke-Check 'gui-log' {
        $log = Join-Path $localApp 'com.sigil.app\logs\sigil.log'
        if (-not (Test-Path -LiteralPath $log)) { throw "no log file at $log" }
        $lines = @(Get-Content -LiteralPath $log -ErrorAction SilentlyContinue)
        $panics = @($lines | Where-Object { $_ -match 'panicked at|stack overflow|fatal runtime error' })
        $errors = @($lines | Where-Object { $_ -match '\sERROR\s' })
        if ($panics.Count -gt 0) { throw ("the log has a panic: " + $panics[0]) }
        if ($errors.Count -gt 0) {
          if ($Strict) { throw ("$($errors.Count) ERROR line(s), first: " + $errors[0]) }
          Add-Note ("log has $($errors.Count) ERROR line(s); first: " + $errors[0])
        }
        $warns = @($lines | Where-Object { $_ -match '\sWARN\s' })
        "$($lines.Count) log lines, no panic, $($errors.Count) error(s), $($warns.Count) warning(s)"
      }
    } finally {
      foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process') }
      if ($app) {
        Invoke-Check 'gui-exit' {
          if ($app.HasExited) { throw "the app had already exited with code $($app.ExitCode)" }
          Initialize-WindowHelper
          $app.Refresh()
          $t0 = Get-Date
          # WM_CLOSE to the window titled 'Sigil', like the close button. Process.CloseMainWindow() posts to the first
          # owner-less visible window, which is tao's invisible event-target window once the real one is minimized, and
          # that breaks the shutdown (see SmokeWindows.CloseTitled).
          $sent = [SmokeWindows]::CloseTitled($app.Id, 'Sigil')
          if ($sent -eq 0) { $sent = [int]$app.CloseMainWindow() }
          $asked = 0
          $ended = $app.WaitForExit(10000)
          if (-not $ended) {
            $asked = [SmokeWindows]::CloseTitled($app.Id, 'Sigil')
            # The app's own exit guard (src-tauri/src/exit_guard.rs) ends the process 20 s after the first close request.
            $ended = $app.WaitForExit([math]::Max(1, $ExitWaitSec - 10) * 1000)
          }
          $took = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
          if (-not $ended) {
            $state = ((Get-AppWindows $app.Id | Where-Object { $_.Visible } | ForEach-Object { "'{0}' [{1}] visible, minimized={2}, unresponsive={3}" -f $_.Title, $_.Class, $_.Minimized, $_.Unresponsive }) -join '; ')
            try { $app.Kill() } catch {}
            throw "closing the window did not end the process within $ExitWaitSec s (WM_CLOSE sent to $sent window(s), again to $asked; windows left: $state; killed)"
          }
          $left = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^(llama-server|sigil-bergamot|tts-server|s2-server)' -and $_.CreationDate -gt $appStarted })
          if ($left.Count -gt 0) { $left | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; throw ("helper processes outlived the app: " + (($left | ForEach-Object { $_.Name }) -join ', ')) }
          # Did the app end by itself, or did its exit guard have to end it? The guard logs one WARN line when it fires.
          $guardLog = Join-Path $localApp 'com.sigil.app\logs\sigil.log'
          $fired = @()
          if (Test-Path -LiteralPath $guardLog) { $fired = @(Get-Content -LiteralPath $guardLog -ErrorAction SilentlyContinue | Where-Object { $_ -match 'exit guard:' }) }
          if ($fired.Count -gt 0) {
            $why = ($fired[0] -replace '^.*exit guard:\s*', '')
            if ($Strict) { throw "the process outlived its closed window and only the exit guard ended it: $why" }
            Add-Note "closing the window did NOT end Sigil by itself; the app's exit guard ended it after $took s (exit code $($app.ExitCode)): $why"
            return "closing the window ended Sigil only through the exit guard after $took s (exit code $($app.ExitCode)); no helper process left"
          }
          if ($asked -gt 0) { Add-Note "the first WM_CLOSE did not end the app within 10 s; the second one did" }
          "closing the window ended Sigil in $took s (exit code $($app.ExitCode)); no helper process left"
        }
        if (-not $app.HasExited) { try { $app.Kill() } catch {} }
      }
    }
  }

  # --- the real profile stayed untouched (Extract / Portable / Layout on a user's PC)
  if (-not (Test-Disposable)) {
    Invoke-Check 'real-profile' {
      $after = if (Test-Path -LiteralPath $realSettings) { (Get-Item -LiteralPath $realSettings).LastWriteTimeUtc.Ticks } else { 0 }
      if ($after -ne $realBefore) { throw "$realSettings changed during the smoke" }
      'the real %APPDATA%\com.sigil.app\user\settings was not touched'
    }
  }
} finally {
  if ($uninstall -and (Test-Path -LiteralPath $uninstall)) {
    try { Start-Process -FilePath $uninstall -ArgumentList '/S' -Wait | Out-Null; Add-Note 'uninstalled' } catch { Add-Note ("uninstall failed: " + $_.Exception.Message) }
  }
}

# --- summary --------------------------------------------------------------------------------------------
$failed = @($script:checks | Where-Object { $_.status -eq 'fail' })
$skipped = @($script:checks | Where-Object { $_.status -eq 'skip' })
$code = 0
if ($failed.Count -gt 0) { $code = 1 }
elseif ($Strict -and $skipped.Count -gt 0) { $code = 1 }
Write-Result $code
Write-Line ''
if ($code -eq 0) { Write-Line ("PASSED: {0} checks passed, {1} skipped. Result: {2}" -f @($script:checks | Where-Object { $_.status -eq 'pass' }).Count, $skipped.Count, (Join-Path $ResultsDir 'smoke-result.json')) 'Green' }
else { Write-Line ("FAILED: {0} failed, {1} skipped{2}. Result: {3}" -f $failed.Count, $skipped.Count, $(if ($Strict) { ' (-Strict)' } else { '' }), (Join-Path $ResultsDir 'smoke-result.json')) 'Red' }
if (-not $KeepFiles -and $code -eq 0 -and $Mode -in @('Extract', 'Portable', 'Installed')) {
  Remove-Item -LiteralPath $work, (Join-Path $ResultsDir 'installed'), (Join-Path $ResultsDir 'online-store') -Recurse -Force -ErrorAction SilentlyContinue
}
exit $code
