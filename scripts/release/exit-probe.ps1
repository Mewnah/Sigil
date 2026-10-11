<#
.SYNOPSIS
  Reproduces the shutdown stall that the app's exit guard exists for: WM_CLOSE posted to the event loop's invisible window.

.DESCRIPTION
  Sigil's event loop (tao) owns an invisible window of class "Tao Thread Event Target" (untitled, 16 x 16, a tool window).
  Windows counts it as a visible, owner-less top-level window, so Process.CloseMainWindow() posts to it once the real window
  is minimized, and so do tools that post WM_CLOSE to every window of a process (taskkill without /F, installers, the Restart
  Manager). WM_CLOSE there destroys the loop's own message window. The real window still closes afterwards, but the process
  does not end until the exit guard (src-tauri/src/exit_guard.rs) terminates it 20 s later.

  This script starts ONE copy of the app minimized without focus (SIGIL_START_MINIMIZED=1) on a scratch Windows profile,
  posts WM_CLOSE to that helper window, then closes the real window (the one titled "Sigil") and reports how long the process
  needed and whether the guard had to end it (its log line). Expected on a build with the guard: the process ends after about
  20 s with exit code 0 and one "exit guard:" line in the log. Without the guard it would not end at all (the script kills it
  after 40 s).

  It is a diagnostic, not a gate: run it once when the shutdown behavior changed. It refuses a Sigil.exe that predates
  SIGIL_START_MINIMIZED (its window would come to the front of whatever you are doing) unless -ShowWindow is given.

.EXAMPLE
  pwsh scripts/release/exit-probe.ps1 -Exe src-tauri\target\release\Sigil.exe
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$Exe,
  # Scratch Windows profile (USERPROFILE / APPDATA / LOCALAPPDATA are redirected to it); deleted first.
  [string]$ProfileDir,
  [string]$Out,
  [switch]$ShowWindow
)
$ErrorActionPreference = 'Stop'
$Exe = (Resolve-Path -LiteralPath $Exe).Path
if (-not $ProfileDir) { $ProfileDir = Join-Path ([System.IO.Path]::GetTempPath()) 'sigil-exit-probe' }

# The window helper lives in the packaged smoke script (one copy of the C#).
$smoke = Join-Path $PSScriptRoot 'packaged-smoke-all.ps1'
$raw = Get-Content -Raw -LiteralPath $smoke
$m = [regex]::Match($raw, "(?s)Add-Type -TypeDefinition @'\r?\n(.*?)\r?\n'@")
if (-not $m.Success) { throw "cannot find the window helper in $smoke" }
Add-Type -TypeDefinition $m.Groups[1].Value
# ... and so are the port helpers (Test-PortFree, Get-FreePort): Sigil shows a modal "Port N is not available!" dialog
# when its port is taken, which is exactly what this script must never put in front of the person at the PC.
$h = [regex]::Match($raw, "(?s)#region testable-helpers\r?\n(.*?)\r?\n#endregion testable-helpers")
if (-not $h.Success) { throw "cannot find the helper functions in $smoke" }
. ([scriptblock]::Create($h.Groups[1].Value))
Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class ProbePost { [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr h, uint m, IntPtr w, IntPtr l); }
'@

if (-not $ShowWindow -and -not [SmokeWindows]::FileContains($Exe, 'SIGIL_START_MINIMIZED')) {
  throw 'this Sigil.exe predates SIGIL_START_MINIMIZED: starting it would put a window in front of whatever you are doing (use -ShowWindow to do it anyway)'
}

$lines = New-Object System.Collections.ArrayList
function Say([string]$t) { [void]$lines.Add($t); Write-Host $t }
function Show-Windows([int]$ProcId) { foreach ($row in [SmokeWindows]::Describe($ProcId)) { Say ('   window: ' + $row) } }

Remove-Item -LiteralPath $ProfileDir -Recurse -Force -ErrorAction SilentlyContinue
$roam = Join-Path $ProfileDir 'AppData\Roaming'
$loc = Join-Path $ProfileDir 'AppData\Local'
New-Item -ItemType Directory -Force -Path (Join-Path $roam 'com.sigil.app\user'), $loc | Out-Null
# The one-time native offers count as answered (they are dialogs), the server binds this PC only.
$seed = @{
  id = 'exit-probe'; uiLanguage = 'en'; onboardingComplete = $true; lanAccess = @{ mode = 'localhost' }
  services = @{
    stt = @{ data = @{ backend = 'local'; autoStart = $false; desktopDefaultMigrated = $true; localOfferShown = $true } }
    tts = @{ data = @{ backend = 'localNeural'; desktopDefaultMigrated = $true; localOfferShown = $true } }
  }
}
($seed | ConvertTo-Json -Depth 8) | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $roam 'com.sigil.app\user\settings')
$log = Join-Path $loc 'com.sigil.app\logs\sigil.log'

$names = 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'SIGIL_START_MINIMIZED', 'SIGIL_APP_DATA_DIR', 'SIGIL_MODEL_CATALOG_OVERRIDE', 'SIGIL_LOCAL_ASSETS'
$saved = @{}
foreach ($k in $names) { $saved[$k] = [Environment]::GetEnvironmentVariable($k, 'Process') }
$env:USERPROFILE = $ProfileDir; $env:APPDATA = $roam; $env:LOCALAPPDATA = $loc
foreach ($k in 'SIGIL_APP_DATA_DIR', 'SIGIL_MODEL_CATALOG_OVERRIDE', 'SIGIL_LOCAL_ASSETS') { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
if ($ShowWindow) { Remove-Item Env:SIGIL_START_MINIMIZED -ErrorAction SilentlyContinue } else { $env:SIGIL_START_MINIMIZED = '1' }
# A port the OS just reported free, checked again at the moment of the launch; never launch into a taken port.
$port = 0
for ($try = 0; $try -lt 3 -and $port -eq 0; $try++) {
  $candidate = Get-FreePort
  if (Test-PortFree $candidate) { $port = $candidate }
}
if ($port -eq 0) {
  foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process') }
  throw 'no free loopback port to start the app on: not launching it (its start-up check would show a dialog)'
}
$app = Start-Process -FilePath $Exe -ArgumentList '--port', "$port" -PassThru
foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process') }
try { $app.PriorityClass = 'BelowNormal' } catch {}
try {
  Say "started pid $($app.Id) on port $port"
  Start-Sleep -Seconds 14
  Show-Windows $app.Id

  # A: WM_CLOSE to the invisible tool window only (visible flag set, no title, tool window).
  $target = $null
  foreach ($row in [SmokeWindows]::Describe($app.Id)) {
    $f = $row -split '\|', 9
    if ($f[1] -eq '1' -and $f[8] -eq '' -and (([int]$f[4] -band 0x80) -ne 0)) { $target = [int64]$f[0] }
  }
  if (-not $target) { Say 'no untitled visible tool window found: nothing to test'; return }
  [void][ProbePost]::PostMessage([IntPtr]$target, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero)
  Say "A: WM_CLOSE posted to window $target (the event loop's helper window)"
  Start-Sleep -Seconds 4
  $app.Refresh()
  Say "A: after 4 s the process is $(if ($app.HasExited) { 'gone' } else { 'still running' })"
  if (-not $app.HasExited) { Show-Windows $app.Id }

  # B: close the real window the way the close button does.
  if (-not $app.HasExited) {
    $t1 = Get-Date
    $n = [SmokeWindows]::CloseTitled($app.Id, 'Sigil')
    Say "B: WM_CLOSE posted to $n window(s) titled Sigil"
    if ($app.WaitForExit(40000)) { Say ("B: the process ended after {0} s (exit code {1})" -f [math]::Round(((Get-Date) - $t1).TotalSeconds, 1), $app.ExitCode) }
    else { Say 'B: the process was still running after 40 s (killed)'; Show-Windows $app.Id }
  }
  if (Test-Path -LiteralPath $log) {
    $g = @(Get-Content -LiteralPath $log | Where-Object { $_ -match 'exit guard:' })
    if ($g.Count -gt 0) { Say ('log: ' + $g[0].Substring(0, [math]::Min(300, $g[0].Length))) } else { Say 'log: no exit guard line (the app ended by itself)' }
  }
} finally {
  if (-not $app.HasExited) { try { $app.Kill() } catch {} }
  if ($Out) { $lines | Set-Content -Encoding UTF8 -LiteralPath $Out }
}
