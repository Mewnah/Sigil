#Requires -Version 7.0
<#
.SYNOPSIS
  Starts the release Sigil in the footprint harness's scratch profile with the WebView's remote-debugging port open, then asks
  the page (through `cdp-probe.mjs`) what the host services report: does the frontend boot, did speech-to-text start, what is its
  status. Use it when a footprint scenario measures nothing (an exe built without the frontend serves a directory listing).

  Like footprint.ps1 it starts the app with SIGIL_START_MINIMIZED=1 (never in front of what the user is doing), refuses an exe that
  does not know the switch (-ShowWindow starts it visibly anyway) and stops only the llama-server that descends from the app.

.EXAMPLE
  pwsh scripts/speech-bench/campaign/footprint/diagnose.ps1 -Exe src-tauri/target/release/sigil.exe -Wait 70
#>
param(
  [string]$Exe = $(if ($env:SIGIL_BENCH_EXE) { $env:SIGIL_BENCH_EXE } else { Join-Path $PSScriptRoot '../../../../src-tauri/target/release/sigil.exe' }),
  [string]$ScratchProfile = "$env:LOCALAPPDATA/sigil-dev-cache/p7/footprint-profile",
  [string]$OutDir = '',
  [int]$Wait = 60,
  [switch]$StartStt,
  # start the window the normal way (in front, then minimized); only for an exe that predates SIGIL_START_MINIMIZED
  [switch]$ShowWindow
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'launch-guard.ps1')
$Exe = (Resolve-Path $Exe).Path
# The user works on this PC: an exe that does not know SIGIL_START_MINIMIZED would take the foreground (checked before anything starts).
Assert-StartsMinimized -Exe $Exe -ShowWindow:$ShowWindow
$origLocal = $env:LOCALAPPDATA
$saved = @{ USERPROFILE = $env:USERPROFILE; APPDATA = $env:APPDATA; LOCALAPPDATA = $env:LOCALAPPDATA; SIGIL_APP_DATA_DIR = $env:SIGIL_APP_DATA_DIR; SIGIL_START_MINIMIZED = $env:SIGIL_START_MINIMIZED; WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS = $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS; PLAYWRIGHT_BROWSERS_PATH = $env:PLAYWRIGHT_BROWSERS_PATH }
$fp = $ScratchProfile.Replace('\', '/')
if (-not $OutDir) { $OutDir = Join-Path $PSScriptRoot '../../../../bench-results/speech/footprint' }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$app = $null; $appStart = Get-Date; $known = @()
try {
  $env:USERPROFILE = $fp.Replace('/', '\'); $env:APPDATA = "$($fp.Replace('/', '\'))\AppData\Roaming"; $env:LOCALAPPDATA = "$($fp.Replace('/', '\'))\AppData\Local"
  Remove-Item Env:SIGIL_APP_DATA_DIR -ErrorAction SilentlyContinue
  $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS = '--remote-debugging-port=9333'
  Add-Type -Namespace W -Name U -MemberDefinition '[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);' -ErrorAction SilentlyContinue
  # Minimized from the first frame and never in the foreground (the user works on this PC while it runs).
  Set-StartMinimizedEnv -ShowWindow:$ShowWindow
  $app = Start-Process $Exe -PassThru
  try { $appStart = $app.StartTime } catch {}
  try { $app.PriorityClass = 'BelowNormal' } catch {}
  $t0 = Get-Date; $minimized = $false
  while (((Get-Date) - $t0).TotalSeconds -lt $Wait) {
    Start-Sleep -Seconds 2
    # SW_SHOWMINNOACTIVE (7): minimized without activating it (SW_MINIMIZE, 6, would hand the focus to the next window)
    if (-not $minimized) { $app.Refresh(); if ($app.MainWindowHandle -ne 0) { Start-Sleep -Seconds 2; [W.U]::ShowWindow($app.MainWindowHandle, 7) | Out-Null; $minimized = $true } }
  }
  $known = @(Get-OwnedProcess -RootPid $app.Id -RootStart $appStart -Name 'llama-server*')
  $env:PLAYWRIGHT_BROWSERS_PATH = "$origLocal/ms-playwright"
  node (Join-Path $PSScriptRoot 'cdp-probe.mjs') 9333 "$OutDir/diagnose.png" $(if ($StartStt) { '1' } else { '0' }) 2>&1 | Tee-Object -FilePath "$OutDir/diagnose.txt"
  $log = "$fp/AppData/Local/com.sigil.app/logs/sigil.log"
  if (Test-Path $log) { Get-Content $log -Tail 60 | Set-Content "$OutDir/diagnose.log.txt" }
}
finally {
  foreach ($k in $saved.Keys) { if ($null -eq $saved[$k]) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue } else { Set-Item "Env:$k" $saved[$k] } }
  if ($app -and -not $app.HasExited) { . (Join-Path $PSScriptRoot 'close-window.ps1'); Close-AppWindow -ProcessId $app.Id -Title 'Sigil' | Out-Null; Start-Sleep -Seconds 3; if (-not $app.HasExited) { $app.Kill() } }
  # only the translation server THIS run started (a descendant of the app), never one the user has running
  if ($app) { Stop-OwnedProcess -RootPid $app.Id -RootStart $appStart -Name 'llama-server*' -Known $known }
}
