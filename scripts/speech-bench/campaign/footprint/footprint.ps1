#Requires -Version 7.0
<#
.SYNOPSIS
  P7-A app footprint: launches the release Sigil in a scratch Windows profile and samples its process tree (WebView2 and
  helper processes included) for -Seconds after a warm-up.

.DESCRIPTION
  The scratch profile redirects USERPROFILE / APPDATA / LOCALAPPDATA, so the real %APPDATA%\com.sigil.app is never touched
  (the script checks that and reports it). `models` and `runtimes` of the profile are junctions to the P7 model store.

  Scenarios
    idle                 app open, local STT off
    captions             local STT on, English, CPU tier, two passes, interim results, nobody speaking
    captions-nointerim   same without interim results (one engine less)
    captions-auto        the default setting of a new install (language auto) on the CPU tier
    obs                  captions + the OBS caption page open in headless Chromium + simulated caption traffic
    translate-cpu        obs + the bundled Hy-MT2 1.8B translator (llama-server on the CPU), English to Japanese, interim translations FORCED ON
                         (the shipped `interim: "auto"` holds them off on the CPU, so this is the cost of the policy P7-I0 replaced)
    translate-cpu-finals the same with interim translations off (final lines only; what `auto` does on the CPU)
    translate-cpu-slow   the same as translate-cpu with at most one interim translation request per 1.5 s (default 300 ms)
    translate-vulkan     obs + the same translator on the Vulkan GPU variant, interim translations forced on
    translate-auto       obs + the translator with the variant and `interim` both left on `auto` (what a new install gets: the governor picks)
  Load
    none                 nothing else
    game                 four Node processes spinning 60 % of every 100 ms (the benchmark's h3_cpu_game profile, outside the app)

  The app is started with SIGIL_START_MINIMIZED=1 (minimized, never in front of what the user is doing) and the script refuses an
  exe whose build does not know that switch (-ShowWindow starts it visibly anyway). Clean-up stops only the llama-server that
  descends from the app it started.

  The microphone is the profile's capture device (-Device, default: the system default) and hears the room; the engines idle
  between captions exactly as they do when nobody speaks. The cost of speech itself is the `speed_pipeline` figure.

.EXAMPLE
  pwsh scripts/bench/gpu-lock.ps1 -Run "P7-A footprint" -Wait 240 -ExpectedMinutes 5 -Command "& scripts/speech-bench/campaign/footprint/footprint.ps1 -Scenario obs -Load game"
#>
param(
  [Parameter(Mandatory)][ValidateSet('idle', 'captions', 'captions-nointerim', 'captions-auto', 'obs', 'translate-cpu', 'translate-cpu-finals', 'translate-cpu-slow', 'translate-vulkan', 'translate-auto')][string]$Scenario,
  [ValidateSet('none', 'game')][string]$Load = 'none',
  [int]$Seconds = 120,
  [int]$Warmup = 60,
  [string]$Affinity = 'FFF',
  [string]$OutDir = '',
  # the release sigil.exe (frontend embedded: build it with `pnpm tauri build --no-bundle` or TAURI_CONFIG frontendDist)
  [string]$Exe = $(if ($env:SIGIL_BENCH_EXE) { $env:SIGIL_BENCH_EXE } else { Join-Path $PSScriptRoot '../../../../src-tauri/target/release/sigil.exe' }),
  [string]$Store = "$env:LOCALAPPDATA/sigil-dev-cache/appdata-p7",
  [string]$ScratchProfile = "$env:LOCALAPPDATA/sigil-dev-cache/p7/footprint-profile",
  [string]$Device = '',
  # start the window the normal way (in front, then minimized); only for an exe that predates SIGIL_START_MINIMIZED
  [switch]$ShowWindow
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
. (Join-Path $here 'launch-guard.ps1')
$sampler = Join-Path $here '../sample-tree.ps1'
$driverJs = Join-Path $here 'driver.mjs'
$Exe = (Resolve-Path $Exe).Path
# The user works on this PC: an exe that does not know SIGIL_START_MINIMIZED would take the foreground (checked before anything is written or started).
Assert-StartsMinimized -Exe $Exe -ShowWindow:$ShowWindow
$origLocal = $env:LOCALAPPDATA
$savedEnv = @{ USERPROFILE = $env:USERPROFILE; APPDATA = $env:APPDATA; LOCALAPPDATA = $env:LOCALAPPDATA; SIGIL_APP_DATA_DIR = $env:SIGIL_APP_DATA_DIR; SIGIL_START_MINIMIZED = $env:SIGIL_START_MINIMIZED; PLAYWRIGHT_BROWSERS_PATH = $env:PLAYWRIGHT_BROWSERS_PATH }
$realSettings = "$env:APPDATA/com.sigil.app/user/settings"
$realBefore = if (Test-Path $realSettings) { (Get-Item $realSettings).LastWriteTimeUtc.Ticks } else { 0 }
$fp = $ScratchProfile.Replace('\', '/')
if (-not $OutDir) { $OutDir = Join-Path $here '../../../../bench-results/speech/footprint' }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$name = "$Scenario-$Load"
$roam = "$fp/AppData/Roaming/com.sigil.app"
New-Item -ItemType Directory -Force "$roam/user", "$fp/AppData/Local" | Out-Null
foreach ($d in 'models', 'runtimes') {
  if (-not (Test-Path "$roam/$d")) { cmd /c mklink /J "$($roam.Replace('/', '\'))\$d" "$($Store.Replace('/', '\'))\$d" | Out-Null }
}

# --- settings of the scenario
$language = if ($Scenario -eq 'captions-auto') { 'auto' } else { 'en' }
$local = @{ tier = 'cpu'; language = $language; twoPass = $true; interim = ($Scenario -ne 'captions-nointerim') }
if ($Device) { $local.device = $Device }
$stt = @{ backend = 'local'; autoStart = ($Scenario -ne 'idle'); desktopDefaultMigrated = $true; localOfferShown = $true; local = $local }
# TTS as a new desktop install has it (Kokoro, idle). Both one-time "try the local engine" offers count as already
# answered: they are native dialogs, and each run rewrites this profile, so they would pop up over the user's desktop
# on every launch.
$tts = @{ backend = 'localNeural'; desktopDefaultMigrated = $true; localOfferShown = $true }
$services = @{ stt = @{ showActionButton = $true; data = $stt }; tts = @{ data = $tts } }
if ($Scenario -like 'translate-*') {
  $variant = switch -Wildcard ($Scenario) { 'translate-cpu*' { 'cpu' } 'translate-vulkan' { 'vulkan' } default { 'auto' } }
  $tr = @{ backend = 'auto'; autoStart = $true; sourceLanguage = 'en'; targetLanguage = 'ja'; llm = @{ serverVariant = $variant } }
  # The shipped `interim` setting is "auto" (P7-I0): off while the bundled translator runs on the processor. Every scenario states
  # it, so a run does not depend on the default of the build: translate-cpu / -slow / -vulkan force interim translations ON (the
  # policy the doc's rows are labelled with; with "auto" translate-cpu would measure finals only), translate-cpu-finals forces them
  # OFF, and translate-auto says "auto" because it measures what a new install gets.
  $interim = switch ($Scenario) { 'translate-cpu-finals' { $false } 'translate-auto' { 'auto' } default { $true } }
  $tr.auto = @{ interim = $interim }
  if ($Scenario -eq 'translate-cpu-slow') { $tr.partials = @{ throttleMs = 1500 } }        # at most one interim request per 1.5 s
  $services.translation = @{ data = $tr }
}
$settings = @{ id = 'p7a-footprint'; uiLanguage = 'en'; onboardingComplete = $true; services = $services }
if ($Device) { $settings.audioInputDevice = $Device }
($settings | ConvertTo-Json -Depth 12) | Set-Content -Encoding UTF8 "$roam/user/settings"

$app = $null; $drv = $null; $burners = @(); $t0 = Get-Date; $appStart = $t0; $known = @()
try {
  # --- environment of the app
  $env:USERPROFILE = $fp.Replace('/', '\'); $env:APPDATA = "$($fp.Replace('/', '\'))\AppData\Roaming"; $env:LOCALAPPDATA = "$($fp.Replace('/', '\'))\AppData\Local"
  Remove-Item Env:SIGIL_APP_DATA_DIR -ErrorAction SilentlyContinue

  # --- the game stand-in
  $burners = @()
  if ($Load -eq 'game') {
    for ($i = 0; $i -lt 4; $i++) {
      $js = 'const d=0.6;setInterval(()=>{const t=Date.now()+100*d;while(Date.now()<t){}},100);'
      $b = Start-Process node -ArgumentList @('-e', "`"$js`"") -PassThru -WindowStyle Hidden
      try { $b.PriorityClass = 'BelowNormal' } catch {}
      $burners += $b
    }
  }

  # --- the app
  # Minimized from the first frame and never in the foreground (the user works on this PC while it runs).
  Set-StartMinimizedEnv -ShowWindow:$ShowWindow
  $app = Start-Process $Exe -PassThru
  try { $appStart = $app.StartTime } catch {}
  try { $app.PriorityClass = 'BelowNormal' } catch {}
  if ($Affinity) { try { $app.ProcessorAffinity = [IntPtr][Convert]::ToInt64($Affinity, 16) } catch { Write-Warning "affinity failed: $_" } }
  Add-Type -Namespace W -Name U -MemberDefinition '[DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);' -ErrorAction SilentlyContinue
  $t0 = Get-Date
  $minimized = $false
  while (((Get-Date) - $t0).TotalSeconds -lt $Warmup) {
    Start-Sleep -Seconds 2
    if (-not $minimized) { $app.Refresh(); if ($app.MainWindowHandle -ne 0) { Start-Sleep -Seconds 2; [W.U]::ShowWindow($app.MainWindowHandle, 7) | Out-Null; $minimized = $true } }
  }
  $port = (Get-NetTCPConnection -OwningProcess $app.Id -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1).LocalPort

  # --- measure
  $res = [ordered]@{ scenario = $Scenario; load = $Load; affinity = $Affinity; port = $port }
  $jobs = @()
  $jobs += Start-Job -ScriptBlock { param($f, $p, $s, $o) & $f -RootPid $p -Seconds $s -IntervalMs 2000 -Out $o } -ArgumentList $sampler, $app.Id, $Seconds, "$OutDir/$name.app.csv"
  $drv = $null
  if ($Scenario -eq 'obs' -or $Scenario -like 'translate-*') {
    $env:PLAYWRIGHT_BROWSERS_PATH = "$origLocal/ms-playwright"
    $drv = Start-Process node -ArgumentList @($driverJs, $port, $Seconds, '0.4', '1') -PassThru -WindowStyle Hidden -RedirectStandardOutput "$OutDir/$name.driver.txt" -RedirectStandardError "$OutDir/$name.driver.err.txt"
    Start-Sleep -Seconds 6
    $jobs += Start-Job -ScriptBlock { param($f, $p, $s, $o) & $f -RootPid $p -Seconds $s -IntervalMs 2000 -Out $o } -ArgumentList $sampler, $drv.Id, ($Seconds - 8), "$OutDir/$name.page.csv"
  }
  $jobs | Wait-Job -Timeout ($Seconds + 120) | Out-Null
  $k = 0
  foreach ($j in $jobs) { $r = Receive-Job $j; $res[$(if ($k -eq 0) { 'app' } else { 'page' })] = ($r | ConvertFrom-Json); $k++ }
  $jobs | Remove-Job -Force

  # --- what the app logged about its engines (the plan it ran), then clean up
  $log = "$fp/AppData/Local/com.sigil.app/logs/sigil.log"
  if (Test-Path $log) { Get-Content $log -Tail 400 | Select-String -Pattern 'route|plan|engine|model|loaded|llama' | Select-Object -Last 25 | ForEach-Object { $_.Line } | Set-Content "$OutDir/$name.log.txt" }
  if ($drv) { $drv.WaitForExit(25000) | Out-Null; if (-not $drv.HasExited) { $drv.Kill() } }
  # the translation server(s) this run started: a snapshot while the app is alive (the parent chain is gone once it exits)
  $known = @(Get-OwnedProcess -RootPid $app.Id -RootStart $appStart -Name 'llama-server*')
  # WM_CLOSE to the window titled 'Sigil' of THIS process (Process.CloseMainWindow can hit tao's invisible event window once the app is minimized and leave the process lingering)
  . (Join-Path $PSScriptRoot 'close-window.ps1'); Close-AppWindow -ProcessId $app.Id -Title 'Sigil' | Out-Null
  Start-Sleep -Seconds 4
  if (-not $app.HasExited) { $app.Kill() }
  # only the translation server THIS run started (a descendant of the app), never one the user has running
  Stop-OwnedProcess -RootPid $app.Id -RootStart $appStart -Name 'llama-server*' -Known $known
  foreach ($b in $burners) { try { $b.Kill() } catch {} }
  $realAfter = if (Test-Path $realSettings) { (Get-Item $realSettings).LastWriteTimeUtc.Ticks } else { 0 }
  $res.real_settings_untouched = ($realBefore -eq $realAfter)
  $res.fake_profile_used = (Test-Path "$fp/AppData/Local/com.sigil.app")
  ($res | ConvertTo-Json -Depth 6) | Tee-Object -FilePath "$OutDir/$name.json"
} finally {
  # the redirected profile must not leak into the caller's session (the next scenario of the same session reads the real one)
  foreach ($k in $savedEnv.Keys) { if ($null -eq $savedEnv[$k]) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue } else { Set-Item "Env:$k" $savedEnv[$k] } }
  foreach ($b in $burners) { try { $b.Kill() } catch {} }
  if ($drv -and -not $drv.HasExited) { try { $drv.Kill() } catch {} }
  if ($app -and -not $app.HasExited) { try { $app.Kill() } catch {} }
  # an error in the middle of a run must not leave the run's own translation server behind either
  if ($app) { try { Stop-OwnedProcess -RootPid $app.Id -RootStart $appStart -Name 'llama-server*' -Known $known } catch {} }
}
