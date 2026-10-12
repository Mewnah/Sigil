<#
.SYNOPSIS
  Verify a Sigil Windows release build: GPU provider DLLs and the Visual C++ runtime staged and
  inside both installers, no new load-time DLL imports, and (optionally) the installed layout
  passing the headless EP probe.

.DESCRIPTION
  Run after `pnpm tauri build --config scripts/release/tauri.release.conf.json`. Checks:
    1. Sigil.exe exists; the staged DLLs (src-tauri/target/native-bundle: the three ONNX Runtime
       provider DLLs and the four Visual C++ runtime DLLs vcruntime140, vcruntime140_1, msvcp140 and
       msvcp140_1) match their manifest (written by stage-native-dlls.mjs).
    2. `dumpbin /DEPENDENTS Sigil.exe`: no CUDA / cuDNN / ONNX Runtime / ggml / OpenVR / sherpa DLL
       is a load-time import (they must load lazily so Sigil starts on any PC), every other
       import is in scripts/release/expected-imports.txt (new ones are warnings, errors with -Strict),
       and every Visual C++ runtime import is one of the four DLLs the installers carry (a clean
       Windows has none of them: without them Sigil.exe exits with 0xC0000135).
    3. MSI (administrative install into a temp dir) and NSIS (listed/extracted with 7-Zip) put
       the three provider DLLs and the four Visual C++ runtime DLLs next to Sigil.exe,
       byte-identical to the staged copies.
    4. Unless -SkipProbe: runs the installed-layout `Sigil.exe --headless ep-probe` and checks that
       the provider DLLs are found next to the exe; prints the EP test results. -RequireCuda makes a
       failed CUDA test an error (manual acceptance on an NVIDIA PC with the runtime pack installed).

.EXAMPLE
  pwsh scripts/release/verify-bundle.ps1
  pwsh scripts/release/verify-bundle.ps1 -RequireCuda      # dev rig, NVIDIA runtime pack installed
#>
[CmdletBinding()]
param(
  [string]$RepoRoot,
  [string]$TargetDir,
  [ValidateSet('release', 'debug')][string]$BuildProfile = 'release',
  [switch]$SkipInstallers,
  [switch]$SkipProbe,
  [switch]$RequireCuda,
  [switch]$Strict
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $RepoRoot) { $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
if (-not $TargetDir) {
  if ($env:CARGO_TARGET_DIR) { $TargetDir = $env:CARGO_TARGET_DIR } else { $TargetDir = Join-Path $RepoRoot 'src-tauri\target' }
}
$profileDir = Join-Path $TargetDir $BuildProfile
$stagingDir = Join-Path $RepoRoot 'src-tauri\target\native-bundle'
$manifestPath = Join-Path $RepoRoot 'src-tauri\target\native-bundle.json'
$expectedImportsPath = Join-Path $PSScriptRoot 'expected-imports.txt'
$providerDlls = @('onnxruntime_providers_shared.dll', 'onnxruntime_providers_cuda.dll', 'DirectML.dll')
# The Visual C++ runtime, app-local: load-time imports of Sigil.exe (expected-imports.txt) that neither a clean Windows
# nor the old installers had (stage-native-dlls.mjs VC_RUNTIME_DLLS).
$vcRuntimeDlls = @('vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'msvcp140_1.dll')
$bundledDlls = @($providerDlls) + @($vcRuntimeDlls)
# Shipped by the MSI only: the WiX bundler adds every *.dll next to the built exe, and ort-sys copies both TensorRT
# providers there. They are small, load nothing at start-up and need TensorRT runtime DLLs the installer does not
# carry (the TensorRT-RTX route takes them from the opt-in runtime/tensorrt-rtx pack, neural_audio/gpu/trt.rs).
$knownExtraDlls = @('onnxruntime_providers_tensorrt.dll', 'onnxruntime_providers_nv_tensorrt_rtx.dll')

$script:failures = New-Object System.Collections.Generic.List[string]
$script:warnings = New-Object System.Collections.Generic.List[string]
function Pass([string]$m) { Write-Host "[ok]   $m" -ForegroundColor Green }
function Warn([string]$m) { $script:warnings.Add($m); Write-Host "[warn] $m" -ForegroundColor Yellow }
function Fail([string]$m) { $script:failures.Add($m); Write-Host "[FAIL] $m" -ForegroundColor Red }
function Hash([string]$p) { (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLowerInvariant() }

# --- 1. Build outputs and staged DLLs --------------------------------------------------------------
$exe = Join-Path $profileDir 'Sigil.exe'
if (-not (Test-Path -LiteralPath $exe)) { Fail "Sigil.exe not found at $exe"; Write-Host "`nFAILED"; exit 1 }
Pass "Sigil.exe $([math]::Round((Get-Item -LiteralPath $exe).Length / 1MB, 1)) MB"

$staged = @{}
if (-not (Test-Path -LiteralPath $manifestPath)) {
  Fail "staging manifest missing ($manifestPath): build with --config scripts/release/tauri.release.conf.json"
} else {
  $manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
  foreach ($name in $bundledDlls) {
    $entry = @($manifest.files | Where-Object { $_.name -ieq $name })
    $path = Join-Path $stagingDir $name
    if ($entry.Count -eq 0) { Fail "$name is not in the staging manifest"; continue }
    if (-not (Test-Path -LiteralPath $path)) { Fail "$name is not staged in $stagingDir"; continue }
    $h = Hash $path
    if ($h -ne $entry[0].sha256) { Fail "$name in the staging dir does not match its manifest"; continue }
    $staged[$name.ToLowerInvariant()] = $h
    if ($vcRuntimeDlls -contains $name) { Pass "staged $name ($([math]::Round($entry[0].size / 1KB)) KB, Visual C++ runtime)" }
    else { Pass "staged $name ($([math]::Round($entry[0].size / 1MB, 1)) MB)" }
  }
  $vcInfo = $manifest.PSObject.Properties['vc_redist']
  if ($vcInfo -and $vcInfo.Value) { Write-Host "       Visual C++ runtime $($vcInfo.Value.version) from $($vcInfo.Value.dir)" }
  $extra = @(Get-ChildItem -LiteralPath $stagingDir -File | Where-Object { $bundledDlls -notcontains $_.Name })
  foreach ($x in $extra) { Fail "unexpected file in the staging dir (it would be installed): $($x.Name)" }
}

# --- 2. Load-time imports --------------------------------------------------------------------------
function Find-Dumpbin {
  $cmd = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (Test-Path -LiteralPath $vswhere) {
    $found = & $vswhere -latest -products * -find 'VC\Tools\MSVC\**\bin\Hostx64\x64\dumpbin.exe' 2>$null | Select-Object -First 1
    if ($found) { return $found }
  }
  return $null
}

$forbidden = '^(cudart|cublas|cublaslt|cufft|curand|cusparse|cudnn|nvcuda|nvml|nvrtc|onnxruntime|tensorrt|nvinfer|sherpa|ggml|llama|whisper|transcribe|sigil_|openvr_api|vulkan-1|libtorch|torch|python)'
$dumpbin = Find-Dumpbin
if (-not $dumpbin) {
  Warn 'dumpbin.exe not found (install the MSVC build tools); load-time imports not checked'
} else {
  $expected = @(Get-Content -LiteralPath $expectedImportsPath | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ -and -not $_.StartsWith('#') })
  $out = & $dumpbin /NOLOGO /DEPENDENTS $exe
  $inList = $false
  $imports = New-Object System.Collections.Generic.List[string]
  foreach ($line in $out) {
    if ($line -match 'following dependencies') { $inList = $true; continue }
    if ($inList -and $line -match 'Summary') { break }
    if ($inList -and $line.Trim() -match '\.dll$') { $imports.Add($line.Trim().ToLowerInvariant()) }
  }
  $imports = @($imports | Sort-Object -Unique)
  if ($imports.Count -eq 0) { Fail "could not read the import table of $exe" }
  $system32 = Join-Path $env:SystemRoot 'System32'
  foreach ($dll in $imports) {
    if ($dll -match $forbidden) { Fail "Sigil.exe load-time imports $dll (GPU/runtime DLLs must be loaded lazily)"; continue }
    # The Visual C++ runtime is in System32 on a PC that ever installed a program built with it, which is why a dev PC
    # never notices: a clean Windows has none, so every import of it has to be one of the four DLLs shipped next to Sigil.exe.
    if ($dll -match '^(vcruntime|msvcp|concrt|vcomp|vccorlib|vcamp)' -and ($vcRuntimeDlls -notcontains $dll)) {
      Fail "Sigil.exe load-time imports $dll (Visual C++ runtime), which the installers do not carry: add it to VC_RUNTIME_DLLS in stage-native-dlls.mjs, to verify-bundle.ps1 and to expected-imports.txt"
      continue
    }
    $known = $false
    foreach ($pattern in $expected) { if ($dll -like $pattern) { $known = $true; break } }
    if ($known) { continue }
    if (Test-Path -LiteralPath (Join-Path $system32 $dll)) {
      $msg = "new load-time import $dll (a Windows DLL): review it and add it to expected-imports.txt"
      if ($Strict) { Fail $msg } else { Warn $msg }
    } else {
      Fail "Sigil.exe load-time imports $dll, which is not a Windows system DLL: Sigil would not start without it"
    }
  }
  Pass "load-time imports checked ($($imports.Count) DLLs)"
}

# --- 3. Installer contents -------------------------------------------------------------------------
$tmpRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("sigil-verify-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $tmpRoot | Out-Null
$installedExe = $null

function Test-InstalledLayout([string]$label, [string]$root) {
  $exes = @(Get-ChildItem -LiteralPath $root -Recurse -Filter 'Sigil.exe' -File -ErrorAction SilentlyContinue)
  if ($exes.Count -eq 0) { Fail "${label}: Sigil.exe not found in the package"; return $null }
  $dir = $exes[0].DirectoryName
  foreach ($name in $bundledDlls) {
    $p = Join-Path $dir $name
    if (-not (Test-Path -LiteralPath $p)) { Fail "${label}: $name is not next to Sigil.exe"; continue }
    $want = $staged[$name.ToLowerInvariant()]
    if (-not $want) { Fail "${label}: $name has no staged copy to compare with"; continue }
    if ((Hash $p) -ne $want) { Fail "${label}: $name differs from the staged copy"; continue }
    Pass "${label}: $name next to Sigil.exe"
  }
  $extras = @(Get-ChildItem -LiteralPath $dir -Filter '*.dll' -File | Where-Object { $bundledDlls -notcontains $_.Name })
  foreach ($x in $extras) {
    if ($knownExtraDlls -contains $x.Name) { Write-Host "       ${label}: also ships $($x.Name) (unused)" }
    else { Warn "${label}: unexpected DLL next to Sigil.exe: $($x.Name)" }
  }
  return $exes[0].FullName
}

try {
  if (-not $SkipInstallers) {
    $bundleDir = Join-Path $profileDir 'bundle'
    $msi = @(Get-ChildItem -LiteralPath (Join-Path $bundleDir 'msi') -Filter '*.msi' -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
    $nsis = @(Get-ChildItem -LiteralPath (Join-Path $bundleDir 'nsis') -Filter '*-setup.exe' -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)

    if ($msi.Count -eq 0) { Fail "no MSI in $bundleDir\msi" } else {
      $msiOut = Join-Path $tmpRoot 'msi'
      $p = Start-Process -FilePath 'msiexec.exe' -ArgumentList @('/a', "`"$($msi[0].FullName)`"", '/qn', "TARGETDIR=`"$msiOut`"") -Wait -PassThru
      if ($p.ExitCode -ne 0) { Fail "MSI administrative install failed (msiexec exit $($p.ExitCode))" }
      else { $installedExe = Test-InstalledLayout "MSI $($msi[0].Name)" $msiOut }
    }

    if ($nsis.Count -eq 0) { Fail "no NSIS installer in $bundleDir\nsis" } else {
      $sevenZip = $null
      $cmd7z = Get-Command 7z.exe -ErrorAction SilentlyContinue
      if ($cmd7z) { $sevenZip = $cmd7z.Source }
      elseif (Test-Path -LiteralPath (Join-Path $env:ProgramFiles '7-Zip\7z.exe')) { $sevenZip = Join-Path $env:ProgramFiles '7-Zip\7z.exe' }
      if (-not $sevenZip) { Warn '7-Zip not found; NSIS installer contents not checked' }
      else {
        $nsisOut = Join-Path $tmpRoot 'nsis'
        & $sevenZip x -y "-o$nsisOut" $nsis[0].FullName | Out-Null
        if ($LASTEXITCODE -ne 0) { Fail "7-Zip could not extract $($nsis[0].Name)" }
        else {
          $nsisExe = Test-InstalledLayout "NSIS $($nsis[0].Name)" $nsisOut
          if (-not $installedExe) { $installedExe = $nsisExe }
        }
      }
    }
  } else {
    $installedExe = $exe
  }

  # --- 4. Headless EP probe on the installed layout -----------------------------------------------
  if (-not $SkipProbe -and $installedExe) {
    $probeOut = Join-Path $tmpRoot 'ep-probe.json'
    $p = Start-Process -FilePath $installedExe -ArgumentList @('--headless', 'ep-probe', '--out', "`"$probeOut`"") -PassThru -WindowStyle Hidden
    if (-not $p.WaitForExit(180000)) { try { $p.Kill() } catch {} ; Fail 'the headless EP probe did not finish within 180 s' }
    elseif (-not (Test-Path -LiteralPath $probeOut)) { Fail "the headless EP probe wrote no report (exit $($p.ExitCode))" }
    else {
      $report = Get-Content -Raw -LiteralPath $probeOut | ConvertFrom-Json
      foreach ($name in $providerDlls) {
        $d = @($report.provider_dlls | Where-Object { $_.name -ieq $name })
        if ($d.Count -eq 0 -or $d[0].source -ne 'exe_dir') { Fail "probe: $name not found next to the installed Sigil.exe" }
      }
      if ($report.directml_loaded_from) {
        $dmlDir = Split-Path -Parent $report.directml_loaded_from
        if ($dmlDir -ieq (Split-Path -Parent $installedExe)) { Pass "probe: DirectML.dll loaded from the install dir" }
        else { Warn "probe: DirectML.dll loaded from $($report.directml_loaded_from), not the bundled copy" }
      }
      foreach ($c in @($report.ep_checks)) {
        $label = "probe: $($c.ep) on $($c.adapter) (device $($c.device_id))"
        if ($c.ok) { Pass "$label works in $($c.millis) ms" }
        elseif ($c.ep -eq 'cuda' -and $RequireCuda) { Fail "$label failed: $($c.error)" }
        else { Warn "$label failed: $($c.error)" }
      }
      if ($RequireCuda -and -not @($report.ep_checks | Where-Object { $_.ep -eq 'cuda' -and $_.ok })) {
        Fail 'probe: CUDA is not usable (install the NVIDIA GPU runtime pack first)'
      }
      Pass "probe: recommended EP '$($report.recommended_ep)'"
      foreach ($n in @($report.notes)) { Write-Host "       note: $n" }
    }
  }
} finally {
  Remove-Item -LiteralPath $tmpRoot -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ''
if ($script:failures.Count -gt 0) {
  Write-Host "FAILED: $($script:failures.Count) problem(s), $($script:warnings.Count) warning(s)" -ForegroundColor Red
  exit 1
}
Write-Host "PASSED with $($script:warnings.Count) warning(s)" -ForegroundColor Green
exit 0
