<#
.SYNOPSIS
  Verify a Sigil Windows release build: GPU provider DLLs and the Visual C++ runtime staged and
  inside both installers, no new load-time DLL imports, and (optionally) the installed layout
  passing the headless EP probe.

.DESCRIPTION
  Run after `pnpm tauri build --config scripts/release/tauri.release.conf.json`. Checks:
    1. Sigil.exe exists; the staged DLLs (src-tauri/target/native-bundle: the three ONNX Runtime
       provider DLLs and the four Visual C++ runtime DLLs vcruntime140, vcruntime140_1, msvcp140 and
       msvcp140_1) match their manifest (written by stage-native-dlls.mjs). The WebView2 install mode
       (embedBootstrapper: Microsoft's small bootstrapper rides inside both installers, so no download
       of the bootstrapper itself) is the one the overlay asks for, in the NSIS script and the WiX
       source the bundler generated; when an installer of a kind exists (or with -Strict) its generated
       file must be where the bundler leaves it, otherwise that is an error, not a warning. The
       bootstrapper file the build fetched must exist, be 1-3 MB and carry a valid Authenticode
       signature of Microsoft Corporation (nothing else pins it), and step 3 checks that both
       installers really contain it. The embedded bootstrapper still downloads the WebView2 runtime
       itself: a PC with neither the runtime nor internet cannot install (only offlineInstaller, about
       +130 MB per installer, could).
    2. `dumpbin /DEPENDENTS Sigil.exe`: no CUDA / cuDNN / ONNX Runtime / ggml / OpenVR / sherpa DLL
       is a load-time import (they must load lazily so Sigil starts on any PC), every other
       import is in scripts/release/expected-imports.txt (new ones are warnings, errors with -Strict),
       and every Visual C++ runtime import is one of the four DLLs the installers carry (a clean
       Windows has none of them: without them Sigil.exe exits with 0xC0000135).
    3. MSI (administrative install into a temp dir) and NSIS (listed/extracted with 7-Zip) put
       the three provider DLLs and the four Visual C++ runtime DLLs next to Sigil.exe,
       byte-identical to the staged copies, and both carry the WebView2 bootstrapper (the NSIS
       setup as a file, the MSI as a Binary-table stream; 7-Zip lists both).
    4. Unless -SkipProbe: runs the installed-layout `Sigil.exe --headless ep-probe` and checks that
       the provider DLLs are found next to the exe; prints the EP test results. -RequireCuda makes a
       failed CUDA test an error (manual acceptance on an NVIDIA PC with the runtime pack installed).
    5. Secrets: `check-no-secrets.mjs` compares the values of every SECRET / API_KEY / TOKEN / PASSWORD
       variable of .env and the environment with dist/, Sigil.exe and both extracted installers. Client secrets and API keys are compiled into Sigil.exe
       obfuscated (src-tauri/build.rs), so a plain-text hit anywhere is a leak. It prints variable names
       only. -RequireSecretValues (CI) fails when there is no secret value to compare, instead of passing
       without checking anything.

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
  [switch]$RequireSecretValues,
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

# The WebView2 install mode of the release installers (tauri.release.conf.json, bundle.windows.webviewInstallMode).
$expectedWebViewMode = 'embedBootstrapper'

#region webview2-install-mode
# A property of nested JSON objects, or $null when any step is missing (safe under Set-StrictMode).
function Get-JsonPath($obj, [string[]]$names) {
  foreach ($n in $names) {
    if ($null -eq $obj) { return $null }
    $prop = $obj.PSObject.Properties[$n]
    if (-not $prop) { return $null }
    $obj = $prop.Value
  }
  return $obj
}

# Whether the installers carry the WebView2 install mode the release asked for. Reads the release overlay and the two files
# the bundler generates from the merged config (the NSIS script and the WiX source): a build that did not get the overlay
# (plain `tauri build`) or an overlay that no longer asks for $Mode both show. A generated file that is not where it is
# expected is a failure when the installer built from it exists ($NsisRequired / $WixRequired: the bundler's output layout
# changed, so the build cannot be shown to have used the mode) and a warning otherwise (nothing was built yet). Returns
# Failures, Warnings and the bootstrapper files the generated sources point at (for Get-BootstrapperFileFindings).
function Get-WebView2ModeFindings([string]$Mode, [string]$OverlayPath, [string]$NsisScriptPath, [string]$WixPath, [bool]$NsisRequired = $false, [bool]$WixRequired = $false) {
  $fail = New-Object System.Collections.Generic.List[string]
  $warn = New-Object System.Collections.Generic.List[string]
  $paths = New-Object System.Collections.Generic.List[string]

  if (-not (Test-Path -LiteralPath $OverlayPath)) {
    $fail.Add("release overlay not found: $OverlayPath")
  } else {
    $overlay = $null
    try { $overlay = Get-Content -Raw -LiteralPath $OverlayPath | ConvertFrom-Json } catch { $fail.Add("release overlay is not valid JSON: $OverlayPath") }
    if ($overlay) {
      $type = Get-JsonPath $overlay @('bundle', 'windows', 'webviewInstallMode', 'type')
      if (-not $type) { $fail.Add("the release overlay sets no bundle.windows.webviewInstallMode (the installers would download the bootstrapper at install time); expected '$Mode'") }
      elseif ($type -cne $Mode) { $fail.Add("the release overlay sets webviewInstallMode '$type', expected '$Mode'") }
    }
  }

  if (-not (Test-Path -LiteralPath $NsisScriptPath)) {
    if ($NsisRequired) { $fail.Add("generated NSIS script not found ($NsisScriptPath) although the NSIS setup was built: the bundler's output layout changed, so its WebView2 install mode cannot be shown") }
    else { $warn.Add("generated NSIS script not found ($NsisScriptPath): the NSIS setup's WebView2 install mode is not checked") }
  } else {
    $nsis = Get-Content -Raw -LiteralPath $NsisScriptPath
    $m = [regex]::Match($nsis, '(?m)^\s*!define\s+INSTALLWEBVIEW2MODE\s+"([^"]*)"')
    if (-not $m.Success) { $fail.Add("the generated NSIS script defines no INSTALLWEBVIEW2MODE: $NsisScriptPath") }
    elseif ($m.Groups[1].Value -cne $Mode) { $fail.Add("the NSIS setup was built with WebView2 install mode '$($m.Groups[1].Value)', not '$Mode' (build with --config scripts/release/tauri.release.conf.json)") }
    elseif ($Mode -eq 'embedBootstrapper') {
      $p = [regex]::Match($nsis, '(?m)^\s*!define\s+WEBVIEW2BOOTSTRAPPERPATH\s+"([^"]*)"')
      if (-not $p.Success -or [string]::IsNullOrWhiteSpace($p.Groups[1].Value)) { $fail.Add('the NSIS setup embeds no WebView2 bootstrapper (WEBVIEW2BOOTSTRAPPERPATH is empty)') }
      else { $paths.Add($p.Groups[1].Value) }
    }
  }

  if ($Mode -eq 'embedBootstrapper') {
    if (-not (Test-Path -LiteralPath $WixPath)) {
      if ($WixRequired) { $fail.Add("generated WiX source not found ($WixPath) although the MSI was built: the bundler's output layout changed, so its WebView2 install mode cannot be shown") }
      else { $warn.Add("generated WiX source not found ($WixPath): the MSI's WebView2 install mode is not checked") }
    } else {
      $wix = Get-Content -Raw -LiteralPath $WixPath
      # Either quote style, any attribute order: the bundler's template is not ours to keep stable.
      $binary = [regex]::Match($wix, '<Binary\b[^>]*\bId\s*=\s*[''"]MicrosoftEdgeWebview2Setup\.exe[''"][^>]*>')
      if (-not $binary.Success) { $fail.Add('the MSI embeds no WebView2 bootstrapper (no MicrosoftEdgeWebview2Setup.exe binary in the WiX source)') }
      else {
        $source = [regex]::Match($binary.Value, '\bSourceFile\s*=\s*[''"]([^''"]+)[''"]')
        if ($source.Success) { $paths.Add($source.Groups[1].Value) }
        else { $fail.Add('the MSI embeds no WebView2 bootstrapper (the MicrosoftEdgeWebview2Setup.exe binary in the WiX source has no SourceFile)') }
      }
      if ($wix -match 'Id\s*=\s*[''"]DownloadAndInvokeBootstrapper[''"]') { $fail.Add('the MSI still downloads the WebView2 bootstrapper at install time (DownloadAndInvokeBootstrapper in the WiX source)') }
    }
  }

  [pscustomobject]@{ Failures = $fail.ToArray(); Warnings = $warn.ToArray(); BootstrapperPaths = @($paths | Select-Object -Unique) }
}

# Whether a certificate subject is Microsoft Corporation's ("CN=Microsoft Corporation, O=Microsoft Corporation, L=Redmond, ...").
function Test-MicrosoftSubject([string]$Subject) { return [bool]($Subject -match '(^|,\s*)O=Microsoft Corporation(,|$)') }

# The bootstrapper file the bundler fetched at build time (go.microsoft.com/fwlink/p/?LinkId=2124703, cached under
# %LOCALAPPDATA%\tauri) and put into both installers. Nothing else pins it, so it must exist, be about 1-3 MB (the real one is
# about 1.7 MB) and be Authenticode-signed (Valid) by Microsoft Corporation. Returns the failures.
function Get-BootstrapperFileFindings([string]$Path, [long]$MinBytes = 1000000, [long]$MaxBytes = 3000000) {
  $fail = New-Object System.Collections.Generic.List[string]
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    $fail.Add("the WebView2 bootstrapper the installers embed is not a file: $Path")
    return $fail.ToArray()
  }
  $size = (Get-Item -LiteralPath $Path).Length
  if ($size -lt $MinBytes -or $size -gt $MaxBytes) { $fail.Add("the WebView2 bootstrapper is $size bytes, expected $MinBytes to ${MaxBytes}: $Path") }
  $sig = Get-AuthenticodeSignature -LiteralPath $Path
  if ($sig.Status -ne 'Valid') { $fail.Add("the WebView2 bootstrapper has a $($sig.Status) Authenticode signature, expected Valid: $Path") }
  elseif (-not (Test-MicrosoftSubject $sig.SignerCertificate.Subject)) { $fail.Add("the WebView2 bootstrapper is signed by '$($sig.SignerCertificate.Subject)', not Microsoft Corporation: $Path") }
  return $fail.ToArray()
}

# The NSIS setup carries the bootstrapper when the tree 7-Zip extracted from it holds the file (the generated script writes it
# to $TEMP\MicrosoftEdgeWebview2Setup.exe), the one the build used when $ExpectedSha256 is given.
function Get-NsisBootstrapperFindings([string]$ExtractedDir, [string]$ExpectedSha256 = '') {
  $files = @(Get-ChildItem -LiteralPath $ExtractedDir -Recurse -Filter 'MicrosoftEdgeWebview2Setup.exe' -File -ErrorAction SilentlyContinue)
  if ($files.Count -eq 0) { return @('the NSIS setup does not contain MicrosoftEdgeWebview2Setup.exe (the WebView2 bootstrapper is not embedded)') }
  if ($ExpectedSha256 -and ((Get-FileHash -Algorithm SHA256 -LiteralPath $files[0].FullName).Hash.ToLowerInvariant() -ne $ExpectedSha256.ToLowerInvariant())) {
    return @('the WebView2 bootstrapper inside the NSIS setup is not the file the build used')
  }
  return @()
}

# Stream name -> size in bytes, from the listing `7z l -slt -tCompound <msi>` (an MSI is an OLE compound file; a Binary-table
# row is the stream Binary.<Name>).
function Get-CompoundStreamSizes([string[]]$Listing) {
  $sizes = @{}
  $name = $null
  foreach ($line in $Listing) {
    if ($line -match '^Path = (.+?)\s*$') { $name = $Matches[1] }
    elseif ($name -and $line -match '^Size = (\d+)\s*$') { $sizes[$name] = [long]$Matches[1]; $name = $null }
  }
  return $sizes
}

# The MSI carries the bootstrapper when its Binary table has the row, with the size of the file the build used when
# $ExpectedBytes is given.
function Get-MsiBootstrapperFindings([hashtable]$StreamSizes, [long]$ExpectedBytes = 0) {
  $key = 'Binary.MicrosoftEdgeWebview2Setup.exe'
  if (-not $StreamSizes.ContainsKey($key)) { return @('the MSI does not contain the Binary MicrosoftEdgeWebview2Setup.exe (the WebView2 bootstrapper is not embedded)') }
  if ($ExpectedBytes -gt 0 -and $StreamSizes[$key] -ne $ExpectedBytes) { return @("the WebView2 bootstrapper inside the MSI is $($StreamSizes[$key]) bytes, not the $ExpectedBytes of the file the build used") }
  return @()
}
#endregion webview2-install-mode

# --- 1. Build outputs and staged DLLs --------------------------------------------------------------
$exe = Join-Path $profileDir 'Sigil.exe'
if (-not (Test-Path -LiteralPath $exe)) { Fail "Sigil.exe not found at $exe"; Write-Host "`nFAILED"; exit 1 }
Pass "Sigil.exe $([math]::Round((Get-Item -LiteralPath $exe).Length / 1MB, 1)) MB"

# The paths compilers embed (Rust panic locations, C/C++ __FILE__, the PDB path) must not name the builder's user folder:
# pnpm tauri:release (scripts/release/tauri-release.mjs) remaps it; a plain `tauri build` leaves it in.
$homeDir = [Environment]::GetFolderPath('UserProfile')
$latin1 = [Text.Encoding]::GetEncoding(28591).GetString([IO.File]::ReadAllBytes($exe))
$homeHits = 0
foreach ($needle in @($homeDir, (-join ($homeDir.ToCharArray() | ForEach-Object { "$_`0" })))) {
  $at = $latin1.IndexOf($needle, [StringComparison]::OrdinalIgnoreCase)
  while ($at -ge 0) { $homeHits++; $at = $latin1.IndexOf($needle, $at + $needle.Length, [StringComparison]::OrdinalIgnoreCase) }
}
$latin1 = $null
if ($homeHits -gt 0) { Fail "Sigil.exe names the build machine's user folder $homeHits times (paths the compilers embedded): build with pnpm tauri:release" }
else { Pass "Sigil.exe does not name the build machine's user folder" }

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

# --- 1b. WebView2 install mode ---------------------------------------------------------------------
$bundleDir = Join-Path $profileDir 'bundle'
$msi = @(Get-ChildItem -LiteralPath (Join-Path $bundleDir 'msi') -Filter '*.msi' -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
$nsis = @(Get-ChildItem -LiteralPath (Join-Path $bundleDir 'nsis') -Filter '*-setup.exe' -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
# An installer that exists was built from a generated NSIS script / WiX source that has to be where the bundler leaves it.
$webview = Get-WebView2ModeFindings $expectedWebViewMode (Join-Path $PSScriptRoot 'tauri.release.conf.json') (Join-Path $profileDir 'nsis\x64\installer.nsi') (Join-Path $profileDir 'wix\x64\main.wxs') ([bool]($Strict -or $nsis.Count -gt 0)) ([bool]($Strict -or $msi.Count -gt 0))
foreach ($f in $webview.Failures) { Fail $f }
foreach ($w in $webview.Warnings) { Warn $w }
if ($webview.Failures.Count -eq 0 -and $webview.Warnings.Count -eq 0) { Pass "WebView2 install mode '$expectedWebViewMode' in the release overlay, the NSIS script and the WiX source" }
# The bootstrapper the bundler fetched is embedded in both installers and pinned by nothing else: size and Microsoft signature.
$bootstrapperSha = $null
$bootstrapperBytes = [long]0
foreach ($bootstrapperPath in @($webview.BootstrapperPaths)) {
  $problems = @(Get-BootstrapperFileFindings $bootstrapperPath)
  foreach ($problem in $problems) { Fail $problem }
  if ($problems.Count -eq 0) {
    Pass "WebView2 bootstrapper $([math]::Round((Get-Item -LiteralPath $bootstrapperPath).Length / 1MB, 2)) MB, signed by Microsoft Corporation ($bootstrapperPath)"
    if (-not $bootstrapperSha) { $bootstrapperSha = Hash $bootstrapperPath; $bootstrapperBytes = (Get-Item -LiteralPath $bootstrapperPath).Length }
    elseif ((Hash $bootstrapperPath) -ne $bootstrapperSha) { Fail "the NSIS setup and the MSI embed different WebView2 bootstrapper files ($bootstrapperPath)" }
  }
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
$msiOut = $null
$nsisOut = $null

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
    $sevenZip = $null
    $cmd7z = Get-Command 7z.exe -ErrorAction SilentlyContinue
    if ($cmd7z) { $sevenZip = $cmd7z.Source }
    elseif (Test-Path -LiteralPath (Join-Path $env:ProgramFiles '7-Zip\7z.exe')) { $sevenZip = Join-Path $env:ProgramFiles '7-Zip\7z.exe' }

    if ($msi.Count -eq 0) { Fail "no MSI in $bundleDir\msi" } else {
      $msiOut = Join-Path $tmpRoot 'msi'
      $p = Start-Process -FilePath 'msiexec.exe' -ArgumentList @('/a', "`"$($msi[0].FullName)`"", '/qn', "TARGETDIR=`"$msiOut`"") -Wait -PassThru
      if ($p.ExitCode -ne 0) { Fail "MSI administrative install failed (msiexec exit $($p.ExitCode))" }
      else { $installedExe = Test-InstalledLayout "MSI $($msi[0].Name)" $msiOut }
      # The bootstrapper is a Binary-table row, which an administrative install does not write out: list the MSI's streams.
      if (-not $sevenZip) { Warn '7-Zip not found; the WebView2 bootstrapper inside the MSI not checked' }
      else {
        $streams = @(& $sevenZip l -slt -tCompound $msi[0].FullName)
        if ($LASTEXITCODE -ne 0) { Fail "7-Zip could not list the streams of $($msi[0].Name)" }
        else {
          $msiProblems = @(Get-MsiBootstrapperFindings (Get-CompoundStreamSizes $streams) $bootstrapperBytes)
          foreach ($problem in $msiProblems) { Fail "MSI $($msi[0].Name): $problem" }
          if ($msiProblems.Count -eq 0) { Pass "MSI $($msi[0].Name): carries the WebView2 bootstrapper" }
        }
      }
    }

    if ($nsis.Count -eq 0) { Fail "no NSIS installer in $bundleDir\nsis" } else {
      if (-not $sevenZip) { Warn '7-Zip not found; NSIS installer contents not checked' }
      else {
        $nsisOut = Join-Path $tmpRoot 'nsis'
        & $sevenZip x -y "-o$nsisOut" $nsis[0].FullName | Out-Null
        if ($LASTEXITCODE -ne 0) { Fail "7-Zip could not extract $($nsis[0].Name)" }
        else {
          $nsisExe = Test-InstalledLayout "NSIS $($nsis[0].Name)" $nsisOut
          if (-not $installedExe) { $installedExe = $nsisExe }
          $nsisProblems = @(Get-NsisBootstrapperFindings $nsisOut $bootstrapperSha)
          foreach ($problem in $nsisProblems) { Fail "NSIS $($nsis[0].Name): $problem" }
          if ($nsisProblems.Count -eq 0) { Pass "NSIS $($nsis[0].Name): carries the WebView2 bootstrapper" }
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

  # --- 5. No secret value in the frontend bundle, the exe or the extracted installers ---------------------
  # Names only are printed. A secret is allowed in exactly one form: XOR-obfuscated inside Sigil.exe (build.rs), which
  # a plain-text search does not match; the frontend bundle (dist/) must not hold any.
  $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
  if (-not $nodeCmd) { Warn 'node not found; scripts/release/check-no-secrets.mjs was not run' }
  else {
    $scanArgs = @((Join-Path $PSScriptRoot 'check-no-secrets.mjs'), '--root', $RepoRoot, $exe)
    foreach ($dir in @($msiOut, $nsisOut)) { if ($dir -and (Test-Path -LiteralPath $dir)) { $scanArgs += $dir } }
    if ($RequireSecretValues) { $scanArgs += '--require-values' }
    $scanOut = @(& $nodeCmd.Source @scanArgs)
    $scanExit = $LASTEXITCODE
    foreach ($line in $scanOut) { Write-Host "       $line" }
    if ($scanExit -eq 0) { Pass 'no secret value in dist, Sigil.exe or the extracted installers' }
    else { Fail "check-no-secrets.mjs found a leaked secret or could not run (exit $scanExit)" }
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
