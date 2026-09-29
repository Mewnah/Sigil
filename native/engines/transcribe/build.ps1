#Requires -Version 7.0
<#
.SYNOPSIS
  Builds sigil_transcribe.dll runtime packs (transcribe.cpp behind the sigil_engine ABI v1).

.DESCRIPTION
  Reproducible, local-only pack build for P1-G2 (docs/adr/0003b-transcribe-cpp.md):
    1. fetches the transcribe.cpp archive pinned in UPSTREAM.json ONCE into the shared dev cache
       (%LOCALAPPDATA%\sigil-dev-cache\transcribe.cpp), refusing it unless size and sha256 match,
       and extracts it without upstream's agent / CI folders;
    2. configures each variant with Ninja + MSVC (Visual Studio 2022 toolset; CUDA 13 for cuda13);
    3. builds sigil_transcribe.dll and checks its exports (exactly the 16 sigil_* functions) and
       imports (no ggml*.dll, nothing outside the variant's allow-list);
    4. stages runtime packs: <OutDir>\transcribe-<variant>\ (DLL, licenses, sigil-pack.json with
       sha256s and the DLL's own engine info, as native_runtime expects)
       and a deterministic zip sigil-transcribe-<version>-<variant>.zip for local import
       (Models tab) or SIGIL_LOCAL_ASSETS. Packs stay local (catalog status "unpublished").
    5. -SelfTest runs sigil_transcribe_selftest.exe against the staged pack (ABI checks; a real
       transcription too with -Model/-Wav).

  Nothing else touches the network. Build trees live outside the repo (short paths keep ggml's
  Vulkan shader generator under MAX_PATH).

.PARAMETER Variant
  One or more of cpu, cpu-avx2, cpu-avx512, cuda13, vulkan ("cpu,cuda13" also works).
  cuda13 needs the CUDA 13 toolkit; vulkan needs the Vulkan SDK (VULKAN_SDK).

.PARAMETER FetchOnly
  Only fetch, verify and extract the pinned source into the dev cache (what
  scripts/native/build-runtime-packs.ps1 -Engine transcribe expects to find), then exit.

.PARAMETER CudaArchitectures
  cuda13 only: build for these CUDA architectures instead of UPSTREAM.json's pinned list (a
  faster development build). Needs -Dev unless it equals the pin. Every configure passes the
  effective list, so an override never outlives the build that asked for it.

.PARAMETER Dev
  Allow a development build whose CUDA architectures differ from the pin. Its zip is named
  sigil-transcribe-<version>-<variant>-dev.zip (never picked up as the catalog pack) and its
  sigil-pack.json says "dev_build": true. The staged folder stays transcribe-<variant> so the
  engine probe and the ignored tests can use it.

.EXAMPLE
  pwsh native/engines/transcribe/build.ps1 -Variant cpu,cuda13 -SelfTest
.EXAMPLE
  pwsh native/engines/transcribe/build.ps1 -FetchOnly
.EXAMPLE
  pwsh native/engines/transcribe/build.ps1 -Variant cuda13 -CudaArchitectures 89-real -Dev -SelfTest `
       -Model "$env:APPDATA\com.sigil.app\whisper\ggml-tiny.en.bin" -Wav <16k wav>
#>
[CmdletBinding()]
param(
    [string[]]$Variant = @('cpu', 'cpu-avx2'),
    [string]$OutDir,
    [string]$BuildRoot,
    [string]$SourceDir,
    [string]$CudaArchitectures,
    [int]$Jobs = 0,
    [switch]$SelfTest,
    [string]$Model,
    [string]$Wav,
    [switch]$Clean,
    [switch]$NoZip,
    [switch]$Cli,
    [switch]$FetchOnly,
    [switch]$Dev
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$EngineDir = $PSScriptRoot
$Upstream = Get-Content -Raw -LiteralPath (Join-Path $EngineDir 'UPSTREAM.json') | ConvertFrom-Json
$ShortCommit = $Upstream.commit.Substring(0, 7)
$EngineJson = Get-Content -Raw -LiteralPath (Join-Path $EngineDir 'engine.json') | ConvertFrom-Json
$CatalogVersion = "$($Upstream.version)-s$($Upstream.shim_revision)"
if ($EngineJson.version -ne $CatalogVersion) {
    throw "engine.json version '$($EngineJson.version)' must be '$CatalogVersion' (UPSTREAM.json version and shim_revision)"
}
# sigil_engine_get_info's version (= engine.json = the catalog version; the commit is in build_info).
$EngineVersion = $CatalogVersion
$PinnedCudaArchs = [string]$Upstream.cuda.architectures
$CudaArchs = if ($CudaArchitectures) { $CudaArchitectures.Trim() } else { $PinnedCudaArchs }
$CudaDevBuild = $CudaArchs -ne $PinnedCudaArchs
$Variants = @($Variant | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$Known = @('cpu', 'cpu-avx2', 'cpu-avx512', 'cuda13', 'vulkan')
foreach ($v in $Variants) {
    if ($Known -notcontains $v) { throw "unknown variant '$v' (expected: $($Known -join ', '))" }
}
if ($CudaDevBuild -and ($Variants -contains 'cuda13') -and -not $Dev) {
    throw "-CudaArchitectures '$CudaArchs' differs from UPSTREAM.json's pinned '$PinnedCudaArchs': add -Dev for a development build (its zip is named ...-cuda13-dev.zip), or drop -CudaArchitectures to build the shippable pack"
}
if (-not $BuildRoot) {
    $BuildRoot = if ($env:SIGIL_NATIVE_BUILD_ROOT) { $env:SIGIL_NATIVE_BUILD_ROOT } else { Join-Path $env:LOCALAPPDATA 'sigil-native' }
}
if (-not $OutDir) { $OutDir = Join-Path $BuildRoot 'packs' }
New-Item -ItemType Directory -Force -Path $BuildRoot, $OutDir | Out-Null
if ($Jobs -le 0) { $Jobs = [Environment]::ProcessorCount }

function Write-Step([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }

# ---------------------------------------------------------------------------------------------
# 1. Pinned source
# ---------------------------------------------------------------------------------------------

function Get-PinnedSource {
    $cache = if ($env:SIGIL_DEV_CACHE) { Join-Path $env:SIGIL_DEV_CACHE 'transcribe.cpp' } else { Join-Path $env:LOCALAPPDATA 'sigil-dev-cache\transcribe.cpp' }
    New-Item -ItemType Directory -Force -Path $cache | Out-Null
    $archive = Join-Path $cache $Upstream.archive.file
    $tree = Join-Path $cache "src-$ShortCommit"
    $marker = Join-Path $tree '.sigil-extracted'
    if ((Test-Path -LiteralPath $marker) -and ((Get-Content -Raw -LiteralPath $marker).Trim() -eq $Upstream.archive.sha256)) {
        return (Join-Path $tree $Upstream.archive.root)
    }
    if (-not (Test-Path -LiteralPath $archive)) {
        Write-Step "Downloading transcribe.cpp $($Upstream.tag) ($ShortCommit, $([math]::Round($Upstream.archive.size / 1MB, 1)) MB, one request)"
        $partial = "$archive.partial"
        Invoke-WebRequest -Uri $Upstream.archive.url -OutFile $partial -UseBasicParsing -MaximumRetryCount 2 -RetryIntervalSec 5
        Move-Item -Force -LiteralPath $partial -Destination $archive
    }
    $item = Get-Item -LiteralPath $archive
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant()
    if ($item.Length -ne [int64]$Upstream.archive.size -or $hash -ne $Upstream.archive.sha256) {
        throw "transcribe.cpp archive mismatch: $($item.Length) bytes, sha256 $hash (UPSTREAM.json pins $($Upstream.archive.size) / $($Upstream.archive.sha256)). Delete $archive to re-download; if GitHub regenerated the archive, re-pin UPSTREAM.json after reviewing the source."
    }
    Write-Step "Extracting $($Upstream.archive.file)"
    $staging = "$tree.staging"
    if (Test-Path -LiteralPath $staging) { Remove-Item -Recurse -Force -LiteralPath $staging }
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    # Upstream's agent-instruction and CI folders are not needed to build (ggml/.pi holds a symlink
    # Windows cannot create without Developer Mode).
    $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
    & $tar -xzf $archive -C $staging --exclude '*/.claude' --exclude '*/.pi' --exclude '*/.github' --exclude '*/.cargo'
    if ($LASTEXITCODE -ne 0) { throw "tar failed ($LASTEXITCODE) extracting $archive" }
    if (Test-Path -LiteralPath $tree) { Remove-Item -Recurse -Force -LiteralPath $tree }
    Move-Item -LiteralPath $staging -Destination $tree
    Set-Content -NoNewline -LiteralPath $marker -Value $Upstream.archive.sha256
    return (Join-Path $tree $Upstream.archive.root)
}

if ($SourceDir) {
    $Src = (Resolve-Path -LiteralPath $SourceDir).Path
} else {
    $Src = Get-PinnedSource
}
if ($FetchOnly) {
    Write-Step "Pinned source ready: $Src"
    exit 0
}
if (-not (Test-Path -LiteralPath (Join-Path $Src 'include\transcribe.h'))) { throw "$Src is not a transcribe.cpp source tree" }
Write-Step "transcribe.cpp source: $Src"

# ---------------------------------------------------------------------------------------------
# 2. Toolchain (MSVC 2022 environment, Ninja, CMake, CUDA 13)
# ---------------------------------------------------------------------------------------------

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw 'vswhere.exe not found: install Visual Studio 2022 Build Tools with the C++ workload' }
# CUDA 13.x supports the VS 2022 (v143) host compiler; prefer it for every variant so all packs share one toolset.
$vs = & $vswhere -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath -latest
if (-not $vs) {
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw 'No Visual Studio installation with the C++ x64 tools was found' }
    Write-Warning "Visual Studio 2022 not found; using $vs (nvcc may reject a newer host compiler)"
}
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
if (-not $env:VCToolsInstallDir) {
    Write-Step "Importing the MSVC x64 environment from $vs"
    $lines = & cmd.exe /d /c "`"$vcvars`" >nul 2>&1 && set"
    if ($LASTEXITCODE -ne 0) { throw "vcvars64.bat failed ($LASTEXITCODE)" }
    foreach ($line in $lines) {
        $i = $line.IndexOf('=')
        if ($i -gt 0) { Set-Item -Path "env:$($line.Substring(0, $i))" -Value $line.Substring($i + 1) }
    }
}
$ninja = (Get-Command ninja -ErrorAction SilentlyContinue)?.Source
if (-not $ninja) { $ninja = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe' }
if (-not (Test-Path -LiteralPath $ninja)) { throw 'ninja.exe not found (install the "C++ CMake tools" component or put ninja on PATH)' }
$cmake = (Get-Command cmake -ErrorAction SilentlyContinue)?.Source
if (-not $cmake) { $cmake = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' }
if (-not (Test-Path -LiteralPath $cmake)) { throw 'cmake.exe not found' }
$dumpbin = (Get-Command dumpbin -ErrorAction SilentlyContinue)?.Source
$msvcVersion = $env:VCToolsVersion

function Find-Cuda13 {
    $candidates = @()
    foreach ($name in (Get-ChildItem env: | Where-Object { $_.Name -match '^CUDA_PATH_V13_\d+$' } | Sort-Object Name -Descending)) { $candidates += $name.Value }
    if ($env:CUDA_PATH) { $candidates += $env:CUDA_PATH }
    $root = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path -LiteralPath $root) {
        $candidates += Get-ChildItem -LiteralPath $root -Directory | Where-Object { $_.Name -match '^v13\.\d+$' } |
            Sort-Object { [version]$_.Name.Substring(1) } -Descending | ForEach-Object FullName
    }
    foreach ($c in $candidates) {
        $nvcc = Join-Path $c 'bin\nvcc.exe'
        if ((Test-Path -LiteralPath $nvcc) -and ((& $nvcc --version) -join "`n") -match 'release 13\.') { return $c }
    }
    throw 'CUDA 13 toolkit not found (install CUDA 13.x or set CUDA_PATH_V13_x)'
}

# ---------------------------------------------------------------------------------------------
# 3-5. Build, check, stage, self-test
# ---------------------------------------------------------------------------------------------

$BaseImports = @('KERNEL32.dll', 'ADVAPI32.dll', 'VCRUNTIME140.dll', 'VCRUNTIME140_1.dll', 'MSVCP140.dll')
$VariantImports = @{
    'cuda13' = @('cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll', 'nvcuda.dll')
    'vulkan' = @('vulkan-1.dll')
}
$CpuFloor = @{ 'cpu' = 'sse4.2'; 'cpu-avx2' = 'avx2'; 'cpu-avx512' = 'avx512'; 'cuda13' = 'avx2'; 'vulkan' = 'avx2' }

function Get-PeImports([string]$Dll) {
    if (-not $dumpbin) { return $null }
    $out = & $dumpbin /nologo /dependents $Dll
    return @($out | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '\.dll$' -and $_ -notmatch '[\\/:]' })
}

function Get-PeExports([string]$Dll) {
    if (-not $dumpbin) { return $null }
    $out = & $dumpbin /nologo /exports $Dll
    $names = @()
    foreach ($line in $out) {
        if ($line -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)') { $names += $Matches[1] }
    }
    return $names
}

function Write-DeterministicZip([string]$SourceDir, [string]$ZipPath) {
    Add-Type -AssemblyName System.IO.Compression
    if (Test-Path -LiteralPath $ZipPath) { Remove-Item -Force -LiteralPath $ZipPath }
    $stamp = [DateTimeOffset]::new(2026, 1, 1, 0, 0, 0, [TimeSpan]::Zero)
    $fs = [System.IO.File]::Open($ZipPath, [System.IO.FileMode]::CreateNew)
    try {
        $zip = [System.IO.Compression.ZipArchive]::new($fs, [System.IO.Compression.ZipArchiveMode]::Create)
        try {
            foreach ($f in (Get-ChildItem -LiteralPath $SourceDir -File | Sort-Object Name)) {
                $entry = $zip.CreateEntry($f.Name, [System.IO.Compression.CompressionLevel]::Optimal)
                $entry.LastWriteTime = $stamp
                $in = [System.IO.File]::OpenRead($f.FullName)
                $es = $entry.Open()
                try { $in.CopyTo($es) } finally { $es.Dispose(); $in.Dispose() }
            }
        } finally { $zip.Dispose() }
    } finally { $fs.Dispose() }
}

$cudaRoot = $null
$results = @()
foreach ($v in $Variants) {
    $bdir = Join-Path $BuildRoot "transcribe\$v"
    if ($Clean -and (Test-Path -LiteralPath $bdir)) { Remove-Item -Recurse -Force -LiteralPath $bdir }
    $cfg = @('-S', $EngineDir, '-B', $bdir, '-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$ninja", '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl', "-DTRANSCRIBE_SOURCE_DIR=$Src", "-DSIGIL_VARIANT=$v",
        "-DSIGIL_TRANSCRIBE_CLI=$(if ($Cli) { 'ON' } else { 'OFF' })")
    if ($v -eq 'cuda13') {
        if (-not $cudaRoot) { $cudaRoot = Find-Cuda13 }
        $cfg += @("-DCMAKE_CUDA_COMPILER=$(Join-Path $cudaRoot 'bin\nvcc.exe')", "-DCUDAToolkit_ROOT=$cudaRoot")
        # Always set (empty = the pin): a build tree reused after a -Dev build returns to the pin.
        $cfg += "-DSIGIL_TRANSCRIBE_CUDA_ARCHS_OVERRIDE=$(if ($CudaDevBuild) { $CudaArchs } else { '' })"
    }
    if ($v -eq 'vulkan' -and -not ($env:VULKAN_SDK -and (Test-Path -LiteralPath $env:VULKAN_SDK))) {
        throw 'the vulkan variant needs the Vulkan SDK (VULKAN_SDK not set); build it on a machine with the SDK (ADR 0003b)'
    }
    Write-Step "Configuring $v"
    & $cmake @cfg
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed for $v" }
    # The self-test tool is always built: it reads the DLL's own sigil_engine_get_info for the manifest.
    $targets = @('sigil_transcribe', 'sigil_transcribe_selftest')
    if ($Cli) { $targets += 'transcribe-cli' }
    Write-Step "Building $v ($($targets -join ', '), $Jobs jobs)"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $cmake --build $bdir --target @targets --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw "build failed for $v" }
    $buildSeconds = [math]::Round($sw.Elapsed.TotalSeconds)
    $dll = Join-Path $bdir 'out\sigil_transcribe.dll'
    $selftestExe = Join-Path $bdir 'tools\sigil_transcribe_selftest.exe'
    if (-not (Test-Path -LiteralPath $dll)) { throw "$dll was not produced" }

    # Export / import checks.
    $exports = Get-PeExports $dll
    $imports = Get-PeImports $dll
    if ($null -ne $exports) {
        $unexpected = @($exports | Where-Object { $_ -notmatch '^sigil_(engine|stream|free)_' })
        if ($exports.Count -ne 16 -or $unexpected.Count -gt 0) {
            throw "sigil_transcribe.dll ($v) exports $($exports.Count) symbols; expected exactly the 16 sigil_* functions. Unexpected: $($unexpected -join ', ')"
        }
    }
    if ($null -ne $imports) {
        $allowed = $BaseImports + ($VariantImports[$v] ?? @())
        $bad = @($imports | Where-Object { $_ -notlike 'api-ms-win-*' -and ($allowed -notcontains $_) })
        if ($bad.Count -gt 0) { throw "sigil_transcribe.dll ($v) imports unexpected DLLs: $($bad -join ', ')" }
        if (@($imports | Where-Object { $_ -like 'ggml*' }).Count -gt 0) { throw "sigil_transcribe.dll ($v) imports a ggml DLL" }
    }

    # Stage the pack.
    $pack = Join-Path $OutDir "transcribe-$v"
    if (Test-Path -LiteralPath $pack) { Remove-Item -Recurse -Force -LiteralPath $pack }
    New-Item -ItemType Directory -Force -Path $pack | Out-Null
    Copy-Item -LiteralPath $dll -Destination $pack
    Copy-Item -LiteralPath (Join-Path $Src 'LICENSE') -Destination (Join-Path $pack 'LICENSE-transcribe.cpp.txt')
    Copy-Item -LiteralPath (Join-Path $Src 'THIRD-PARTY-LICENSES.md') -Destination (Join-Path $pack 'THIRD-PARTY-LICENSES-transcribe.cpp.md')
    Copy-Item -LiteralPath (Join-Path $EngineDir 'NOTICE.txt') -Destination (Join-Path $pack 'NOTICE.txt')

    # The DLL's own info (loading it also proves every import resolves on this PC).
    $dllDirs = @()
    if ($v -eq 'cuda13') { $dllDirs = @('--dll-dir', (Join-Path $cudaRoot 'bin\x64'), '--dll-dir', (Join-Path $cudaRoot 'bin')) }
    $infoFile = Join-Path $bdir 'engine-info.json'
    if (Test-Path -LiteralPath $infoFile) { Remove-Item -Force -LiteralPath $infoFile }
    $abiOut = & $selftestExe $pack @dllDirs --info-json $infoFile
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $infoFile)) {
        $abiOut | Write-Host
        throw "sigil_transcribe.dll ($v) failed its ABI self-check"
    }
    $info = Get-Content -Raw -LiteralPath $infoFile | ConvertFrom-Json
    if ($info.version -ne $EngineVersion -or $info.variant -ne $v -or $info.name -ne 'transcribe') {
        throw "sigil_transcribe.dll reports $($info.name) $($info.version) [$($info.variant)], expected transcribe $EngineVersion [$v]"
    }
    $isDevPack = $false
    if ($v -eq 'cuda13') {
        # The DLL's own build info names the architectures it was compiled for: never stage a pack
        # whose kernels differ from what was asked (a stale build tree, an edited pin).
        $builtArchs = if ($info.build_info -match 'archs ([^,(]+)') { ($Matches[1].Trim() -split '\s+') -join ';' } else { '' }
        if ($builtArchs -ne $CudaArchs) {
            Remove-Item -Recurse -Force -LiteralPath $pack
            throw "sigil_transcribe.dll (cuda13) was built for CUDA architectures '$builtArchs', expected '$CudaArchs'; no pack was staged (rebuild with -Clean)"
        }
        $isDevPack = $CudaDevBuild
    }

    # sigil-pack.json: the native_runtime pack manifest (schema 1; every file hashed; checked by
    # Sigil before the DLL is loaded and against the DLL's own info after). Extra keys are ours.
    $files = @(Get-ChildItem -LiteralPath $pack -File | Sort-Object Name | ForEach-Object {
            [ordered]@{ path = $_.Name; size = $_.Length; sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant() }
        })
    $cudaVersion = $null
    if ($v -eq 'cuda13') { $cudaVersion = ((& (Join-Path $cudaRoot 'bin\nvcc.exe') --version) -join ' ') -replace '.*release ([0-9.]+).*', '$1' }
    $manifest = [ordered]@{
        schema          = 1
        engine          = 'transcribe'
        version         = $info.version
        variant         = $v
        abi_version     = [int]$info.abi_version
        library         = 'sigil_transcribe.dll'
        kind            = $info.kind
        capabilities    = [uint64]$info.capabilities
        build_info      = $info.build_info
        files           = $files
        catalog_id      = if ($isDevPack) { $null } else { "runtime/sigil-transcribe-$v" }
        catalog_version = $CatalogVersion
        dev_build       = [bool]$isDevPack
        upstream        = [ordered]@{ repo = $Upstream.repo; tag = $Upstream.tag; commit = $Upstream.commit; archive_sha256 = $Upstream.archive.sha256; ggml_commit = $Upstream.ggml_commit }
        requires        = [ordered]@{ cpu = $CpuFloor[$v]; imports = @($imports ?? @()) }
        toolchain       = [ordered]@{ msvc = $msvcVersion; cmake = ((& $cmake --version | Select-Object -First 1) -replace 'cmake version ', ''); cuda = $cudaVersion }
    }
    if ($v -eq 'cuda13') {
        $manifest.requires['nvidia'] = 'driver R580+, Turing (sm_75) or newer; CUDA 13 runtime DLLs from runtime/nvidia-cuda13'
        $manifest['cuda_architectures'] = $CudaArchs
    }
    ($manifest | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath (Join-Path $pack 'sigil-pack.json') -Encoding utf8NoBOM

    $zipPath = $null
    $zipHash = $null
    if (-not $NoZip) {
        $zipSuffix = if ($isDevPack) { '-dev' } else { '' }
        $zipPath = Join-Path $OutDir "sigil-transcribe-$CatalogVersion-$v$zipSuffix.zip"
        Write-DeterministicZip $pack $zipPath
        $zipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash.ToLowerInvariant()
        Set-Content -LiteralPath "$zipPath.sha256" -Value "$zipHash  $(Split-Path -Leaf $zipPath)" -Encoding ascii
    }

    $testResult = 'abi ok'
    if ($SelfTest) {
        $stArgs = @($pack) + $dllDirs
        if ($Model) { $stArgs += @('--model', $Model) }
        if ($Wav) { $stArgs += @('--wav', $Wav) }
        Write-Step "Self-test $v"
        & $selftestExe @stArgs
        $testResult = if ($LASTEXITCODE -eq 0) { 'pass' } else { "FAIL ($LASTEXITCODE)" }
    }
    $dllItem = Get-Item -LiteralPath (Join-Path $pack 'sigil_transcribe.dll')
    $results += [pscustomobject]@{
        Variant   = $v
        DllMB     = [math]::Round($dllItem.Length / 1MB, 1)
        BuildSec  = $buildSeconds
        SelfTest  = $testResult
        Dev       = $isDevPack
        Zip       = if ($zipPath) { Split-Path -Leaf $zipPath } else { '-' }
        ZipSha256 = $zipHash
    }
}

Write-Step "Packs in $OutDir"
$results | Format-Table -AutoSize | Out-String | Write-Host
if (@($results | Where-Object { $_.SelfTest -like 'FAIL*' }).Count -gt 0) { exit 1 }
