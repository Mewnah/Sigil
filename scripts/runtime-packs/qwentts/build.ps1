#Requires -Version 7.0
<#
.SYNOPSIS
  Builds the runtime/qwentts-server-* packs: sigil-tts-server.exe (Qwen3-TTS through qwentts.cpp,
  Sigil's server build) for local import in Models & Runtimes, and measures it.

.DESCRIPTION
  Local-only pack build for P6-E (asset hosting is local only; the catalog entries stay
  "unpublished"):
    1. fetches the two pinned archives of UPSTREAM.json ONCE into the shared dev cache
       (%LOCALAPPDATA%\sigil-dev-cache\qwentts\archives): qwentts.cpp and its ggml fork submodule,
       one request each, a few seconds apart; both are checked against their pinned size and sha256;
    2. extracts them into one source tree and overlays server/ (tools/sigil-tts-server.cpp and the
       target appended to CMakeLists.txt);
    3. configures each variant with Ninja + MSVC (Visual Studio 2022 toolset; CUDA 13 for cuda13,
       the Vulkan SDK for vulkan): static ggml inside the exe (BUILD_SHARED_LIBS=OFF, no backend
       DLLs), GGML_NATIVE=OFF with the AVX2 baseline (never tuned for the build machine's CPU);
    4. builds sigil-tts-server.exe and checks its imports: Windows and MSVC runtime DLLs, plus
       cudart64_13 / cublas64_13 / cublasLt64_13 / nvcuda (cuda13, from runtime/nvidia-cuda13 and the
       driver) or vulkan-1 (vulkan, from the driver); never a ggml DLL;
    5. stages <OutDir>\qwentts-server-<variant>\ (the exe, sigil-pack.json with the sha256 of every
       file, the licenses) and a deterministic zip sigil-qwentts-server-<version>-<variant>.zip (the
       catalog's file name) plus .zip.sha256, for Import in Models & Runtimes or SIGIL_LOCAL_ASSETS;
    6. -SelfTest runs the exe (--version proves every import resolves). With -Talker and -Codec it
       starts the server on a free loopback port with a random key and logs, per request, the time
       to the first audio and the real-time factor: one sentence per -Text line (preset / design
       talkers speak with -Voice / -Instruct), and on a base talker a cloned voice from -RefWav (or
       upstream's examples/freeman.wav) registered through the same route Sigil uses, then spoken
       with its cached latents. The WAVs and a JSON report go to <OutDir>\qwentts-selftest\.
       GPU timing runs hold -GpuLock (another phase may be measuring the voice changer on the GPU).

  Nothing else touches the network.

.PARAMETER Variant
  cpu, vulkan, cuda13 (comma-separated or repeated). Default cuda13,cpu (vulkan needs VULKAN_SDK).

.PARAMETER CudaArchitectures
  cuda13 only: build for these architectures instead of UPSTREAM.json's pin (a faster development
  build, e.g. 89-real). Needs -Dev; the zip is then named ...-cuda13-dev.zip and sigil-pack.json has
  no catalog id.

.PARAMETER SkipBuild
  With -SelfTest: test the packs already staged in <OutDir>\qwentts-server-<variant>\ instead of
  building (no compiler needed; a cuda13 pack still takes the CUDA DLLs from the toolkit).

.EXAMPLE
  pwsh scripts/runtime-packs/qwentts/build.ps1 -FetchOnly
.EXAMPLE
  pwsh scripts/runtime-packs/qwentts/build.ps1 -Variant cuda13,cpu
.EXAMPLE
  pwsh scripts/runtime-packs/qwentts/build.ps1 -Variant cuda13 -CudaArchitectures 89-real -Dev -SelfTest `
      -Talker "$env:APPDATA\com.sigil.app\models\tts\qwen3-tts-1.7b-base\<ver>\qwen-talker-1.7b-base-Q8_0.gguf" `
      -Codec "$env:APPDATA\com.sigil.app\models\tts\qwen3-tts-tokenizer\<ver>\qwen-tokenizer-12hz-Q8_0.gguf"
.EXAMPLE
  pwsh scripts/runtime-packs/qwentts/build.ps1 -Variant cuda13 -SkipBuild -SelfTest -Talker <gguf> -Codec <gguf>
#>
[CmdletBinding()]
param(
    [string[]]$Variant = @('cuda13', 'cpu'),
    [string]$OutDir,
    [string]$BuildRoot,
    [string]$SourceDir,
    [int]$Jobs = 0,
    [string]$CudaArchitectures,
    [switch]$Dev,
    [switch]$FetchOnly,
    [switch]$Clean,
    [switch]$NoZip,
    [switch]$SelfTest,
    [switch]$SkipBuild,
    [string]$Talker,
    [string]$Codec,
    [string[]]$Text = @(
        'Hello! This is Sigil speaking with Qwen three text to speech, streamed sentence by sentence.',
        'The quick brown fox jumps over the lazy dog, and then it takes a well deserved nap in the warm afternoon sun.'
    ),
    [string]$Language = 'english',
    [string]$Voice = '',
    [string]$Instruct = '',
    [string]$RefWav,
    [string]$RefText,
    [string]$GpuLock,
    [int]$LoadTimeoutSec = 300
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$PackDir = $PSScriptRoot
$Upstream = Get-Content -Raw -LiteralPath (Join-Path $PackDir 'UPSTREAM.json') | ConvertFrom-Json
$ShortCommit = $Upstream.commit.Substring(0, 7)
$CatalogVersion = [string]$Upstream.catalog_version
if ($CatalogVersion -ne "$ShortCommit-s$($Upstream.shim_revision)") {
    throw "UPSTREAM.json catalog_version '$CatalogVersion' must be '$ShortCommit-s$($Upstream.shim_revision)'"
}
$PinnedCudaArchs = [string]$Upstream.cuda.architectures
$CudaArchs = if ($CudaArchitectures) { $CudaArchitectures.Trim() } else { $PinnedCudaArchs }
$CudaDevBuild = $CudaArchs -ne $PinnedCudaArchs
$Variants = @($Variant | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$Known = @('cpu', 'vulkan', 'cuda13')
foreach ($v in $Variants) {
    if ($Known -notcontains $v) { throw "unknown variant '$v' (expected: $($Known -join ', '))" }
}
if ($SkipBuild -and -not ($SelfTest -and $Talker)) {
    throw '-SkipBuild only runs the self-test: add -SelfTest -Talker <gguf> -Codec <gguf>'
}
if ($CudaDevBuild -and ($Variants -contains 'cuda13') -and -not $Dev -and -not $SkipBuild) {
    throw "-CudaArchitectures '$CudaArchs' differs from UPSTREAM.json's pinned '$PinnedCudaArchs': add -Dev for a development build (its zip is named ...-cuda13-dev.zip)"
}
if (-not $BuildRoot) {
    $BuildRoot = if ($env:SIGIL_NATIVE_BUILD_ROOT) { $env:SIGIL_NATIVE_BUILD_ROOT } else { Join-Path $env:LOCALAPPDATA 'sigil-native' }
}
if (-not $OutDir) { $OutDir = Join-Path $BuildRoot 'packs' }
New-Item -ItemType Directory -Force -Path $BuildRoot, $OutDir | Out-Null
if ($Jobs -le 0) { $Jobs = [Math]::Max(1, [Environment]::ProcessorCount - 2) }
$CacheRoot = if ($env:SIGIL_DEV_CACHE) { Join-Path $env:SIGIL_DEV_CACHE 'qwentts' } else { Join-Path $env:LOCALAPPDATA 'sigil-dev-cache\qwentts' }

function Write-Step([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }

# ---------------------------------------------------------------------------------------------
# 1-2. Pinned sources plus the Sigil overlay
# ---------------------------------------------------------------------------------------------

$script:LastRequest = [DateTime]::MinValue
function Wait-RequestSpacing {
    # The dev PC's router blocks bursts of connections: at least 3 s between request starts.
    $wait = 3000 - ([DateTime]::UtcNow - $script:LastRequest).TotalMilliseconds
    if ($wait -gt 0) { Start-Sleep -Milliseconds ([int]$wait) }
    $script:LastRequest = [DateTime]::UtcNow
}

function Get-PinnedArchive($Pin, [string]$Label) {
    $dir = Join-Path $CacheRoot 'archives'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $file = Join-Path $dir $Pin.file
    if (-not (Test-Path -LiteralPath $file)) {
        Wait-RequestSpacing
        Write-Step "Downloading $Label ($([math]::Round($Pin.size / 1MB, 1)) MB, one request)"
        $partial = "$file.partial"
        Invoke-WebRequest -Uri $Pin.url -OutFile $partial -UseBasicParsing -MaximumRetryCount 2 -RetryIntervalSec 5
        Move-Item -Force -LiteralPath $partial -Destination $file
    }
    $size = (Get-Item -LiteralPath $file).Length
    $sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash.ToLowerInvariant()
    if ($size -ne [int64]$Pin.size -or $sha -ne $Pin.sha256) {
        throw "$Label archive mismatch: $size bytes, sha256 $sha (UPSTREAM.json pins $($Pin.size) / $($Pin.sha256)). Delete $file to download it again; if GitHub regenerated the archive, re-pin after reviewing the source."
    }
    return $file
}

function Get-OverlayHash {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Collections.Generic.List[byte]]::new()
        foreach ($f in (Get-ChildItem -LiteralPath (Join-Path $PackDir 'server') -File | Sort-Object Name)) {
            $bytes.AddRange([System.Text.Encoding]::UTF8.GetBytes($f.Name))
            $bytes.AddRange([System.IO.File]::ReadAllBytes($f.FullName))
        }
        return ([System.BitConverter]::ToString($sha.ComputeHash($bytes.ToArray())) -replace '-', '').ToLowerInvariant()
    } finally { $sha.Dispose() }
}

function Expand-Archive-Into([string]$Archive, [string]$Destination) {
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
    # Drop the archive's top folder; skip CI and agent folders (not needed to build).
    & $tar -xzf $Archive -C $Destination --strip-components=1 --exclude '*/.github' --exclude '*/.claude' --exclude '*/.pi'
    if ($LASTEXITCODE -ne 0) { throw "tar failed ($LASTEXITCODE) extracting $Archive" }
}

# Bump when the preparation below changes, so trees prepared by an older build.ps1 are redone.
$PrepRevision = 2

# Upstream's tools/version.cmake stamps `git rev-parse` into qt_version(); an archive has no .git
# (it would say "unknown (unknown)"), so the prepared tree stamps the pinned commit and its date.
function Get-VersionCmake {
    $stamp = "$ShortCommit ($($Upstream.commit_date))"
    return @"
# Sigil (scripts/runtime-packs/qwentts/build.ps1): this tree is the pinned archive of UPSTREAM.json,
# not a git checkout, so the version is the pinned commit instead of ``git rev-parse``.
# Usage (as upstream): cmake -DSRC_DIR=... -DOUTPUT=... -P version.cmake
set(CONTENT "#pragma once\n#define QWEN_VERSION \"$stamp\"\n")
if(EXISTS "`${OUTPUT}")
    file(READ "`${OUTPUT}" EXISTING)
    if("`${EXISTING}" STREQUAL "`${CONTENT}")
        return()
    endif()
endif()
file(WRITE "`${OUTPUT}" "`${CONTENT}")
"@
}

function Get-SourceTree {
    $qArchive = Get-PinnedArchive $Upstream.archive "qwentts.cpp $ShortCommit"
    $gArchive = Get-PinnedArchive $Upstream.ggml.archive "ggml fork $($Upstream.ggml.commit.Substring(0, 7))"
    if ($FetchOnly) { return $null }
    $tree = Join-Path $CacheRoot "src-$ShortCommit-s$($Upstream.shim_revision)"
    $marker = Join-Path $tree '.sigil-prepared'
    $expected = "prep$PrepRevision,$($Upstream.archive.sha256),$($Upstream.ggml.archive.sha256),$(Get-OverlayHash)"
    if ((Test-Path -LiteralPath $marker) -and ((Get-Content -Raw -LiteralPath $marker).Trim() -eq $expected)) { return $tree }
    Write-Step "Preparing the source tree $tree"
    $staging = "$tree.staging"
    if (Test-Path -LiteralPath $staging) { Remove-Item -Recurse -Force -LiteralPath $staging }
    Expand-Archive-Into $qArchive $staging
    $ggml = Join-Path $staging 'ggml'
    if (Test-Path -LiteralPath $ggml) { Remove-Item -Recurse -Force -LiteralPath $ggml }
    Expand-Archive-Into $gArchive $ggml
    if (-not (Test-Path -LiteralPath (Join-Path $ggml 'CMakeLists.txt'))) { throw 'the ggml archive has no CMakeLists.txt' }
    # Sigil overlay: our server source next to upstream's tools, its target after upstream's.
    Copy-Item -LiteralPath (Join-Path $PackDir 'server\sigil-tts-server.cpp') -Destination (Join-Path $staging 'tools\sigil-tts-server.cpp')
    $append = Get-Content -Raw -LiteralPath (Join-Path $PackDir 'server\CMakeLists.append.cmake')
    Add-Content -LiteralPath (Join-Path $staging 'CMakeLists.txt') -Value $append -Encoding utf8NoBOM
    $versionScript = Join-Path $staging 'tools\version.cmake'
    if (-not (Test-Path -LiteralPath $versionScript)) { throw 'the qwentts.cpp archive has no tools/version.cmake (re-read upstream before upgrading)' }
    Set-Content -LiteralPath $versionScript -Value (Get-VersionCmake) -Encoding utf8NoBOM
    if (Test-Path -LiteralPath $tree) { Remove-Item -Recurse -Force -LiteralPath $tree }
    Move-Item -LiteralPath $staging -Destination $tree
    Set-Content -NoNewline -LiteralPath $marker -Value $expected
    return $tree
}

$Src = if ($SourceDir) { (Resolve-Path -LiteralPath $SourceDir).Path } else { Get-SourceTree }
if ($FetchOnly) {
    Write-Step 'Pinned archives ready'
    exit 0
}
if (-not (Test-Path -LiteralPath (Join-Path $Src 'tools\sigil-tts-server.cpp'))) { throw "$Src is not a prepared qwentts.cpp tree (tools\sigil-tts-server.cpp is missing)" }
Write-Step "qwentts.cpp source: $Src"

# ---------------------------------------------------------------------------------------------
# 3. Toolchain (MSVC 2022 environment, Ninja, CMake, CUDA 13, Vulkan SDK)
# ---------------------------------------------------------------------------------------------

$ninja = $null
$cmake = $null
$dumpbin = $null
$msvcVersion = $null
if (-not $SkipBuild) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { throw 'vswhere.exe not found: install Visual Studio 2022 Build Tools with the C++ workload' }
    $vs = & $vswhere -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath -latest
    if (-not $vs) {
        $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if (-not $vs) { throw 'No Visual Studio installation with the C++ x64 tools was found' }
        Write-Warning "Visual Studio 2022 not found; using $vs (nvcc may reject a newer host compiler)"
    }
    if (-not $env:VCToolsInstallDir) {
        Write-Step "Importing the MSVC x64 environment from $vs"
        $vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
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
}

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
# 4-5. Build, check, stage
# ---------------------------------------------------------------------------------------------

$BaseImports = @('KERNEL32.dll', 'ADVAPI32.dll', 'WS2_32.dll', 'CRYPT32.dll', 'USER32.dll', 'SHELL32.dll', 'ole32.dll', 'OLEAUT32.dll',
    'VCRUNTIME140.dll', 'VCRUNTIME140_1.dll', 'MSVCP140.dll')
$VariantImports = @{
    'cuda13' = @('cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll', 'nvcuda.dll')
    'vulkan' = @('vulkan-1.dll')
}
$Exe = 'sigil-tts-server.exe'

function Get-PeImports([string]$Path) {
    if (-not $dumpbin) { return $null }
    $out = & $dumpbin /nologo /dependents $Path
    return @($out | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '\.dll$' -and $_ -notmatch '[\\/:]' })
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

function Write-ThirdPartyLicenses([string]$Path) {
    $yyjsonHeader = (Get-Content -LiteralPath (Join-Path $Src 'vendor\yyjson\yyjson.h') -TotalCount 24) -join "`n"
    $parts = @(
        "Third-party components of sigil-tts-server.exe (runtime/qwentts-server-*)`n",
        "== qwentts.cpp ($($Upstream.repo), commit $($Upstream.commit)) ==",
        (Get-Content -Raw -LiteralPath (Join-Path $Src 'LICENSE')),
        "Upstream model: Qwen3-TTS by Alibaba / Qwen team, Apache 2.0. Audio codec: Qwen3-TTS-Tokenizer-12Hz, Apache 2.0.`n",
        "== ggml ($($Upstream.ggml.repo), commit $($Upstream.ggml.commit)) ==",
        (Get-Content -Raw -LiteralPath (Join-Path $Src 'ggml\LICENSE')),
        '== cpp-httplib (vendor/cpp-httplib) ==',
        (Get-Content -Raw -LiteralPath (Join-Path $Src 'vendor\cpp-httplib\LICENSE')),
        '== yyjson (vendor/yyjson) ==',
        $yyjsonHeader
    )
    ($parts -join "`n") | Set-Content -LiteralPath $Path -Encoding utf8NoBOM
}

$cudaRoot = $null
$results = @()
$builtExe = @{}
foreach ($v in $Variants) {
    if ($SkipBuild) {
        $staged = Join-Path $OutDir "qwentts-server-$v\$Exe"
        if (-not (Test-Path -LiteralPath $staged)) { throw "-SkipBuild: $staged is not built yet (run without -SkipBuild first)" }
        if ($v -eq 'cuda13' -and -not $cudaRoot) { $cudaRoot = Find-Cuda13 }
        $builtExe[$v] = $staged
        continue
    }
    $bdir = Join-Path $BuildRoot "qwentts\$v"
    if ($Clean -and (Test-Path -LiteralPath $bdir)) { Remove-Item -Recurse -Force -LiteralPath $bdir }
    $cfg = @('-S', $Src, '-B', $bdir, '-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$ninja", '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl', '-DBUILD_SHARED_LIBS=OFF', '-DGGML_BACKEND_DL=OFF',
        '-DGGML_NATIVE=OFF', '-DGGML_AVX=ON', '-DGGML_AVX2=ON', '-DGGML_FMA=ON', '-DGGML_F16C=ON',
        '-DGGML_BUILD_TESTS=OFF', '-DGGML_BUILD_EXAMPLES=OFF', '-DGGML_OPENMP=OFF',
        "-DSIGIL_QWENTTS_COMMIT=$ShortCommit", "-DSIGIL_PACK_VERSION=$CatalogVersion", "-DSIGIL_PACK_VARIANT=$v")
    switch ($v) {
        'cuda13' {
            if (-not $cudaRoot) { $cudaRoot = Find-Cuda13 }
            $cfg += @('-DGGML_CUDA=ON', '-DGGML_VULKAN=OFF', "-DCMAKE_CUDA_COMPILER=$(Join-Path $cudaRoot 'bin\nvcc.exe')", "-DCUDAToolkit_ROOT=$cudaRoot",
                "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchs")
        }
        'vulkan' {
            if (-not ($env:VULKAN_SDK -and (Test-Path -LiteralPath $env:VULKAN_SDK))) {
                throw 'the vulkan variant needs the Vulkan SDK (VULKAN_SDK not set); build it on a machine with the SDK'
            }
            $cfg += @('-DGGML_VULKAN=ON', '-DGGML_CUDA=OFF')
        }
        default { $cfg += @('-DGGML_CUDA=OFF', '-DGGML_VULKAN=OFF') }
    }
    Write-Step "Configuring $v"
    & $cmake @cfg
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed for $v" }
    Write-Step "Building $v (sigil-tts-server, $Jobs jobs)"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $cmake --build $bdir --target sigil-tts-server --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw "build failed for $v" }
    $buildSeconds = [math]::Round($sw.Elapsed.TotalSeconds)
    $exePath = Join-Path $bdir $Exe
    if (-not (Test-Path -LiteralPath $exePath)) { throw "$exePath was not produced" }

    $imports = Get-PeImports $exePath
    if ($null -ne $imports) {
        $allowed = $BaseImports + ($VariantImports[$v] ?? @())
        $bad = @($imports | Where-Object { $_ -notlike 'api-ms-win-*' -and -not (@($allowed) -icontains $_) })
        if ($bad.Count -gt 0) { throw "$Exe ($v) imports unexpected DLLs: $($bad -join ', ') (expected only $($allowed -join ', '))" }
        if (@($imports | Where-Object { $_ -like 'ggml*' }).Count -gt 0) { throw "$Exe ($v) imports a ggml DLL (static ggml expected)" }
    }

    $pack = Join-Path $OutDir "qwentts-server-$v"
    if (Test-Path -LiteralPath $pack) { Remove-Item -Recurse -Force -LiteralPath $pack }
    New-Item -ItemType Directory -Force -Path $pack | Out-Null
    Copy-Item -LiteralPath $exePath -Destination $pack
    Copy-Item -LiteralPath (Join-Path $PackDir 'NOTICE.txt') -Destination (Join-Path $pack 'NOTICE.txt')
    Write-ThirdPartyLicenses (Join-Path $pack 'THIRD-PARTY-LICENSES.txt')

    # --version loads the exe: proves every import resolves on this PC (CUDA from the toolkit here;
    # Sigil puts runtime/nvidia-cuda13 on the helper's PATH instead).
    $savedPath = $env:PATH
    try {
        if ($v -eq 'cuda13') { $env:PATH = "$(Join-Path $cudaRoot 'bin\x64');$(Join-Path $cudaRoot 'bin');$savedPath" }
        $versionLine = (& (Join-Path $pack $Exe) --version 2>&1 | Select-Object -First 1)
        if ($LASTEXITCODE -ne 0) { throw "$Exe ($v) --version failed ($LASTEXITCODE): $versionLine" }
    } finally { $env:PATH = $savedPath }
    if ("$versionLine" -notmatch [regex]::Escape($CatalogVersion)) { throw "$Exe ($v) reports '$versionLine', expected version $CatalogVersion" }
    # The library stamp of the prepared tree (a -SourceDir checkout stamps its own git commit).
    if (-not $SourceDir -and "$versionLine" -notmatch [regex]::Escape("library $ShortCommit")) {
        throw "$Exe ($v) reports '$versionLine': its qwentts.cpp library is not the pinned $ShortCommit (delete $Src to prepare it again)"
    }

    $isDevPack = ($v -eq 'cuda13') -and $CudaDevBuild
    $files = @(Get-ChildItem -LiteralPath $pack -File | Sort-Object Name | ForEach-Object {
            [ordered]@{ path = $_.Name; size = $_.Length; sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant() }
        })
    $cudaVersion = $null
    if ($v -eq 'cuda13') { $cudaVersion = ((& (Join-Path $cudaRoot 'bin\nvcc.exe') --version) -join ' ') -replace '.*release ([0-9.]+).*', '$1' }
    $manifest = [ordered]@{
        schema          = 1
        engine          = 'qwentts-server'
        kind            = 'helper'
        version         = $CatalogVersion
        variant         = $v
        program         = $Exe
        files           = $files
        catalog_id      = if ($isDevPack) { $null } else { "runtime/qwentts-server-$v" }
        catalog_version = $CatalogVersion
        dev_build       = [bool]$isDevPack
        version_line    = "$versionLine"
        upstream        = [ordered]@{ repo = $Upstream.repo; commit = $Upstream.commit; archive_sha256 = $Upstream.archive.sha256; ggml_commit = $Upstream.ggml.commit; ggml_archive_sha256 = $Upstream.ggml.archive.sha256 }
        requires        = [ordered]@{ cpu = 'avx2'; imports = @($imports ?? @()) }
        toolchain       = [ordered]@{ msvc = $msvcVersion; cmake = ((& $cmake --version | Select-Object -First 1) -replace 'cmake version ', ''); cuda = $cudaVersion }
    }
    if ($v -eq 'cuda13') {
        $manifest.requires['nvidia'] = 'driver R580+, Turing (sm_75) or newer; cudart64_13 / cublas64_13 / cublasLt64_13 from runtime/nvidia-cuda13'
        $manifest['cuda_architectures'] = $CudaArchs
    }
    if ($v -eq 'vulkan') { $manifest.requires['vulkan'] = 'Vulkan 1.2 driver (vulkan-1.dll ships with every GPU driver)' }
    ($manifest | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath (Join-Path $pack 'sigil-pack.json') -Encoding utf8NoBOM

    $zipPath = $null
    $zipHash = $null
    if (-not $NoZip) {
        $zipSuffix = if ($isDevPack) { '-dev' } else { '' }
        $zipPath = Join-Path $OutDir "sigil-qwentts-server-$CatalogVersion-$v$zipSuffix.zip"
        Write-DeterministicZip $pack $zipPath
        $zipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash.ToLowerInvariant()
        Set-Content -LiteralPath "$zipPath.sha256" -Value "$zipHash  $(Split-Path -Leaf $zipPath)" -Encoding ascii
    }
    $builtExe[$v] = Join-Path $pack $Exe
    $exeItem = Get-Item -LiteralPath (Join-Path $pack $Exe)
    $results += [pscustomobject]@{
        Variant   = $v
        ExeMB     = [math]::Round($exeItem.Length / 1MB, 1)
        BuildSec  = $buildSeconds
        Dev       = $isDevPack
        Zip       = if ($zipPath) { Split-Path -Leaf $zipPath } else { '-' }
        ZipSha256 = $zipHash
    }
}

Write-Step "Packs in $OutDir"
$results | Format-Table -AutoSize | Out-String | Write-Host

# ---------------------------------------------------------------------------------------------
# 6. Self-test: time to first audio and real-time factor through the server
# ---------------------------------------------------------------------------------------------

if (-not $SelfTest -or -not $Talker) { exit 0 }
if (-not $Codec) { throw '-SelfTest with -Talker needs -Codec (qwen-tokenizer-12hz-*.gguf)' }
foreach ($p in @($Talker, $Codec)) { if (-not (Test-Path -LiteralPath $p)) { throw "$p not found" } }

function Get-FreePort {
    $l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $l.Start(); $port = ([System.Net.IPEndPoint]$l.LocalEndpoint).Port; $l.Stop()
    return $port
}

function Enter-GpuLock {
    if (-not $GpuLock) { return }
    while (Test-Path -LiteralPath $GpuLock) {
        Write-Host "    waiting for the GPU timing lock ($GpuLock): $((Get-Content -Raw -LiteralPath $GpuLock -ErrorAction SilentlyContinue))"
        Start-Sleep -Seconds 15
    }
    Set-Content -LiteralPath $GpuLock -Value "P6-E qwentts self-test (pid $PID) since $([DateTime]::Now.ToString('s'))" -Encoding utf8NoBOM
}

function Exit-GpuLock { if ($GpuLock -and (Test-Path -LiteralPath $GpuLock)) { Remove-Item -Force -LiteralPath $GpuLock } }

$testDir = Join-Path $OutDir 'qwentts-selftest'
New-Item -ItemType Directory -Force -Path $testDir | Out-Null
$report = @()
$http = [System.Net.Http.HttpClient]::new()
$http.Timeout = [TimeSpan]::FromMinutes(10)

function Invoke-Speech([string]$Base, [string]$Key, [hashtable]$Body, [string]$OutWav) {
    $req = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Post, "$Base/v1/audio/speech")
    $req.Headers.Authorization = [System.Net.Http.Headers.AuthenticationHeaderValue]::new('Bearer', $Key)
    $req.Content = [System.Net.Http.StringContent]::new(($Body | ConvertTo-Json -Compress), [System.Text.Encoding]::UTF8, 'application/json')
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $res = $http.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
    if (-not $res.IsSuccessStatusCode) { throw "speech request failed: $([int]$res.StatusCode) $($res.Content.ReadAsStringAsync().GetAwaiter().GetResult())" }
    $stream = $res.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
    $buf = [byte[]]::new(65536)
    $pcm = [System.IO.MemoryStream]::new()
    $firstMs = $null
    while (($n = $stream.Read($buf, 0, $buf.Length)) -gt 0) {
        if ($null -eq $firstMs) { $firstMs = $sw.Elapsed.TotalMilliseconds }
        $pcm.Write($buf, 0, $n)
    }
    $totalMs = $sw.Elapsed.TotalMilliseconds
    $bytes = $pcm.ToArray()
    $audioSec = ($bytes.Length / 2) / 24000.0
    # WAV header (s16le mono 24 kHz).
    $wav = [System.IO.MemoryStream]::new()
    $w = [System.IO.BinaryWriter]::new($wav)
    $w.Write([System.Text.Encoding]::ASCII.GetBytes('RIFF')); $w.Write([int](36 + $bytes.Length)); $w.Write([System.Text.Encoding]::ASCII.GetBytes('WAVEfmt '))
    $w.Write([int]16); $w.Write([int16]1); $w.Write([int16]1); $w.Write([int]24000); $w.Write([int]48000); $w.Write([int16]2); $w.Write([int16]16)
    $w.Write([System.Text.Encoding]::ASCII.GetBytes('data')); $w.Write([int]$bytes.Length); $w.Write($bytes); $w.Flush()
    [System.IO.File]::WriteAllBytes($OutWav, $wav.ToArray())
    return [pscustomobject]@{ FirstAudioMs = [math]::Round($firstMs ?? $totalMs); TotalMs = [math]::Round($totalMs); AudioSec = [math]::Round($audioSec, 2); Rtf = if ($audioSec -gt 0) { [math]::Round(($totalMs / 1000.0) / $audioSec, 3) } else { $null } }
}

foreach ($v in $builtExe.Keys) {
    $exePath = $builtExe[$v]
    $port = Get-FreePort
    $key = -join ((1..32) | ForEach-Object { '{0:x2}' -f (Get-Random -Maximum 256) })
    $base = "http://127.0.0.1:$port"
    $log = Join-Path $testDir "server-$v.log"
    $psi = [System.Diagnostics.ProcessStartInfo]::new($exePath)
    foreach ($a in @('--model', $Talker, '--codec', $Codec, '--host', '127.0.0.1', '--port', "$port", '--alias', 'selftest')) { $psi.ArgumentList.Add($a) }
    $psi.UseShellExecute = $false
    $psi.RedirectStandardError = $true
    $psi.RedirectStandardOutput = $true
    $psi.Environment['QWENTTS_API_KEY'] = $key
    if ($v -eq 'cuda13') { $psi.Environment['PATH'] = "$(Join-Path $cudaRoot 'bin\x64');$(Join-Path $cudaRoot 'bin');$env:PATH" }
    Enter-GpuLock
    $proc = $null
    try {
        Write-Step "Self-test $($v): starting the server on $base"
        $loadSw = [Diagnostics.Stopwatch]::StartNew()
        $proc = [System.Diagnostics.Process]::Start($psi)
        $errTask = $proc.StandardError.ReadToEndAsync()
        $outTask = $proc.StandardOutput.ReadToEndAsync()
        $ready = $false
        while ($loadSw.Elapsed.TotalSeconds -lt $LoadTimeoutSec -and -not $proc.HasExited) {
            try { $r = $http.GetAsync("$base/health").GetAwaiter().GetResult(); if ($r.IsSuccessStatusCode) { $ready = $true; break } } catch { }
            Start-Sleep -Milliseconds 250
        }
        if (-not $ready) { throw "the server did not become ready (exited: $($proc.HasExited))" }
        $loadMs = [math]::Round($loadSw.Elapsed.TotalMilliseconds)
        # The key is enforced: a request without it must be refused.
        $noKey = $http.GetAsync("$base/v1/models").GetAwaiter().GetResult()
        if ([int]$noKey.StatusCode -ne 401) { throw "GET /v1/models without the key answered $([int]$noKey.StatusCode), expected 401" }
        $infoReq = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Get, "$base/sigil/info")
        $infoReq.Headers.Authorization = [System.Net.Http.Headers.AuthenticationHeaderValue]::new('Bearer', $key)
        $info = $http.SendAsync($infoReq).GetAwaiter().GetResult().Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json
        Write-Host "    loaded in $loadMs ms: $($info.model_type), languages $($info.languages -join ' '), speakers $($info.speakers -join ' ')"

        $voiceName = $Voice
        if ($info.model_type -eq 'base') {
            $wavPath = if ($RefWav) { $RefWav } else { Join-Path $Src 'examples\freeman.wav' }
            $refTextValue = if ($RefText) { $RefText } elseif (-not $RefWav) { (Get-Content -Raw -LiteralPath (Join-Path $Src 'examples\freeman.txt')).Trim() } else { '' }
            $regBody = @{ name = 'sigil-selftest'; ref_text = $refTextValue; wav_b64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($wavPath)) }
            $regReq = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Post, "$base/v1/audio/voices")
            $regReq.Headers.Authorization = [System.Net.Http.Headers.AuthenticationHeaderValue]::new('Bearer', $key)
            $regReq.Content = [System.Net.Http.StringContent]::new(($regBody | ConvertTo-Json -Compress), [System.Text.Encoding]::UTF8, 'application/json')
            $regSw = [Diagnostics.Stopwatch]::StartNew()
            $reg = $http.SendAsync($regReq).GetAwaiter().GetResult()
            $regText = $reg.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            if (-not $reg.IsSuccessStatusCode) { throw "voice registration failed: $regText" }
            $regJson = $regText | ConvertFrom-Json
            Write-Host "    voice registered from $(Split-Path -Leaf $wavPath) in $([math]::Round($regSw.Elapsed.TotalMilliseconds)) ms (ref_T $($regJson.ref_T), spk_dim $($regJson.spk_dim), latents returned: $([bool]($regJson.PSObject.Properties.Name -contains 'spk_b64')))"
            # Register the returned latents under a second name: the path Sigil takes for a cached voice.
            $latReq = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Post, "$base/v1/audio/voices")
            $latReq.Headers.Authorization = [System.Net.Http.Headers.AuthenticationHeaderValue]::new('Bearer', $key)
            $latReq.Content = [System.Net.Http.StringContent]::new((@{ name = 'sigil-selftest-cached'; ref_text = $refTextValue; spk_b64 = $regJson.spk_b64; rvq_b64 = $regJson.rvq_b64 } | ConvertTo-Json -Compress), [System.Text.Encoding]::UTF8, 'application/json')
            $lat = $http.SendAsync($latReq).GetAwaiter().GetResult()
            if (-not $lat.IsSuccessStatusCode) { throw "registering the cached latents failed: $($lat.Content.ReadAsStringAsync().GetAwaiter().GetResult())" }
            $voiceName = 'sigil-selftest-cached'
        } elseif (-not $voiceName -and @($info.speakers).Count -gt 0) {
            $voiceName = @($info.speakers)[0]
        }

        # Warm-up (first GPU kernels / pipelines), not reported.
        $null = Invoke-Speech $base $key @{ input = 'Warm up.'; voice = $voiceName; language = $Language; response_format = 'pcm' } (Join-Path $testDir "warmup-$v.wav")
        $i = 0
        foreach ($line in $Text) {
            $i++
            $body = @{ input = $line; language = $Language; response_format = 'pcm' }
            if ($voiceName) { $body.voice = $voiceName }
            if ($Instruct -and $info.model_type -ne 'base') { $body.instructions = $Instruct }
            $m = Invoke-Speech $base $key $body (Join-Path $testDir "$v-$($info.model_type)-$i.wav")
            Write-Host ("    [{0}] first audio {1} ms, {2} s of audio in {3} ms, RTF {4}" -f $i, $m.FirstAudioMs, $m.AudioSec, $m.TotalMs, $m.Rtf)
            $report += [pscustomobject]@{ variant = $v; model_type = $info.model_type; talker = (Split-Path -Leaf $Talker); voice = $voiceName; sentence = $i; load_ms = $loadMs; first_audio_ms = $m.FirstAudioMs; total_ms = $m.TotalMs; audio_s = $m.AudioSec; rtf = $m.Rtf }
        }
    } finally {
        if ($proc -and -not $proc.HasExited) { $proc.Kill($true); $proc.WaitForExit(5000) | Out-Null }
        Exit-GpuLock
        if ($proc) {
            Set-Content -LiteralPath $log -Value ("{0}`n{1}" -f $outTask.GetAwaiter().GetResult(), $errTask.GetAwaiter().GetResult()) -Encoding utf8NoBOM
        }
    }
}
$reportPath = Join-Path $testDir 'report.json'
($report | ConvertTo-Json -Depth 4) | Set-Content -LiteralPath $reportPath -Encoding utf8NoBOM
Write-Step "Self-test report: $reportPath"
$report | Format-Table -AutoSize | Out-String | Write-Host
