#Requires -Version 7.0
<#
.SYNOPSIS
  Builds the runtime/s2-server-* packs (Fish Audio S2 Pro through s2.cpp) for local import in
  Sigil's Models & Runtimes.

.DESCRIPTION
  Local-only pack build for P6-F (asset hosting is local only; the catalog entries stay
  "unpublished"). Upstream publishes no Windows binaries, so:
    1. fetches the pinned s2.cpp archive of UPSTREAM.json ONCE into the shared dev cache
       (%LOCALAPPDATA%\sigil-dev-cache\s2\archives), one request; a commit with a ggml submodule
       also fetches that archive (its commit read from the GitHub contents API), a few seconds
       apart. Both are checked against UPSTREAM.json's size and sha256; -PinFirstFetch records them
       there the first time (review the source before trusting a new pin);
    2. extracts them into one source tree (unmodified upstream);
    3. configures each variant with Ninja + MSVC (Visual Studio 2022; CUDA 13 for cuda13, the
       Vulkan SDK for vulkan): static ggml (BUILD_SHARED_LIBS=OFF, no backend DLLs),
       GGML_NATIVE=OFF with the AVX2 baseline;
    4. builds, finds s2.exe, checks its imports (Windows and MSVC runtime DLLs, plus CUDA 13 or
       vulkan-1), runs `s2.exe --help` and records the GPU switches it offers in sigil-pack.json
       (`server_args`: how this variant selects the GPU; `low_vram_args`: layers and codec on the
       processor for 8 GB cards), so Sigil never guesses them;
    5. stages <OutDir>\s2-server-<variant>\ (s2.exe, any DLL the build produced, sigil-pack.json
       with every file's sha256, NOTICE.txt, LICENSE) and a deterministic zip
       sigil-s2-server-<catalog_version>-<variant>.zip (the catalog's file name) plus .zip.sha256.

  The speak test runs through Sigil's own path (process host, adapter, timing): see README.md
  (`cargo test real_ggml_engine_speaks -- --ignored`).

.PARAMETER Variant
  cpu, vulkan, cuda13 (comma-separated or repeated). Default cuda13,cpu (vulkan needs VULKAN_SDK).

.EXAMPLE
  pwsh scripts/runtime-packs/s2/build.ps1 -FetchOnly -PinFirstFetch
.EXAMPLE
  pwsh scripts/runtime-packs/s2/build.ps1 -Variant cuda13,cpu
.EXAMPLE
  pwsh scripts/runtime-packs/s2/build.ps1 -Variant cuda13 -CudaArchitectures 89-real -Dev
#>
[CmdletBinding()]
param(
    [string[]]$Variant = @('cuda13', 'cpu'),
    [string]$OutDir,
    [string]$BuildRoot,
    [int]$Jobs = 0,
    [string]$CudaArchitectures,
    [switch]$Dev,
    [switch]$FetchOnly,
    [switch]$PinFirstFetch,
    [switch]$Clean,
    [switch]$NoZip
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$PackDir = $PSScriptRoot
$UpstreamPath = Join-Path $PackDir 'UPSTREAM.json'
$Upstream = Get-Content -Raw -LiteralPath $UpstreamPath | ConvertFrom-Json
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
if ($CudaDevBuild -and ($Variants -contains 'cuda13') -and -not $Dev) {
    throw "-CudaArchitectures '$CudaArchs' differs from UPSTREAM.json's pinned '$PinnedCudaArchs': add -Dev for a development build (its zip is named ...-cuda13-dev.zip)"
}
if (-not $BuildRoot) {
    $BuildRoot = if ($env:SIGIL_NATIVE_BUILD_ROOT) { $env:SIGIL_NATIVE_BUILD_ROOT } else { Join-Path $env:LOCALAPPDATA 'sigil-native' }
}
if (-not $OutDir) { $OutDir = Join-Path $BuildRoot 'packs' }
New-Item -ItemType Directory -Force -Path $BuildRoot, $OutDir | Out-Null
if ($Jobs -le 0) { $Jobs = [Math]::Max(1, [Environment]::ProcessorCount - 2) }
$CacheRoot = if ($env:SIGIL_DEV_CACHE) { Join-Path $env:SIGIL_DEV_CACHE 's2' } else { Join-Path $env:LOCALAPPDATA 'sigil-dev-cache\s2' }

function Write-Step([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }

# ---------------------------------------------------------------------------------------------
# 1-2. Pinned sources
# ---------------------------------------------------------------------------------------------

$script:LastRequest = [DateTime]::MinValue
function Wait-RequestSpacing {
    # The dev PC's router blocks bursts of connections: at least 3 s between request starts.
    $wait = 3000 - ([DateTime]::UtcNow - $script:LastRequest).TotalMilliseconds
    if ($wait -gt 0) { Start-Sleep -Milliseconds ([int]$wait) }
    $script:LastRequest = [DateTime]::UtcNow
}

function Save-Upstream {
    ($Upstream | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath $UpstreamPath -Encoding utf8NoBOM
}

# Download `$Pin.url` once into the cache and check it against `$Pin.size` / `$Pin.sha256`; with
# -PinFirstFetch an unpinned archive gets its pin written to UPSTREAM.json.
function Get-PinnedArchive($Pin, [string]$Label) {
    $dir = Join-Path $CacheRoot 'archives'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $file = Join-Path $dir $Pin.file
    if (-not (Test-Path -LiteralPath $file)) {
        Wait-RequestSpacing
        Write-Step "Downloading $Label (one request)"
        $partial = "$file.partial"
        Invoke-WebRequest -Uri $Pin.url -OutFile $partial -UseBasicParsing -MaximumRetryCount 2 -RetryIntervalSec 5
        Move-Item -Force -LiteralPath $partial -Destination $file
    }
    $size = (Get-Item -LiteralPath $file).Length
    $sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash.ToLowerInvariant()
    if (-not $Pin.sha256) {
        if (-not $PinFirstFetch) {
            throw "$Label is not pinned yet ($size bytes, sha256 $sha). Review the source, then run again with -PinFirstFetch to record the pin in UPSTREAM.json."
        }
        $Pin.sha256 = $sha
        $Pin.size = $size
        Save-Upstream
        Write-Host "    pinned $Label`: $size bytes, sha256 $sha (UPSTREAM.json updated)"
    } elseif ($size -ne [int64]$Pin.size -or $sha -ne $Pin.sha256) {
        throw "$Label archive mismatch: $size bytes, sha256 $sha (UPSTREAM.json pins $($Pin.size) / $($Pin.sha256)). Delete $file to download it again; if GitHub regenerated the archive, re-pin after reviewing the source."
    }
    return $file
}

function Expand-Archive-Into([string]$Archive, [string]$Destination) {
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
    & $tar -xzf $Archive -C $Destination --strip-components=1 --exclude '*/.github'
    if ($LASTEXITCODE -ne 0) { throw "tar failed ($LASTEXITCODE) extracting $Archive" }
}

# The ggml submodule of the pinned commit (GitHub contents API: a submodule entry carries its
# commit and URL), recorded in UPSTREAM.json on the first fetch.
function Resolve-GgmlPin([string]$SubmodulePath) {
    if ($Upstream.ggml) { return $Upstream.ggml }
    if (-not $PinFirstFetch) { throw "the s2.cpp archive has a ggml submodule ($SubmodulePath) that UPSTREAM.json does not pin: run again with -PinFirstFetch" }
    Wait-RequestSpacing
    $repoPath = ($Upstream.repo -replace '^https://github.com/', '')
    $info = Invoke-RestMethod -Uri "https://api.github.com/repos/$repoPath/contents/$SubmodulePath`?ref=$($Upstream.commit)" -Headers @{ 'User-Agent' = 'Sigil-build (s2 pack)' }
    if ($info.type -ne 'submodule' -or -not $info.submodule_git_url -or -not $info.sha) { throw "$SubmodulePath is not a submodule at $($Upstream.commit)" }
    $gitUrl = ($info.submodule_git_url -replace '\.git$', '')
    if ($gitUrl -notmatch '^https://github.com/([^/]+/[^/]+)$') { throw "the ggml submodule is not on GitHub: $($info.submodule_git_url)" }
    $ggmlRepo = $Matches[1]
    $pin = [pscustomobject]@{
        path    = $SubmodulePath
        repo    = "https://github.com/$ggmlRepo"
        license = 'MIT'
        commit  = $info.sha
        archive = [pscustomobject]@{
            url    = "https://codeload.github.com/$ggmlRepo/tar.gz/$($info.sha)"
            file   = "ggml-$($info.sha.Substring(0, 12)).tar.gz"
            sha256 = $null
            size   = $null
        }
    }
    $Upstream.ggml = $pin
    Save-Upstream
    return $pin
}

function Get-SourceTree {
    $archive = Get-PinnedArchive $Upstream.archive "s2.cpp $ShortCommit"
    $tree = Join-Path $CacheRoot "src-$ShortCommit"
    $marker = Join-Path $tree '.sigil-prepared'
    $staging = "$tree.staging"
    if (Test-Path -LiteralPath $staging) { Remove-Item -Recurse -Force -LiteralPath $staging }
    Expand-Archive-Into $archive $staging
    $ggmlArchive = $null
    $gitmodules = Join-Path $staging '.gitmodules'
    if (Test-Path -LiteralPath $gitmodules) {
        $path = (Select-String -LiteralPath $gitmodules -Pattern '^\s*path\s*=\s*(.*ggml\S*)\s*$' | Select-Object -First 1)?.Matches[0].Groups[1].Value
        if ($path -and -not (Test-Path -LiteralPath (Join-Path $staging "$path\CMakeLists.txt"))) {
            $pin = Resolve-GgmlPin $path
            $ggmlArchive = Get-PinnedArchive $pin.archive "ggml $($pin.commit.Substring(0, 7))"
            $dest = Join-Path $staging $path
            if (Test-Path -LiteralPath $dest) { Remove-Item -Recurse -Force -LiteralPath $dest }
            Expand-Archive-Into $ggmlArchive $dest
        }
    }
    if ($FetchOnly) {
        Remove-Item -Recurse -Force -LiteralPath $staging
        return $null
    }
    $expected = "$($Upstream.archive.sha256),$(if ($Upstream.ggml) { $Upstream.ggml.archive.sha256 } else { '-' })"
    if ((Test-Path -LiteralPath $marker) -and ((Get-Content -Raw -LiteralPath $marker).Trim() -eq $expected)) {
        Remove-Item -Recurse -Force -LiteralPath $staging
        return $tree
    }
    if (Test-Path -LiteralPath $tree) { Remove-Item -Recurse -Force -LiteralPath $tree }
    Move-Item -LiteralPath $staging -Destination $tree
    Set-Content -NoNewline -LiteralPath $marker -Value $expected
    return $tree
}

$Src = Get-SourceTree
if ($FetchOnly) {
    Write-Step 'Pinned archives ready'
    exit 0
}
if (-not (Test-Path -LiteralPath (Join-Path $Src 'CMakeLists.txt'))) { throw "$Src has no CMakeLists.txt" }
Write-Step "s2.cpp source: $Src"

# ---------------------------------------------------------------------------------------------
# 3. Toolchain (MSVC 2022 environment, Ninja, CMake, CUDA 13, Vulkan SDK)
# ---------------------------------------------------------------------------------------------

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
    'VCRUNTIME140.dll', 'VCRUNTIME140_1.dll', 'MSVCP140.dll', 'bcrypt.dll', 'WINMM.dll')
$VariantImports = @{
    'cuda13' = @('cudart64_13.dll', 'cublas64_13.dll', 'cublasLt64_13.dll', 'nvcuda.dll')
    'vulkan' = @('vulkan-1.dll')
}
$Exe = 's2.exe'

function Get-PeImports([string]$Path) {
    if (-not $dumpbin) { return $null }
    $out = & $dumpbin /nologo /dependents $Path
    return @($out | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '\.dll$' -and $_ -notmatch '[\\/:]' })
}

# The GPU switches of this build's `--help` (s2.cpp selects the backend on the command line).
function Get-ServerArgs([string]$HelpText, [string]$V) {
    $server = @()
    $flag = switch ($V) { 'cuda13' { '--cuda' } 'vulkan' { '--vulkan' } default { $null } }
    if ($flag) {
        $m = [regex]::Match($HelpText, "(?m)^\s*$([regex]::Escape($flag))(?:[ =]+(?<arg>[<\[]?\w+[>\]]?))?")
        if ($m.Success) {
            # A value placeholder (<N>, [id], N, DEVICE, 0) means the switch takes a device index;
            # a description word ("use the CUDA backend") means it is a plain switch.
            $arg = $m.Groups['arg'].Value
            $takesIndex = $arg -and ($arg -match '^[<\[].*[>\]]$' -or $arg -cmatch '^(N|ID|IDX|INDEX|DEV|DEVICE|GPU|NUM|INT|\d+)$')
            $server = if ($takesIndex) { @($flag, '0') } else { @($flag) }
        } else {
            Write-Warning "$Exe --help does not mention ${flag}: the $V build runs with the default backend choice of s2.cpp"
        }
    }
    $low = @()
    if ($HelpText -match '(?m)^\s*--gpu-layers') { $low += @('--gpu-layers', '18') }
    if ($HelpText -match '(?m)^\s*--codec-cpu') { $low += @('--codec-cpu') }
    return @{ server = $server; low = $low }
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
    $bdir = Join-Path $BuildRoot "s2\$v"
    if ($Clean -and (Test-Path -LiteralPath $bdir)) { Remove-Item -Recurse -Force -LiteralPath $bdir }
    $cfg = @('-S', $Src, '-B', $bdir, '-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$ninja", '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl', '-DBUILD_SHARED_LIBS=OFF', '-DGGML_BACKEND_DL=OFF',
        '-DGGML_NATIVE=OFF', '-DGGML_AVX=ON', '-DGGML_AVX2=ON', '-DGGML_FMA=ON', '-DGGML_F16C=ON',
        '-DGGML_BUILD_TESTS=OFF', '-DGGML_BUILD_EXAMPLES=OFF', '-DGGML_OPENMP=OFF')
    switch ($v) {
        'cuda13' {
            if (-not $cudaRoot) { $cudaRoot = Find-Cuda13 }
            $cfg += @('-DS2_CUDA=ON', '-DS2_VULKAN=OFF', '-DGGML_CUDA=ON', "-DCMAKE_CUDA_COMPILER=$(Join-Path $cudaRoot 'bin\nvcc.exe')",
                "-DCUDAToolkit_ROOT=$cudaRoot", "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchs")
        }
        'vulkan' {
            if (-not ($env:VULKAN_SDK -and (Test-Path -LiteralPath $env:VULKAN_SDK))) {
                throw 'the vulkan variant needs the Vulkan SDK (VULKAN_SDK not set); build it on a machine with the SDK'
            }
            $cfg += @('-DS2_VULKAN=ON', '-DS2_CUDA=OFF', '-DGGML_VULKAN=ON')
        }
        default { $cfg += @('-DS2_CUDA=OFF', '-DS2_VULKAN=OFF') }
    }
    Write-Step "Configuring $v"
    & $cmake @cfg
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed for $v" }
    Write-Step "Building $v ($Jobs jobs)"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $cmake --build $bdir --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw "build failed for $v" }
    $buildSeconds = [math]::Round($sw.Elapsed.TotalSeconds)
    $exeItem = Get-ChildItem -LiteralPath $bdir -Recurse -File -Filter $Exe | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $exeItem) { throw "$Exe was not produced in $bdir (check the target name in upstream's CMakeLists.txt)" }

    $imports = Get-PeImports $exeItem.FullName
    $dlls = @(Get-ChildItem -LiteralPath $exeItem.DirectoryName -File -Filter '*.dll')
    if ($null -ne $imports) {
        $allowed = $BaseImports + ($VariantImports[$v] ?? @()) + @($dlls | ForEach-Object Name)
        $bad = @($imports | Where-Object { $_ -notlike 'api-ms-win-*' -and -not (@($allowed) -icontains $_) })
        if ($bad.Count -gt 0) { throw "$Exe ($v) imports unexpected DLLs: $($bad -join ', ') (expected only $($allowed -join ', '))" }
    }
    if ($dlls.Count -gt 0) { Write-Warning "$Exe ($v) ships with DLLs from the build: $($dlls.Name -join ', ') (static ggml was asked for)" }

    $pack = Join-Path $OutDir "s2-server-$v"
    if (Test-Path -LiteralPath $pack) { Remove-Item -Recurse -Force -LiteralPath $pack }
    New-Item -ItemType Directory -Force -Path $pack | Out-Null
    Copy-Item -LiteralPath $exeItem.FullName -Destination $pack
    foreach ($d in $dlls) { Copy-Item -LiteralPath $d.FullName -Destination $pack }
    Copy-Item -LiteralPath (Join-Path $PackDir 'NOTICE.txt') -Destination (Join-Path $pack 'NOTICE.txt')
    $license = Get-ChildItem -LiteralPath $Src -File | Where-Object { $_.Name -match '^LICENSE' } | Select-Object -First 1
    if ($license) { Copy-Item -LiteralPath $license.FullName -Destination (Join-Path $pack 'LICENSE.md') }

    # --help loads the exe (every import resolves) and tells how this build selects the GPU.
    $savedPath = $env:PATH
    try {
        if ($v -eq 'cuda13') { $env:PATH = "$(Join-Path $cudaRoot 'bin\x64');$(Join-Path $cudaRoot 'bin');$savedPath" }
        $help = (& (Join-Path $pack $Exe) --help 2>&1 | Out-String)
    } finally { $env:PATH = $savedPath }
    if ($help -notmatch '--server') { throw "$Exe ($v) --help does not offer --server:`n$help" }
    $gpuArgs = Get-ServerArgs $help $v

    $isDevPack = ($v -eq 'cuda13') -and $CudaDevBuild
    $files = @(Get-ChildItem -LiteralPath $pack -File | Sort-Object Name | ForEach-Object {
            [ordered]@{ path = $_.Name; size = $_.Length; sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant() }
        })
    $manifest = [ordered]@{
        schema          = 1
        engine          = 's2-server'
        kind            = 'helper'
        version         = $CatalogVersion
        variant         = $v
        program         = $Exe
        files           = $files
        server_args     = @($gpuArgs.server)
        low_vram_args   = @($gpuArgs.low)
        catalog_id      = if ($isDevPack) { $null } else { "runtime/s2-server-$v" }
        catalog_version = $CatalogVersion
        dev_build       = [bool]$isDevPack
        upstream        = [ordered]@{ repo = $Upstream.repo; commit = $Upstream.commit; archive_sha256 = $Upstream.archive.sha256; ggml = $Upstream.ggml }
        requires        = [ordered]@{ cpu = 'avx2'; imports = @($imports ?? @()) }
        help_excerpt    = @($help -split "`r?`n" | Where-Object { $_ -match '--(cuda|vulkan|gpu|codec|server|host|port|threads)' } | Select-Object -First 24)
    }
    if ($v -eq 'cuda13') {
        $manifest.requires['nvidia'] = 'driver R580+, Turing (sm_75) or newer; cudart64_13 / cublas64_13 / cublasLt64_13 from runtime/nvidia-cuda13'
        $manifest['cuda_architectures'] = $CudaArchs
    }
    ($manifest | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath (Join-Path $pack 'sigil-pack.json') -Encoding utf8NoBOM

    $zipPath = $null
    $zipHash = $null
    if (-not $NoZip) {
        $zipSuffix = if ($isDevPack) { '-dev' } else { '' }
        $zipPath = Join-Path $OutDir "sigil-s2-server-$CatalogVersion-$v$zipSuffix.zip"
        Write-DeterministicZip $pack $zipPath
        $zipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash.ToLowerInvariant()
        Set-Content -LiteralPath "$zipPath.sha256" -Value "$zipHash  $(Split-Path -Leaf $zipPath)" -Encoding ascii
    }
    $results += [pscustomobject]@{
        Variant    = $v
        ExeMB      = [math]::Round($exeItem.Length / 1MB, 1)
        BuildSec   = $buildSeconds
        ServerArgs = ($gpuArgs.server -join ' ')
        LowVram    = ($gpuArgs.low -join ' ')
        Zip        = if ($zipPath) { Split-Path -Leaf $zipPath } else { '-' }
        ZipSha256  = $zipHash
    }
}

Write-Step "Packs in $OutDir"
$results | Format-Table -AutoSize | Out-String | Write-Host
Write-Host '    Install: Models & Runtimes -> Fish S2 Pro server (<variant>) -> Import -> the zip (or put it in a SIGIL_LOCAL_ASSETS folder).'
