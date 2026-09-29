#Requires -Version 7.2
<#
.SYNOPSIS
  Build, check and package Sigil's native runtime packs (sigil_engine DLLs) on this PC.

.DESCRIPTION
  For every requested engine (native/engines/<name>/) and variant (cpu, cpu-avx2, cpu-avx512,
  vulkan, cuda13) this script
    1. configures a CMake tree (native/build/<engine>/<variant>) with -DSIGIL_VARIANT and builds
       sigil_<engine>.dll plus sigil_engine_check.exe, then takes both from the exact output path
       CMake recorded for -Config (<tree>/sigil-outputs/<config>/<target>.path; a Visual Studio tree
       keeps Debug/ and Release/ side by side, so the tree is never searched for "the newest DLL");
    2. validates the DLL the way Sigil loads it (sigil_engine_check: ABI version first, all 16
       exports, sigil_engine_get_info, imports, create + {"op":"info"}), refuses debug CRT and
       OpenMP runtime imports, and compares it with engine.json. cuda13 checks resolve the CUDA
       runtime from the toolkit (--dll-dir). create is skipped, and only the structural checks run,
       when this build PC cannot run the variant: a CPU without its instruction sets (the checker
       detects it), no NVIDIA driver / Vulkan loader, a GPU backend the engine reports unavailable,
       or -NoCreate;
    3. stages <out>/dev/<engine>/<variant>/ (the DLL and sigil-pack.json with sizes and sha256) for
       `Sigil.exe --headless engine-probe <engine> --runtime-dir <that folder>`;
    4. writes a reproducible zip <out>/packs/sigil-<engine>-<version>-<variant>.zip and copies it
       to <out>/local-assets/runtime/sigil-<engine>-<variant>/ so that SIGIL_LOCAL_ASSETS=<out>/local-assets
       installs it through the model store (`Sigil.exe --headless fetch runtime/sigil-<engine>-<variant>`);
    5. checks the catalog entry (src-tauri/model-catalog/*.json) and prints the pin to use when the
       pack is published.

  -Config Debug links the debug CRT (it exists only on PCs with Visual Studio): such a build is
  staged as <out>/dev/<engine>/<variant>-debug/ for engine-probe on this PC, and no pack is written.

  -Engine ort-cuda-provider stages the ONNX Runtime CUDA provider (runtime/ort-cuda13-provider)
  from the ort-sys build output or pyke's dist cache instead, after checking its pinned sha256.

  Nothing is downloaded and nothing is published (asset hosting is local-only). Outputs are
  gitignored (native/build, native/out).

.PARAMETER Engine
  Engine folder names under native/engines, 'all', or 'ort-cuda-provider'. Default: echo.
.PARAMETER Variant
  Variants to build. Default: the engine's default_variants from engine.json.
.PARAMETER Config
  CMake configuration. Default: Release. Debug builds are staged for local probing only.
.PARAMETER BuildRoot
  CMake build trees. Default: native/build.
.PARAMETER OutDir
  Staged packs. Default: native/out.
.PARAMETER Generator
  CMake generator (default: CMake's default, the newest Visual Studio, with -A x64).
.PARAMETER Clean
  Delete each build tree before configuring.
.PARAMETER NoCreate
  Structural checks only (load, ABI, exports, info, imports): never create an engine instance on
  this build PC. Needed only when an engine's create would crash rather than fail on hardware the
  build PC lacks; the usual cases are detected automatically.
.PARAMETER List
  Show the engines, their versions and variants, then exit.

.EXAMPLE
  pwsh scripts/native/build-runtime-packs.ps1 -Engine echo -Variant cpu
.EXAMPLE
  pwsh scripts/native/build-runtime-packs.ps1 -Engine echo            # cpu + cpu-avx2
.EXAMPLE
  pwsh scripts/native/build-runtime-packs.ps1 -Engine ort-cuda-provider
#>
[CmdletBinding()]
param(
    [string[]] $Engine = @('echo'),
    [string[]] $Variant = @(),
    [ValidateSet('Release', 'RelWithDebInfo', 'Debug')] [string] $Config = 'Release',
    [string] $BuildRoot,
    [string] $OutDir,
    [string] $Generator,
    [switch] $Clean,
    [switch] $NoCreate,
    [switch] $List
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 3.0

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..' '..')).Path
$NativeDir = Join-Path $RepoRoot 'native'
$CatalogDir = Join-Path $RepoRoot 'src-tauri' 'model-catalog'
if (-not $BuildRoot) { $BuildRoot = Join-Path $NativeDir 'build' }
if (-not $OutDir) { $OutDir = Join-Path $NativeDir 'out' }
$BuildRoot = [IO.Path]::GetFullPath($BuildRoot)
$OutDir = [IO.Path]::GetFullPath($OutDir)
$Versions = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'versions.json') | ConvertFrom-Json

function Write-Step([string] $Message) { Write-Host "==> $Message" -ForegroundColor Cyan }

function Get-Sha256([string] $Path) {
    (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Reset-Directory([string] $Path) {
    if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Assert-CMake {
    $cmake = Get-Command cmake -ErrorAction SilentlyContinue
    if (-not $cmake) { throw 'cmake was not found on PATH (install CMake 3.25 or newer, or the Visual Studio C++ CMake tools).' }
    $line = (& cmake --version | Select-Object -First 1)
    if ($line -notmatch '(\d+\.\d+(\.\d+)?)') { throw "cannot read the CMake version from '$line'" }
    $found = [version]$Matches[1]
    if ($found -lt [version]$Versions.toolchain.cmake_min) {
        throw "CMake $found is too old; $($Versions.toolchain.cmake_min) or newer is required."
    }
    return $found
}

function Get-CatalogEntries {
    $entries = @()
    foreach ($f in Get-ChildItem -LiteralPath $CatalogDir -Filter '*.json' | Where-Object { -not $_.Name.StartsWith('_') }) {
        $doc = Get-Content -Raw -LiteralPath $f.FullName | ConvertFrom-Json
        foreach ($e in $doc.entries) { $entries += $e }
    }
    return , $entries
}

function Get-EngineManifest([string] $Name) {
    $path = Join-Path $NativeDir 'engines' $Name 'engine.json'
    if (-not (Test-Path -LiteralPath $path)) { throw "native/engines/$Name/engine.json does not exist" }
    $m = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
    if ($m.name -ne $Name) { throw "engine.json of '$Name' names '$($m.name)'" }
    if ($m.version -notmatch '^\d+\.\d+\.\d+') { throw "engine.json of '$Name' has an invalid version '$($m.version)'" }
    foreach ($v in $m.variants) {
        if ($v -notin $Versions.variants) { throw "engine.json of '$Name' lists an unknown variant '$v'" }
    }
    return $m
}

function Get-AllEngines {
    Get-ChildItem -LiteralPath (Join-Path $NativeDir 'engines') -Directory |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'engine.json') } |
        ForEach-Object { $_.Name } | Sort-Object
}

function Find-CudaToolkit {
    $min = [version]$Versions.cuda13.toolkit_min
    $roots = @()
    foreach ($var in Get-ChildItem Env: | Where-Object { $_.Name -match '^CUDA_PATH(_V13_\d+)?$' }) { $roots += $var.Value }
    $base = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path -LiteralPath $base) { $roots += (Get-ChildItem -LiteralPath $base -Directory | ForEach-Object FullName) }
    $best = $null
    foreach ($r in ($roots | Select-Object -Unique)) {
        if ((Split-Path -Leaf $r) -notmatch '^v?(\d+\.\d+)$') { continue }
        $ver = [version]$Matches[1]
        $nvcc = @('bin\nvcc.exe', 'bin\x64\nvcc.exe') | ForEach-Object { Join-Path $r $_ } | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
        if ($nvcc -and $ver -ge $min -and $ver.Major -eq 13 -and (-not $best -or $ver -gt $best.Version)) {
            $best = [pscustomobject]@{ Root = $r; Version = $ver }
        }
    }
    if (-not $best) { throw "The cuda13 variant needs the CUDA 13 toolkit ($min or newer); none was found." }
    return $best
}

# What building and checking variant $V needs on this PC:
#   Configure - extra CMake configure arguments;
#   DllDirs   - folders the checker adds with --dll-dir (Sigil gets the same DLLs from runtime packs);
#   NoCreate  - why the checker must not create an engine here (the backend's driver is missing), or $null.
# A CPU without the variant's instruction sets is detected by the checker itself (--variant).
function Assert-VariantPrerequisites([string] $V) {
    $prereq = [pscustomobject]@{ Configure = @(); DllDirs = @(); NoCreate = $null }
    $system32 = Join-Path ($env:SystemRoot ?? 'C:\Windows') 'System32'
    switch ($V) {
        'cuda13' {
            $cuda = Find-CudaToolkit
            Write-Host "    CUDA toolkit $($cuda.Version) at $($cuda.Root)"
            $prereq.Configure += "-DCUDAToolkit_ROOT=$($cuda.Root)"
            if (-not $Generator -or $Generator -like 'Visual Studio*') { $prereq.Configure += @('-T', "cuda=$($cuda.Root)") }
            # The engine links the dynamic cudart / cuBLAS (Sigil preloads them from runtime/nvidia-cuda13);
            # the checker resolves them from the toolkit (CUDA 13 keeps its DLLs in bin\x64).
            $prereq.DllDirs = @(@('bin\x64', 'bin') | ForEach-Object { Join-Path $cuda.Root $_ } | Where-Object {
                    (Test-Path -LiteralPath $_) -and (Get-ChildItem -LiteralPath $_ -Filter 'cudart64_*.dll' -ErrorAction SilentlyContinue)
                })
            if ($prereq.DllDirs.Count -eq 0) {
                Write-Warning "no cudart64_*.dll under $($cuda.Root)\bin\x64 or \bin: the ABI check will not be able to load the DLL"
            }
            if (-not (Test-Path -LiteralPath (Join-Path $system32 'nvcuda.dll'))) {
                $prereq.NoCreate = 'this build PC has no NVIDIA driver (nvcuda.dll)'
            }
        }
        'vulkan' {
            if (-not $env:VULKAN_SDK -or -not (Test-Path -LiteralPath $env:VULKAN_SDK)) {
                throw 'The vulkan variant needs the Vulkan SDK (VULKAN_SDK is not set).'
            }
            Write-Host "    Vulkan SDK at $env:VULKAN_SDK"
            if (-not (Test-Path -LiteralPath (Join-Path $system32 'vulkan-1.dll'))) {
                # The loader normally comes with the GPU driver; the SDK's copy lets the DLL load.
                $sdkBin = Join-Path $env:VULKAN_SDK 'Bin'
                if (Test-Path -LiteralPath (Join-Path $sdkBin 'vulkan-1.dll')) { $prereq.DllDirs = @($sdkBin) }
                $prereq.NoCreate = 'this build PC has no Vulkan driver (System32\vulkan-1.dll)'
            }
        }
    }
    if ($NoCreate -and -not $prereq.NoCreate) { $prereq.NoCreate = 'requested with -NoCreate' }
    return $prereq
}

# Output of $Target for -Config, from the path CMake recorded at generate time (sigil_record_output in
# native/cmake/SigilEngine.cmake). Never search the tree: a Visual Studio tree keeps Debug/ and
# Release/ side by side, and an up-to-date Release DLL is older than a later Debug build.
function Get-BuildOutput([string] $BuildDir, [string] $Target) {
    $record = Join-Path $BuildDir 'sigil-outputs' $Config "$Target.path"
    if (-not (Test-Path -LiteralPath $record)) {
        throw "$record is missing: CMake did not record the $Config output of $Target (rerun with -Clean)"
    }
    $path = (Get-Content -Raw -LiteralPath $record).Trim()
    if (-not $path -or -not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "the $Config build produced no $Target ($path)"
    }
    return Get-Item -LiteralPath $path
}

# A property of a JSON object, or $null (StrictMode refuses reading a missing property).
function Get-JsonProperty($Object, [string] $Name) {
    if ($null -eq $Object) { return $null }
    $p = $Object.PSObject.Properties[$Name]
    if ($p) { return $p.Value }
    return $null
}

# Debug CRT DLLs (they exist only on PCs with Visual Studio) and the OpenMP runtime (not
# redistributed with packs): a pack DLL importing any of them fails to load on users' PCs.
$DebugCrtPattern = '^(ucrtbased|vcruntime\d+(_\d+)?d|msvcp\d+(_\d+)?d(_\w+)?|concrt\d+d|vccorlib\d+d|vcomp\d+d|vcamp\d+d)\.dll$'
$OpenMpPattern = '^(vcomp\d+|libomp[\w.]*|libiomp5md)\.dll$'

# Run sigil_engine_check on $Dll; returns its JSON result. A checker that dies without printing
# (an illegal instruction, a crash in the engine) is reported with its exit status, never as a
# JSON parse error.
function Invoke-EngineCheck([string] $Checker, [string] $Dll, [string] $V, [string[]] $DllDirs, [string] $NoCreateReason) {
    $checkArgs = @($Dll, '--variant', $V)
    foreach ($d in $DllDirs) { $checkArgs += @('--dll-dir', $d) }
    if ($NoCreateReason) { $checkArgs += '--no-create' }
    $errFile = [IO.Path]::GetTempFileName()
    try {
        $raw = & $Checker @checkArgs 2> $errFile
        $code = $LASTEXITCODE
        # An empty file reads as nothing ($null even through a [string] cast).
        $stderr = (Get-Content -Raw -LiteralPath $errFile -ErrorAction SilentlyContinue) ?? ''
    } finally {
        Remove-Item -LiteralPath $errFile -Force -ErrorAction SilentlyContinue
    }
    $text = ($raw | Out-String).Trim()
    $check = $null
    if ($text) {
        try { $check = $text | ConvertFrom-Json } catch { $check = $null }
    }
    if ($null -eq $check -or $null -eq (Get-JsonProperty $check 'ok')) {
        $status = '0x{0:X8}' -f $code
        $why = if ($code -eq -1073741795) {
            # STATUS_ILLEGAL_INSTRUCTION
            "it died with an illegal instruction ($status): this build PC cannot run the $V code; rerun with -NoCreate for the structural checks only"
        } elseif ($code -eq -1073741819) {
            # STATUS_ACCESS_VIOLATION
            "it crashed with an access violation ($status) inside the engine"
        } else {
            "it exited with code $code ($status) without a valid result"
        }
        $tail = ($stderr.Trim() -split "`r?`n" | Select-Object -Last 20) -join "`n"
        $out = if ($text) { "`nstdout: $text" } else { '' }
        throw "sigil_engine_check could not check $(Split-Path -Leaf $Dll): $why$out$(if ($tail) { "`nstderr:`n$tail" })"
    }
    return $check
}

function New-ReproducibleZip([string] $SourceDir, [string[]] $Files, [string] $Destination) {
    Add-Type -AssemblyName System.IO.Compression
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    if (Test-Path -LiteralPath $Destination) { Remove-Item -LiteralPath $Destination -Force }
    $stamp = [DateTimeOffset]::new(2000, 1, 1, 0, 0, 0, [TimeSpan]::Zero)
    $fs = [IO.File]::Open($Destination, [IO.FileMode]::CreateNew)
    try {
        $zip = [IO.Compression.ZipArchive]::new($fs, [IO.Compression.ZipArchiveMode]::Create)
        try {
            foreach ($name in ($Files | Sort-Object)) {
                $entry = $zip.CreateEntry($name, [IO.Compression.CompressionLevel]::Optimal)
                $entry.LastWriteTime = $stamp
                $out = $entry.Open()
                $in = [IO.File]::OpenRead((Join-Path $SourceDir $name))
                try { $in.CopyTo($out) } finally { $in.Dispose(); $out.Dispose() }
            }
        } finally { $zip.Dispose() }
    } finally { $fs.Dispose() }
}

function Invoke-Native([string] $Exe, [string[]] $Arguments) {
    # Out-Host: the tool's output goes to the console, never into the caller's return value (the
    # build summary would otherwise collect CMake's log lines as results).
    & $Exe @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

# Engines with their own build script (native/engines/<name>/build.ps1) own their pinned-source fetch,
# import allow-lists, CUDA DLL folders for the self-test and the license files their packs must carry
# (transcribe: LICENSE-transcribe.cpp.txt, THIRD-PARTY-LICENSES-transcribe.cpp.md, NOTICE.txt). This
# script delegates to it and puts the packs under <OutDir>/local-assets (SIGIL_LOCAL_ASSETS finds a
# catalog file by its plain name there).
function Invoke-EngineBuildScript([string] $Name, [string] $Script, [string[]] $Variants) {
    Write-Step "$Name ($($Variants -join ', ')): delegating to native/engines/$Name/build.ps1"
    $packs = Join-Path $OutDir 'local-assets'
    $pwshArgs = @('-NoProfile', '-File', $Script, '-Variant', ($Variants -join ','), '-OutDir', $packs)
    if ($Clean) { $pwshArgs += '-Clean' }
    $pwshExe = (Get-Process -Id $PID).Path
    & $pwshExe @pwshArgs | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "native/engines/$Name/build.ps1 failed with exit code $LASTEXITCODE" }
    $rows = @()
    foreach ($v in $Variants) {
        $zip = Get-ChildItem -LiteralPath $packs -Filter "sigil-$Name-*-$v.zip" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $rows += [ordered]@{
            id         = "runtime/sigil-$Name-$v"
            engine     = $Name
            variant    = $v
            delegated  = $Script
            pack_dir   = (Join-Path $packs "$Name-$v")
            zip        = if ($zip) { $zip.FullName } else { $null }
            zip_sha256 = if ($zip) { Get-Sha256 $zip.FullName } else { $null }
        }
    }
    return , $rows
}

function Build-EnginePack([string] $Name, [string] $V, $Manifest, $Catalog) {
    Write-Step "$Name $($Manifest.version) ($V)"
    if ($V -notin $Manifest.variants) {
        throw "engine '$Name' is not built for '$V' (engine.json variants: $($Manifest.variants -join ', '))"
    }
    $prereq = Assert-VariantPrerequisites $V
    $bdir = Join-Path $BuildRoot $Name $V
    if ($Clean -and (Test-Path -LiteralPath $bdir)) { Remove-Item -LiteralPath $bdir -Recurse -Force }

    $configure = @('-S', $NativeDir, '-B', $bdir, "-DSIGIL_VARIANT=$V", "-DSIGIL_ENGINES=$Name")
    if ($Generator) { $configure += @('-G', $Generator) }
    if (-not $Generator -or $Generator -like 'Visual Studio*') { $configure += @('-A', 'x64') }
    if ($Generator -and $Generator -notlike 'Visual Studio*') { $configure += "-DCMAKE_BUILD_TYPE=$Config" }
    $configure += $prereq.Configure
    Invoke-Native cmake $configure
    Invoke-Native cmake @('--build', $bdir, '--config', $Config, '--target', "sigil_$Name", 'sigil_engine_check', '--parallel')

    $dllName = "sigil_$Name.dll"
    $dll = Get-BuildOutput $bdir "sigil_$Name"
    $checker = Get-BuildOutput $bdir 'sigil_engine_check'
    if ($dll.Name -ne $dllName) { throw "the build produced $($dll.Name), not $dllName" }

    $check = Invoke-EngineCheck -Checker $checker.FullName -Dll $dll.FullName -V $V -DllDirs $prereq.DllDirs -NoCreateReason $prereq.NoCreate
    if (-not $check.ok) { throw "$dllName failed the ABI check: $($check | ConvertTo-Json -Depth 6 -Compress)" }
    $info = $check.info
    $problems = @()
    if ($info.name -ne $Name) { $problems += "name '$($info.name)'" }
    if ($info.version -ne $Manifest.version) { $problems += "version '$($info.version)' (engine.json: $($Manifest.version))" }
    if ($info.variant -ne $V) { $problems += "variant '$($info.variant)'" }
    if ([int]$info.abi_version -ne [int]$Versions.abi_version) { $problems += "ABI v$($info.abi_version)" }
    if ($problems) { throw "$dllName does not match what was built: $($problems -join ', ')" }
    Write-Host "    ABI v$($info.abi_version), capabilities $($info.capabilities), $($info.build_info)"

    # Imports: a debug CRT or the OpenMP runtime does not exist on users' PCs.
    $imports = @(Get-JsonProperty $check 'imports')
    $openmp = @($imports | Where-Object { $_ -match $OpenMpPattern })
    if ($openmp) { throw "$dllName imports the OpenMP runtime ($($openmp -join ', ')), which packs do not carry (GGML_OPENMP must stay OFF)" }
    $debugCrt = @($imports | Where-Object { $_ -match $DebugCrtPattern })
    $devOnly = $false
    if ($debugCrt) {
        if ($Config -ne 'Debug') {
            throw "$dllName ($Config) imports the debug CRT ($($debugCrt -join ', ')); rebuild with -Clean"
        }
        $devOnly = $true
    }
    $skipped = Get-JsonProperty (Get-JsonProperty $check 'create') 'skipped'
    if ($skipped) {
        $reason = if ($prereq.NoCreate) { $prereq.NoCreate } else { $skipped }
        Write-Warning "$dllName ($V): create was not run ($reason); only the structural checks passed. Probe the pack on a PC that can run $V before relying on it."
    }

    # Development folder: DLL + manifest (engine-probe --runtime-dir). A Debug build gets its own
    # folder and never becomes a pack.
    $label = if ($devOnly) { "$V-debug" } else { $V }
    $stage = Join-Path $OutDir 'dev' $Name $label
    Reset-Directory $stage
    Copy-Item -LiteralPath $dll.FullName -Destination (Join-Path $stage $dllName)
    $pdb = [IO.Path]::ChangeExtension($dll.FullName, '.pdb')
    if (Test-Path -LiteralPath $pdb) {
        $symbols = Join-Path $OutDir 'symbols' $Name $label
        New-Item -ItemType Directory -Force -Path $symbols | Out-Null
        Copy-Item -LiteralPath $pdb -Destination $symbols -Force
    }
    $staged = Join-Path $stage $dllName
    $pack = [ordered]@{
        schema       = 1
        engine       = $Name
        version      = [string]$Manifest.version
        variant      = $V
        abi_version  = [int]$info.abi_version
        library      = $dllName
        kind         = [string]$info.kind
        capabilities = [uint64]$info.capabilities
        build_info   = [string]$info.build_info
        files        = @([ordered]@{ path = $dllName; size = (Get-Item -LiteralPath $staged).Length; sha256 = (Get-Sha256 $staged) })
    }
    ($pack | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath (Join-Path $stage 'sigil-pack.json') -Encoding utf8NoBOM

    $id = "runtime/sigil-$Name-$V"
    if ($devOnly) {
        Write-Warning ("$dllName is a Debug build linked with the debug CRT ($($debugCrt -join ', ')), which exists only on PCs " +
            "with Visual Studio: staged $stage for engine-probe on this PC; no pack was written (use -Config Release or RelWithDebInfo).")
        return [ordered]@{
            id             = $id
            engine         = $Name
            version        = [string]$Manifest.version
            variant        = $V
            config         = $Config
            dll            = $staged
            dev_dir        = $stage
            zip            = $null
            create_skipped = $skipped
            catalog        = 'not packed (Debug build)'
            capabilities   = [uint64]$info.capabilities
        }
    }

    # Pack zip + local assets layout (<dir>/<entry id>/<file path>).
    $zipName = "sigil-$Name-$($Manifest.version)-$V.zip"
    $zip = Join-Path $OutDir 'packs' $zipName
    New-ReproducibleZip -SourceDir $stage -Files @($dllName, 'sigil-pack.json') -Destination $zip
    $assetDir = Join-Path $OutDir 'local-assets' ($id -replace '/', [IO.Path]::DirectorySeparatorChar)
    New-Item -ItemType Directory -Force -Path $assetDir | Out-Null
    Copy-Item -LiteralPath $zip -Destination (Join-Path $assetDir $zipName) -Force

    $zipSize = (Get-Item -LiteralPath $zip).Length
    $zipSha = Get-Sha256 $zip
    $entry = $Catalog | Where-Object { $_.id -eq $id } | Select-Object -First 1
    $catalogNote = 'ok'
    if (-not $entry) {
        $catalogNote = "missing: add $id (variant '$V') to a src-tauri/model-catalog fragment"
    } elseif ($entry.version -ne $Manifest.version) {
        $catalogNote = "version mismatch: the catalog has $($entry.version), engine.json $($Manifest.version)"
    } elseif ($entry.files[0].path -ne $zipName) {
        $catalogNote = "file mismatch: the catalog installs $($entry.files[0].path), the build produced $zipName"
    }
    if ($catalogNote -ne 'ok') { Write-Warning "catalog entry $id : $catalogNote" }
    Write-Host "    $zip ($zipSize bytes, sha256 $zipSha)"

    return [ordered]@{
        id             = $id
        engine         = $Name
        version        = [string]$Manifest.version
        variant        = $V
        config         = $Config
        dll            = $staged
        dev_dir        = $stage
        zip            = $zip
        zip_size       = $zipSize
        zip_sha256     = $zipSha
        create_skipped = $skipped
        catalog        = $catalogNote
        capabilities   = [uint64]$info.capabilities
    }
}

function Stage-OrtCudaProvider($Catalog) {
    $p = $Versions.ort_cuda_provider
    Write-Step "ONNX Runtime $($p.onnxruntime) CUDA provider ($($p.catalog_id))"
    $entry = $Catalog | Where-Object { $_.id -eq $p.catalog_id } | Select-Object -First 1
    if (-not $entry) { throw "the catalog has no $($p.catalog_id) entry" }
    $file = $entry.files | Where-Object { $_.path -eq $p.file } | Select-Object -First 1
    if (-not $file -or -not $file.sha256) { throw "$($p.catalog_id) does not pin $($p.file)" }
    $candidates = @(
        (Join-Path $RepoRoot 'src-tauri' 'target' 'release' $p.file),
        (Join-Path $RepoRoot 'src-tauri' 'target' 'debug' $p.file),
        (Join-Path $env:LOCALAPPDATA 'ort.pyke.io' 'dfbin' 'x86_64-pc-windows-msvc' $p.pyke_dist $p.file)
    )
    $source = $null
    foreach ($c in $candidates) {
        if (-not (Test-Path -LiteralPath $c)) { continue }
        $sha = Get-Sha256 $c
        if ($sha -eq $file.sha256) { $source = $c; break }
        Write-Warning "$c has sha256 $sha, not the pinned $($file.sha256) (another ORT build?)"
    }
    if (-not $source) {
        throw "no $($p.file) matching the catalog pin was found; build Sigil once (cargo build) so ort-sys fetches ONNX Runtime $($p.onnxruntime)"
    }
    $dir = Join-Path $OutDir 'local-assets' ($p.catalog_id -replace '/', [IO.Path]::DirectorySeparatorChar)
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $dest = Join-Path $dir $p.file
    Copy-Item -LiteralPath $source -Destination $dest -Force
    Write-Host "    $source -> $dest ($((Get-Item -LiteralPath $dest).Length) bytes, sha256 $($file.sha256))"
    return [ordered]@{ id = $p.catalog_id; file = $dest; sha256 = $file.sha256; source = $source }
}

# ---------------------------------------------------------------------------------------------------

# `pwsh -File` passes "a,b" as one string: accept comma-separated lists too.
$Engine = @($Engine | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$Variant = @($Variant | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($v in $Variant) {
    if ($v -notin $Versions.variants) { throw "unknown variant '$v' (variants: $($Versions.variants -join ', '))" }
}

$catalog = Get-CatalogEntries
$allEngines = @(Get-AllEngines)

if ($List) {
    foreach ($name in $allEngines) {
        $m = Get-EngineManifest $name
        Write-Host ("{0,-16} {1,-10} variants: {2} (default: {3})" -f $name, $m.version, ($m.variants -join ', '), ($m.default_variants -join ', '))
    }
    Write-Host ("{0,-16} {1,-10} stages the pinned ONNX Runtime CUDA provider" -f 'ort-cuda-provider', $Versions.ort_cuda_provider.onnxruntime)
    return
}

$engines = if ($Engine -contains 'all') { $allEngines } else { $Engine }
$results = @()
$needsCMake = $engines | Where-Object { $_ -ne 'ort-cuda-provider' }
if ($needsCMake) {
    $cmakeVersion = Assert-CMake
    Write-Host "CMake $cmakeVersion; build trees in $BuildRoot; packs in $OutDir"
}
foreach ($name in $engines) {
    if ($name -eq 'ort-cuda-provider') {
        $results += Stage-OrtCudaProvider $catalog
        continue
    }
    if ($name -notin $allEngines) { throw "unknown engine '$name' (available: $($allEngines -join ', '), ort-cuda-provider)" }
    $manifest = Get-EngineManifest $name
    $variants = if ($Variant.Count -gt 0) { $Variant } else { @($manifest.default_variants) }
    $ownScript = Join-Path $NativeDir 'engines' $name 'build.ps1'
    if (Test-Path -LiteralPath $ownScript) {
        foreach ($v in $variants) {
            if ($v -notin $manifest.variants) {
                throw "engine '$name' is not built for '$v' (engine.json variants: $($manifest.variants -join ', '))"
            }
        }
        $results += Invoke-EngineBuildScript -Name $name -Script $ownScript -Variants $variants
        continue
    }
    foreach ($v in $variants) {
        $results += Build-EnginePack -Name $name -V $v -Manifest $manifest -Catalog $catalog
    }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$summary = [ordered]@{
    built_at     = (Get-Date).ToUniversalTime().ToString('o')
    config       = $Config
    local_assets = (Join-Path $OutDir 'local-assets')
    results      = $results
}
($summary | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath (Join-Path $OutDir 'build-summary.json') -Encoding utf8NoBOM

Write-Step 'Done'
Write-Host "Summary: $(Join-Path $OutDir 'build-summary.json')"
Write-Host "Install locally (model store):  `$env:SIGIL_LOCAL_ASSETS='$(Join-Path $OutDir 'local-assets')'; Sigil.exe --headless fetch <id>"
foreach ($r in $results | Where-Object { $_.Contains('dev_dir') }) {
    Write-Host "Probe without installing:       Sigil.exe --headless engine-probe $($r.engine) --runtime-dir `"$($r.dev_dir)`""
}
