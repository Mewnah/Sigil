#Requires -Version 7.0
<#
.SYNOPSIS
  Builds the runtime/bergamot pack: sigil-bergamot.exe (Mozilla's Bergamot translation engine behind
  Sigil's JSON-lines protocol) for local import in Models & Runtimes.

.DESCRIPTION
  Local-only pack build for P4-E (docs/adr/0008-bergamot.md):
    1. fetches the pinned sources listed in helper/UPSTREAM.json ONCE into the shared dev cache
       (%LOCALAPPDATA%\sigil-dev-cache\bergamot): the mozilla/translations archive (only its inference/
       folder is extracted) and one archive per git submodule the native build needs, sequentially
       and a few seconds apart; every archive is checked against its pinned size and sha256
       (-PinArchives records them the first time);
    2. configures helper/CMakeLists.txt with Ninja + MSVC (Visual Studio 2022 toolset preferred;
       static CRT, the WebAssembly-compatible sources: intgemm for the int8 models, no MKL/BLAS);
    3. builds sigil-bergamot.exe and checks it imports only Windows system DLLs;
    4. stages <OutDir>\bergamot-<arch>\ (the exe, sigil-pack.json with protocol, arch and sha256 of
       every file, and each component's license) plus a deterministic zip
       sigil-bergamot-<version>-cpu.zip (the catalog's file name; other -Arch values add a suffix)
       for local import or SIGIL_LOCAL_ASSETS. Packs stay local (catalog status "unpublished").
    5. -SelfTest drives the staged exe through the protocol. With -Models it also translates a
       fixed set of lines en->ja and ja->en and fails when the 95th-percentile line takes longer
       than -MaxP95Ms (150 ms, the P4-E acceptance target).

  -Fake builds only the protocol shell with a fake engine: no network, no upstream sources. Its
  self-test runs automatically (protocol conformance of main.cpp / json.cpp).

  Nothing else touches the network.

.PARAMETER Arch
  avx2 (default: every PC since about 2015), sse2 (compatibility build for older CPUs; intgemm
  still picks SSSE3/AVX2 kernels at run time) or avx512.

.PARAMETER Models
  A folder with one subfolder per direction key (en_ja, ja_en), each holding the Mozilla files as
  the model store installs them (model.*.intgemm.alphas.bin, lex.*.s2t.bin, vocab/srcvocab/trgvocab
  *.spm; a version subfolder in between is fine), e.g. <app data>\models\mt-bergamot.

.EXAMPLE
  pwsh scripts/runtime-packs/bergamot/build.ps1 -FetchOnly -PinArchives
.EXAMPLE
  pwsh scripts/runtime-packs/bergamot/build.ps1 -SelfTest -Models "$env:APPDATA\com.sigil.app\models\mt-bergamot"
.EXAMPLE
  pwsh scripts/runtime-packs/bergamot/build.ps1 -Fake
#>
[CmdletBinding()]
param(
    [ValidateSet('avx2', 'sse2', 'avx512')]
    [string]$Arch = 'avx2',
    [string]$OutDir,
    [string]$BuildRoot,
    [string]$SourceDir,
    [int]$Jobs = 0,
    [switch]$Fake,
    [switch]$FetchOnly,
    [switch]$PinArchives,
    [switch]$SelfTest,
    [string]$Models,
    [int]$MaxP95Ms = 150,
    [switch]$Clean,
    [switch]$NoZip
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$PackDir = $PSScriptRoot
$HelperDir = Join-Path $PackDir 'helper'
$UpstreamPath = Join-Path $HelperDir 'UPSTREAM.json'
$Upstream = Get-Content -Raw -LiteralPath $UpstreamPath | ConvertFrom-Json
$CatalogVersion = [string]$Upstream.catalog_version
$ExpectedVersion = "$($Upstream.upstream_version)-s$($Upstream.shim_revision)"
if ($CatalogVersion -ne $ExpectedVersion) { throw "UPSTREAM.json catalog_version '$CatalogVersion' must be '$ExpectedVersion'" }
$RootCommit = [string]$Upstream.root.commit
$ShortCommit = $RootCommit.Substring(0, 7)
$UpstreamLabel = "$($Upstream.root.repo) $ShortCommit"

if (-not $BuildRoot) {
    $BuildRoot = if ($env:SIGIL_NATIVE_BUILD_ROOT) { $env:SIGIL_NATIVE_BUILD_ROOT } else { Join-Path $env:LOCALAPPDATA 'sigil-native' }
}
if (-not $OutDir) { $OutDir = Join-Path $BuildRoot 'packs' }
New-Item -ItemType Directory -Force -Path $BuildRoot, $OutDir | Out-Null
if ($Jobs -le 0) { $Jobs = [Environment]::ProcessorCount }
$CacheRoot = if ($env:SIGIL_DEV_CACHE) { Join-Path $env:SIGIL_DEV_CACHE 'bergamot' } else { Join-Path $env:LOCALAPPDATA 'sigil-dev-cache\bergamot' }

function Write-Step([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }

# ---------------------------------------------------------------------------------------------
# 1. Pinned sources
# ---------------------------------------------------------------------------------------------

$script:LastRequest = [DateTime]::MinValue
function Wait-RequestSpacing {
    # The dev PC's router blocks bursts of connections: at least 3 s between request starts.
    $wait = 3000 - ([DateTime]::UtcNow - $script:LastRequest).TotalMilliseconds
    if ($wait -gt 0) { Start-Sleep -Milliseconds ([int]$wait) }
    $script:LastRequest = [DateTime]::UtcNow
}

function Get-Archive($Item, [string]$Name) {
    $archives = Join-Path $CacheRoot 'archives'
    New-Item -ItemType Directory -Force -Path $archives | Out-Null
    $file = Join-Path $archives "$Name-$($Item.commit.Substring(0, 12)).tar.gz"
    if (-not (Test-Path -LiteralPath $file)) {
        if (-not $Item.sha256 -and -not $PinArchives) {
            throw "$Name is not pinned in UPSTREAM.json (sha256 null): run with -FetchOnly -PinArchives once to download and record it"
        }
        Wait-RequestSpacing
        Write-Step "Downloading $($Item.repo) @ $($Item.commit.Substring(0, 12)) (one request)"
        $partial = "$file.partial"
        Invoke-WebRequest -Uri $Item.url -OutFile $partial -UseBasicParsing -MaximumRetryCount 2 -RetryIntervalSec 5
        Move-Item -Force -LiteralPath $partial -Destination $file
    }
    $size = (Get-Item -LiteralPath $file).Length
    $sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $file).Hash.ToLowerInvariant()
    if ($Item.sha256) {
        if ($sha -ne $Item.sha256 -or ($Item.size -and $size -ne [long]$Item.size)) {
            Remove-Item -Force -LiteralPath $file
            throw "$Name archive does not match UPSTREAM.json (got $size bytes, sha256 $sha); deleted, fetch again"
        }
    } else {
        $Item.sha256 = $sha
        $Item.size = $size
        $script:Pinned = $true
        Write-Host "    pinned $Name : $size bytes, sha256 $sha"
    }
    return $file
}

function Expand-Into([string]$Archive, [string]$Destination, [string]$Member) {
    # bsdtar (Windows' tar.exe): drop the archive's top folder; optionally only one member folder.
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $top = (& tar -tzf $Archive | Select-Object -First 1) -replace '/.*$', ''
    if (-not $top) { throw "cannot read $Archive" }
    $strip = 1
    $members = @()
    if ($Member) {
        $members = @("$top/$Member")
        $strip = 1 + ($Member -split '/').Count
    }
    & tar -xzf $Archive -C $Destination "--strip-components=$strip" @members
    if ($LASTEXITCODE -ne 0) { throw "tar failed to extract $Archive" }
}

function Get-PinnedSource {
    $tree = Join-Path $CacheRoot "src-$ShortCommit"
    $inference = Join-Path $tree 'inference'
    $marker = Join-Path $tree '.sigil-extracted'
    $expected = (@($Upstream.root.sha256) + @($Upstream.submodules | ForEach-Object { $_.sha256 })) -join ','
    if ((Test-Path -LiteralPath $marker) -and $Upstream.root.sha256 -and ((Get-Content -Raw -LiteralPath $marker).Trim() -eq $expected)) {
        return $inference
    }
    $script:Pinned = $false
    $rootArchive = Get-Archive $Upstream.root 'translations'
    $subArchives = @()
    foreach ($s in $Upstream.submodules) {
        $subArchives += [pscustomobject]@{ Item = $s; File = (Get-Archive $s (($s.repo -split '/')[1])) }
    }
    if ($script:Pinned) {
        ($Upstream | ConvertTo-Json -Depth 8) | Set-Content -LiteralPath $UpstreamPath -Encoding utf8NoBOM
        Write-Step "Recorded the archive digests in $UpstreamPath (review and keep the change)"
    }
    if (Test-Path -LiteralPath $tree) { Remove-Item -Recurse -Force -LiteralPath $tree }
    Write-Step "Extracting inference/ and $($subArchives.Count) submodules into $tree"
    Expand-Into $rootArchive $inference $Upstream.root.subdir
    foreach ($s in $subArchives) {
        $dest = Join-Path $tree ($s.Item.path -replace '/', '\')
        if (Test-Path -LiteralPath $dest) { Remove-Item -Recurse -Force -LiteralPath $dest }
        Expand-Into $s.File $dest $null
    }
    $expected = (@($Upstream.root.sha256) + @($Upstream.submodules | ForEach-Object { $_.sha256 })) -join ','
    Set-Content -LiteralPath $marker -Value $expected -Encoding ascii
    return $inference
}

$Src = $null
if (-not $Fake) {
    $Src = if ($SourceDir) { (Resolve-Path -LiteralPath $SourceDir).Path } else { Get-PinnedSource }
    if (-not (Test-Path -LiteralPath (Join-Path $Src 'marian-fork'))) { throw "$Src is not the inference/ folder of mozilla/translations" }
    Write-Step "Bergamot source: $Src"
}
if ($FetchOnly) { return }

# ---------------------------------------------------------------------------------------------
# 2. Toolchain (MSVC environment, Ninja, CMake)
# ---------------------------------------------------------------------------------------------

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) { throw 'vswhere.exe not found: install Visual Studio 2022 Build Tools with the C++ workload' }
$vs = & $vswhere -version '[17.0,18.0)' -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath -latest
if (-not $vs) {
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw 'No Visual Studio installation with the C++ x64 tools was found' }
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

# ---------------------------------------------------------------------------------------------
# 3. Configure and build
# ---------------------------------------------------------------------------------------------

$flavor = if ($Fake) { 'fake' } else { $Arch }
$bdir = Join-Path $BuildRoot "bergamot\$flavor"
if ($Clean -and (Test-Path -LiteralPath $bdir)) { Remove-Item -Recurse -Force -LiteralPath $bdir }
$cfg = @('-S', $HelperDir, '-B', $bdir, '-G', 'Ninja', "-DCMAKE_MAKE_PROGRAM=$ninja", '-DCMAKE_BUILD_TYPE=Release',
    '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl',
    # Upstream's third-party trees still declare old cmake_minimum_required versions (CMake 4 refuses < 3.5).
    '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
    "-DSIGIL_BERGAMOT_VERSION=$CatalogVersion", "-DSIGIL_BERGAMOT_UPSTREAM=$UpstreamLabel")
if ($Fake) {
    $cfg += '-DSIGIL_BERGAMOT_FAKE=ON'
} else {
    $cfg += @('-DSIGIL_BERGAMOT_FAKE=OFF', "-DBERGAMOT_SOURCE_DIR=$Src", "-DSIGIL_BERGAMOT_ARCH=$Arch")
}
Write-Step "Configuring sigil-bergamot ($flavor)"
& $cmake @cfg
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($flavor)" }
Write-Step "Building sigil-bergamot ($flavor, $Jobs jobs)"
$sw = [Diagnostics.Stopwatch]::StartNew()
& $cmake --build $bdir --target sigil-bergamot --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw "build failed ($flavor)" }
$buildSeconds = [math]::Round($sw.Elapsed.TotalSeconds)
$exe = Join-Path $bdir 'sigil-bergamot.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "$exe was not produced" }
if ($Fake) {
    Write-Step 'JSON unit tests'
    & $cmake --build $bdir --target sigil-bergamot-json-test --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw 'building the JSON unit tests failed' }
    & (Join-Path $bdir 'sigil-bergamot-json-test.exe')
    if ($LASTEXITCODE -ne 0) { throw 'the JSON unit tests failed' }
}

# Static CRT: only Windows system DLLs may be imported (no VCRUNTIME / MSVCP, nothing from the build PC).
$imports = @()
if ($dumpbin) {
    $imports = @(& $dumpbin /nologo /dependents $exe | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '\.dll$' -and $_ -notmatch '[\\/:]' })
    $system = @('KERNEL32.dll', 'ADVAPI32.dll', 'USER32.dll', 'SHELL32.dll', 'OLE32.dll', 'WS2_32.dll', 'SHLWAPI.dll', 'bcrypt.dll', 'dbghelp.dll', 'PSAPI.DLL')
    $bad = @($imports | Where-Object { $_ -notlike 'api-ms-win-*' -and ($system -notcontains $_) })
    if ($bad.Count -gt 0) { throw "sigil-bergamot.exe imports unexpected DLLs: $($bad -join ', ') (the pack must be self-contained)" }
}

# ---------------------------------------------------------------------------------------------
# 4. Stage the pack
# ---------------------------------------------------------------------------------------------

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

$pack = Join-Path $OutDir "bergamot-$flavor"
if (Test-Path -LiteralPath $pack) { Remove-Item -Recurse -Force -LiteralPath $pack }
New-Item -ItemType Directory -Force -Path $pack | Out-Null
Copy-Item -LiteralPath $exe -Destination $pack

$components = @()
if (-not $Fake) {
    $treeRoot = Split-Path -Parent $Src
    $licenseSources = @(
        @{ Name = 'bergamot-translator (mozilla/translations inference)'; Dir = $Src; Out = 'LICENSE-bergamot.txt'; License = $Upstream.root.license; Commit = $RootCommit },
        @{ Name = 'marian (Mozilla fork)'; Dir = (Join-Path $Src 'marian-fork'); Out = 'LICENSE-marian.txt'; License = 'MIT'; Commit = $RootCommit }
    )
    foreach ($s in $Upstream.submodules) {
        $short = ($s.repo -split '/')[1]
        $licenseSources += @{ Name = $s.repo; Dir = (Join-Path $treeRoot ($s.path -replace '/', '\')); Out = "LICENSE-$short.txt"; License = $s.license; Commit = $s.commit }
    }
    foreach ($l in $licenseSources) {
        $file = Get-ChildItem -LiteralPath $l.Dir -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^(LICEN[CS]E|COPYING)(\.md|\.txt)?$' } | Select-Object -First 1
        if (-not $file -and $l.Dir -eq $Src) {
            # The inference/ folder is covered by the repository's root license (MPL-2.0).
            $file = Get-ChildItem -LiteralPath $treeRoot -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^LICEN[CS]E' } | Select-Object -First 1
        }
        if ($file) { Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $pack $l.Out) }
        else { Write-Warning "no license file found for $($l.Name) in $($l.Dir)" }
        $components += "| $($l.Name) | ``$($l.Commit.Substring(0, 12))`` | $($l.License) | $(if ($file) { $l.Out } else { '(not found)' }) |"
    }
}
$notice = @(
    '# sigil-bergamot third-party notices',
    '',
    "sigil-bergamot $CatalogVersion ($flavor) is Sigil's JSON-lines front end (AGPL-3.0) for Mozilla's Bergamot translation engine, built from $UpstreamLabel.",
    'Each component''s license text is included in this pack as listed below.',
    '',
    '| Component | Commit | License | File |',
    '|---|---|---|---|'
) + $components
if ($Fake) { $notice += '| (fake engine build: no third-party code) | | | |' }
$notice | Set-Content -LiteralPath (Join-Path $pack 'THIRD-PARTY-NOTICES.md') -Encoding utf8NoBOM

$files = @(Get-ChildItem -LiteralPath $pack -File | Sort-Object Name | ForEach-Object {
        [ordered]@{ path = $_.Name; size = $_.Length; sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant() }
    })
$manifest = [ordered]@{
    schema          = 1
    name            = 'sigil-bergamot'
    version         = $CatalogVersion
    catalog_id      = 'runtime/bergamot'
    protocol        = [int]$Upstream.protocol
    arch            = $(if ($Fake) { 'sse2' } else { $Arch })
    fake            = [bool]$Fake
    files           = $files
    upstream        = [ordered]@{ repo = $Upstream.root.repo; commit = $RootCommit; archive_sha256 = $Upstream.root.sha256 }
    imports         = $imports
    toolchain       = [ordered]@{ msvc = $env:VCToolsVersion; cmake = ((& $cmake --version | Select-Object -First 1) -replace 'cmake version ', '') }
}
($manifest | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath (Join-Path $pack 'sigil-pack.json') -Encoding utf8NoBOM

$zipPath = $null
$zipHash = $null
if (-not $NoZip -and -not $Fake) {
    $suffix = if ($Arch -eq 'avx2') { '' } else { "-$Arch" }
    $zipPath = Join-Path $OutDir "sigil-bergamot-$CatalogVersion-cpu$suffix.zip"
    Write-DeterministicZip $pack $zipPath
    $zipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zipPath).Hash.ToLowerInvariant()
    Set-Content -LiteralPath "$zipPath.sha256" -Value "$zipHash  $(Split-Path -Leaf $zipPath)" -Encoding ascii
}

# ---------------------------------------------------------------------------------------------
# 5. Self-test over the protocol
# ---------------------------------------------------------------------------------------------

class Helper {
    [System.Diagnostics.Process]$Process
    [int]$NextId = 1

    Helper([string]$Exe, [int]$Threads) {
        $psi = [System.Diagnostics.ProcessStartInfo]::new($Exe)
        $psi.ArgumentList.Add('--jsonl')
        $psi.ArgumentList.Add('--threads')
        $psi.ArgumentList.Add([string]$Threads)
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.RedirectStandardInput = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $utf8 = [System.Text.UTF8Encoding]::new($false)
        $psi.StandardInputEncoding = $utf8
        $psi.StandardOutputEncoding = $utf8
        $psi.StandardErrorEncoding = $utf8
        $this.Process = [System.Diagnostics.Process]::Start($psi)
        # Drain stderr in the background so the helper never blocks on it.
        $this.Process.BeginErrorReadLine()
    }

    [void] SendRaw([string]$Line) {
        $this.Process.StandardInput.Write($Line + "`n")
        $this.Process.StandardInput.Flush()
    }

    [int] Send([hashtable]$Body) {
        $id = $this.NextId++
        $Body['id'] = $id
        $this.SendRaw(($Body | ConvertTo-Json -Compress -Depth 5))
        return $id
    }

    [object] Read() {
        $task = $this.Process.StandardOutput.ReadLineAsync()
        if (-not $task.Wait(60000)) {
            $this.Process.Kill()
            throw 'self-test: no reply from the helper within 60 s'
        }
        $line = $task.Result
        if ($null -eq $line) { throw 'the helper closed stdout' }
        return ($line | ConvertFrom-Json)
    }

    [object] Call([hashtable]$Body) {
        $id = $this.Send($Body)
        while ($true) {
            $r = $this.Read()
            if ($r.id -eq $id) { return $r }
        }
        return $null
    }

    [int] Close() {
        $this.Process.StandardInput.Close()
        if (-not $this.Process.WaitForExit(10000)) { $this.Process.Kill(); throw 'the helper did not exit after stdin closed' }
        return $this.Process.ExitCode
    }
}

function Assert-True([bool]$Condition, [string]$Message) { if (-not $Condition) { throw "self-test: $Message" } }

function Find-DirectionFolder([string]$Root, [string]$Key) {
    $dir = Join-Path $Root $Key
    if (-not (Test-Path -LiteralPath $dir)) { return $null }
    $model = Get-ChildItem -LiteralPath $dir -Recurse -File -Filter 'model.*.bin' | Select-Object -First 1
    if (-not $model) { return $null }
    return $model.DirectoryName
}

function Get-LoadRequest([string]$Key, [string]$Dir) {
    $model = (Get-ChildItem -LiteralPath $Dir -File -Filter 'model.*.bin' | Select-Object -First 1).FullName
    $lex = (Get-ChildItem -LiteralPath $Dir -File -Filter 'lex.*.bin' | Select-Object -First 1)?.FullName
    $shared = (Get-ChildItem -LiteralPath $Dir -File -Filter 'vocab.*.spm' | Select-Object -First 1)?.FullName
    $vocabs = if ($shared) { @($shared, $shared) } else {
        @((Get-ChildItem -LiteralPath $Dir -File -Filter 'srcvocab.*.spm').FullName, (Get-ChildItem -LiteralPath $Dir -File -Filter 'trgvocab.*.spm').FullName)
    }
    $gemm = if ($model -like '*.intgemm8.bin') { 'int8shiftAll' } else { 'int8shiftAlphaAll' }
    return @{ op = 'load'; key = $Key; model = $model; lex = $lex; vocabs = $vocabs; gemm = $gemm }
}

function Invoke-ProtocolSelfTest([string]$Exe) {
    $h = [Helper]::new($Exe, 1)
    try {
        $hello = $h.Call(@{ op = 'hello'; protocol = 1 })
        Assert-True ($hello.ok -and $hello.result.protocol -eq 1) "hello: $($hello | ConvertTo-Json -Compress)"
        $h.SendRaw('{ not json')
        $bad = $h.Read()
        Assert-True ((-not $bad.ok) -and $null -eq $bad.id -and $bad.error.kind -eq 'invalid_request') 'invalid JSON is answered with id null'
        $r = $h.Call(@{ op = 'translate'; keys = @('en_ja'); text = 'x' })
        Assert-True ((-not $r.ok) -and $r.error.kind -eq 'not_loaded') 'translate before load: not_loaded'
        $r = $h.Call(@{ op = 'frobnicate' })
        Assert-True ((-not $r.ok) -and $r.error.kind -eq 'invalid_request') 'unknown op'
        $r = $h.Call(@{ op = 'load'; key = 'Bad Key'; model = 'x'; vocabs = @('y') })
        Assert-True ((-not $r.ok) -and $r.error.kind -eq 'invalid_request') 'invalid key'
        $r = $h.Call(@{ op = 'stats' })
        Assert-True ($r.ok -and $r.result.threads -eq 1) 'stats'
        if ($hello.result.engine -eq 'fake') {
            $tmp = Join-Path ([IO.Path]::GetTempPath()) "sigil-bergamot-selftest-$PID-日本"
            New-Item -ItemType Directory -Force -Path $tmp | Out-Null
            foreach ($f in 'model.enja.intgemm.alphas.bin', 'lex.50.50.enja.s2t.bin', 'vocab.enja.spm') { Set-Content -LiteralPath (Join-Path $tmp $f) -Value 'x' }
            try {
                $r = $h.Call((Get-LoadRequest 'en_ja' $tmp))
                Assert-True $r.ok "load from a non-ASCII folder: $($r | ConvertTo-Json -Compress)"
                $r = $h.Call(@{ op = 'translate'; keys = @('en_ja'); text = "héllo ""quoted""`n日本 😀"; priority = 'final' })
                Assert-True ($r.ok -and $r.result.text -eq "<en_ja>héllo ""quoted""`n日本 😀") "translate round-trips UTF-8 and escapes: $($r | ConvertTo-Json -Compress)"
                # Cancel a queued interim behind a slow one.
                $slow = $h.Send(@{ op = 'translate'; keys = @('en_ja'); text = '__slow__'; priority = 'interim' })
                $queued = $h.Send(@{ op = 'translate'; keys = @('en_ja'); text = 'queued'; priority = 'interim' })
                $cancel = $h.Send(@{ op = 'cancel'; target = $queued })
                $seen = @{}
                while ($seen.Count -lt 3) { $x = $h.Read(); $seen[[string]$x.id] = $x }
                Assert-True ($seen[[string]$queued].error.kind -eq 'cancelled') 'the queued request was cancelled'
                Assert-True ($seen[[string]$cancel].result.cancelled -eq $true) 'cancel reports success'
                Assert-True ($seen[[string]$slow].result.text -eq '<en_ja>__slow__') 'the running request finished'
                $r = $h.Call(@{ op = 'unload'; key = 'en_ja' })
                Assert-True $r.ok 'unload'
            } finally { Remove-Item -Recurse -Force -LiteralPath $tmp -ErrorAction SilentlyContinue }
        }
    } finally {
        $code = $h.Close()
    }
    Assert-True ($code -eq 0) "exit code $code after stdin EOF"
    return 'protocol ok'
}

function Invoke-TranslationSelfTest([string]$Exe, [string]$Root) {
    $lines = @{
        'en_ja' = @(
            'Hello, can you hear me?', 'Let''s meet at the plaza in five minutes.', 'I think the event starts at eight tonight.',
            'Thank you for coming to the stream!', 'Wait, which world are we going to next?', 'That avatar looks amazing.',
            'Could you say that one more time, slowly?', 'I''m going to grab some water, be right back.',
            'The music in this world is really nice.', 'See you tomorrow, good night!')
        'ja_en' = @(
            'こんにちは、聞こえますか？', '五分後に広場で会いましょう。', 'イベントは今夜八時に始まると思います。',
            '配信に来てくれてありがとう！', '次はどのワールドに行くの？', 'そのアバター、すごくかわいいね。',
            'もう一度ゆっくり言ってもらえますか？', 'ちょっと水を取ってくるね、すぐ戻る。', 'このワールドの音楽、本当にいいね。', 'また明日、おやすみ！')
    }
    $results = @()
    $h = [Helper]::new($Exe, 1)
    try {
        $hello = $h.Call(@{ op = 'hello'; protocol = 1 })
        Assert-True ($hello.ok -and $hello.result.protocol -eq 1) 'hello'
        foreach ($key in $lines.Keys | Sort-Object) {
            $dir = Find-DirectionFolder $Root $key
            if (-not $dir) { Write-Warning "no $key model under $Root; skipped"; continue }
            $sw = [Diagnostics.Stopwatch]::StartNew()
            $r = $h.Call((Get-LoadRequest $key $dir))
            Assert-True $r.ok "load $key : $($r | ConvertTo-Json -Compress)"
            $loadMs = $sw.ElapsedMilliseconds
            # Warm-up line (first call allocates the workspace), then the timed set.
            $null = $h.Call(@{ op = 'translate'; keys = @($key); text = $lines[$key][0]; priority = 'final' })
            $times = @()
            foreach ($text in $lines[$key]) {
                $sw.Restart()
                $r = $h.Call(@{ op = 'translate'; keys = @($key); text = $text; priority = 'final' })
                $ms = $sw.Elapsed.TotalMilliseconds
                Assert-True ($r.ok -and $r.result.text.Trim().Length -gt 0) "translate $key : $($r | ConvertTo-Json -Compress)"
                $times += $ms
                Write-Host ("    {0,7:N1} ms  {1}  ->  {2}" -f $ms, $text, $r.result.text)
            }
            $sorted = @($times | Sort-Object)
            $p50 = $sorted[[math]::Floor(($sorted.Count - 1) * 0.5)]
            $p95 = $sorted[[math]::Ceiling(($sorted.Count - 1) * 0.95)]
            $results += [pscustomobject]@{ Direction = $key; LoadMs = $loadMs; P50Ms = [math]::Round($p50, 1); P95Ms = [math]::Round($p95, 1); MaxMs = [math]::Round(($sorted | Select-Object -Last 1), 1) }
        }
    } finally {
        $code = $h.Close()
    }
    Assert-True ($code -eq 0) "exit code $code after stdin EOF"
    $results | Format-Table -AutoSize | Out-String | Write-Host
    $slow = @($results | Where-Object { $_.P95Ms -gt $MaxP95Ms })
    if ($results.Count -eq 0) { return 'no models found' }
    if ($slow.Count -gt 0) { return "FAIL: p95 over $MaxP95Ms ms for $($slow.Direction -join ', ')" }
    return "pass (p95 <= $MaxP95Ms ms)"
}

$testResult = '-'
if ($SelfTest -or $Fake) {
    Write-Step "Self-test ($flavor)"
    $packExe = Join-Path $pack 'sigil-bergamot.exe'
    $testResult = Invoke-ProtocolSelfTest $packExe
    if ($Models) { $testResult += '; ' + (Invoke-TranslationSelfTest $packExe $Models) }
}

Write-Step "Pack in $pack"
[pscustomobject]@{
    Flavor    = $flavor
    ExeMB     = [math]::Round((Get-Item -LiteralPath (Join-Path $pack 'sigil-bergamot.exe')).Length / 1MB, 2)
    BuildSec  = $buildSeconds
    SelfTest  = $testResult
    Zip       = if ($zipPath) { Split-Path -Leaf $zipPath } else { '-' }
    ZipSha256 = $zipHash
} | Format-List | Out-String | Write-Host
if ($testResult -like '*FAIL*') { exit 1 }
