#Requires -Version 7.0
<#
.SYNOPSIS
  Converts NVIDIA Magpie TTS Multilingual v2607 (Korean, Arabic and Brazilian Portuguese on top of
  v2602's languages) for Sigil's Magpie voices: tts/magpie-tts-357m-v2607 (the GGUF) and
  tts/magpie-tts-357m-v2607-tokenizer (a zip of the tokenizer files).

.DESCRIPTION
  NVIDIA publishes v2607 only as a NeMo .nemo archive (factor-2 frame stacking), which
  NeMo-Speech.cpp must convert locally (docs/tts/models.md). A maintainer step, run once:

    1. the .nemo at revision v2607 (-Nemo, or one download into the dev cache);
    2. NeMo-Speech.cpp's converter at the commit the Rust engine was written against (one archive
       download), in a virtual environment of its own (pip installs requirements.txt: torch CPU,
       gguf, numpy, ... about 1 GB; one package at a time);
    3. convert_model.py -> magpie_tts_multilingual_357m.v2607.f16.gguf;
    4. every member of the .nemo except the weights -> magpie-tts-357m-v2607-tokenizer.zip;
    5. prints both files' size and sha256 (pin them in tts-ggml-experimental.json when publishing).

  Then, in Sigil: Models & Runtimes -> Import the GGUF for "Magpie TTS Multilingual 357M (v2607)"
  and the zip for "Magpie TTS v2607 text tokenizer" (or use -LocalAssets for SIGIL_LOCAL_ASSETS).

.EXAMPLE
  pwsh scripts/runtime-packs/nemo-speech/convert-magpie-v2607.ps1
.EXAMPLE
  pwsh scripts/runtime-packs/nemo-speech/convert-magpie-v2607.ps1 -Nemo D:\models\magpie_tts_multilingual_357m.nemo -LocalAssets D:\sigil-assets
#>
[CmdletBinding()]
param(
    [string]$Nemo,
    [string]$WorkDir = (Join-Path $env:LOCALAPPDATA 'sigil-dev-cache\nemo-speech\magpie-v2607'),
    [string]$Python = 'python',
    [string]$LocalAssets
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# The NeMo-Speech.cpp commit whose server and converter tts_helpers::magpie targets (the release
# v0.1.0 packs read GGUFs of this converter generation).
$NemoSpeechCommit = '97a15afa5caa9bce5baaa86c1184103877af4101'
$NemoUrl = 'https://huggingface.co/nvidia/magpie_tts_multilingual_357m/resolve/v2607/magpie_tts_multilingual_357m.nemo'
$GgufName = 'magpie_tts_multilingual_357m.v2607.f16.gguf'
$ZipName = 'magpie-tts-357m-v2607-tokenizer.zip'

function Write-Step([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$tar = Join-Path $env:SystemRoot 'System32\tar.exe'
$curl = Join-Path $env:SystemRoot 'System32\curl.exe'

# 1. The checkpoint.
if (-not $Nemo) {
    $Nemo = Join-Path $WorkDir 'magpie_tts_multilingual_357m-v2607.nemo'
    if (-not (Test-Path -LiteralPath $Nemo)) {
        Write-Step 'Downloading the v2607 .nemo (one request, several hundred MB; resumable)'
        & $curl --fail --location --show-error --continue-at - --retry 2 --retry-delay 5 --output "$Nemo.partial" $NemoUrl
        if ($LASTEXITCODE -ne 0) { throw "curl failed ($LASTEXITCODE) fetching $NemoUrl" }
        Move-Item -Force -LiteralPath "$Nemo.partial" -Destination $Nemo
    }
}
if (-not (Test-Path -LiteralPath $Nemo)) { throw "$Nemo not found" }
Write-Host "    .nemo sha256 $((Get-FileHash -Algorithm SHA256 -LiteralPath $Nemo).Hash.ToLowerInvariant())"

# 2. The converter (pinned source archive) in its own virtual environment.
$src = Join-Path $WorkDir "nemo-speech-$($NemoSpeechCommit.Substring(0, 7))"
if (-not (Test-Path -LiteralPath (Join-Path $src 'convert_model.py'))) {
    $archive = "$src.tar.gz"
    if (-not (Test-Path -LiteralPath $archive)) {
        Write-Step 'Downloading the NeMo-Speech.cpp source archive (one request)'
        & $curl --fail --location --silent --show-error --output "$archive.partial" "https://codeload.github.com/NVIDIA/NeMo-Speech.cpp/tar.gz/$NemoSpeechCommit"
        if ($LASTEXITCODE -ne 0) { throw "curl failed ($LASTEXITCODE) fetching the NeMo-Speech.cpp archive" }
        Move-Item -Force -LiteralPath "$archive.partial" -Destination $archive
    }
    Write-Host "    archive sha256 $((Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLowerInvariant())"
    New-Item -ItemType Directory -Force -Path $src | Out-Null
    & $tar -xzf $archive -C $src --strip-components=1 --exclude '*/test_files' --exclude '*/ggml' --exclude '*/llama.cpp' --exclude '*/third_party'
    if ($LASTEXITCODE -ne 0) { throw "tar failed ($LASTEXITCODE)" }
}
$venv = Join-Path $WorkDir 'venv'
$py = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) {
    Write-Step "Creating the converter's virtual environment"
    & $Python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "python -m venv failed ($LASTEXITCODE): install Python 3.10-3.12" }
    & $py -m pip install --upgrade pip --progress-bar off
    Write-Step 'Installing the converter requirements (torch CPU and friends)'
    & $py -m pip install --progress-bar off -r (Join-Path $src 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw "pip install failed ($LASTEXITCODE)" }
}

# 3. The GGUF.
$gguf = Join-Path $WorkDir $GgufName
if (-not (Test-Path -LiteralPath $gguf)) {
    Write-Step 'Converting the checkpoint (f16 GGUF)'
    Push-Location $src
    try {
        & $py convert_model.py $Nemo --outfile $gguf
        if ($LASTEXITCODE -ne 0) { throw "convert_model.py failed ($LASTEXITCODE)" }
    } finally { Pop-Location }
}

# 4. The tokenizer files: every member except the weights.
$tokDir = Join-Path $WorkDir 'tokenizer'
if (Test-Path -LiteralPath $tokDir) { Remove-Item -Recurse -Force -LiteralPath $tokDir }
New-Item -ItemType Directory -Force -Path $tokDir | Out-Null
Write-Step 'Extracting the tokenizer files'
& $tar -xf $Nemo -C $tokDir --exclude '*model_weights.ckpt'
if ($LASTEXITCODE -ne 0) { throw "tar failed ($LASTEXITCODE) reading $Nemo" }
if (-not (Test-Path -LiteralPath (Join-Path $tokDir 'model_config.yaml'))) { throw "the .nemo has no model_config.yaml at its top level" }
$zip = Join-Path $WorkDir $ZipName
if (Test-Path -LiteralPath $zip) { Remove-Item -Force -LiteralPath $zip }
Add-Type -AssemblyName System.IO.Compression
$stamp = [DateTimeOffset]::new(2026, 1, 1, 0, 0, 0, [TimeSpan]::Zero)
$fs = [System.IO.File]::Open($zip, [System.IO.FileMode]::CreateNew)
try {
    $archiveOut = [System.IO.Compression.ZipArchive]::new($fs, [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($f in (Get-ChildItem -LiteralPath $tokDir -File -Recurse | Sort-Object FullName)) {
            $rel = [System.IO.Path]::GetRelativePath($tokDir, $f.FullName) -replace '\\', '/'
            $entry = $archiveOut.CreateEntry($rel, [System.IO.Compression.CompressionLevel]::Optimal)
            $entry.LastWriteTime = $stamp
            $in = [System.IO.File]::OpenRead($f.FullName)
            $es = $entry.Open()
            try { $in.CopyTo($es) } finally { $es.Dispose(); $in.Dispose() }
        }
    } finally { $archiveOut.Dispose() }
} finally { $fs.Dispose() }

# 5. Report (and SIGIL_LOCAL_ASSETS layout).
foreach ($p in @($gguf, $zip)) {
    $item = Get-Item -LiteralPath $p
    Write-Host ("    {0}: {1} bytes, sha256 {2}" -f $item.Name, $item.Length, (Get-FileHash -Algorithm SHA256 -LiteralPath $p).Hash.ToLowerInvariant())
}
if ($LocalAssets) {
    foreach ($pair in @(@('tts/magpie-tts-357m-v2607', $gguf), @('tts/magpie-tts-357m-v2607-tokenizer', $zip))) {
        $dest = Join-Path $LocalAssets ($pair[0] -replace '/', '\')
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        Copy-Item -Force -LiteralPath $pair[1] -Destination $dest
    }
    Write-Host "    Copied for SIGIL_LOCAL_ASSETS=$LocalAssets"
}
Write-Step "Done: import $GgufName and $ZipName in Models & Runtimes (Magpie v2607 entries)."
