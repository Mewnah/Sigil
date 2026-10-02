#Requires -Version 7.0
<#
.SYNOPSIS
  Fetches the Magpie TTS v2602 tokenizer files (catalog entry tts/magpie-tts-357m-v2602-tokenizer)
  for Sigil's Magpie voices: the first 32 MiB of NVIDIA's .nemo archive, one request.

.DESCRIPTION
  NVIDIA ships Magpie's pronunciation dictionaries and model_config.yaml only inside the
  magpie_tts_multilingual_357m.nemo archive (a tar of several hundred MB). The tokenizer members sit
  before the weights, so NeMo-Speech.cpp's own model index fetches just the archive prefix; this
  script does the same, once, sequentially:

    1. downloads bytes 0-33554431 of the .nemo at the pinned revision with one ranged request
       (curl.exe --range, refused if the server would send more) into the dev cache;
    2. checks the prefix against NeMo-Speech.cpp's pinned sha256;
    3. walks the tar headers and extracts the seven members the catalog lists, checking each one's
       size and sha256 against src-tauri/model-catalog/tts-ggml-experimental.json;
    4. writes them to -OutDir (and, with -LocalAssets, to <LocalAssets>\tts\magpie-tts-357m-v2602-tokenizer\
       for SIGIL_LOCAL_ASSETS).

  Then, in Sigil: Models & Runtimes -> "Magpie TTS v2602 text tokenizer" -> Import -> choose -OutDir.

.EXAMPLE
  pwsh scripts/runtime-packs/nemo-speech/fetch-magpie-tokenizer.ps1
.EXAMPLE
  pwsh scripts/runtime-packs/nemo-speech/fetch-magpie-tokenizer.ps1 -LocalAssets D:\sigil-assets
#>
[CmdletBinding()]
param(
    [string]$OutDir = (Join-Path $env:LOCALAPPDATA 'sigil-dev-cache\nemo-speech\magpie-tts-357m-v2602-tokenizer'),
    [string]$LocalAssets,
    # Use an already downloaded prefix instead of fetching it.
    [string]$Prefix
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# NeMo-Speech.cpp models/index.json (commit 97a15af): the tokenizer artifact of
# nvidia/magpie_tts_multilingual_357m, type "tar-prefix".
$Revision = '34d7e40da85cabc97f92198889b65cea27bc7fd1'
$Url = "https://huggingface.co/nvidia/magpie_tts_multilingual_357m/resolve/$Revision/magpie_tts_multilingual_357m.nemo"
$PrefixBytes = 33554432
$PrefixSha256 = '9efde603de69818ea8dc449ab4165c39bae661c8aa7f798a2c9891961e8b0dd0'
$EntryId = 'tts/magpie-tts-357m-v2602-tokenizer'

function Write-Step([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$catalog = Get-Content -Raw -LiteralPath (Join-Path $repo 'src-tauri\model-catalog\tts-ggml-experimental.json') | ConvertFrom-Json
$entry = $catalog.entries | Where-Object { $_.id -eq $EntryId }
if (-not $entry) { throw "$EntryId is not in tts-ggml-experimental.json" }
$wanted = @{}
foreach ($f in $entry.files) { $wanted[$f.path] = $f }

# 1-2. The archive prefix.
$cache = Join-Path $env:LOCALAPPDATA 'sigil-dev-cache\nemo-speech'
New-Item -ItemType Directory -Force -Path $cache | Out-Null
if (-not $Prefix) {
    $Prefix = Join-Path $cache "magpie_tts_multilingual_357m-$($Revision.Substring(0, 7)).nemo.prefix"
    if (-not (Test-Path -LiteralPath $Prefix)) {
        Write-Step "Downloading the first 32 MiB of the Magpie .nemo archive (one request)"
        $partial = "$Prefix.partial"
        $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
        & $curl --fail --location --silent --show-error --range "0-$($PrefixBytes - 1)" --max-filesize ($PrefixBytes + 1024) `
            --retry 2 --retry-delay 5 --output $partial $Url
        if ($LASTEXITCODE -ne 0) { throw "curl failed ($LASTEXITCODE) fetching $Url" }
        Move-Item -Force -LiteralPath $partial -Destination $Prefix
    }
}
$len = (Get-Item -LiteralPath $Prefix).Length
$sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $Prefix).Hash.ToLowerInvariant()
if ($len -ne $PrefixBytes -or $sha -ne $PrefixSha256) {
    throw "the archive prefix does not match ($len bytes, sha256 $sha; expected $PrefixBytes / $PrefixSha256). Delete $Prefix and run again."
}

# 3. Walk the tar headers (ustar and old-style; GNU long names are not used by these members).
$bytes = [System.IO.File]::ReadAllBytes($Prefix)
function Read-Field([int]$Offset, [int]$Length) {
    $end = $Offset
    while ($end -lt $Offset + $Length -and $bytes[$end] -ne 0) { $end++ }
    return [System.Text.Encoding]::ASCII.GetString($bytes, $Offset, $end - $Offset)
}
$found = @{}
$pos = 0
while ($pos + 512 -le $bytes.Length) {
    $empty = $true
    for ($i = $pos; $i -lt $pos + 512; $i++) { if ($bytes[$i] -ne 0) { $empty = $false; break } }
    if ($empty) { break }
    $name = Read-Field $pos 100
    $sizeText = (Read-Field ($pos + 124) 12).Trim()
    $size = if ($sizeText) { [Convert]::ToInt64($sizeText, 8) } else { 0 }
    $type = [char]$bytes[$pos + 156]
    if ((Read-Field ($pos + 257) 5) -eq 'ustar') {
        $prefixName = Read-Field ($pos + 345) 155
        if ($prefixName) { $name = "$prefixName/$name" }
    }
    $name = $name -replace '^\./', ''
    $data = $pos + 512
    if (($type -eq '0' -or $type -eq [char]0) -and $wanted.ContainsKey($name)) {
        if ($data + $size -gt $bytes.Length) { throw "$name is cut off by the 32 MiB prefix (the archive layout changed?)" }
        $member = [byte[]]::new($size)
        [Array]::Copy($bytes, $data, $member, 0, $size)
        $found[$name] = $member
    }
    $pos = $data + [int64]([Math]::Ceiling($size / 512.0) * 512)
}
$missing = @($wanted.Keys | Where-Object { -not $found.ContainsKey($_) })
if ($missing.Count -gt 0) { throw "the archive prefix lacks: $($missing -join ', ')" }

# 4. Check and write.
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$sha256 = [System.Security.Cryptography.SHA256]::Create()
try {
    foreach ($name in ($wanted.Keys | Sort-Object)) {
        $data = $found[$name]
        $hash = ([System.BitConverter]::ToString($sha256.ComputeHash($data)) -replace '-', '').ToLowerInvariant()
        $pin = $wanted[$name]
        if ($data.Length -ne [int64]$pin.size -or $hash -ne $pin.sha256) {
            throw "$name does not match the catalog ($($data.Length) bytes, $hash; expected $($pin.size) / $($pin.sha256))"
        }
        [System.IO.File]::WriteAllBytes((Join-Path $OutDir $name), $data)
        if ($LocalAssets) {
            $dest = Join-Path $LocalAssets ($EntryId -replace '/', '\')
            New-Item -ItemType Directory -Force -Path $dest | Out-Null
            [System.IO.File]::WriteAllBytes((Join-Path $dest $name), $data)
        }
    }
} finally { $sha256.Dispose() }

Write-Step "The Magpie v2602 tokenizer ($($wanted.Count) files, every hash checked) is in $OutDir"
Write-Host "    In Sigil: Models & Runtimes -> Magpie TTS v2602 text tokenizer -> Import -> choose that folder."
if ($LocalAssets) { Write-Host "    Also copied for SIGIL_LOCAL_ASSETS=$LocalAssets" }
