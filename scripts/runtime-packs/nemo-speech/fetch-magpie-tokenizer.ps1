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

  -Version v2607 (catalog entry tts/magpie-tts-357m-v2607-tokenizer, one zip): NVIDIA's own
  NeMo-Speech.cpp model index (models/index.json at v0.2.0 and main) names the tokenizer members of
  the v2607 .nemo with their sizes and sha256, and the two byte ranges that hold them. This mode
  fetches exactly those ranges of the .nemo at the pinned revision (about 31 MB in two requests, 3 s
  apart; not the 1.47 GB archive and no Python), checks the concatenated ranges and every member
  against the index, and writes magpie-tts-357m-v2607-tokenizer.zip (members at the zip root, fixed
  timestamps, so the same bytes give the same zip) plus a .sha256 file next to it. Upload that zip
  as the release asset of the catalog entry.

.EXAMPLE
  pwsh scripts/runtime-packs/nemo-speech/fetch-magpie-tokenizer.ps1
.EXAMPLE
  pwsh scripts/runtime-packs/nemo-speech/fetch-magpie-tokenizer.ps1 -LocalAssets D:\sigil-assets
.EXAMPLE
  pwsh scripts/runtime-packs/nemo-speech/fetch-magpie-tokenizer.ps1 -Version v2607
#>
[CmdletBinding()]
param(
    [ValidateSet('v2602', 'v2607')]
    [string]$Version = 'v2602',
    [string]$OutDir = (Join-Path $env:LOCALAPPDATA 'sigil-dev-cache\nemo-speech\magpie-tts-357m-v2602-tokenizer'),
    [string]$LocalAssets,
    # Use an already downloaded prefix instead of fetching it (v2602 only).
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

# ---------------------------------------------------------------------------------------------
# -Version v2607: one zip of the tokenizer members, from the two byte ranges NVIDIA's index names.
# ---------------------------------------------------------------------------------------------
if ($Version -eq 'v2607') {
    # NeMo-Speech.cpp models/index.json (v0.2.0 = main), model nvidia/magpie_tts_multilingual_357m,
    # artifact role "tokenizer" (type "tar-ranges"): the revision, the ranges, the sha256 of the
    # ranges concatenated, and every member's size and sha256.
    $Rev2607 = '19806879b16d3f2ccf28fb112b1bcd16a3c7923e'
    $Url2607 = "https://huggingface.co/nvidia/magpie_tts_multilingual_357m/resolve/$Rev2607/magpie_tts_multilingual_357m.nemo"
    $Ranges = @(
        @{ Start = [int64]0; End = [int64]20619263 },
        @{ Start = [int64]1459750400; End = [int64]1470207999 }
    )
    $RangesSha256 = '169144a64f6d775178696914e5aad8282db44820e73341d5f9a4e7865fac1f26'
    $Members = [ordered]@{
        '05adc40366e149b69319acd4b28a4919_pt_br_prondict-v1.0.dict' = @(3580284, '6492abc404db16bbad83dd9d7f9a60eb617699f0a3ff54b3be6184e14507b745')
        '339da71c54b046f98cbcf38ef6d4ff67_hindi_phoneme_merged_phoneme_dict.dict' = @(7319864, '3b50cc2c21cd8d34983d8492c91308e9db4a58c1194d6506fdbb2b97aead75fc')
        '41913ebaa70342058574da74293b7630_magpie_tts_multilingual_357m.nemo.speakers.json' = @(68, '36cdcf01ebc0afb506660ac71f2d0a211236374ad5c872d7d8d985a3f9f6ccaf')
        '61d57a4ccb064d1e8e3d9f871c571932_heteronyms-052722' = @(1606, 'b701909aedf753172eff223950f8859cd4b9b4c80199cf0a6e9ac4a307c8f8ec')
        '74b3e832fe914a569fcd42b51d06b2f1_ipa_dict_nv23.05.txt' = @(5419, '252e8eaf60dfd891520759913dc8534393eebfa8d57192365f76f9229f294e9e')
        '7dbc31751f224f2486090d59dc95b9f7_es_ES_nv230301.dict' = @(2230910, '94c9a25bb359f733cd863c887d0112e7ae19a744b0333ef64b6a88e576428669')
        'c5e4ec2af5a14ce294f4b9edcc936535_de_nv230119.heteronym' = @(44566, '771cc585a574fd35bd14f4ce6108edf1e0e512a8dbc810db7de4c1165ba9d0ef')
        'cf01ab5c48c84f3282ef7888263361e5_de_nv230119.dict' = @(4313907, '5c7bbf3346ebd6dc57769b5cc805124215cb1bcc3b8174c4254e9e928c5d6094')
        'dc7d60d6b15a4651b21c9ca2932b62c6_ipa_cmudict-0.7b_nv23.01.txt' = @(3093097, '7ea5c4c2c59cc780748b24d4ab60bd689779c4ac4ea7e27c0a8c3e20f300c0fa')
        'model_config.yaml' = @(8358, '9577bfa4769981849feccd6a54d38425d81fff41ac16c282f60be97faf46ebe4')
    }
    $ZipName = 'magpie-tts-357m-v2607-tokenizer.zip'
    $cache2607 = Join-Path $env:LOCALAPPDATA 'sigil-dev-cache\nemo-speech'
    New-Item -ItemType Directory -Force -Path $cache2607 | Out-Null
    $curl = Join-Path $env:SystemRoot 'System32\curl.exe'

    # The ranges (two requests, 3 s apart; a range already in the cache is not fetched again).
    $rangeFiles = @()
    $fetched = 0
    for ($r = 0; $r -lt $Ranges.Count; $r++) {
        $rg = $Ranges[$r]
        $want = $rg.End - $rg.Start + 1
        $file = Join-Path $cache2607 ("magpie_tts_multilingual_357m-$($Rev2607.Substring(0, 7)).nemo.range$r-$($rg.Start)-$($rg.End)")
        if (-not (Test-Path -LiteralPath $file)) {
            if ($fetched -gt 0) { Start-Sleep -Seconds 3 }
            Write-Step "Downloading bytes $($rg.Start)-$($rg.End) of the v2607 .nemo ($([math]::Round($want / 1MB, 1)) MB, one request)"
            $partial = "$file.partial"
            & $curl --fail --location --silent --show-error --range "$($rg.Start)-$($rg.End)" --max-filesize ($want + 1024) `
                --retry 2 --retry-delay 5 --output $partial $Url2607
            if ($LASTEXITCODE -ne 0) { throw "curl failed ($LASTEXITCODE) fetching $Url2607" }
            Move-Item -Force -LiteralPath $partial -Destination $file
            $fetched++
        }
        $have = (Get-Item -LiteralPath $file).Length
        if ($have -ne $want) { throw "$file is $have bytes, expected $want. Delete it and run again." }
        $rangeFiles += $file
    }

    # Provenance: the ranges concatenated must hash to the index's sha256.
    $hasher = [System.Security.Cryptography.IncrementalHash]::CreateHash([System.Security.Cryptography.HashAlgorithmName]::SHA256)
    try {
        foreach ($f in $rangeFiles) { $hasher.AppendData([System.IO.File]::ReadAllBytes($f)) }
        $got = ([System.BitConverter]::ToString($hasher.GetHashAndReset()) -replace '-', '').ToLowerInvariant()
    } finally { $hasher.Dispose() }
    if ($got -ne $RangesSha256) { throw "the fetched ranges do not match NVIDIA's index (sha256 $got, expected $RangesSha256)" }
    Write-Host "    ranges sha256 $got (matches NeMo-Speech.cpp's index)"

    # Walk the tar headers from the start of each range (both start on a header). The .nemo is the
    # v2602 tar with the members v2607 changed (two dictionaries and model_config.yaml) appended
    # after the weights: as with tar extraction, a later member of the same name wins.
    $found = @{}
    for ($r = 0; $r -lt $rangeFiles.Count; $r++) {
        $bytes = [System.IO.File]::ReadAllBytes($rangeFiles[$r])
        $pos = 0
        while ($pos + 512 -le $bytes.Length) {
            $empty = $true
            for ($i = $pos; $i -lt $pos + 512; $i++) { if ($bytes[$i] -ne 0) { $empty = $false; break } }
            if ($empty) { break }
            $end = $pos
            while ($end -lt $pos + 100 -and $bytes[$end] -ne 0) { $end++ }
            $name = ([System.Text.Encoding]::ASCII.GetString($bytes, $pos, $end - $pos)) -replace '^\./', ''
            $sizeText = ([System.Text.Encoding]::ASCII.GetString($bytes, $pos + 124, 12)).Trim([char]0, ' ')
            if ($sizeText -notmatch '^[0-7]*$') { Write-Host "    range ${r}: no tar header at offset $pos, stopping"; break }
            $size = if ($sizeText) { [Convert]::ToInt64($sizeText, 8) } else { 0 }
            $type = [char]$bytes[$pos + 156]
            $data = $pos + 512
            $isMember = $Members.Contains($name)
            if ($name -ne './@PaxHeader' -and $name -ne '@PaxHeader' -and $name) {
                Write-Host ("    range {0}: {1} ({2} bytes){3}" -f $r, $name, $size, $(if ($isMember) { '' } else { ' - not a tokenizer member' }))
            }
            if ($isMember -and ($type -eq '0' -or $type -eq [char]0)) {
                if ($data + $size -gt $bytes.Length) { throw "$name is cut off by its range (the archive layout changed?)" }
                $member = [byte[]]::new($size)
                [Array]::Copy($bytes, $data, $member, 0, $size)
                $found[$name] = $member
            }
            $next = $data + [int64]([Math]::Ceiling($size / 512.0) * 512)
            if ($next -gt $bytes.Length) { break }
            $pos = $next
        }
    }
    $missing = @($Members.Keys | Where-Object { -not $found.ContainsKey($_) })
    if ($missing.Count -gt 0) { throw "the fetched ranges lack: $($missing -join ', ')" }

    # Every member against the index, then the zip.
    if (-not $PSBoundParameters.ContainsKey('OutDir')) { $OutDir = Join-Path $cache2607 'magpie-tts-357m-v2607-tokenizer' }
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $zip = Join-Path $OutDir $ZipName
    if (Test-Path -LiteralPath $zip) { Remove-Item -Force -LiteralPath $zip }
    Add-Type -AssemblyName System.IO.Compression
    $stamp = [DateTimeOffset]::new(2026, 1, 1, 0, 0, 0, [TimeSpan]::Zero)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    $fs = [System.IO.File]::Open($zip, [System.IO.FileMode]::CreateNew)
    try {
        $archiveOut = [System.IO.Compression.ZipArchive]::new($fs, [System.IO.Compression.ZipArchiveMode]::Create)
        try {
            foreach ($name in ($Members.Keys | Sort-Object -CaseSensitive)) {
                $data = $found[$name]
                $hash = ([System.BitConverter]::ToString($sha256.ComputeHash($data)) -replace '-', '').ToLowerInvariant()
                $pin = $Members[$name]
                if ($data.Length -ne [int64]$pin[0] -or $hash -ne $pin[1]) {
                    throw "$name does not match NVIDIA's index ($($data.Length) bytes, $hash; expected $($pin[0]) / $($pin[1]))"
                }
                $entry = $archiveOut.CreateEntry($name, [System.IO.Compression.CompressionLevel]::Optimal)
                $entry.LastWriteTime = $stamp
                $es = $entry.Open()
                try { $es.Write($data, 0, $data.Length) } finally { $es.Dispose() }
            }
        } finally { $archiveOut.Dispose() }
    } finally { $fs.Dispose(); $sha256.Dispose() }

    $zipItem = Get-Item -LiteralPath $zip
    $zipHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant()
    "$zipHash  $ZipName  $($zipItem.Length)" | Set-Content -LiteralPath "$zip.sha256" -Encoding ascii
    if ($LocalAssets) {
        $dest = Join-Path $LocalAssets 'tts\magpie-tts-357m-v2607-tokenizer'
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        Copy-Item -Force -LiteralPath $zip -Destination $dest
        Write-Host "    Also copied for SIGIL_LOCAL_ASSETS=$LocalAssets"
    }
    Write-Step "The Magpie v2607 tokenizer ($($Members.Count) members, every hash checked) is in $zip"
    Write-Host ("    {0}: {1} bytes, sha256 {2}" -f $ZipName, $zipItem.Length, $zipHash)
    Write-Host "    Pin it in tts-ggml-experimental.json (size and sha256 of tts/magpie-tts-357m-v2607-tokenizer), then upload the zip to the assets-v1 release."
    return
}

$repo =(Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
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
