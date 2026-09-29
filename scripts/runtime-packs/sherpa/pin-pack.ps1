<#
.SYNOPSIS
  Print the catalog pin (size + sha256) of a sherpa-onnx Windows runtime archive.

.DESCRIPTION
  Maintainer tool for upgrading src-tauri/model-catalog/runtime-sherpa.json. Makes ONE request to
  the GitHub releases API for the tag and prints the `files[0]` object for the chosen asset, using
  GitHub's own sha256 asset digest. With -Verify it also downloads the archive once (into
  %LOCALAPPDATA%\sigil-dev-cache\sherpa-onnx, ~8 MB) and re-hashes it locally. Nothing is written
  into the repo: paste the printed object into the fragment and bump "version" by hand.

  The pack is the static-CRT (/MT) shared Release build, '-lib' variant: onnxruntime.dll and
  sherpa-onnx-c-api.dll with no executables, no VC++ redistributable dependency, and no
  execution-provider DLLs (sherpa stays CPU-only; Sigil's own ONNX Runtime owns the GPU).

.EXAMPLE
  pwsh scripts/runtime-packs/sherpa/pin-pack.ps1 -Version 1.13.8 -Verify
#>
[CmdletBinding()]
param(
  [string]$Version = "1.13.8",
  [string]$Flavor = "shared-MT-Release-lib",
  [switch]$Verify
)
$ErrorActionPreference = "Stop"
$tag = "v$Version"
$asset = "sherpa-onnx-$tag-win-x64-$Flavor.tar.bz2"
$api = "https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/$tag"
Write-Host "GET $api"
$release = Invoke-RestMethod -Uri $api -Headers @{ Accept = "application/vnd.github+json" }
$a = $release.assets | Where-Object { $_.name -eq $asset }
if (-not $a) { throw "release $tag has no asset $asset" }
if (-not ($a.digest -match '^sha256:([0-9a-f]{64})$')) { throw "GitHub reports no sha256 digest for $asset" }
$sha = $Matches[1]

if ($Verify) {
  $cache = Join-Path $env:LOCALAPPDATA "sigil-dev-cache\sherpa-onnx"
  New-Item -ItemType Directory -Force $cache | Out-Null
  $file = Join-Path $cache $asset
  if (-not (Test-Path $file)) {
    Write-Host "Downloading $($a.browser_download_url)"
    Invoke-WebRequest -Uri $a.browser_download_url -OutFile "$file.partial" -UseBasicParsing
    Move-Item -Force "$file.partial" $file
  }
  $local = (Get-FileHash -Algorithm SHA256 $file).Hash.ToLowerInvariant()
  if ($local -ne $sha) { throw "local sha256 $local differs from GitHub's $sha" }
  if ((Get-Item $file).Length -ne $a.size) { throw "local size differs from GitHub's $($a.size)" }
  Write-Host "Verified $file"
}

$entry = [ordered]@{
  path    = $asset
  size    = $a.size
  sha256  = $sha
  sources = @([ordered]@{ kind = "github-release"; url = $a.browser_download_url; revision = $tag })
  archive = [ordered]@{ format = "tar.bz2"; strip = 2; include = @("onnxruntime.dll", "sherpa-onnx-c-api.dll") }
}
$entry | ConvertTo-Json -Depth 5
