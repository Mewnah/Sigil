<#
.SYNOPSIS
  Upload the staged assets to the `assets-v1` GitHub release. Prints the plan; uploads only with -Execute.

.DESCRIPTION
  Sigil-built packs, exports and templates are catalog entries that start `unpublished` and name their
  files as assets of https://github.com/Mewnah/Sigil-assets/releases/tag/assets-v1. This script is the one
  place that puts them there. It does nothing by itself: you run it, and without -Execute it only
  prints what it would do.

  Plan (always, offline): scripts/models/assets-v1.mjs compares the staging folder with the catalog:
  the files that would be uploaded (present, named by a catalog entry, size equal to the catalog pin),
  the files that are blocked (no pin yet, wrong size, wrong hash), the catalog assets that are not
  staged, and files in the folder that no entry names (ignored; SHA256SUMS.txt and MANIFEST.json are
  never uploaded). -Verify also hashes every candidate against its pin first (reads all of them).

  -Execute (needs the GitHub CLI, installed and logged in by you with `gh auth login`; this script
  never reads, stores or asks for a token):
    1. creates the release `assets-v1` when it does not exist (a pre-release with a short note);
    2. lists the assets already on it;
    3. uploads the planned files one at a time, with a pause between uploads, skipping a file that is
       already there with the same size. A file there with another size stops that file unless you
       pass -Replace (then it is uploaded with --clobber).
  It refuses to start while any staged file is blocked: fix the pin first
  (node scripts/models/pin-staged.mjs --staging <dir> --write).

  Afterwards flip the entries:
    node scripts/models/publish-entries.mjs --ids all-staged --staging <dir> --write

.PARAMETER Staging
  The staging folder (files named exactly like the release assets).
.PARAMETER Execute
  Really upload. Without it nothing leaves this PC.
.PARAMETER Verify
  Hash every staged candidate against the catalog pin before planning (slow for GBs; sequential).
.PARAMETER Replace
  Upload over an asset that exists with a different size (gh release upload --clobber).
.PARAMETER DelaySeconds
  Pause between uploads (default 3, minimum 2): the dev network blocks request bursts.
.PARAMETER CatalogDir
  Catalog fragments to compare with (default src-tauri/model-catalog; for tests).

.EXAMPLE
  pwsh scripts/release/publish-assets.ps1 -Staging D:\sigil-assets-v1-staging
  pwsh scripts/release/publish-assets.ps1 -Staging D:\sigil-assets-v1-staging -Verify -Execute
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Staging,
    [switch]$Execute,
    [switch]$Verify,
    [switch]$Replace,
    [ValidateRange(2, 600)][int]$DelaySeconds = 3,
    [string]$CatalogDir
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Staging = (Resolve-Path -LiteralPath $Staging).Path

function Fail([string]$message) {
    Write-Host "ERROR: $message" -ForegroundColor Red
    exit 1
}

function Format-Size([long]$n) {
    if ($n -ge 1GB) { return ('{0:N2} GiB' -f ($n / 1GB)) }
    if ($n -ge 1MB) { return ('{0:N1} MiB' -f ($n / 1MB)) }
    if ($n -ge 1KB) { return ('{0:N1} KiB' -f ($n / 1KB)) }
    return "$n B"
}

if (-not (Get-Command node -ErrorAction SilentlyContinue)) { Fail 'node is not on PATH (it reads the catalog).' }

# 1. The plan: node does the catalog work, this script only prints it and (with -Execute) uploads it.
$planArgs = @('scripts/models/assets-v1.mjs', 'plan', '--staging', $Staging, '--json')
if ($Verify) { $planArgs += '--hash' }
if ($CatalogDir) { $planArgs += @('--catalog-dir', $CatalogDir) }
Push-Location $Root
try {
    $raw = & node @planArgs
    $planExit = $LASTEXITCODE
} finally {
    Pop-Location
}
if (-not $raw) { Fail "assets-v1.mjs plan printed nothing (exit $planExit)." }
$plan = ($raw -join "`n") | ConvertFrom-Json
$release = "$($plan.release.owner)/$($plan.release.repo)"
$tag = $plan.release.tag

Write-Host "staging : $($plan.staging)"
Write-Host "release : $release  tag $tag"
Write-Host ("to upload: {0} file(s), {1}" -f $plan.totals.files, (Format-Size $plan.totals.bytes))
foreach ($f in $plan.upload) {
    Write-Host ('  {0,-66} {1,11}  {2}' -f $f.asset, (Format-Size $f.size), ($f.entries -join ', '))
}
if ($plan.blocked.Count -gt 0) {
    Write-Host "blocked ($($plan.blocked.Count)): not uploaded until the pin and the file agree" -ForegroundColor Yellow
    foreach ($b in $plan.blocked) { Write-Host "  $($b.asset): $($b.state) $($b.detail)" -ForegroundColor Yellow }
}
if ($plan.notStaged.Count -gt 0) {
    # One line per entry: a partly staged entry cannot be published, so it stays import-only.
    $byEntry = @{}
    foreach ($n in $plan.notStaged) { foreach ($e in $n.entries) { $byEntry[$e] = 1 + [int]$byEntry[$e] } }
    Write-Host "named by the catalog, not staged: $($plan.notStaged.Count) file(s) of $($byEntry.Count) entries stay import-only" -ForegroundColor DarkGray
    foreach ($e in ($byEntry.Keys | Sort-Object)) { Write-Host "  $e ($($byEntry[$e]) file(s))" -ForegroundColor DarkGray }
}
if ($plan.optionalMirrors.Count -gt 0) {
    Write-Host "optional mirrors not staged ($($plan.optionalMirrors.Count)): the files also have an upstream host" -ForegroundColor DarkGray
    foreach ($n in $plan.optionalMirrors) { Write-Host "  $($n.asset)  ($($n.entries -join ', '))" -ForegroundColor DarkGray }
}
if ($plan.extras.Count -gt 0) { Write-Host "in the folder but named by no entry (ignored): $($plan.extras -join ', ')" -ForegroundColor DarkGray }
foreach ($p in $plan.problems) { Write-Host "$($p.level.ToUpper()): $($p.message)" -ForegroundColor Red }

if (-not $Execute) {
    Write-Host ''
    Write-Host 'Dry run: nothing was uploaded.' -ForegroundColor Green
    Write-Host "To upload: pwsh scripts/release/publish-assets.ps1 -Staging `"$($plan.staging)`" -Execute   (needs the GitHub CLI, logged in with: gh auth login)"
    exit 0
}

# 2. Preconditions for a real upload.
if ($planExit -ne 0 -or $plan.problems.Count -gt 0) { Fail 'The plan has errors (above). Nothing was uploaded.' }
if ($plan.blocked.Count -gt 0) { Fail 'Some staged files disagree with the catalog (above). Fix the pins first: node scripts/models/pin-staged.mjs --staging <dir> --write. Nothing was uploaded.' }
if ($plan.upload.Count -eq 0) { Fail 'Nothing to upload.' }
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) { Fail 'The GitHub CLI (gh) is not installed or not on PATH. Install it from https://cli.github.com/ and run: gh auth login' }
& gh auth status *> $null
if ($LASTEXITCODE -ne 0) { Fail 'gh is not logged in. Run: gh auth login (this script never handles tokens).' }

# 3. The release: create it once, as a pre-release.
$view = & gh release view $tag --repo $release --json 'tagName,assets' 2>&1
if ($LASTEXITCODE -ne 0) {
    if (($view -join ' ') -match 'release not found') {
        Write-Host "creating release $tag (pre-release)"
        $notes = 'Model and runtime assets that Sigil downloads (the files named by src-tauri/model-catalog). Every file is pinned by sha256 in the catalog; do not edit or re-upload by hand. Uploaded by scripts/release/publish-assets.ps1.'
        & gh release create $tag --repo $release --prerelease --title 'Sigil assets (v1)' --notes $notes
        if ($LASTEXITCODE -ne 0) { Fail 'gh release create failed.' }
        $existing = @{}
    } else {
        Fail ("gh release view failed: " + ($view -join ' '))
    }
} else {
    $existing = @{}
    foreach ($a in (($view -join "`n") | ConvertFrom-Json).assets) { $existing[$a.name] = [long]$a.size }
}

# 4. Upload one file at a time.
$sent = 0; $skipped = 0; $failed = @()
$first = $true
foreach ($f in $plan.upload) {
    if ($existing.ContainsKey($f.asset)) {
        if ($existing[$f.asset] -eq [long]$f.size) {
            Write-Host "skip   $($f.asset) (already there, same size)"
            $skipped++
            continue
        }
        if (-not $Replace) {
            Write-Host "STOP   $($f.asset): the release has $($existing[$f.asset]) bytes, the staged file $($f.size). Use -Replace to overwrite it." -ForegroundColor Red
            $failed += $f.asset
            continue
        }
    }
    if (-not $first) { Start-Sleep -Seconds $DelaySeconds }
    $first = $false
    Write-Host ("upload {0} ({1})" -f $f.asset, (Format-Size $f.size))
    $ghArgs = @('release', 'upload', $tag, $f.file, '--repo', $release)
    if ($existing.ContainsKey($f.asset)) { $ghArgs += '--clobber' }
    & gh @ghArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED $($f.asset) (gh exit $LASTEXITCODE)" -ForegroundColor Red
        $failed += $f.asset
    } else {
        $sent++
    }
}

Write-Host ''
Write-Host "uploaded $sent, skipped $skipped, failed $($failed.Count)"
if ($failed.Count -gt 0) { Write-Host "failed: $($failed -join ', ')" -ForegroundColor Red; exit 1 }
Write-Host 'Next: prove the uploads and flip the entries:'
Write-Host "  node scripts/models/publish-entries.mjs --ids all-staged --staging `"$($plan.staging)`" --write"
exit 0
