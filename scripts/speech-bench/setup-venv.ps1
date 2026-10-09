<#
.SYNOPSIS
  Create the P7 speech-bench Python environment (light packages only) outside the repo.

.DESCRIPTION
  Creates a virtual environment (default %LOCALAPPDATA%\sigil-dev-cache\p7\venv) and installs the
  pinned packages of requirements.txt ONE AT A TIME, without dependency resolution (--no-deps: the
  file pins the full transitive set), pausing between packages so the network never sees a burst
  (the dev PC's router blocks bursts of connections). Packages already installed at the pinned
  version are skipped, so re-running is cheap and offline.

  No torch, no transformers: MetricX-24 has its own optional recipe (README.md, "MetricX-24").

.EXAMPLE
  pwsh scripts/speech-bench/setup-venv.ps1
  & "$env:LOCALAPPDATA\sigil-dev-cache\p7\venv\Scripts\python.exe" -m unittest discover -s scripts/speech-bench -p "test_*.py"

.EXAMPLE
  # Offline, from a folder of wheels downloaded earlier (pip download -r requirements.txt --no-deps -d wheels)
  pwsh scripts/speech-bench/setup-venv.ps1 -WheelDir D:\wheels
#>
[CmdletBinding()]
param(
    # Interpreter that creates the venv (Python 3.11 or 3.12).
    [string]$Python = '',
    [string]$VenvDir = (Join-Path $env:LOCALAPPDATA 'sigil-dev-cache\p7\venv'),
    [string]$Requirements = (Join-Path $PSScriptRoot 'requirements.txt'),
    # Seconds between two package installs (network gentleness).
    [double]$PauseSeconds = 2,
    # Install from this folder only (no index access).
    [string]$WheelDir = ''
)

$ErrorActionPreference = 'Stop'
try { (Get-Process -Id $PID).PriorityClass = 'BelowNormal' } catch { }

# A command line such as 'py -3.12' as { Exe; Lead } (the leading arguments, always an array).
function Split-Command([string]$commandLine) {
    $parts = @($commandLine.Split(' ', [StringSplitOptions]::RemoveEmptyEntries))
    $rest = @()
    if ($parts.Count -gt 1) { $rest = @($parts[1..($parts.Count - 1)]) }
    return [pscustomobject]@{ Exe = $parts[0]; Lead = $rest }
}

function Find-Python {
    if ($Python) { return $Python }
    foreach ($candidate in @('py -3.12', 'py -3.11', 'python')) {
        $cmd = Split-Command $candidate
        $exe = $cmd.Exe; $lead = $cmd.Lead
        try {
            $v = & $exe @lead -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>$null
            if ($LASTEXITCODE -eq 0 -and $v -match '^3\.(11|12)$') { return $candidate }
        } catch { }
    }
    throw 'No Python 3.11 / 3.12 found: pass -Python <path to python.exe>'
}

$venvPython = Join-Path $VenvDir 'Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    $creator = Find-Python
    Write-Host "Creating $VenvDir with $creator"
    New-Item -ItemType Directory -Force (Split-Path $VenvDir) | Out-Null
    $cmd = Split-Command $creator
    $exe = $cmd.Exe; $lead = $cmd.Lead
    & $exe @lead -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed ($LASTEXITCODE)" }
}

# Pinned name==version lines, in file order (comments and blanks skipped).
$pins = Get-Content $Requirements | ForEach-Object { ($_ -replace '#.*$', '').Trim() } | Where-Object { $_ -match '^[A-Za-z0-9_.\-]+==[A-Za-z0-9_.\-+]+$' }

$installedJson = & $venvPython -c 'import json, importlib.metadata as m; print(json.dumps({d.metadata["Name"].lower().replace("_", "-"): d.version for d in m.distributions()}))'
$installed = $installedJson | ConvertFrom-Json -AsHashtable

$first = $true
foreach ($pin in $pins) {
    $name, $version = $pin -split '==', 2
    $key = $name.ToLower().Replace('_', '-')
    if ($installed.ContainsKey($key) -and $installed[$key] -eq $version) {
        Write-Host "ok      $pin"
        continue
    }
    if (-not $first -and -not $WheelDir) { Start-Sleep -Seconds $PauseSeconds }
    $first = $false
    Write-Host "install $pin"
    $pipArgs = @('-m', 'pip', 'install', '--no-deps', '--disable-pip-version-check', '--no-input', $pin)
    if ($WheelDir) { $pipArgs += @('--no-index', '--find-links', $WheelDir) }
    & $venvPython @pipArgs
    if ($LASTEXITCODE -ne 0) { throw "pip install $pin failed ($LASTEXITCODE)" }
}

& $venvPython -m pip check --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { Write-Warning 'pip check reported a problem (see above)' }

$info = & $venvPython -c 'import json, sys, importlib.metadata as m; print(json.dumps({"python": sys.version.split()[0], "executable": sys.executable, "packages": {d.metadata["Name"]: d.version for d in sorted(m.distributions(), key=lambda d: d.metadata["Name"].lower())}}, indent=2))'
$info | Set-Content -Encoding utf8 (Join-Path $VenvDir 'sigil-venv-info.json')
Write-Host "Ready: $venvPython"
