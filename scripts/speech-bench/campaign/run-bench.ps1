#Requires -Version 7.0
<#
.SYNOPSIS
  Runs one speech-bench spec the way the P7-A campaign did: BelowNormal priority, the P7 model store, optional CPU-only
  (CUDA hidden) and CPU-affinity slice (a smaller CPU), optional process-tree sampling (CPU / RAM / GPU of Sigil and its
  helper processes, which the runner's own figures do not include).

.EXAMPLE
  pwsh scripts/speech-bench/campaign/run-bench.ps1 -Spec scripts/speech-bench/campaign/specs/speed_utt.json -Out bench-results/speech/speed-utt -Cpu -Affinity FFF
#>
param(
  [Parameter(Mandatory)][string]$Spec,
  [Parameter(Mandatory)][string]$Out,
  # 'release' / 'debug' under -TargetDir, or the full path of a sigil.exe
  [string]$Exe = 'release',
  [string]$TargetDir = $(if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $PSScriptRoot '../../../src-tauri/target' }),
  [switch]$Cpu,            # hide CUDA from the process (CPU-only rows)
  [string]$Affinity = '',  # hex mask: FFF = the first 12 logical processors (a 6-core / 12-thread slice), FF = 4 cores
  [string]$Sample = '',    # CSV path: sample the process tree every 2 s while the run lasts
  [string]$AppData = "$env:LOCALAPPDATA/sigil-dev-cache/appdata-p7"
)
$ErrorActionPreference = 'Stop'
$exePath = if (Test-Path $Exe -PathType Leaf) { $Exe } else { Join-Path $TargetDir "$Exe/sigil.exe" }
$env:SIGIL_APP_DATA_DIR = $AppData
$env:SIGIL_LOCAL_ASSETS = "$env:LOCALAPPDATA/sigil-dev-cache/p7/local-assets"
if (-not $env:SIGIL_BENCH_RIG) { $env:SIGIL_BENCH_RIG = 'H2' }
$ptr = Join-Path $env:LOCALAPPDATA 'sigil-bench/gpu_lock_path.txt'
if (-not $env:SIGIL_GPU_LOCK -and (Test-Path $ptr)) { $env:SIGIL_GPU_LOCK = ((Get-Content $ptr -TotalCount 1).Trim()) }
if ($Cpu) { $env:CUDA_VISIBLE_DEVICES = '-1' } else { Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
New-Item -ItemType Directory -Force (Split-Path -Parent $Out) | Out-Null
$sw = [Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath $exePath -ArgumentList @('--headless', 'speech-bench', $Spec, '--out', $Out) -NoNewWindow -PassThru `
  -RedirectStandardOutput "$Out.stdout.txt" -RedirectStandardError "$Out.stderr.txt"
try { $p.PriorityClass = 'BelowNormal' } catch {}
if ($Affinity) { try { $p.ProcessorAffinity = [IntPtr][Convert]::ToInt64($Affinity, 16) } catch { Write-Host "affinity failed: $_" } }
$job = $null
if ($Sample) {
  $job = Start-Job -ScriptBlock { param($f, $id, $o) & $f -RootPid $id -IntervalMs 2000 -Out $o } -ArgumentList (Join-Path $PSScriptRoot 'sample-tree.ps1'), $p.Id, $Sample
}
$p.WaitForExit()
if ($job) { Wait-Job $job -Timeout 60 | Out-Null; (Receive-Job $job) | Out-File "$Sample.summary.json" -Encoding utf8; Remove-Job $job -Force }
"exit=$($p.ExitCode) elapsed=$($sw.Elapsed)"
exit $p.ExitCode
