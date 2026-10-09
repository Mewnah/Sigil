<#
.SYNOPSIS
  P3B-E dev-rig tests of RMVPE under the GPU timing lock: the per-call plan cost (32 vs 64 frames,
  t32, fp16; ABBA rounds), the t32 parity on the CPU and the stream-path fp16 gate.

.DESCRIPTION
  Runs an already built test binary (never builds under the lock, rule R2): build it first with
      $env:CARGO_TARGET_DIR = "$env:LOCALAPPDATA\sigil-target\p3b-dev"
      cargo test --manifest-path src-tauri/Cargo.toml --no-run
  and pass the `Executable unittests src\main.rs (...)` path as -TestExe. The test binary is copied
  with the ONNX Runtime provider DLLs of its profile folder (ORT's CUDA EP loads them from next to
  the executable) into <Out>/bin, and -AppData (an app data folder with the runtime/nvidia-cuda13
  pack) becomes SIGIL_APP_DATA_DIR, so the CUDA rung is available. The background load
  (rustc processes, CPU %) is printed before the lock is taken; each test's output goes to
  <Out>/<test>.log. `rmvpe_fp16_gate` writes scripts/vc-templates/reports/pitch_rmvpe_fp16_gate.json.

.EXAMPLE
  pwsh scripts/bench/specs/p3b/e/run_rig_tests.ps1 -TestExe <sigil-xxxx.exe> -Out <dir>
  pwsh scripts/bench/specs/p3b/e/run_rig_tests.ps1 -TestExe <exe> -Out <dir> -Tests rmvpe_plan_cost -Rounds 10
#>
param(
    [Parameter(Mandatory = $true)][string]$TestExe,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$AppData = '',
    [string[]]$Tests = @('real_t32_matches_full_on_cpu', 'rmvpe_plan_cost', 'rmvpe_fp16_gate'),
    [int]$Rounds = 6,
    [string]$LockTag = 'P3B-E rmvpe rig tests',
    [int]$LockWait = 180
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..\..')).Path
$lock = Join-Path $repo 'scripts\bench\gpu-lock.ps1'
if ($Tests.Count -eq 1 -and $Tests[0] -match '[;,]') { $Tests = @($Tests[0] -split '[;,]' | ForEach-Object { $_.Trim() } | Where-Object { $_ }) }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
# The test binary next to the provider DLLs of its profile folder (target/<profile>/*.dll).
$bin = Join-Path $Out 'bin'
New-Item -ItemType Directory -Force -Path $bin | Out-Null
$profileDir = Split-Path -Parent (Split-Path -Parent (Resolve-Path -LiteralPath $TestExe).Path)
Copy-Item -LiteralPath $TestExe -Destination $bin -Force
Get-ChildItem -LiteralPath $profileDir -Filter *.dll | Copy-Item -Destination $bin -Force
$TestExe = Join-Path $bin (Split-Path -Leaf $TestExe)
if ($AppData) { $env:SIGIL_APP_DATA_DIR = (Resolve-Path -LiteralPath $AppData).Path }
$env:SIGIL_TEST_MODELS_DIR = Join-Path $env:LOCALAPPDATA 'sigil-dev-cache'
$env:SIGIL_RMVPE_COST_ROUNDS = "$Rounds"

$rustc = @(Get-Process rustc -ErrorAction SilentlyContinue).Count
$cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
Write-Host "background load: rustc $rustc, CPU $cpu %"

& pwsh -NoProfile -File $lock -Acquire $LockTag -Wait $LockWait -ExpectedMinutes 30 -OwnerPid $PID
if ($LASTEXITCODE -ne 0) { throw 'the GPU timing lock was not acquired' }
try {
    foreach ($t in $Tests) {
        $log = Join-Path $Out "$t.log"
        "background load: rustc $(@(Get-Process rustc -ErrorAction SilentlyContinue).Count), CPU $((Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average) %" |
            Out-File -LiteralPath $log -Encoding utf8
        # The test binary runs from src-tauri (CARGO_MANIFEST_DIR paths in the reports).
        Push-Location (Join-Path $repo 'src-tauri')
        try {
            & $TestExe $t --ignored --nocapture --test-threads 1 2>&1 | Out-File -LiteralPath $log -Append -Encoding utf8
        }
        finally { Pop-Location }
        Write-Host "$t exit $LASTEXITCODE"
    }
}
finally {
    & pwsh -NoProfile -File $lock -Release $LockTag
}
