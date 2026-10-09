<#
.SYNOPSIS
  P3B-E measure-first (1): how many F0 calls per minute of speech carry more than 31 audio frames.

.DESCRIPTION
  Materializes onsets.template.json for every clip, runs `Sigil.exe --headless bench` on each under
  one hold of the GPU timing lock (rule R2), then prints per clip and in total: the converted calls,
  the onset calls (F0 calls over 31 frames: whole-window calls after a silence skip, or the first
  call), the clip's speech minutes (converted calls x 40 ms) and the onset calls per minute of
  speech, with the onset / steady call times. The background load (rustc processes, CPU %) at the
  start of the batch is printed too.

.EXAMPLE
  pwsh scripts/bench/specs/p3b/e/run_onsets.ps1 -Exe <exe> -AppData <appdata> -Ep dml -Out <dir>
#>
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$AppData,
    [string]$Slot = '',
    [ValidateSet('cpu', 'dml', 'cuda')][string]$Ep = 'dml',
    [string]$Pitch = 'rmvpe',
    [string[]]$Clips = @(),
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$LockTag = 'P3B-E onsets',
    [int]$LockWait = 180
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..\..')).Path
$lock = Join-Path $repo 'scripts\bench\gpu-lock.ps1'
$devCache = Join-Path $env:LOCALAPPDATA 'sigil-dev-cache'
if (-not $Slot) { $Slot = Join-Path $AppData 'voice-changer\models\default' }
if ($Clips.Count -eq 1 -and $Clips[0] -match '[;,]') { $Clips = @($Clips[0] -split '[;,]' | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() }) }
if ($Clips.Count -eq 0) {
    $Clips = @(Get-ChildItem (Join-Path $devCache 'sherpa-onnx\clips') -Filter *.wav | Sort-Object Name | ForEach-Object FullName) +
             @(Get-ChildItem (Join-Path $devCache 'models\transcribe\samples') -Filter *.wav -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object FullName)
}
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$template = Get-Content -Raw (Join-Path $here 'onsets.template.json')
$env:SIGIL_APP_DATA_DIR = (Resolve-Path -LiteralPath $AppData).Path

$rustc = @(Get-Process rustc -ErrorAction SilentlyContinue).Count
$cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
Write-Host "background load: rustc $rustc, CPU $cpu %"

& pwsh -NoProfile -File $lock -Acquire $LockTag -Wait $LockWait -ExpectedMinutes 10 -OwnerPid $PID
if ($LASTEXITCODE -ne 0) { throw 'the GPU timing lock was not acquired' }
try {
    foreach ($clip in $Clips) {
        $stem = [IO.Path]::GetFileNameWithoutExtension($clip)
        $dir = Join-Path $Out $stem
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $spec = $template.Replace('${NAME}', "onsets-$Ep-$stem").Replace('${CLIP}', ($clip -replace '\\', '/')).
            Replace('${SLOT}', ($Slot -replace '\\', '/')).Replace('${EP}', $Ep).Replace('${PITCH}', $Pitch)
        $specPath = Join-Path $dir 'spec.json'
        Set-Content -LiteralPath $specPath -Value $spec -Encoding utf8
        # Piped, so PowerShell waits for the (GUI-subsystem) executable to exit.
        & $Exe --headless bench $specPath --out $dir 2>&1 | Out-File -LiteralPath (Join-Path $dir 'bench.log') -Encoding utf8
        Write-Host "$stem exit $LASTEXITCODE"
    }
}
finally {
    & pwsh -NoProfile -File $lock -Release $LockTag
}

$tot = @{ calls = 0; onset = 0 }
foreach ($clip in $Clips) {
    $stem = [IO.Path]::GetFileNameWithoutExtension($clip)
    $rep = Join-Path (Join-Path $Out $stem) 'report.json'
    if (-not (Test-Path $rep)) { Write-Host "$stem : no report"; continue }
    $r = Get-Content -Raw $rep | ConvertFrom-Json
    $run = $r.runs[0]
    $m = $run.metrics
    $att = $run.details.attribution
    $conv = [double]$att.converted_calls
    $onset = [double]$m.onset_calls
    $silent = [double]$run.details.silent_ticks
    $min = $conv * 0.04 / 60
    $per = if ($min -gt 0) { $onset / $min } else { 0 }
    $tot.calls += $conv; $tot.onset += $onset
    '{0,-16} converted {1,5} silent {2,4} onset {3,3} ({4,6:N1}/min of speech)  call onset p50 {5,6:N2} steady p50 {6,6:N2} ms  f0 onset p50 {7,6:N2} steady p50 {8,6:N2} ms' -f $stem, $conv, $silent, $onset, $per,
        $m.call_ms_onset_p50, $m.call_ms_steady_p50, $m.f0_ms_onset_p50, $m.f0_ms_steady_p50 | Write-Host
}
$min = $tot.calls * 0.04 / 60
'TOTAL converted {0} onset {1} = {2:N1} onset calls per minute of speech' -f $tot.calls, $tot.onset, ($(if ($min) { $tot.onset / $min } else { 0 })) | Write-Host
