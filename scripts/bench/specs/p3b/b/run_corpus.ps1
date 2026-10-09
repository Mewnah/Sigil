<#
.SYNOPSIS
  P3B-B: the scorer-v2 fidelity corpus run (ADR 0009) over every local clip, on one EP.

.DESCRIPTION
  For each clip (default: the sherpa-onnx test clips and the transcribe samples of the dev cache,
  18 clips) the template fid-v2.template.json is materialized and run with
  `Sigil.exe --headless bench` while holding the GPU timing lock (rule R2): three offsets, ceilings
  A (long shape), B and whole, the floors, and with -Analyzer v2+ablations the oracle ablations.
  Then a rescore over the saved stream / whole-ceiling pairs of every offset measures the
  target-speaker similarity leave one clip out (each clip's offsets against the other clips'
  whole-utterance renditions; the clip's value is the median over its offsets), and
  score_v2/aggregate.py writes the corpus summary (means with bootstrap CIs, the ablation table and
  the T5 v2 rows).

  SIGIL_TEST_MODELS_DIR is set to the dev cache, so the benchmark finds bench/campplus-exact (and
  the legacy bench/campplus) in its vc-assets staging when the app data has no store entry yet.

.EXAMPLE
  pwsh scripts/bench/specs/p3b/b/run_corpus.ps1 -Exe <scratch>/p3b/b/exe-b/sigil.exe -AppData <scratch>/p3f/appdata -Ep cpu -Out <scratch>/p3b/b/corpus-cpu -Analyzer v2+ablations -Clips 'a.wav;b.wav'
#>
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$AppData,
    [string]$Slot = '',
    [ValidateSet('cpu', 'dml', 'cuda')][string]$Ep = 'cpu',
    [string]$Pitch = 'rmvpe',
    [string]$Preset = 'low_latency',
    [double]$ExtraMs = 64,
    [double]$LookaheadMs = 0,
    [ValidateSet('v2', 'v2+ablations')][string]$Analyzer = 'v2',
    [string[]]$Clips = @(),
    [int]$MaxClips = 0,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$LockTag = 'P3B-B',
    [int]$LockWait = 180,
    # Clips run under one hold of the GPU timing lock (1 = take it per clip). Other agents' timing
    # runs hold it 15-40 minutes at a time and the queue alternates owners, so a corpus taken one
    # clip per turn waits that long between 1-2 minute clips; a batch of 4 (about 8-20 minutes of
    # GPU or CPU clips) holds it for a comparable time.
    [int]$Batch = 4,
    [switch]$SkipRescore
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..\..')).Path
$lock = Join-Path $repo 'scripts\bench\gpu-lock.ps1'
$devCache = Join-Path $env:LOCALAPPDATA 'sigil-dev-cache'
if (-not $Slot) { $Slot = Join-Path $AppData 'voice-changer\models\default' }
if (-not (Test-Path $Exe)) { throw "no Sigil executable at $Exe" }
if (-not (Test-Path $Slot)) { throw "no voice slot at $Slot" }
# `pwsh -File` passes a list as one string: split it on ';' (or ',').
if ($Clips.Count -eq 1 -and $Clips[0] -match '[;,]') { $Clips = @($Clips[0] -split '[;,]' | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() }) }
if ($Clips.Count -eq 0) {
    $Clips = @(Get-ChildItem (Join-Path $devCache 'sherpa-onnx\clips') -Filter *.wav | Sort-Object Name | ForEach-Object FullName) +
             @(Get-ChildItem (Join-Path $devCache 'models\transcribe\samples') -Filter *.wav | Sort-Object Name | ForEach-Object FullName)
}
if ($MaxClips -gt 0) { $Clips = $Clips | Select-Object -First $MaxClips }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$env:SIGIL_APP_DATA_DIR = (Resolve-Path $AppData).Path
$env:SIGIL_TEST_MODELS_DIR = $devCache
# The spec contract's scoring.analyzer takes v1 / v2: the ablations are switched on for the bench
# process by SIGIL_BENCH_ABLATIONS (fidelity.rs), the spec says v2.
$specAnalyzer = 'v2'
if ($Analyzer -eq 'v2+ablations') { $env:SIGIL_BENCH_ABLATIONS = '1' } else { Remove-Item Env:SIGIL_BENCH_ABLATIONS -ErrorAction SilentlyContinue }
$template = Get-Content -Raw (Join-Path $here 'fid-v2.template.json')

# The lock file gpu-lock.ps1 uses (its -LockPath rule: $env:SIGIL_GPU_LOCK, else the pointer file, else
# its per-user default; `-PrintPath` prints the resolved one), for the in-process fallback release below.
$lockPath = if ($env:SIGIL_GPU_LOCK) { $env:SIGIL_GPU_LOCK } else {
    (& pwsh -NoProfile -File $lock -PrintPath | Select-Object -First 1)
}
$lockLog = Join-Path $Out 'gpu-lock.log'

function Remove-OwnLock {
    # Fallback when the release through gpu-lock.ps1 failed: delete the lock file only when it is
    # this script's (our tag and our pid), in process (no child process to start).
    try {
        $o = Get-Content -Raw -LiteralPath $lockPath -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
        if ($o.tag -eq $LockTag -and [int]$o.pid -eq $PID) {
            Remove-Item -LiteralPath $lockPath -Force -ErrorAction Stop
            Add-Content -Path $lockLog -Value ("{0:o} released in process ({1})" -f (Get-Date), $LockTag)
        }
    } catch { }
}

function Invoke-Locked([string]$what, [scriptblock]$body) {
    # A short poll: other agents' runs release and retake the lock back to back. The lock records
    # this script's pid, so a lock left behind by a killed run is stale 30 minutes later instead of
    # 4 hours; the release is in `finally` and names our tag (another owner's lock is never
    # released), so an error anywhere after the acquire still frees it. The lock script's output
    # goes to <Out>/gpu-lock.log, not to this script's console: a driver whose console pipe died
    # failed in the child process start-up, after its acquire and before its release (P3B-B).
    try {
        & pwsh -NoProfile -File $lock -Acquire $LockTag -Wait $LockWait -PollSeconds 2 -ExpectedMinutes 20 -OwnerPid $PID *>> $lockLog
        if ($LASTEXITCODE -ne 0) { throw "could not take the GPU timing lock for $what" }
        & $body
    } finally {
        try { & pwsh -NoProfile -File $lock -Release $LockTag *>> $lockLog } catch { }
        Remove-OwnLock
    }
}

$i = 0
$pending = @()
foreach ($clip in $Clips) {
    $i++
    $stem = [IO.Path]::GetFileNameWithoutExtension($clip) -replace '[^A-Za-z0-9_-]', '_'
    $name = "fid2-$Ep-$stem"
    $dir = Join-Path $Out $name
    if (Test-Path (Join-Path $dir 'report.json')) { Write-Host "[$i/$($Clips.Count)] ${name}: done before, kept"; continue }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $spec = $template.
        Replace('${NAME}', $name).
        Replace('${CLIP}', ($clip -replace '\\', '/')).
        Replace('${SLOT}', ($Slot -replace '\\', '/')).
        Replace('${CONTENT_ENCODER}', '').
        Replace('${EP}', $Ep).
        Replace('${PITCH}', $Pitch).
        Replace('${PRESET}', $Preset).
        Replace('${EXTRA_MS}', [string]$ExtraMs).
        Replace('${LOOKAHEAD_MS}', [string]$LookaheadMs).
        Replace('${ANALYZER}', $specAnalyzer).
        Replace('${REFERENCE_CLIPS_JSON}', '[]')
    $specPath = Join-Path $dir 'spec.json'
    Set-Content -Path $specPath -Value $spec -Encoding utf8NoBOM
    $pending += [pscustomobject]@{ Index = $i; Name = $name; Dir = $dir; Spec = $specPath }
}

function Invoke-Clip($c) {
    Write-Host "[$($c.Index)/$($Clips.Count)] $($c.Name)"
    # The background load of the run (P3B: other agents compile concurrently): rustc processes and
    # the total CPU load when the run starts, next to its report.
    $load = [ordered]@{
        start = (Get-Date).ToString('o')
        rustc = @(Get-Process rustc -ErrorAction SilentlyContinue).Count
        cpu_percent = [math]::Round((Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average, 1)
    }
    $load | ConvertTo-Json | Set-Content -Path (Join-Path $c.Dir 'load.json') -Encoding utf8NoBOM
    & $Exe --headless bench $c.Spec --out $c.Dir 2>&1 | Out-File -FilePath (Join-Path $c.Dir 'bench.log') -Encoding utf8
}

$size = [math]::Max(1, $Batch)
for ($k = 0; $k -lt $pending.Count; $k += $size) {
    $group = @($pending[$k..([math]::Min($k + $size, $pending.Count) - 1)])
    Invoke-Locked (($group | ForEach-Object Name) -join ', ') { foreach ($c in $group) { Invoke-Clip $c } }
}

if (-not $SkipRescore) {
    $rescore = Join-Path $Out 'rescore-whole.json'
    & python (Join-Path $repo 'scripts\bench\score_v2\make_rescore_spec.py') $Out --out $rescore --ceiling whole --name "rescore-$Ep-whole"
    if ($LASTEXITCODE -eq 0) {
        $rdir = Join-Path $Out 'rescore-whole'
        Invoke-Locked 'rescore' { & $Exe --headless bench $rescore --out $rdir 2>&1 | Out-File -FilePath (Join-Path $Out 'rescore-whole.log') -Encoding utf8 }
    }
}
& python (Join-Path $repo 'scripts\bench\score_v2\aggregate.py') $Out --json (Join-Path $Out 'summary.json') --md (Join-Path $Out 'summary.md')
