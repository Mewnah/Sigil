<#
.SYNOPSIS
  P3B-H: the scorer-v2 fidelity corpus of one stream geometry (crossfade, lookahead, SOLA search and
  tail rounding), for the sweep behind the Low-latency lookahead table (T5-P5), the crossfade gate and
  the tail-alignment A/B.

.DESCRIPTION
  A copy of P3B-B's run_corpus.ps1 with the crossfade as a variable and the process switches of this
  workstream (SIGIL_VC_SOLA_SEARCH_MS, SIGIL_VC_SOLA_SLACK) set for the harness runs. Each clip is run
  with `Sigil.exe --headless bench` under the GPU timing lock (batches of -Batch clips per hold), three
  offsets per clip; then P3B-B's rescore (leave-one-clip-out target set) and aggregate.py give the
  corpus summary (T5 v2 rows, envelope MCD over its floor, clicks, fed F0, V/UV).

.EXAMPLE
  pwsh scripts/bench/specs/p3b/h/run_fid.ps1 -Exe <exe> -AppData <appdata> -Ep cuda -Out <dir> -Crossfade 10 -Lookahead 0 -MaxClips 8
#>
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$AppData,
    [string]$Slot = '',
    [ValidateSet('cpu', 'dml', 'cuda')][string]$Ep = 'cuda',
    [string]$Pitch = 'rmvpe',
    [string]$Preset = 'low_latency',
    [double]$ExtraMs = 64,
    [double]$Crossfade = 20,
    [double]$LookaheadMs = 0,
    [string]$SolaSearchMs = '',
    [string]$SolaSlack = '',
    [string[]]$Clips = @(),
    [int]$MaxClips = 8,
    [Parameter(Mandatory = $true)][string]$Out,
    [string]$LockTag = 'P3B-H',
    [int]$LockWait = 240,
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
if ($Clips.Count -eq 1 -and $Clips[0] -match '[;,]') { $Clips = @($Clips[0] -split '[;,]' | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() }) }
if ($Clips.Count -eq 0) {
    $Clips = @(Get-ChildItem (Join-Path $devCache 'sherpa-onnx\clips') -Filter *.wav | Sort-Object Name | ForEach-Object FullName) +
             @(Get-ChildItem (Join-Path $devCache 'models\transcribe\samples') -Filter *.wav | Sort-Object Name | ForEach-Object FullName)
}
if ($MaxClips -gt 0) { $Clips = $Clips | Select-Object -First $MaxClips }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$env:SIGIL_APP_DATA_DIR = (Resolve-Path $AppData).Path
$env:SIGIL_TEST_MODELS_DIR = $devCache
Remove-Item Env:SIGIL_BENCH_ABLATIONS -ErrorAction SilentlyContinue
if ($SolaSearchMs -ne '') { $env:SIGIL_VC_SOLA_SEARCH_MS = $SolaSearchMs } else { Remove-Item Env:SIGIL_VC_SOLA_SEARCH_MS -ErrorAction SilentlyContinue }
if ($SolaSlack -ne '') { $env:SIGIL_VC_SOLA_SLACK = $SolaSlack } else { Remove-Item Env:SIGIL_VC_SOLA_SLACK -ErrorAction SilentlyContinue }
$template = Get-Content -Raw (Join-Path $here 'fid-h.template.json')
$lockLog = Join-Path $Out 'gpu-lock.log'
function Invoke-Locked([string]$what, [scriptblock]$body) {
    try {
        & pwsh -NoProfile -File $lock -Acquire $LockTag -Wait $LockWait -PollSeconds 2 -ExpectedMinutes 20 -OwnerPid $PID *>> $lockLog
        if ($LASTEXITCODE -ne 0) { throw "could not take the GPU timing lock for $what" }
        & $body
    } finally {
        try { & pwsh -NoProfile -File $lock -Release $LockTag *>> $lockLog } catch { }
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
        Replace('${CLIP}', $clip.Replace('\', '/')).
        Replace('${SLOT}', $Slot.Replace('\', '/')).
        Replace('${CONTENT_ENCODER}', '').
        Replace('${EP}', $Ep).
        Replace('${PITCH}', $Pitch).
        Replace('${PRESET}', $Preset).
        Replace('${EXTRA_MS}', [string]$ExtraMs).
        Replace('${CROSSFADE_MS}', [string]$Crossfade).
        Replace('${LOOKAHEAD_MS}', [string]$LookaheadMs).
        Replace('${ANALYZER}', 'v2').
        Replace('${REFERENCE_CLIPS_JSON}', '[]')
    $specPath = Join-Path $dir 'spec.json'
    Set-Content -Path $specPath -Value $spec -Encoding utf8NoBOM
    $pending += [pscustomobject]@{ Index = $i; Name = $name; Dir = $dir; Spec = $specPath }
}
function Invoke-Clip($c) {
    Write-Host "[$($c.Index)/$($Clips.Count)] $($c.Name)"
    [ordered]@{ start = (Get-Date).ToString('o'); rustc = @(Get-Process rustc -ErrorAction SilentlyContinue).Count
        cpu_percent = [math]::Round((Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average, 1) } |
        ConvertTo-Json | Set-Content -Path (Join-Path $c.Dir 'load.json') -Encoding utf8NoBOM
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
