<#
.SYNOPSIS
  P3B-C: the T6 dry run of compare v2 (an SVE ablation against SVE on a local corpus).

.DESCRIPTION
  Takes a compare corpus manifest (`sigil-compare-corpus/1`, e.g. the interim local corpus of
  `make-test-clips.py --local-corpus` or the public one of `fetch-public-corpus.py`), adds the dry
  run's rows and writes it next to the run (`<Out>/corpus.dryrun.json`, clip paths absolute):
  - `abl-40`: the SVE ablation at the spec's shape (anchored_noise off, nsf_phase0 off; partial
    decode and the F0 cache on), versus the spec's own SVE row `sigil`;
  - `sve-gui` and `abl-gui`: SVE and its ablation at the nearest SVE shape to RVC WebUI's GUI
    defaults: 200 ms engine and device block (the engine's longest; the GUI's is 250 ms), 50 ms
    crossfade, 2.0 s extra (the app's longest; the GUI's 2.5 s);
  - stand-in references: the voice's whole-utterance offline render with two noise seeds no stream
    uses (until upstream renders exist: rvc_offline_reference.py).
  Then it materializes compare-ablation.template.json and runs `Sigil.exe --headless bench` while
  holding the GPU timing lock (rule R2), records the background load (rustc processes, CPU %) in
  `<Out>/load.json`, and writes the T6 summary with scripts/vc-bench/compare_export.py.

  Ablations are never competitors: their margins say what SVE's time anchoring buys at matched
  shapes, and the paired SD they measure sizes the T6 corpus.

.EXAMPLE
  pwsh scripts/bench/specs/p3b/c/run_dryrun.ps1 -Exe <scratch>/p3b/c/exe-c1/sigil.exe -AppData <scratch>/p3b/d/appdata `
       -Corpus <scratch>/p3b/c/corpus-local/corpus.json -Ep cpu -Pitch fcpe -Out <scratch>/p3b/c/dry-cpu
#>
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$AppData,
    [Parameter(Mandatory = $true)][string]$Corpus,
    [string]$Slot = '',
    [ValidateSet('cpu', 'dml', 'cuda')][string]$Ep = 'cpu',
    [string]$Pitch = 'fcpe',
    [Parameter(Mandatory = $true)][string]$Out,
    [switch]$NoGui,
    [switch]$WriteAudio,
    # Analyzers that must agree for a consensus F0 truth frame (0 = all: the plan's intersection).
    [int]$ConsensusMinAgree = 0,
    [string]$LockTag = 'P3B-C dry run',
    [int]$LockWait = 240
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = (Resolve-Path (Join-Path $here '..\..\..\..\..')).Path
$lock = Join-Path $repo 'scripts\bench\gpu-lock.ps1'
if (-not $Slot) { $Slot = Join-Path $AppData 'voice-changer\models\default' }
foreach ($p in @($Exe, $Slot, $Corpus)) { if (-not (Test-Path $p)) { throw "not found: $p" } }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Out = (Resolve-Path $Out).Path

# The dry run's manifest: the corpus's clips (absolute paths) plus the dry run's rows.
$src = Get-Content -Raw $Corpus | ConvertFrom-Json
$base = Split-Path -Parent (Resolve-Path $Corpus).Path
$clips = foreach ($c in $src.clips) {
    $o = [ordered]@{ id = $c.id; input = (Join-Path $base $c.input) -replace '\\', '/' }
    if ($c.sha256) { $o.sha256 = $c.sha256 }
    if ($c.f0_truth) { $o.f0_truth = (Join-Path $base $c.f0_truth) -replace '\\', '/' }
    if ($c.references) { $o.references = @($c.references | ForEach-Object { (Join-Path $base $_) -replace '\\', '/' }) }
    $o
}
$off = [ordered]@{ anchored_noise = 'off'; nsf_phase0 = 'off'; partial_decode = 'on'; f0_cache = 'on' }
$points = @(
    [ordered]@{ name = 'abl-40'; role = 'ablation'; versus = 'sigil'; engine = [ordered]@{ levers = $off }
        label = "SVE ablation at the spec's shape: anchored noise and phase0 off (fresh noise, phase reset every call)" }
)
if (-not $NoGui) {
    $gui = [ordered]@{ block_frames = 9600; crossfade_ms = 50; extra_context_ms = 2000 }
    $points += [ordered]@{ name = 'sve-gui'; role = 'sigil'; engine = [ordered]@{ block_ms = 200 }; stream = $gui
        label = "SVE at the nearest shape to RVC WebUI's GUI defaults (200 ms block, 50 ms crossfade, 2.0 s extra)" }
    $points += [ordered]@{ name = 'abl-gui'; role = 'ablation'; versus = 'sve-gui'; engine = [ordered]@{ block_ms = 200; levers = $off }; stream = $gui
        label = 'SVE ablation at the GUI-like shape' }
}
$manifest = [ordered]@{
    schema = 'sigil-compare-corpus/1'; name = "dryrun-$($src.name)"
    description = "P3B-C dry run on $($src.name): SVE ablations against SVE (never competitors), stand-in references."
    transpose = 0; clips = @($clips); points = @(); min_clips = 20
    sigil = [ordered]@{ points = $points }
    stand_in_references = [ordered]@{ seeds = @(7001, 7002) }
}
if ($ConsensusMinAgree -gt 0) { $manifest.consensus_min_agree = $ConsensusMinAgree }
$mpath = Join-Path $Out 'corpus.dryrun.json'
$manifest | ConvertTo-Json -Depth 10 | Set-Content -Path $mpath -Encoding utf8NoBOM

$name = "dryrun-$Ep"
$spec = (Get-Content -Raw (Join-Path $here 'compare-ablation.template.json')).
    Replace('${NAME}', $name).
    Replace('${SLOT}', ($Slot -replace '\\', '/')).
    Replace('${EP}', $Ep).
    Replace('${PITCH}', $Pitch).
    Replace('${CORPUS}', ($mpath -replace '\\', '/')).
    Replace('${WRITE_AUDIO}', $(if ($WriteAudio) { 'true' } else { 'false' }))
$specPath = Join-Path $Out 'spec.json'
Set-Content -Path $specPath -Value $spec -Encoding utf8NoBOM

$env:SIGIL_APP_DATA_DIR = (Resolve-Path $AppData).Path
$env:SIGIL_TEST_MODELS_DIR = Join-Path $env:LOCALAPPDATA 'sigil-dev-cache'
& pwsh -NoProfile -File $lock -Acquire $LockTag -Wait $LockWait -PollSeconds 5 -ExpectedMinutes 60 -OwnerPid $PID
if ($LASTEXITCODE -ne 0) { throw 'could not take the GPU timing lock' }
try {
    $load = [ordered]@{
        start = (Get-Date).ToString('o')
        rustc = @(Get-Process rustc -ErrorAction SilentlyContinue).Count
        cpu_percent = [math]::Round((Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average, 1)
    }
    # Piped: Sigil.exe is a GUI-subsystem program, which PowerShell does not wait for otherwise.
    & $Exe --headless bench $specPath --out $Out 2>&1 | Out-File -FilePath (Join-Path $Out 'bench.log') -Encoding utf8
    $load.end = (Get-Date).ToString('o')
    $load.rustc_end = @(Get-Process rustc -ErrorAction SilentlyContinue).Count
    $load.cpu_percent_end = [math]::Round((Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average, 1)
    $load | ConvertTo-Json | Set-Content -Path (Join-Path $Out 'load.json') -Encoding utf8NoBOM
} finally {
    & pwsh -NoProfile -File $lock -Release $LockTag
}
$report = Join-Path $Out 'report.json'
if (Test-Path $report) {
    & python (Join-Path $repo 'scripts\vc-bench\compare_export.py') $report
} else {
    Write-Host "no report: see $(Join-Path $Out 'bench.log')"
}
