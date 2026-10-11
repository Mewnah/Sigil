# Samples CPU / RAM / GPU of a process tree once per interval and writes one CSV row per sample plus a summary.
# Usage: sample_tree.ps1 -RootPid <pid> -Seconds 120 -IntervalMs 1000 -Out file.csv [-Names sigil,msedgewebview2]
param(
  [Parameter(Mandatory)][int]$RootPid,
  [int]$Seconds = 0,
  [int]$IntervalMs = 1000,
  [Parameter(Mandatory)][string]$Out
)
$ErrorActionPreference = 'SilentlyContinue'
$cores = [Environment]::ProcessorCount
function Get-Tree([int]$root) {
  $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, Name
  $ids = New-Object System.Collections.Generic.HashSet[int]
  [void]$ids.Add($root)
  $changed = $true
  while ($changed) {
    $changed = $false
    foreach ($p in $all) { if ($ids.Contains([int]$p.ParentProcessId) -and -not $ids.Contains([int]$p.ProcessId)) { [void]$ids.Add([int]$p.ProcessId); $changed = $true } }
  }
  return $ids
}
$rows = @()
$prev = @{}
$t0 = Get-Date
$end = $t0.AddSeconds($Seconds)
$last = $t0
while ($true) {
  if ($Seconds -gt 0 -and (Get-Date) -ge $end) { break }
  if (-not (Get-Process -Id $RootPid -ErrorAction SilentlyContinue)) { break }
  Start-Sleep -Milliseconds $IntervalMs
  $now = Get-Date
  $dt = ($now - $last).TotalSeconds
  $last = $now
  $ids = Get-Tree $RootPid
  $cpuDelta = 0.0; $ws = 0.0; $priv = 0.0; $n = 0; $per = @()
  foreach ($id in $ids) {
    $p = Get-Process -Id $id
    if (-not $p) { continue }
    $n++
    $ws += $p.WorkingSet64; $priv += $p.PrivateMemorySize64
    $c = $p.TotalProcessorTime.TotalSeconds
    if ($prev.ContainsKey($id)) { $dd = [Math]::Max(0.0, $c - $prev[$id]); $cpuDelta += $dd; $per += ('{0}={1:N2}' -f $p.ProcessName, ($dd / $dt)) }
    $prev[$id] = $c
  }
  # GPU: dedicated memory and engine utilization summed over the tree
  $gm = 0.0; $gu = 0.0
  $cs = (Get-Counter -Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples
  foreach ($cx in $cs) {
    if ($cx.InstanceName -match '^pid_(\d+)_') {
      if ($ids.Contains([int]$Matches[1])) {
        if ($cx.Path -like '*gpu process memory*') { $gm += $cx.CookedValue } else { $gu += $cx.CookedValue }
      }
    }
  }
  $rows += [pscustomobject]@{
    t = [Math]::Round(($now - $t0).TotalSeconds, 1)
    procs = $n
    cpu_cores = [Math]::Round($cpuDelta / $dt, 4)
    ws_mb = [Math]::Round($ws / 1MB, 1)
    private_mb = [Math]::Round($priv / 1MB, 1)
    gpu_mem_mb = [Math]::Round($gm / 1MB, 1)
    gpu_util_pct = [Math]::Round($gu, 2)
    per_process = ($per -join ';')
  }
}
$rows | Export-Csv -NoTypeInformation -Path $Out
$cpu = $rows | Measure-Object cpu_cores -Average -Maximum
$ws = $rows | Measure-Object ws_mb -Maximum
$gm = $rows | Measure-Object gpu_mem_mb -Maximum
$gu = $rows | Measure-Object gpu_util_pct -Average -Maximum
[pscustomobject]@{ samples = $rows.Count; cpu_cores_mean = [Math]::Round($cpu.Average,4); cpu_cores_max = [Math]::Round($cpu.Maximum,3); ws_mb_max = $ws.Maximum; gpu_mem_mb_max = $gm.Maximum; gpu_util_mean = [Math]::Round($gu.Average,2); gpu_util_max = $gu.Maximum } | ConvertTo-Json -Compress
