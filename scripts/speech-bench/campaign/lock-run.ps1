#Requires -Version 7.0
<#
.SYNOPSIS
  run-bench.ps1 under the dev rig's GPU timing lock (scripts/bench/gpu-lock.ps1 -Run). The lock owner is "P7-A <Name>", which
  is what the speed specs' gpu_lock.tag ("P7-A") accepts. Waits up to four hours in the lock's fair queue.
#>
param(
  [Parameter(Mandatory)][string]$Name,
  [Parameter(Mandatory)][string]$Spec,
  [Parameter(Mandatory)][string]$Out,
  [switch]$Cpu,
  [string]$Affinity = '',
  [string]$Sample = '',
  [double]$ExpectedMinutes = 30
)
$cmd = "& '$(Join-Path $PSScriptRoot 'run-bench.ps1')' -Spec '$Spec' -Out '$Out'"
if ($Cpu) { $cmd += ' -Cpu' }
if ($Affinity) { $cmd += " -Affinity $Affinity" }
if ($Sample) { $cmd += " -Sample '$Sample'" }
pwsh -NoProfile -File (Join-Path $PSScriptRoot '../../bench/gpu-lock.ps1') -Run "P7-A $Name" -Wait 240 -ExpectedMinutes $ExpectedMinutes -Command $cmd
exit $LASTEXITCODE
