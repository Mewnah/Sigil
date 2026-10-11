# Windows SAPI (System.Speech) render timing, CPU and RAM for the casual English prompts. Run in Windows PowerShell 5.1.
param([string]$Prompts, [string]$Out, [int]$Limit = 12, [string]$Voice = 'Microsoft Zira Desktop')
Add-Type -AssemblyName System.Speech
$lines = Get-Content $Prompts -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.lang -eq 'en' } | Select-Object -First $Limit
$syn = New-Object System.Speech.Synthesis.SpeechSynthesizer
$syn.SelectVoice($Voice)
$proc = [Diagnostics.Process]::GetCurrentProcess()
$rows = @()
# warm-up (first call loads the engine)
$ms = New-Object System.IO.MemoryStream
$syn.SetOutputToWaveStream($ms); $syn.Speak('warm up'); 
foreach ($l in $lines) {
  $ms = New-Object System.IO.MemoryStream
  $syn.SetOutputToWaveStream($ms)
  $proc.Refresh(); $c0 = $proc.TotalProcessorTime.TotalMilliseconds
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $syn.Speak($l.text)
  $sw.Stop()
  $proc.Refresh(); $c1 = $proc.TotalProcessorTime.TotalMilliseconds
  $bytes = $ms.Length - 44
  $audio_s = $bytes / 2.0 / 22050.0   # the default format is 22.05 kHz 16-bit mono
  $rows += [pscustomobject]@{ id = $l.id; chars = $l.text.Length; render_ms = [Math]::Round($sw.Elapsed.TotalMilliseconds,1); cpu_ms = [Math]::Round($c1 - $c0,1); audio_s = [Math]::Round($audio_s,2); rtf = [Math]::Round($sw.Elapsed.TotalSeconds / [Math]::Max(0.01,$audio_s),3) }
}
$proc.Refresh()
$rows | Export-Csv -NoTypeInformation $Out
$r = $rows | Sort-Object render_ms
$n = $r.Count
[pscustomobject]@{ engine = "sapi:$Voice"; items = $n; render_ms_p50 = $r[[int][Math]::Ceiling($n*0.5)-1].render_ms; render_ms_p95 = $r[[int][Math]::Ceiling($n*0.95)-1].render_ms; rtf_mean = [Math]::Round(($rows | Measure-Object rtf -Average).Average,3); cpu_ms_per_audio_s = [Math]::Round((($rows | Measure-Object cpu_ms -Sum).Sum) / (($rows | Measure-Object audio_s -Sum).Sum),1); ws_mb = [Math]::Round($proc.PeakWorkingSet64/1MB,1) } | ConvertTo-Json -Compress
