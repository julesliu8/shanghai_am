param([Parameter(Mandatory=$true)][string]$RunDirectory)
$Host.UI.RawUI.WindowTitle = 'Shanghai phone CTC - Cloud RTX 4090'
Write-Host 'Shanghai phone CTC training | remote RTX 4090 24 GB' -ForegroundColor Cyan
Write-Host "Local mirror: $RunDirectory"
Write-Host 'Updates every 5 seconds. Closing this view does not stop remote training.'
Write-Host 'step = this run / target; PER = phone error rate (lower is better).'
Write-Host 'dev = latest evaluation; best_group_PER = best dev group average in this run.'
Write-Host ''
$progressFile = Join-Path $RunDirectory 'live_progress.log'
while (-not (Test-Path -LiteralPath $progressFile)) { Start-Sleep -Seconds 1 }
Get-Content -LiteralPath $progressFile -Tail 25 -Wait -Encoding utf8
