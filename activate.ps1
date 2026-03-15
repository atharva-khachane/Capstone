# Activate SL-RAG virtual environment
Write-Host "Activating SL-RAG virtual environment..." -ForegroundColor Green
& .\venv\Scripts\Activate.ps1
Write-Host ""
Write-Host "Virtual environment activated!" -ForegroundColor Green
Write-Host "Python location:" -ForegroundColor Cyan
Get-Command python | Select-Object -ExpandProperty Source
Write-Host ""
Write-Host "To deactivate, type: deactivate" -ForegroundColor Yellow
