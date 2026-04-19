<# Run the FastAPI backend (use in its own terminal). #>
$ProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Get-Location }
Set-Location -LiteralPath $ProjectRoot
& (Join-Path $ProjectRoot "venv\Scripts\Activate.ps1")
Write-Host "Backend: http://127.0.0.1:8000  (docs: /docs)" -ForegroundColor Cyan
uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000
