<# Run the Vite dev server (use in its own terminal). #>
$ProjectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Get-Location }
$fe = Join-Path $ProjectRoot "app\frontend"
Set-Location -LiteralPath $fe
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
npm run dev
