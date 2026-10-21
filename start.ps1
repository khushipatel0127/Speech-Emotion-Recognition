# SpeechSense — Full Application Launcher
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         SPEECHSENSE — SPEECH EMOTION RECOGNITION         " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment not found at $VenvPython" -ForegroundColor Red
    exit 1
}

Write-Host "`n[1/3] Starting FastAPI Backend on http://localhost:8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ScriptDir'; .\.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload"

Write-Host "[2/3] Starting Vite Frontend on http://localhost:5173..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ScriptDir\frontend'; npm run dev"

Write-Host "[3/3] Opening SpeechSense in default browser..." -ForegroundColor Green
Start-Sleep -Seconds 3
Start-Process "http://localhost:5173"

Write-Host "`n SpeechSense is now running!" -ForegroundColor Cyan
Write-Host "Frontend : http://localhost:5173" -ForegroundColor Yellow
Write-Host "Backend  : http://localhost:8000" -ForegroundColor Yellow
Write-Host "API Docs : http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan
