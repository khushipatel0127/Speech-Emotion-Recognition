# SpeechSense — Frontend Launcher
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Starting SpeechSense React + Vite Frontend  " -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$FrontendDir = Join-Path $ScriptDir "frontend"
Set-Location $FrontendDir

Write-Host "Running on http://localhost:5173" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the frontend server.`n" -ForegroundColor Gray

npm run dev
