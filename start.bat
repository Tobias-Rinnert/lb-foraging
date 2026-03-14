@echo off
title LB Foraging App

:: Start backend with auto-reload (watches Python files)
start "LBF Backend" cmd /c "cd /d %~dp0 && .venv\Scripts\python -m uvicorn web.backend.server:app --reload --reload-dir tr_lbf_addon --reload-dir web\backend --port 8000"

:: Wait for backend to be ready
timeout /t 2 /nobreak >nul

:: Start frontend dev server (Vite HMR auto-reloads on file changes)
start "LBF Frontend" cmd /c "cd /d %~dp0\web\frontend && npm run dev"

:: Wait for frontend to be ready
timeout /t 3 /nobreak >nul

:: Open browser
start http://localhost:5173

echo.
echo LB Foraging app is running at http://localhost:5173
echo Close this window or press Ctrl+C to stop.
echo.
pause
