@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\unlock_rtx3060_admin.ps1" %*
exit /b %ERRORLEVEL%
