@REM SPDX-License-Identifier: Apache-2.0
@REM Copyright 2026 Sungwoo Kang
@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run-Latest-DevPortable.ps1" %*
exit /b %ERRORLEVEL%
