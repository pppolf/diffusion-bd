@echo off
set "BD_FULL_TRAIN_TARGET=clean"
call "%~dp0train_full_windows.bat"
exit /b %ERRORLEVEL%
