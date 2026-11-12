@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 goto use_python
py -3 run.py
goto finished
:use_python
where python >nul 2>nul
if errorlevel 1 goto missing
python run.py
goto finished
:missing
echo Please install 64-bit Python 3.12 from https://www.python.org/downloads/
echo Enable Add Python to PATH, then run this file again.
pause
exit /b 1
:finished
if errorlevel 1 pause
