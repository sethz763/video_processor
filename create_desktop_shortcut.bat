@echo off
setlocal

REM Creates a Desktop shortcut for launching the GUI app from this repository.
set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"

set "PYTHONW=%REPO_DIR%\venv\Scripts\pythonw.exe"
set "PYTHON=%REPO_DIR%\venv\Scripts\python.exe"
set "APP=%REPO_DIR%\gui\app.py"
set "SHORTCUT_NAME=Video Processor GUI.lnk"

if not exist "%APP%" (
  echo ERROR: Could not find "%APP%"
  exit /b 1
)

set "TARGET="
if exist "%PYTHONW%" (
  set "TARGET=%PYTHONW%"
) else if exist "%PYTHON%" (
  set "TARGET=%PYTHON%"
) else (
  echo ERROR: Could not find venv Python at:
  echo   "%PYTHONW%"
  echo   or
  echo   "%PYTHON%"
  echo Create the venv first, then run this script again.
  exit /b 1
)

for /f "usebackq delims=" %%D in (`powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"`) do set "DESKTOP_DIR=%%D"
if not defined DESKTOP_DIR (
  echo ERROR: Could not resolve Desktop folder.
  exit /b 1
)

set "SHORTCUT_PATH=%DESKTOP_DIR%\%SHORTCUT_NAME%"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws=New-Object -ComObject WScript.Shell;" ^
  "$s=$ws.CreateShortcut('%SHORTCUT_PATH%');" ^
  "$s.TargetPath='%TARGET%';" ^
  "$s.Arguments='""%APP%""';" ^
  "$s.WorkingDirectory='%REPO_DIR%';" ^
  "$s.IconLocation='%TARGET%,0';" ^
  "$s.Description='Launch Video Processor GUI';" ^
  "$s.Save();"

if errorlevel 1 (
  echo ERROR: Failed to create desktop shortcut.
  exit /b 1
)

echo Created: "%SHORTCUT_PATH%"
echo Target : "%TARGET%" "%APP%"
exit /b 0
