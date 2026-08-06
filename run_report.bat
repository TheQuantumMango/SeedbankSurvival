@echo off
REM Double-click launcher: runs the CLI against the usual local files and
REM opens the resulting report.html automatically. Edit INVENTORY/ACCESSIONS/
REM GENUS below if you're working with different data.
setlocal
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
set "INVENTORY=docs\RawGRINAstragalusExport.xlsx"
set "ACCESSIONS=docs\RawGRINAstragalusExportAccessions.xlsx"
set "GENUS=Astragalus"
set "OUT_DIR=docs\output"

if not exist "%PYTHON%" (
    echo Could not find %PYTHON% -- has the project's virtual environment been set up?
    pause
    exit /b 1
)
if not exist "%INVENTORY%" (
    echo Could not find %INVENTORY%
    echo Put the raw GRIN inventory export there, or edit run_report.bat to point at your file.
    pause
    exit /b 1
)

if exist "%ACCESSIONS%" (
    "%PYTHON%" -m seedbank_survival --inventory "%INVENTORY%" --accessions "%ACCESSIONS%" -g %GENUS% --out-dir "%OUT_DIR%"
) else (
    "%PYTHON%" -m seedbank_survival --inventory "%INVENTORY%" -g %GENUS% --out-dir "%OUT_DIR%"
)

if errorlevel 1 (
    echo.
    echo seedbank-survival failed -- see the error above.
    pause
    exit /b 1
)

start "" "%OUT_DIR%\report.html"
endlocal
