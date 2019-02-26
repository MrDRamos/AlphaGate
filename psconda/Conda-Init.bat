@ECHO off
ECHO %PSModulePath% | findstr %USERPROFILE% >NUL
IF %ERRORLEVEL% EQU 0 GOTO :ISPOWERSHELL

if "%CONDA_PREFIX%"=="" set CONDA_PREFIX=C:\Apps\MiniConda
call %CONDA_PREFIX%\Scripts\activate.bat %CONDA_PREFIX%
if not "%1"=="" call conda activate %1
python --version
GOTO :EOF

:ISPOWERSHELL
ECHO. >&2
ECHO ERROR: This batch file may not be run from a PowerShell prompt >&2
if EXIST %~dpn0.ps1 (ECHO try: %~dpn0.ps1 %* & ECHO or : %~n0 %*)>&2
ECHO. >&2
exit /b 1
