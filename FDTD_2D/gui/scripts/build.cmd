@echo off
setlocal
cd /d "%~dp0\..\..\.."
set "ROOT=%CD%"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%I"
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
set "CMAKE=%ROOT%\.gui-tools\cmake\data\bin\cmake.exe"
set "PATH=%ROOT%\.gui-tools\bin;%PATH%"
"%CMAKE%" -S "%ROOT%\FDTD_2D\gui" -B "%ROOT%\FDTD_2D\gui\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="%ROOT%\.gui-deps\Qt\6.8.3\msvc2022_64;%ROOT%\.gui-deps\vtk" -DBUILD_TESTING=ON
if errorlevel 1 exit /b %errorlevel%
"%CMAKE%" --build "%ROOT%\FDTD_2D\gui\build" --parallel 8
if errorlevel 1 exit /b %errorlevel%
"%ROOT%\.gui-deps\Qt\6.8.3\msvc2022_64\bin\windeployqt.exe" --release --no-translations --no-compiler-runtime "%ROOT%\FDTD_2D\gui\build\fdtd-studio.exe"
if errorlevel 1 exit /b %errorlevel%
copy /y "%ROOT%\.gui-deps\vtk\bin\*.dll" "%ROOT%\FDTD_2D\gui\build\" >nul
if errorlevel 1 exit /b %errorlevel%
copy /y "%ROOT%\.gui-deps\Qt\6.8.3\msvc2022_64\bin\Qt6Test.dll" "%ROOT%\FDTD_2D\gui\build\" >nul
if errorlevel 1 exit /b %errorlevel%
copy /y "%ROOT%\.gui-deps\Qt\6.8.3\msvc2022_64\plugins\platforms\qoffscreen.dll" "%ROOT%\FDTD_2D\gui\build\platforms\" >nul
if errorlevel 1 exit /b %errorlevel%
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\FDTD_2D\gui\scripts\collect-licenses.ps1"
if errorlevel 1 exit /b %errorlevel%
exit /b 0
