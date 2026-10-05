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
"%CMAKE%" -S "%ROOT%\.gui-deps\VTK-9.5.2" -B "%ROOT%\.gui-deps\vtk-build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="%ROOT%\.gui-deps\vtk" -DCMAKE_PREFIX_PATH="%ROOT%\.gui-deps\Qt\6.8.3\msvc2022_64" -DVTK_QT_VERSION=6 -DBUILD_SHARED_LIBS=ON -DVTK_BUILD_TESTING=OFF -DVTK_BUILD_EXAMPLES=OFF -DVTK_BUILD_DOCUMENTATION=OFF -DVTK_WRAP_PYTHON=OFF -DVTK_ENABLE_WRAPPING=OFF -DVTK_GROUP_ENABLE_StandAlone=DONT_WANT -DVTK_GROUP_ENABLE_Rendering=DONT_WANT -DVTK_GROUP_ENABLE_Imaging=DONT_WANT -DVTK_GROUP_ENABLE_Qt=DONT_WANT -DVTK_MODULE_ENABLE_VTK_GUISupportQt=YES -DVTK_MODULE_ENABLE_VTK_RenderingOpenGL2=YES -DVTK_MODULE_ENABLE_VTK_InteractionStyle=YES -DVTK_MODULE_ENABLE_VTK_IOXML=YES -DVTK_MODULE_ENABLE_VTK_IOImage=YES -DVTK_MODULE_ENABLE_VTK_RenderingAnnotation=YES -DVTK_MODULE_ENABLE_VTK_FiltersGeneral=YES
if errorlevel 1 exit /b %errorlevel%
"%CMAKE%" --build "%ROOT%\.gui-deps\vtk-build" --parallel 8
if errorlevel 1 exit /b %errorlevel%
"%CMAKE%" --install "%ROOT%\.gui-deps\vtk-build"
exit /b %errorlevel%
