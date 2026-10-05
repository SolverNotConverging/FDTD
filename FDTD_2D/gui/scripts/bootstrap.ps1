param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
Push-Location $taskRoot
try {
    $taskTools = Join-Path $taskRoot '.gui-tools'
    $taskDeps = Join-Path $taskRoot '.gui-deps'
    New-Item -ItemType Directory -Force -Path $taskDeps | Out-Null
    if (-not (Test-Path -LiteralPath (Join-Path $taskTools 'aqt')) -or
        -not (Test-Path -LiteralPath (Join-Path $taskTools 'cmake/data/bin/cmake.exe')) -or
        -not (Test-Path -LiteralPath (Join-Path $taskTools 'bin/ninja.exe'))) {
        & $Python -m pip install --target $taskTools aqtinstall==3.3.0 cmake==4.4.3 ninja==1.13.2
        if ($LASTEXITCODE) { throw 'Build-tool installation failed.' }
    }
    $taskOriginalPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $taskTools
        if (-not (Test-Path -LiteralPath (Join-Path $taskDeps 'Qt/6.8.3/msvc2022_64/lib/cmake/Qt6/Qt6Config.cmake'))) {
            & $Python -m aqt install-qt windows desktop 6.8.3 win64_msvc2022_64 --archives qtbase --outputdir (Join-Path $taskDeps 'Qt')
            if ($LASTEXITCODE) { throw 'Qt installation failed.' }
        }
    } finally { $env:PYTHONPATH = $taskOriginalPythonPath }
    # The binary SDK omits license texts; keep the matching official source too.
    if (-not (Test-Path -LiteralPath (Join-Path $taskDeps 'qtbase-everywhere-src-6.8.3/LICENSES/LGPL-3.0-only.txt'))) {
        $taskQtArchive = Join-Path $taskDeps 'qtbase-everywhere-src-6.8.3.tar.xz'
        Invoke-WebRequest -Uri 'https://download.qt.io/archive/qt/6.8/6.8.3/submodules/qtbase-everywhere-src-6.8.3.tar.xz' -OutFile $taskQtArchive
        & tar -xf $taskQtArchive -C $taskDeps
        if ($LASTEXITCODE) { throw 'Qt source extraction failed.' }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $taskDeps 'VTK-9.5.2/CMakeLists.txt'))) {
        $taskArchive = Join-Path $taskDeps 'VTK-9.5.2.tar.gz'
        Invoke-WebRequest -Uri 'https://www.vtk.org/files/release/9.5/VTK-9.5.2.tar.gz' -OutFile $taskArchive
        & tar -xzf $taskArchive -C $taskDeps
        if ($LASTEXITCODE) { throw 'VTK extraction failed.' }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $taskDeps 'vtk/lib/cmake/vtk-9.5/vtk-config.cmake'))) {
        & $env:ComSpec /c (Join-Path $PSScriptRoot 'build-vtk.cmd')
        if ($LASTEXITCODE) { throw 'VTK build failed.' }
    }
    & $env:ComSpec /c (Join-Path $PSScriptRoot 'build.cmd')
    if ($LASTEXITCODE) { throw 'FDTD Studio build failed.' }
    Write-Host "Built: $taskRoot/FDTD_2D/gui/build/fdtd-studio.exe"
} finally { Pop-Location }
