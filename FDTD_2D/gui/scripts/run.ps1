param([string]$Project = '', [string]$Python = '')
$taskRoot = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
$taskExe = Join-Path $taskRoot 'FDTD_2D/gui/build/fdtd-studio.exe'
if (-not (Test-Path -LiteralPath $taskExe)) { throw 'Build first with FDTD_2D/gui/scripts/build.cmd.' }
if ($Python) { $env:FDTD_PYTHON = $Python }
$taskArgs = @()
if ($Project) { $taskArgs += '--project'; $taskArgs += (Resolve-Path -LiteralPath $Project).Path }
& $taskExe @taskArgs
