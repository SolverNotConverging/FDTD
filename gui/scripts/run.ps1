param([string]$Project = '', [string]$Python = '')
$taskRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$taskExe = Join-Path $taskRoot 'gui/build/fdtd-studio.exe'
if (-not (Test-Path -LiteralPath $taskExe)) { throw 'Build first with gui/scripts/build.cmd.' }
if ($Python) { $env:FDTD_PYTHON = $Python }
$taskArgs = @()
if ($Project) { $taskArgs += '--project'; $taskArgs += (Resolve-Path -LiteralPath $Project).Path }
& $taskExe @taskArgs
