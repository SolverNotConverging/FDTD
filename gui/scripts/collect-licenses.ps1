$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$taskQt = Join-Path $taskRoot '.gui-deps/qtbase-everywhere-src-6.8.3'
$taskVtk = Join-Path $taskRoot '.gui-deps/VTK-9.5.2'
$taskOutput = Join-Path $taskRoot 'gui/build/licenses'
if (-not (Test-Path -LiteralPath (Join-Path $taskQt 'LICENSES/LGPL-3.0-only.txt'))) {
    throw 'Matching Qt source notices are missing. Run gui/scripts/bootstrap.ps1.'
}
if (-not (Test-Path -LiteralPath (Join-Path $taskVtk 'Copyright.txt'))) {
    throw 'Matching VTK source notices are missing. Run gui/scripts/bootstrap.ps1.'
}
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
Copy-Item -LiteralPath (Join-Path $taskRoot 'gui/THIRD_PARTY.md') -Destination $taskOutput
# Preserve relative paths to avoid overwriting identically named vendor notices.
foreach ($taskPair in @(@($taskQt, 'Qt'), @($taskVtk, 'VTK'))) {
    $taskSource = (Resolve-Path -LiteralPath $taskPair[0]).Path
    $taskTarget = Join-Path $taskOutput $taskPair[1]
    Get-ChildItem -LiteralPath $taskSource -File -Recurse | Where-Object {
        $_.FullName -match '[\\/]LICENSES[\\/]' -or
        $_.Name -match '(?i)(license|copying|copyright|notice)' -or $_.Name -eq 'qt_attribution.json'
    } | ForEach-Object {
        $taskRelative = $_.FullName.Substring($taskSource.Length + 1)
        $taskDestination = Join-Path $taskTarget $taskRelative
        New-Item -ItemType Directory -Force -Path (Split-Path $taskDestination -Parent) | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $taskDestination
    }
}
$taskSbom = Join-Path $taskRoot '.gui-deps/Qt/6.8.3/msvc2022_64/sbom'
if (Test-Path -LiteralPath $taskSbom) {
    Copy-Item -LiteralPath $taskSbom -Destination (Join-Path $taskOutput 'Qt') -Recurse -Force
}
Write-Host "Dependency notices: $taskOutput"
