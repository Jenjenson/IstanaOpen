<#
.SYNOPSIS
Build the native Unreal Engine 5.5 editor and/or Windows game.
.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\Tools\build.ps1 -Target All
#>
[CmdletBinding()]
param(
    [string]$EngineRoot,
    [ValidateSet('All', 'Editor', 'Game')][string]$Target = 'All',
    [ValidateSet('Development', 'Shipping')][string]$Configuration = 'Development'
)
$ErrorActionPreference = 'Stop'
try {
    . (Join-Path $PSScriptRoot 'build-common.ps1')
    $projectRoot = Get-IstanaProjectRoot
    Assert-IstanaSource $projectRoot
    $engine = Resolve-IstanaEngine $EngineRoot
    $project = Join-Path $projectRoot 'IstanaOpen.uproject'
    $build = Join-Path $engine 'Engine\Build\BatchFiles\Build.bat'
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    if ($Target -in @('All', 'Editor')) {
        Invoke-IstanaNative $build @('IstanaOpenEditor', 'Win64', 'Development', "-Project=$project", '-WaitMutex', '-NoHotReloadFromIDE') (Join-Path $projectRoot "Saved\BuildLogs\editor-$stamp.log")
    }
    if ($Target -in @('All', 'Game')) {
        Invoke-IstanaNative $build @('IstanaOpen', 'Win64', $Configuration, "-Project=$project", '-WaitMutex', '-NoHotReloadFromIDE') (Join-Path $projectRoot "Saved\BuildLogs\game-$stamp.log")
    }
    Write-Host "Build complete: $Target ($Configuration)."
}
catch { Write-Error $_ -ErrorAction Continue; exit 1 }
