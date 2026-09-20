<#
.SYNOPSIS
Create the isolated Blue Team Python environment using Python supplied with UE 5.5.
#>
[CmdletBinding()]
param([string]$EngineRoot, [string]$PythonExe)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'build-common.ps1')
$projectRoot = Get-IstanaProjectRoot
if (!$PythonExe) {
    $engine = Resolve-IstanaEngine $EngineRoot
    $PythonExe = Join-Path $engine 'Engine\Binaries\ThirdParty\Python3\Win64\python.exe'
}
if (!(Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python executable not found: $PythonExe" }
& $PythonExe -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
if ($LASTEXITCODE -ne 0) { throw 'Python version check failed.' }
$environment = Join-Path $projectRoot 'RL\BlueTeam\.venv'
$python = Join-Path $environment 'Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) {
    & $PythonExe -m venv $environment
    if ($LASTEXITCODE -ne 0) { throw 'Creating the Python environment failed.' }
}
& $python -m pip install -e ((Join-Path $projectRoot 'RL\BlueTeam\Python') + '[test,media]')
if ($LASTEXITCODE -ne 0) { throw 'Installing the Blue Team dependencies failed.' }
Write-Host "Blue Team Python is ready: $python"
