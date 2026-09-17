<#
.SYNOPSIS
Start the local recorded-replay and experimental Unreal telemetry console.
#>
[CmdletBinding()]
param([ValidateRange(1024, 65535)][int]$Port = 9048,
      [ValidateRange(1024, 65535)][int]$BridgePort = 8765)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot 'RL\BlueTeam\.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Run Tools\setup_blue_python.ps1 first to create the Python environment.'
}
& $python (Join-Path $projectRoot 'RL\BlueTeam\Python\simulation_console.py') --port $Port --bridge-port $BridgePort
if ($LASTEXITCODE -ne 0) { throw "Simulation console exited with code $LASTEXITCODE" }
