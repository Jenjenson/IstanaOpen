<#
.SYNOPSIS
Open the compiled UE 5.5 Istana scene with the live Red/Blue bridge.
#>
[CmdletBinding()]
param([string]$EngineRoot, [ValidateRange(1024, 65535)][int]$Port = 8765, [switch]$WarningApproachV2)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'build-common.ps1')
$projectRoot = Get-IstanaProjectRoot
$engine = Resolve-IstanaEngine $EngineRoot
$editor = Join-Path $engine 'Engine\Binaries\Win64\UnrealEditor.exe'
$module = Join-Path $projectRoot 'Binaries\Win64\UnrealEditor-IstanaOpen.dll'
if (!(Test-Path -LiteralPath $module -PathType Leaf)) {
    throw 'Build the editor first: powershell -ExecutionPolicy Bypass -File Tools\build.ps1 -Target Editor'
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Bridge port $Port is already in use. Reuse or close the existing live scene before launching another."
}
$project = Join-Path $projectRoot 'IstanaOpen.uproject'
$log = Join-Path $projectRoot 'Saved\Logs\BlueLive.log'
$arguments = @(('"' + $project + '"'), '/Game/Maps/Istana', '-game', '-IstanaBlueLive',
    "-IstanaBluePort=$Port", '-windowed', '-ResX=1280', '-ResY=720', '-IstanaMedium',
    '-IstanaCaptureView=3', ('-abslog="' + $log + '"'))
if ($WarningApproachV2) { $arguments += '-IstanaWarningApproachV2' }
$viewerProcess = Start-Process -FilePath $editor -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Normal -PassThru
Write-Host "Started live Istana scene (process $($viewerProcess.Id)); bridge 127.0.0.1:$Port."
Write-Host 'Open http://127.0.0.1:9048/ and choose Live Unreal, Connect, then Plan new episode.'
Write-Host 'If the console is not running, launch Tools\start_simulation_console.ps1 separately.'
