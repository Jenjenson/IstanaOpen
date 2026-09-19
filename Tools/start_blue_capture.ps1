<#
.SYNOPSIS
Start a rendered, offscreen Istana scene for native 3D timelapse capture.
#>
[CmdletBinding()]
param([string]$EngineRoot, [ValidateRange(1024,65535)][int]$Port = 8766, [switch]$WarningApproachV2)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'build-common.ps1')
$projectRoot = Get-IstanaProjectRoot
$engine = Resolve-IstanaEngine $EngineRoot
$editor = Join-Path $engine 'Engine\Binaries\Win64\UnrealEditor.exe'
if (!(Test-Path -LiteralPath (Join-Path $projectRoot 'Binaries\Win64\UnrealEditor-IstanaOpen.dll'))) {
    throw 'Build the editor first: Tools\build.ps1 -Target Editor'
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Bridge port $Port is occupied; reuse or close the existing scene."
}
$project = Join-Path $projectRoot 'IstanaOpen.uproject'
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $projectRoot "Saved\Logs\BlueCapture-$stamp.log"
$arguments = @(('"'+$project+'"'),'/Game/Maps/Istana','-game','-IstanaBlueLive',
    "-IstanaBluePort=$Port",'-IstanaAllowCapture','-RenderOffscreen','-unattended',
    '-nosound','-nop4','-IstanaHigh','-IstanaPhoto','-windowed','-ResX=1920','-ResY=1080',
    '-ForceRes',('-abslog="'+$log+'"'))
if ($WarningApproachV2) { $arguments += '-IstanaWarningApproachV2' }
$captureProcess = Start-Process -FilePath $editor -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
$deadline = [DateTime]::UtcNow.AddSeconds(45)
while ([DateTime]::UtcNow -lt $deadline) {
    if ($captureProcess.HasExited) { throw "Capture scene exited; inspect $log" }
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        Write-Host "Rendered capture scene ready on 127.0.0.1:$Port (process $($captureProcess.Id))."
        Write-Host 'Run capture_warning_3d.py; the browser console must not own this bridge.'
        return
    }
    Start-Sleep -Milliseconds 250
}
throw "Scene is still starting (process $($captureProcess.Id)); inspect $log before retrying."
