<# Launch the separate synthetic long-approach benchmark without rendering. #>
[CmdletBinding()]
param([string]$EngineRoot, [ValidateRange(1024,65535)][int]$Port = 8767)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'build-common.ps1')
$projectRoot = Get-IstanaProjectRoot
$engine = Resolve-IstanaEngine $EngineRoot
if (!(Test-Path -LiteralPath (Join-Path $projectRoot 'Binaries\Win64\UnrealEditor-IstanaOpen.dll'))) {
    throw 'Build the editor first: Tools\build.ps1 -Target Editor'
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is occupied."
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$trainingLog = Join-Path $projectRoot "Saved\Logs\WarningTraining-$stamp.log"
$argsList = @(('"' + (Join-Path $projectRoot 'IstanaOpen.uproject') + '"'),
    '/Game/Maps/Istana', '-game', '-IstanaBlueLive', '-IstanaWarningApproachV2',
    "-IstanaBluePort=$Port", '-nullrhi', '-unattended', '-nosound', '-nop4',
    ('-abslog="' + $trainingLog + '"'))
$trainingProcess = Start-Process -FilePath (Join-Path $engine 'Engine\Binaries\Win64\UnrealEditor.exe') -ArgumentList $argsList -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
$deadline = [DateTime]::UtcNow.AddSeconds(45)
while ([DateTime]::UtcNow -lt $deadline) {
    if ($trainingProcess.HasExited) { throw "Training scene exited; inspect $trainingLog" }
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        Write-Host "Warning approach v2 ready: process $($trainingProcess.Id); port $Port; log $trainingLog"
        return
    }
    Start-Sleep -Milliseconds 250
}
throw "Scene is still starting (process $($trainingProcess.Id)); inspect $trainingLog before retrying."
