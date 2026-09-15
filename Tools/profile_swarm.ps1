param(
    [string]$EngineRoot = 'C:\Program Files\Epic Games\UE_5.5',
    [ValidateRange(0,21)][int]$Swarms = 5,
    [ValidateRange(60,10000)][int]$Frames = 360,
    [switch]$NoDebug,
    [switch]$NoVisuals
)
$ErrorActionPreference = 'Stop'
$swarmProject = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../IstanaOpen.uproject')).Path
$swarmExecutable = Join-Path $EngineRoot 'Engine/Binaries/Win64/UnrealEditor-Cmd.exe'
$swarmLog = Join-Path (Split-Path $swarmProject) 'Saved/Logs/SwarmGraphicsProfile.log'
$swarmArgs = @($swarmProject, '/Game/Maps/Istana', '-game', '-unattended', '-nosound', '-nop4',
    '-RenderOffscreen', "-abslog=$swarmLog", '-IstanaSwarmProfile', "-SwarmProfileCount=$Swarms",
    "-ExecCmds=r.GPUCsvStatsEnabled 1,csvprofile FRAMES=$Frames", '-csvStatCounts', '-csvCompression=0', '-ExitAfterCsvProfiling')
if ($NoDebug) { $swarmArgs += '-SwarmProfileNoDebug' }
if ($NoVisuals) { $swarmArgs += '-SwarmProfileNoVisuals' }
& $swarmExecutable @swarmArgs
if ($LASTEXITCODE -ne 0) { throw "Swarm profile exited with $LASTEXITCODE" }
Write-Host 'CSV timing output is under Saved/Profiling/CSV. Map assets were not saved.'
