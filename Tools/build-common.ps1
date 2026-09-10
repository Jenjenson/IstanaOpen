Set-StrictMode -Version Latest

function Get-IstanaProjectRoot {
    return [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
}

function Resolve-IstanaEngine {
    param([string]$EngineRoot)
    $candidates = [System.Collections.Generic.List[string]]::new()
    if ($EngineRoot) { $candidates.Add($EngineRoot) }
    elseif ($env:UE_ENGINE_ROOT) { $candidates.Add($env:UE_ENGINE_ROOT) }
    else {
        $launcher = Join-Path $env:ProgramData 'Epic\UnrealEngineLauncher\LauncherInstalled.dat'
        if (Test-Path -LiteralPath $launcher) {
            $installed = Get-Content -LiteralPath $launcher -Raw | ConvertFrom-Json
            foreach ($entry in $installed.InstallationList) {
                if ($entry.AppName -eq 'UE_5.5') { $candidates.Add($entry.InstallLocation) }
            }
        }
        $registry = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\EpicGames\Unreal Engine\5.5' -ErrorAction SilentlyContinue
        if ($registry -and $registry.PSObject.Properties['InstalledDirectory']) {
            $candidates.Add($registry.InstalledDirectory)
        }
        $candidates.Add((Join-Path $env:ProgramFiles 'Epic Games\UE_5.5'))
        foreach ($drive in Get-PSDrive -PSProvider FileSystem) {
            $candidates.Add((Join-Path $drive.Root 'Epic Games\UE_5.5'))
        }
    }
    foreach ($candidate in $candidates | Select-Object -Unique) {
        $root = [IO.Path]::GetFullPath($candidate)
        if ((Split-Path $root -Leaf) -eq 'Engine') { $root = Split-Path $root -Parent }
        $versionPath = Join-Path $root 'Engine\Build\Build.version'
        $buildPath = Join-Path $root 'Engine\Build\BatchFiles\Build.bat'
        if ((Test-Path -LiteralPath $versionPath) -and (Test-Path -LiteralPath $buildPath)) {
            $version = Get-Content -LiteralPath $versionPath -Raw | ConvertFrom-Json
            if ($version.MajorVersion -eq 5 -and $version.MinorVersion -eq 5) { return $root }
        }
    }
    throw 'Unreal Engine 5.5 was not found. Install it, or supply -EngineRoot "D:\Epic Games\UE_5.5" (UE_ENGINE_ROOT also works).'
}

function Assert-IstanaSource {
    param([string]$ProjectRoot)
    $required = @(
        'IstanaOpen.uproject', 'Config\DefaultEngine.ini',
        'Source\IstanaOpen.Target.cs', 'Source\IstanaOpenEditor.Target.cs',
        'Source\IstanaOpen\IstanaOpen.Build.cs', 'Source\IstanaOpen\IstanaOpen.cpp',
        'Source\IstanaOpen\IstanaGameMode.h', 'Source\IstanaOpen\IstanaGameMode.cpp'
    )
    foreach ($relative in $required) {
        $path = Join-Path $ProjectRoot $relative
        if (!(Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required project file is missing: $path" }
        if ((Get-Item -LiteralPath $path).Length -eq 0) { throw "Required project file is empty: $path" }
    }
}

function Invoke-IstanaNative {
    param([string]$Command, [string[]]$Arguments, [string]$LogPath)
    New-Item -ItemType Directory -Force -Path (Split-Path $LogPath -Parent) | Out-Null
    Write-Host "Running $Command"
    $previousErrorPreference = $ErrorActionPreference
    try {
        # Native stderr can be informational; the process exit code is authoritative.
        $ErrorActionPreference = 'Continue'
        & $Command @Arguments 2>&1 | Tee-Object -FilePath $LogPath | ForEach-Object { Write-Host $_ }
        $nativeExitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previousErrorPreference }
    if ($nativeExitCode -ne 0) { throw "Command failed with exit code $nativeExitCode. Log: $LogPath" }
}

function Assert-IstanaChildPath {
    param([string]$Path, [string]$Parent)
    $fullParent = [IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
    $fullPath = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    if (!$fullPath.StartsWith($fullParent + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Output path must stay within its chosen directory: $fullPath (parent $fullParent)"
    }
    return $fullPath
}
