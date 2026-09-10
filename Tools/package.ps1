<#
.SYNOPSIS
Cook a portable Windows release, including prerequisites, launchers and notices.
.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\Tools\package.ps1
.EXAMPLE
powershell -ExecutionPolicy Bypass -File .\Tools\package.ps1 -RebuildScene -EngineRoot "D:\Epic Games\UE_5.5"
#>
[CmdletBinding()]
param(
    [string]$EngineRoot,
    [string]$OutputRoot,
    [ValidateSet('Development', 'Shipping')][string]$Configuration = 'Development',
    [switch]$RebuildScene
)
$ErrorActionPreference = 'Stop'
try {
    . (Join-Path $PSScriptRoot 'build-common.ps1')
    $projectRoot = Get-IstanaProjectRoot
    Assert-IstanaSource $projectRoot
    $engine = Resolve-IstanaEngine $EngineRoot
    $project = Join-Path $projectRoot 'IstanaOpen.uproject'
    $python = Join-Path $engine 'Engine\Binaries\ThirdParty\Python3\Win64\python.exe'
    if (!(Test-Path -LiteralPath $python)) { throw "Unreal's bundled Python was not found: $python" }
    $verifier = Join-Path $PSScriptRoot 'verify_release.py'
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    Invoke-IstanaNative $python @($verifier, '--source-root', $projectRoot, '--source-only') (Join-Path $projectRoot "Saved\BuildLogs\source-check-$stamp.log")

    if ($RebuildScene) {
        $build = Join-Path $engine 'Engine\Build\BatchFiles\Build.bat'
        Invoke-IstanaNative $build @('IstanaOpenEditor', 'Win64', 'Development', "-Project=$project", '-WaitMutex', '-NoHotReloadFromIDE') (Join-Path $projectRoot "Saved\BuildLogs\scene-editor-$stamp.log")
        $sceneScript = Join-Path $PSScriptRoot 'build_scene.py'
        $report = Join-Path $projectRoot 'Saved\build-report.json'
        $errorReport = Join-Path $projectRoot 'Saved\build-error.txt'
        foreach ($receipt in @($report, $errorReport)) {
            if (Test-Path -LiteralPath $receipt) { Remove-Item -LiteralPath $receipt -Force }
        }
        $sceneLog = Join-Path $projectRoot "Saved\BuildLogs\scene-$stamp.log"
        $editor = Join-Path $engine 'Engine\Binaries\Win64\UnrealEditor.exe'
        $editorArgs = @('"' + $project + '"', '-ExecutePythonScript="' + $sceneScript + '"', '-unattended', '-d3d12', '-RenderOffscreen', '-nosound', '-NoSplash', '-abslog="' + $sceneLog + '"')
        $sceneProcess = Start-Process -FilePath $editor -ArgumentList $editorArgs -WindowStyle Hidden -PassThru -Wait
        if ($sceneProcess.ExitCode -ne 0 -or (Test-Path -LiteralPath $errorReport) -or !(Test-Path -LiteralPath $report)) {
            throw "Scene generation failed or did not produce its success receipt. Inspect $sceneLog and $errorReport"
        }
    }
    $map = Join-Path $projectRoot 'Content\Maps\Istana.umap'
    if (!(Test-Path -LiteralPath $map) -or (Get-Item -LiteralPath $map).Length -lt 1000) {
        throw 'The saved Istana map is missing. Download the Git LFS assets, or run package.ps1 -RebuildScene after preparing the source assets.'
    }
    $uat = Join-Path $engine 'Engine\Build\BatchFiles\RunUAT.bat'
    # Archive freshly into an isolated run directory, so a failed cook never replaces a working release.
    $archiveRoot = Join-Path $projectRoot "Saved\PackageArchives\$stamp"
    $uatArgs = @('BuildCookRun', "-project=$project", '-noP4', '-platform=Win64', "-clientconfig=$Configuration", '-build', '-cook', '-stage', '-pak', '-iostore', '-compressed', '-prereqs', '-archive', "-archivedirectory=$archiveRoot", '-map=/Game/Maps/Istana', '-unattended', '-utf8output', '-NoDebugInfo')
    Invoke-IstanaNative $uat $uatArgs (Join-Path $projectRoot "Saved\BuildLogs\package-$stamp.log")
    $archive = Assert-IstanaChildPath (Join-Path $archiveRoot 'Windows') $archiveRoot
    if (!(Test-Path -LiteralPath (Join-Path $archive 'IstanaOpen.exe'))) {
        throw "AutomationTool did not produce the expected Windows release: $archive"
    }

    $normalLauncher = @'
@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0IstanaOpen.exe" (
  echo Extract the entire release folder before starting Istana.
  pause
  exit /b 1
)
start "" "%~dp0IstanaOpen.exe" -windowed -ResX=1600 -ResY=900 %*
'@
    $compatLauncher = @'
@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0IstanaOpen.exe" (
  echo Extract the entire release folder before starting Istana.
  pause
  exit /b 1
)
start "" "%~dp0IstanaOpen.exe" -d3d11 -sm5 -IstanaMedium -windowed -ResX=1600 -ResY=900 %*
'@
    $prereqLauncher = @'
@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0Engine\Extras\Redist\en-us\UEPrereqSetup_x64.exe" (
  echo The prerequisite installer is missing. Extract the entire release archive again.
  pause
  exit /b 1
)
start /wait "" "%~dp0Engine\Extras\Redist\en-us\UEPrereqSetup_x64.exe"
'@
    Set-Content -LiteralPath (Join-Path $archive 'Start_Istana.cmd') -Value $normalLauncher -Encoding ASCII
    Set-Content -LiteralPath (Join-Path $archive 'Start_Compatibility.cmd') -Value $compatLauncher -Encoding ASCII
    Set-Content -LiteralPath (Join-Path $archive 'Install_Prerequisites.cmd') -Value $prereqLauncher -Encoding ASCII
    $releaseInstructions = @'
ISTANA OPEN - WINDOWS

Extract the complete folder, then double-click Start_Istana.cmd.
The release runs locally and does not need Unreal Editor, an account or internet access.
Start_Compatibility.cmd selects DirectX 11 with simplified mesh detail.
If Windows requests runtime components, run Install_Prerequisites.cmd once.

CONTROLS
Mouse + WASD: explore    E / Q: rise / descend
Shift: fast             Ctrl: precise movement
1 / 2 / 3 / 4: palace / garden / aerial / approach
T: tour                 F: Medium / High quality
P: photo mode           F1: controls
Esc: release mouse      Click or Enter: resume      Esc again: quit

Startup is 1600 x 900, Medium. Press F for High, or pass -IstanaHigh to the launcher.
The scene is an artistic public-exterior reconstruction of Singapore's Istana.
Map data: (c) OpenStreetMap contributors, ODbL. See included project notices.
'@
    Set-Content -LiteralPath (Join-Path $archive 'START_HERE.txt') -Value $releaseInstructions -Encoding UTF8
    foreach ($file in Get-ChildItem -LiteralPath $projectRoot -File) {
        if ($file.Name -match '^(README|LICENSE|NOTICE|ATTRIBUTION|THIRD_PARTY|END_USER|EULA)') {
            Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $archive $file.Name)
        }
    }
    foreach ($folder in @('Docs', 'ThirdParty')) {
        $from = Join-Path $projectRoot $folder
        if (Test-Path -LiteralPath $from) { Copy-Item -LiteralPath $from -Destination $archive -Recurse }
    }
    $openData = Join-Path $archive 'OpenData'
    New-Item -ItemType Directory -Force -Path $openData | Out-Null
    foreach ($sourcePair in @(
        @('SourceAssets\Geodata\osm_context.overpass.json', 'osm_context.overpass.json'),
        @('SourceAssets\Environment\context_provenance.json', 'context_provenance.json'),
        @('Tools\generate_environment.py', 'generate_environment.py')
    )) {
        $sourceFile = Join-Path $projectRoot $sourcePair[0]
        if (!(Test-Path -LiteralPath $sourceFile)) { throw "Open geographic source file is missing: $sourceFile" }
        Copy-Item -LiteralPath $sourceFile -Destination (Join-Path $openData $sourcePair[1])
    }
    $openDataInstructions = @'
# Geographic source and transformations

Map data is (c) OpenStreetMap contributors, under ODbL 1.0.
https://www.openstreetmap.org/copyright
https://opendatacommons.org/licenses/odbl/1-0/

The bundled raw extract and context_provenance.json identify the input snapshot,
retained source IDs and authored transformations. The transformation script is
included here so this source offer does not depend on a hosted repository.

Optional regeneration, using Python 3.11 or later from the release directory:

    python OpenData\generate_environment.py --input "%CD%\OpenData\osm_context.overpass.json" --output "%CD%\OpenData\Regenerated"

Run that command in Windows Command Prompt. It creates OBJ meshes, an instance
list and provenance records; Python is not required to play the application.
Generated geographic data remains subject to ODbL. Heights without mapped values,
facade modules, vegetation and terrain treatments are artistic approximations.
'@
    Set-Content -LiteralPath (Join-Path $openData 'README.md') -Value $openDataInstructions -Encoding UTF8
    Copy-Item -LiteralPath (Join-Path $projectRoot 'SourceAssets\manifest.json') -Destination (Join-Path $archive 'source-assets-manifest.json')
    $releaseReadmePath = Join-Path $archive 'README.md'
    if (Test-Path -LiteralPath $releaseReadmePath) {
        $releaseReadmeText = (Get-Content -LiteralPath $releaseReadmePath -Raw).Replace('(SourceAssets/manifest.json)', '(source-assets-manifest.json)')
        Set-Content -LiteralPath $releaseReadmePath -Value $releaseReadmeText -Encoding UTF8
    }
    Invoke-IstanaNative $python @($verifier, '--release', $archive, '--write-manifest') (Join-Path $projectRoot "Saved\BuildLogs\release-check-$stamp.log")

    if (!$OutputRoot) { $OutputRoot = Join-Path $projectRoot 'Releases' }
    $OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
    New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    $release = Assert-IstanaChildPath (Join-Path $OutputRoot 'Windows') $OutputRoot
    if (Test-Path -LiteralPath $release) {
        $previous = Assert-IstanaChildPath (Join-Path $OutputRoot "Windows-previous-$stamp") $OutputRoot
        if (Test-Path -LiteralPath $previous) { throw "Previous-release backup already exists: $previous" }
        Move-Item -LiteralPath $release -Destination $previous
        Write-Host "Previous release preserved at $previous"
    }
    # Both resolved absolute paths have been checked against their intended parent directories.
    Move-Item -LiteralPath $archive -Destination $release
    Write-Host "Release complete: $release"
    Write-Host 'Launch Start_Istana.cmd. The verifier checks package structure; run the packaged game to verify rendering on the destination computer.'
}
catch { Write-Error $_ -ErrorAction Continue; exit 1 }
