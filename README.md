# Istana Open

Explore an offline reconstruction of Singapore's Istana and its surroundings.

## Download and run on Windows

1. Download **[IstanaOpen-Windows-2026-09-10.zip](https://github.com/Jenjenson/IstanaOpen/releases/download/v1.0.0/IstanaOpen-Windows-2026-09-10.zip)**
   from the [latest release](https://github.com/Jenjenson/IstanaOpen/releases/latest).
2. Right-click the ZIP in File Explorer and choose **Extract All**. Open the
   extracted `IstanaOpen-Windows` folder containing `Start_Istana.cmd`.
3. Double-click **Start_Istana.cmd**. Keep the entire folder together, including
   its `Engine` and `IstanaOpen` subfolders. Do not run it from inside the ZIP.

**You do not need Unreal Engine, Visual Studio, Git or Python to play.** Once
downloaded, the application runs locally without an account, API token,
internet connection, AirSim installation or Cesium subscription.

If Windows reports missing Microsoft runtime components, double-click
**Install_Prerequisites.cmd** once, complete its installer, and launch again.
The bundled installer is also available at
`Engine/Extras/Redist/en-us/UEPrereqSetup_x64.exe`.

This release is for **64-bit Windows** and was tested on Windows 11. There is
no macOS or Linux build. The normal launcher uses DirectX 12. If that does not
work with your graphics hardware, try **Start_Compatibility.cmd**, which uses
DirectX 11 with simpler rendering and mesh detail.

![Packaged Istana Open palace view](Docs/Evidence/packaged-medium.png)

The scene includes a newly authored palace exterior, formal landscape,
scanned vegetation, 4K architectural materials, and surrounding buildings and
roads derived from OpenStreetMap.

## Controls

| Control | Action |
| --- | --- |
| Mouse, W A S D | Look and fly |
| E / Q | Rise / descend |
| Shift / Ctrl | Fast / precise movement |
| 1 / 2 / 3 / 4 | Palace / garden / aerial / approach |
| T | Start or stop the orbit tour |
| F | Medium / High quality |
| P | Hide interface for photography |
| F1 | Controls |
| Esc | Release mouse; press again to quit |
| Click or Enter | Resume |

The default is a windowed **1600 × 900, Medium** profile with a 60 FPS cap.
Press F to switch between Medium and High. Packaged rendering was tested on
an AMD Ryzen 7 5700X3D computer with **32 GB RAM and an NVIDIA TITAN V with
12 GB VRAM**, averaging approximately 59 FPS in the tested views. Those are
measurements from one computer, not minimum specifications or a guarantee
for arbitrary PCs. See [validation and evidence](Docs/VALIDATION.md), including
the limitation that manual keyboard and mouse testing could not be completed.

## Troubleshooting

- **Missing files or the window closes immediately:** extract the complete
  Windows release again. Moving only the EXE or running inside the ZIP leaves
  required files behind.
- **Missing runtime/DLL message:** run `Install_Prerequisites.cmd`, then retry.
- **DirectX or graphics-device error:** try `Start_Compatibility.cmd` and check
  that your graphics driver supports the selected DirectX version.
- **Slow rendering:** use Medium quality. For a smaller 1280 × 720 window,
  open PowerShell in the extracted folder containing `IstanaOpen.exe` and run:

  ```powershell
  .\IstanaOpen.exe -IstanaMedium -windowed -ResX=1280 -ResY=720
  ```

  For DirectX 11 at that resolution, use:

  ```powershell
  .\IstanaOpen.exe -d3d11 -sm5 -IstanaMedium -windowed -ResX=1280 -ResY=720
  ```

- **Further diagnosis:** after a launch, check the newest log under
  `IstanaOpen/Saved/Logs` inside the extracted release. Include it, your Windows
  version and GPU model when reporting an issue. A failure before the game
  starts may not produce a log.
- **No playable EXE in your download:** you downloaded the source project.
  Use the named **Windows** release asset linked above.

## Edit or build

The editable source requires Unreal Engine; the playable download above does
not. Choose either the complete
**[IstanaOpen-Source-2026-09-10.zip](https://github.com/Jenjenson/IstanaOpen/releases/download/v1.0.0/IstanaOpen-Source-2026-09-10.zip)**
release asset, or clone the repository with **Git and Git LFS installed**:

```powershell
git lfs install
git clone https://github.com/Jenjenson/IstanaOpen.git
cd IstanaOpen
git lfs pull
```

The named source release ZIP includes the actual source assets. GitHub's
automatic **Source code (zip)** or **Code → Download ZIP** may contain Git LFS
pointers instead of the large meshes and textures; those snapshots are not
the playable application. Use the named source asset or `git lfs pull` to
obtain the full editable content.

On Windows, install **Unreal Engine 5.5.4**, **Visual Studio 2022** with the
**Desktop development with C++** and **Game development with C++** workloads,
and a Windows SDK. Open PowerShell in the source folder containing
`IstanaOpen.uproject` (`IstanaOpen-Source` when using the source release ZIP),
then build:

```powershell
powershell -ExecutionPolicy Bypass -File Tools/build.ps1
```

Create the playable Windows package with:

```powershell
powershell -ExecutionPolicy Bypass -File Tools/package.ps1
```

The completed package is written to `Releases/Windows`. If Unreal is installed
in a location the scripts cannot discover, pass `-EngineRoot` with your local
Unreal 5.5 installation folder to either script.

Add `-RebuildScene` to reimport the bundled source assets and rebuild the saved
level before packaging. This uses Unreal's bundled editor Python. To reproduce
the generated geometry first, install Python **3.11 or later** and run
`python Tools/generate_istana.py` and `python Tools/generate_environment.py`
before `Tools/package.ps1 -RebuildScene`. Neither step downloads map tiles.

Open IstanaOpen.uproject and Content/Maps/Istana to edit. The saved Content
assets and level load without regenerating geometry or importing the raw
source assets; ordinary builds do not regenerate the world.
Tools/generate_istana.py and Tools/generate_environment.py reproduce the
original geometry with Python 3.11's standard library. Tools/build_scene.py
imports it using Unreal's editor Python. Tools/prepare_assets.py records the
one-time migration of individually verified scans from the former local
cache; it is not a dependency for consumers of this repository.

## Fidelity and data

The palace is a modeled interpretation of the publicly visible exterior,
including arcades, colonnades, shutters, cornices, a mansard tower, dormers and
steps. It is not a surveyed digital twin. Surrounding footprints and road
alignments come from the included August 2026 OSM snapshot; many building
heights and all generic facade treatments are inferred. Terrain is an artistic
hill, and planting and garden detail are composed for the presentation.
Interiors, current construction works and physical sensor simulation are not
modeled. The camera flies freely through the scene.

This is an independent project and is not an official Istana product. No
Presidential Crest, Presidential Standard, restricted map imagery, or
commercial marketplace scene pack is included.

## Redistribution

Original code: MIT. Original art and selected Poly Haven/ambientCG scans: CC0.
Map data: © OpenStreetMap contributors, ODbL. The raw snapshot, provenance,
generation code and attribution are included. Engine source, editor and
installed plugins are excluded from the public source tree. A compiled release
contains Epic's runtime under the Unreal Engine EULA.

See [licensing details](Docs/LICENSING.md), [third-party notices](ThirdParty/README.md)
and [asset manifest](SourceAssets/manifest.json). These are redistributable
components with applicable terms, not a claim that Unreal has no license.
