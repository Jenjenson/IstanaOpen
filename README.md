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

## Shared simulation data structures (contributors)

The simulation foundation is implemented as Blueprint-accessible C++ value types,
reusable Data Assets, validation functions, and a no-op policy interface.
A separate [synthetic swarm module](Docs/SWARM_SIMULATION.md) now implements
seeded multi-swarm spawning, boid movement, level-collision pathfinding and shared
objective following through `ARedTeamManager`. Sensor behavior, the
operator panel, optimization, and RL remain future work. The existing architectural
viewer still runs as before.

Start with the **[shared contracts usage guide](Docs/SIMULATION_CONTRACTS.md)** for
field definitions, C++ and Blueprint examples, ownership, validation, and tests.

| Contract | Unreal type | What it contains |
| --- | --- | --- |
| Scenario configuration | `FIstanaScenarioConfig` | Seed, duration, fixed timestep, map, sensor budget, swarms, conditions, initial sensors |
| Drone state | `FIstanaDroneState` | Stable drone/group IDs, position, velocity, active flag; simulator ground truth |
| Sensor configuration | `FIstanaSensorConfig` | Sensor ID/type, placement, abstract model parameters, sampling interval |
| Detection event | `FIstanaDetectionEvent` | Sensor ID, sequence number, sample/delivery steps, reported position, confidence |
| Episode result | `FIstanaEpisodeResult` | Run/scenario/version identifiers, seed, executed steps, metrics, completion reason |
| Policy interface | `IIstanaPolicyInterface` | Reset, receive permitted observations, produce actions; schema 1 actions are no-op only |

### Use the contracts

1. Rebuild the editor after pulling C++ changes, then open the project in UE 5.5.4.
2. In **Content > Simulation > Examples**, duplicate `DA_SyntheticContractExample`
   or `DA_AbstractSensorExample` for your task. Assign a fictional map to the
   scenario; the example intentionally leaves it unassigned.
3. Call **Create Runtime Config** on the scenario asset, store the returned copy,
   and edit that copy. For sensors, call **Create Sensor Config** on the sensor preset.
4. Call **Validate Scenario** or **Validate Sensor** before using the configuration.
   Show the returned field paths and messages in your UI or logs.
5. For policies, use `BP_NoOpPolicyExample` as a starting point and follow
   **Reset Policy → Receive Observations → Produce Actions**. The future coordinator
   validates returned actions before applying anything.

C++ headers live in `Source/IstanaOpen/Simulation`; include them with paths such as
`Simulation/IstanaSimulationTypes.h`. Positions use centimetres, velocities use
centimetres/second, and timestamps use zero-based simulation steps. See the guide
before adding fields or interpreting metrics.

### Working on another machine

- Use **UE 5.5.4**, the C++ toolchain described above, and Git LFS. From an existing
  clone, run `git lfs install` and `git lfs pull` before building.
- Keep `EngineAssociation` in `IstanaOpen.uproject` set to **`5.5`**. Unreal may
  replace it with a local installation GUID; do not commit that GUID. The descriptor
  uses `5.5` while this project's tested patch version is `5.5.4`.
- Build using `Tools/build.ps1`; if discovery fails, supply `-EngineRoot` or set
  `UE_ENGINE_ROOT` in your own shell. Engine installation paths belong to each
  developer's environment, not shared source or configuration.
- If the launcher offers conversion despite using 5.5.4, cancel and launch that
  installation's `Engine/Binaries/Win64/UnrealEditor.exe` with the full path to your
  local `IstanaOpen.uproject`. Do not upgrade the shared project to another version.
- The enabled browser helpers (`ContentBrowserAssetDataSource`,
  `ContentBrowserClassDataSource`, `EngineAssetDefinitions`) are bundled UE editor
  plugins. Keep all three enabled; the project disables other engine plugins by
  default. They are restricted to editor targets and require no marketplace install.
- Commit source, docs, the project descriptor, and new example `.uasset` files
  together. Binary assets use the existing Git LFS rules. Build products, caches,
  generated solution files, and local editor settings under `Saved` are ignored.
- Close Unreal before full builds involving reflected headers. Each teammate builds
  locally; copying another developer's DLL is not part of the setup.

Run the `Istana.Simulation.Contracts` automation tests after changing contracts.
Cross-machine behavior still needs verification on each teammate's setup; tests on
one Windows machine do not establish support for every toolchain or platform.

## Synthetic swarm demo

The [Red Team Manager README](Source/IstanaOpen/Simulation/RedTeam/README.md)
covers the implemented multi-swarm spawner, shared-objective controller, configuration,
runtime ownership and troubleshooting.

To follow a marker in a fictional level, place `BP_SwarmObjective` (or a standard
Target Point) and assign it to the swarm manager's **Objective Target** property.
Keep **Follow Objective** enabled. The [swarm guide](Docs/SWARM_SIMULATION.md)
explains arrival, retargeting and valid collision-free destinations.

For the swarm demo, open **Content > Simulation > Maps > SyntheticSwarmArena** and
press Play. **Space** pauses, **R** resets, and **V** toggles debug. Configure groups
on the swarm manager actor and movement through `DA_SwarmMovementExample`.
See the [swarm guide](Docs/SWARM_SIMULATION.md) for setup, commands, integration,
tests and model limitations. This demo uses a fictional arena, not the palace map.

## Multiple swarms around one objective

1. Place **Red Team Manager** and assign a Target Point or `BP_SwarmObjective` to
   **Objective Target**. The objective's position anchors spawning.
2. Set **Number Of Swarms**, **Drones Per Swarm**, **Min/Max Spawn Radius Cm**,
   **Swarm Spread Radius Cm**, and **Spawn Height Offset Cm**.
3. Keep **Auto Initialize**, **Auto Advance**, **Follow Objective** and **Spawn Visuals**
   enabled, then Play. The inherited manual Swarms array is not used by this actor.

Defaults spawn three groups of 12 drones with centers 30�60 meters from the objective.
Editor distance fields use centimeters. An assigned movement preset overrides inline
settings; its Max Drones limits total population (default 128, hard cap 256).

Moving the marker retargets all groups without respawning; removing it brakes them.
Markers on or inside geometry are accepted: drones approach reachable positions and
retry blocked paths while keeping collision active. Partial approach does not count
as exact route completion. There are no arena min/max limits or predefined obstacles;
navigation uses level objects that block the selected collision channel.

See the [Red Team Manager README](Source/IstanaOpen/Simulation/RedTeam/README.md) for
setup and lifecycle, the [swarm guide](Docs/SWARM_SIMULATION.md) for navigation tuning,
and [validation](Docs/VALIDATION.md#swarm-and-red-team-validation-12-september-2026)
for the successful editor/game builds and 13-test automation run. Learning, sensors,
rewards and full episode coordination remain future work.

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
