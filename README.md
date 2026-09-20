# Istana Open

Explore an offline reconstruction of Singapore's Istana and its surroundings.

**New computer?** Use the [complete source setup below](#set-up-the-browser-interface-and-live-3d-simulation-windows)
for the sensor/drone simulation and interface. The older downloadable EXE is a
landscape viewer, not the current integrated simulation.

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

## Sensor simulation console (experimental)

The [Blue + Red console](Docs/BLUE_TEAM_LIVE.md) provides local browser playback
of 18 published synthetic sensor-placement evaluations, including sensor ranges,
drone paths, timeline controls and results. These recorded cases are not live
Istana runs. Experimental live Unreal telemetry controls are also included;
the integration has passed native builds, 18 native tests and live headless
smoke episodes on the Istana map. Use the source project's live launcher,
not the older landscape-only downloadable release.

### Set up the browser interface and live 3D simulation (Windows)

The **browser is the control/telemetry interface**; a **separate Unreal window
renders the running 3D scene**. Capture mode also supports native 3D saved-layout
previews directly inside the original interface. For these modes use the current Git repository, not
the September 10 source ZIP or landscape-only Windows release.

**Install once:**

- **64-bit Windows**, tested on Windows 11; PowerShell and a modern desktop
  browser (Edge/Chrome). The integrated Windows workflow is not validated on
  macOS/Linux or Windows ARM.
- [Git for Windows](https://git-scm.com/downloads/win) and
  [Git LFS](https://git-lfs.com/), for source and map/mesh/texture assets.
- Unreal Engine **5.5.4** through [Epic Games Launcher](https://www.unrealengine.com/download).
  Select the 5.5 engine line in Library; do not upgrade the project to another
  engine version for this setup. Keep the Windows engine components and bundled
  Python. The tested default install is `C:\Program Files\Epic Games\UE_5.5`.
- Visual Studio **2022 Build Tools** (or Visual Studio 2022) with C++ build tools:
  **MSVC v143 VS 2022 C++ x64/x86 build tools v14.38**, **Windows 11 SDK
  10.0.22621.0**, **.NET Framework 4.8 SDK**, and **.NET Framework 4.8 targeting
  pack**. In Visual Studio Installer, use **Modify → Individual components**
  to find these. VS Code is optional; it does not replace the compiler.
- In Build Tools select **Desktop development with C++**, then verify those
  individual components. For the full Visual Studio IDE, **Game development
  with C++** is also useful. Use the 2022 installer, not simply the newest major
  release offered on the download homepage; Microsoft's
  [2022 release history](https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history)
  links the 2022 installers. Epic's [UE 5.5 toolchain guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/setting-up-visual-studio-development-environment-for-cplusplus-projects-in-unreal-engine?application_version=5.5)
  documents compatible versions. UE's build uses bundled .NET 8; the **.NET
  Framework 4.8 SDK and targeting pack are separate components**, not replaced
  by installing the .NET 8 runtime.
- A GPU/driver capable of running UE 5.5's Windows renderer. This project was
  tested with **32 GB RAM / 12 GB VRAM**; these are reference machine specs,
  not established minimums. Check installer disk estimates and leave substantial
  additional SSD space for assets, shader caches, builds and video frames.
  Native capture still needs a working GPU even though it has no visible window.

No separate Python installation is needed for this route: the setup script
uses Unreal's bundled Python and creates an isolated environment. Internet is
needed for the initial downloads/dependency installation; the simulation then
runs locally. Docker is not required for the browser + 3D workflow.
Neither Node/npm, CUDA, PyTorch, AirSim, ROS, Cesium nor cloud/API credentials
are required by this workflow. Epic sign-in is needed to download the engine;
GitHub credentials are only needed for private access or pushing changes.

**1. Get the current project and build it.** In PowerShell, for a fresh checkout:

```powershell
git lfs install
git clone --branch codex/istana-open https://github.com/Jenjenson/IstanaOpen.git
cd IstanaOpen
git lfs pull
git lfs fsck
powershell -ExecutionPolicy Bypass -File .\Tools\setup_blue_python.ps1
powershell -ExecutionPolicy Bypass -File .\Tools\build.ps1 -Target Editor
```

The repository's default branch is `codex/istana-open`. If you already have a
checkout, update that checkout instead of cloning again. Run all commands below
from the folder containing `IstanaOpen.uproject`. Close the project's Unreal
window before rebuilding. If the engine is not discovered, pass
`-EngineRoot 'D:\Epic Games\UE_5.5'` (using your actual path) to the setup,
build and live-scene scripts.

Use a short local folder, for example `C:\Projects\IstanaOpen`, rather than a
cloud-synced directory. Restart PowerShell after installing Git/compiler tools.
The setup script installs `triad-rl` plus its `test` and `media` extras into
`RL\BlueTeam\.venv`: NumPy, Gymnasium, PettingZoo, pytest, Pillow and
imageio-ffmpeg. The Windows imageio-ffmpeg wheel supplies FFmpeg; no separate
system FFmpeg install is required. Run the setup script again after updating
dependencies. Do not copy a virtual environment from another computer.

For browser-only replay on a device without Unreal, install Python **3.11+**
(3.11 is the tested bundled version), then use your actual interpreter path:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\setup_blue_python.ps1 -PythonExe 'C:\Path\To\Python311\python.exe'
```

That alternative does **not** provide native 3D; native mode still requires UE
and the compiled project. No source asset regeneration is required after a
complete Git LFS checkout.

**2. Start the browser server.** Keep this PowerShell window open:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\start_simulation_console.ps1
```

Open **[http://127.0.0.1:9048/](http://127.0.0.1:9048/)** in your browser.
Recorded replay mode works without launching the 3D scene. It shows archived
synthetic evaluations, not a live Unreal simulation.

**3. Start the live 3D scene.** Open another PowerShell window in the project
folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\start_blue_live.ps1
```

Wait for the Istana scene to finish loading; first-time shader compilation can
take several minutes. The launcher starts the compiled project with the live
bridge on **127.0.0.1:8765**. Do not launch a second scene on the same port.

**4. Deploy and run from the browser:**

1. Choose **Live Unreal → Connect to Unreal**.
2. Under **Blue planner**, choose a trained checkpoint (406/407/408) or
   **Greedy · non-RL**. Greedy chooses the legal sensor/site with the greatest
   predicted marginal return, repeating within the budget.
3. Click **Plan new episode**. This places Blue sensors and scripted Red drones
   in the Unreal scene. An empty scene before this step is expected.
4. Click **Run episode**, **Pause**, or **Step** to control the shared simulation
   clock. Changing the planner only takes effect on **Plan new episode**.

In Unreal, press **1** for a closer palace view or **3** for an aerial view;
use **WASD + mouse** to explore and **E/Q** to move up/down. Sensors have cyan
labels and visible masts resting on supported ground/roofs. Unsupported sites,
steep slopes and narrow edges are excluded. Drone markers are grouped and
labelled. Detailed sensor heads and quadcopter visuals are presentation assets,
not validated physical hardware models.

**5. Use native 3D previews inside the actual interface (optional).**

This is the mode used for the model-switching presentation. Close/disconnect any
other bridge client first. Stop the browser server with Ctrl+C, then start a
capture-enabled Unreal instance and point the original console at its port:

```powershell
# Terminal 1, repository root: starts Unreal offscreen on a separate port.
powershell -ExecutionPolicy Bypass -File .\Tools\start_blue_capture.ps1 -Port 8766

# Terminal 2, repository root: leave running.
powershell -ExecutionPolicy Bypass -File .\Tools\start_simulation_console.ps1 -Port 9048 -BridgePort 8766
```

Open port 9048, select **Live Unreal → Connect to Unreal**. The model selector
retains the live **406 / 407 / 408 / Greedy** choices. Under **Archived native
layouts (no inference)** choose a saved output and press **Apply saved layout**.
Unreal validates and replaces the sensors; the console shows the native overhead
capture. Enable **Presentation view** to expose the whole list and hide metrics.
These previews do not advance Red or run a new policy. Ordinary live choices
still use **Plan new episode** and the existing telemetry workflow.

The three small saved-layout fixtures are included in
`RL/BlueTeam/Results/model-switch-demo`; no old `Saved` folder is needed for
them. Their source hashes are documented alongside them. They are **not model
weights**. The trained temporal weights for 406/407/408 and 18 archived replay
cases are also in the repository. The 256-episode warning experiment is a
different model family and is not silently substituted into this dropdown.

**Ports and ownership:**

| Port | Purpose | When needed |
| --- | --- | --- |
| 9048 | Original browser console | Replay, live controls, native previews |
| 8765 | Interactive Unreal bridge | Separate live 3D window |
| 8766 | Capture-enabled Unreal bridge | Native preview/capture |
| 9050 | Optional video/report server | Viewing generated reports |
| 9051 | Earlier standalone demo selector | Not required for the original UI |
| 8767–8770 | Optional independent benchmark workers | Training only |

These services bind to loopback; no firewall port-forwarding is needed. Only
one client should control each bridge. Do not attach the console and a capture
or training script to the same bridge at the same time. To stop, disconnect in
the console, Ctrl+C its server terminal, and close the Unreal window. The
offscreen launcher prints its PID; stop **that specific process** when finished.

**6. Verify the new installation.** From the repository root:

```powershell
.\RL\BlueTeam\.venv\Scripts\python.exe -c "import numpy, gymnasium, pettingzoo, PIL, imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
Push-Location .\RL\BlueTeam\Python
..\.venv\Scripts\python.exe -m pytest tests/test_simulation_console.py tests/test_model_switch_demo.py -q
Pop-Location
```

Check that recorded playback loads, Unreal connects, and a saved preview shows
the selected model name and a changed native sensor layout. First-run shaders
can take several minutes. Automated Python tests do not replace this GPU/native
smoke check; results can differ on a new driver/device.

**7. Transfer optional training results and videos separately.**

Git intentionally excludes `Saved`, `Binaries`, `Intermediate`, caches and
virtual environments. Cloning includes code/assets/published temporal models,
but **not** locally generated MP4s, native frame dumps or the later warning
checkpoints. To continue those exact runs, copy the desired complete run folders
from `Saved/WarningTraining` on the old machine to the same relative location
on the new one (including `policy-*.json`, evaluation JSON, summary and protocol).
For existing footage, copy the relevant video/report folder and its sidecars.
Do not copy compiled DLLs or `.venv`; rebuild/reinstall them on the new device.
Some capture manifests contain absolute frame paths from the original machine;
playing a copied MP4 is portable, but rerendering those manifests may require
relocating the frame references or making a new native capture.

To serve a copied report folder, from `RL/BlueTeam/Python` run:

```powershell
..\.venv\Scripts\python.exe serve_warning_report.py 'C:\Path\To\CopiedReportFolder' --port 9050
```

Open port 9050 (the folder needs its generated `index.html`). The console does
not require this report server. See the training guide below for optional
synthetic benchmark scripts; no retraining is necessary for the interface demo.

**Common fixes:**

- **Browser does not load:** keep the console-server terminal running; check
  that port 9048 is free. Stop that server with Ctrl+C when finished.
- **Cannot connect to Unreal:** use `start_blue_live.ps1`, wait for the map to
  load, then reconnect. The old packaged viewer does not include the bridge.
- **No drones/sensors:** click **Plan new episode**, then **Run episode**.
  If the planner chose STOP, check the deployment panel for an empty layout.
- **Connection timed out after pausing:** the bridge has a five-minute idle
  timeout. Reconnect and plan a new episode.
- **Build fails:** check the four Visual Studio components above and inspect
  `Saved/BuildLogs`. Runtime logs are in `Saved/Logs/BlueLive.log`.
- **Missing `PIL` / `imageio_ffmpeg`:** rerun `setup_blue_python.ps1`; use the
  repository `.venv` interpreter, not another Python on PATH.
- **Native preview rejected:** it needs `start_blue_capture.ps1`, not the normal
  interactive launcher; the latter does not enable screenshot requests.
- **Preview choices missing:** update the repository, confirm the three JSON
  fixtures under `Results/model-switch-demo`, then restart the console and reload.
- **Grey/missing meshes or tiny asset files:** run `git lfs pull` and
  `git lfs fsck`; an unresolved LFS pointer is not a usable Unreal asset.
- **Port already in use:** reuse the matching service or close its specific
  process. Use `Get-NetTCPConnection -LocalPort 9048,8765,8766 -State Listen`
  to identify owners; do not indiscriminately stop all Python/Unreal processes.

The live Red controller is scripted, not learned. Sensor capabilities/rewards
are synthetic, and sensing does not model terrain occlusion. See the
[live integration guide](Docs/BLUE_TEAM_LIVE.md) for validation and limitations.

### Train Blue for warning time and watch the timelapse

The [native warning-time training guide](Docs/WARNING_TIME_TRAINING.md) explains
how to train a separate Blue RL policy, compare fixed checkpoints against
untrained and greedy controls, and render an MP4 with sensor placements and
measured warning times. It also includes a command to replay the trained policy
in the actual Unreal 3D scene. It supports both the lightweight top-down replay
and an **actual 1080p Unreal 3D timelapse** with detailed surface-mounted sensor
models and native close-ups. Team warning and per-drone warning are separate;
arrival means entering the target zone, not a physical crash. Improvement is
measured rather than assumed.

The new opt-in **long-approach benchmark** moves starts beyond sensor range,
uses several fixed synthetic approach sectors, and logs spawn-clearance and
first-look saturation checks. Its paired evaluation verifies unchanged Red
trajectories and sensor/terrain constraints across checkpoints. The guide also
includes **clean 3D footage**: no marker boxes or burned-in metric panels;
timings and results sit beside/below the player and can be hidden.

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
objective following through `ARedTeamManager`. The red team also supports
[agent-controlled placement](Docs/RED_TEAM_AGENT.md), an external Python runtime bridge,
and a bounded [initial-placement RL workflow](Docs/RED_TEAM_RL.md).
Behavior-preserving optimizations and measured limits are described in
[optimization results](Docs/SWARM_OPTIMIZATION_RESULTS.md). Sensors, the operator panel
and live synthetic evaluation are integrated; real-world calibration and a learned
flight controller remain future work. The architectural viewer still runs as before.

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

Defaults spawn three groups of 12 drones with centers 30–60 meters from the objective.
Editor distance fields use centimeters. An assigned movement preset overrides inline
settings; its Max Drones limits total population (default 128, hard cap 256).

Moving the marker retargets all groups without respawning; removing it brakes them.
Markers on or inside geometry are accepted: drones approach reachable positions and
retry blocked paths while keeping collision active. Partial approach does not count
as exact route completion. There are no arena min/max limits or predefined obstacles;
navigation uses level objects that block the selected collision channel.

See the [Red Team Manager README](Source/IstanaOpen/Simulation/RedTeam/README.md) for
setup and lifecycle, the [swarm guide](Docs/SWARM_SIMULATION.md) for navigation tuning,
and [agent integration](Docs/RED_TEAM_AGENT.md) for explicit placement, fixed-step episodes
and the runtime bridge. See [validation](Docs/VALIDATION.md) and
[optimization results](Docs/SWARM_OPTIMIZATION_RESULTS.md) for measured scope and limits.
Learning, sensors, task rewards and full-world episode coordination remain future work.

## Blue Team reinforcement learning

The [Blue Team RL package](RL/BlueTeam/README.md) adds a CPU Python trainer for
choosing sensor profiles and continuous placements, a trained TRIAD checkpoint,
and recorded evaluation results. Its new [adaptive experiment](RL/BlueTeam/ADAPTIVE.md)
adds randomized threats/weather, joint sensor/site decisions, a shared external
input contract, held-out baseline comparisons and an offline replay demo.
Start with its Unreal-free quickstart. Live
training requires the separate original TRIAD host. The experimental live
Istana adapter now connects initial Blue placement to the Red Team Manager;
it does not implement joint Red/Blue training. See the
[integration guide](RL/BlueTeam/INTEGRATION.md) and
[experiment results](RL/BlueTeam/RESULTS.md).

## Run in Docker (any machine)

The Python side of this project — trainer, planner, evaluators, the 66-module
test suite and the published experiment evidence — runs in one CPU-only image.
No GPU, no network, no account, no API key. It builds from a plain `git clone`;
`git lfs pull` is **not** required, because the image excludes `Content/` and
`SourceAssets/`.

The image builds and runs natively on **linux/amd64 and linux/arm64**, so Intel
and AMD machines, Apple Silicon Macs, and ARM servers all use the same commands
with no emulation and no extra flags.

### Requirements

| Host | Install |
| --- | --- |
| macOS (Apple Silicon or Intel) | [Docker Desktop](https://docs.docker.com/desktop/install/mac-install/) |
| Windows 10/11 | [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/) with the WSL 2 backend |
| Linux | [Docker Engine](https://docs.docker.com/engine/install/) plus the Compose plugin |

Roughly 3 GB of free disk. Nothing else: the toolchain, the pinned dependencies
and the evidence tree are all inside the image.

### Quick start

```bash
git clone https://github.com/Jenjenson/IstanaOpen.git
cd IstanaOpen

docker compose run --rm test      # full suite, including byte-exact evidence checks
docker compose run --rm plan      # sensor layout from a public snapshot -> ./runs
docker compose run --rm smoke     # train -> evaluate -> replay -> plan, end to end
```

The first command builds the image (a few minutes), then runs 1,682 checks, most
of which re-verify the archived experiment artifacts rather than merely
exercising code. Budget about 20 minutes. None of it touches the network, and
you can prove that by running the built image with networking switched off:

```bash
docker run --rm --network none istana-rl:local test
```

Without Compose:

```bash
docker build -t istana-rl .
docker run --rm --network none istana-rl test
docker run --rm -v "$PWD/runs:/workspace/runs" istana-rl smoke
```

On Windows PowerShell use `${PWD}` in place of `$PWD`.

### Apple Silicon and other arm64 hosts

Nothing special is needed. `docker compose run --rm test` builds an arm64 image
and runs it natively. Two things are worth knowing:

- **Only the optional Unreal image is amd64-only.** Epic publishes
  `ghcr.io/epicgames/unreal-engine` for amd64 alone, and emulating a full engine
  build is not practical. Build the native C++ modules on an x86-64 machine or
  on Windows with `Tools/build.ps1`. Everything in the table below is unaffected.
- **To reproduce the archived numbers bit-for-bit, force amd64.** The published
  evidence was generated on x86-64. The suite passes on arm64, but floating-point
  reduction order is not guaranteed identical across architectures, so if you are
  checking hashes rather than behaviour:

  ```bash
  DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose run --rm test
  ```

  That runs under emulation and is several times slower. For everyday use,
  running natively on arm64 is the right choice.

### Every command

`docker run --rm istana-rl help` prints this list. Arguments after the command
pass straight through to the underlying script.

**Verification**

| Command | What it does |
| --- | --- |
| `test [pytest args]` | Full suite, `-q` by default. Includes byte-exact checks of the published artifacts. |
| `smoke` | End-to-end pipeline: train, evaluate, replay, recommend. Writes `runs/smoke`. |
| `versions` | Interpreter, architecture and resolved dependency versions. |

**Planning**

| Command | What it does |
| --- | --- |
| `recommend [args]` | Temporal planner on a public snapshot. With no arguments, uses the bundled example and writes `runs/temporal-plan.json`. |
| `recommend-adaptive [args]` | The same through the `adaptive-v1` checkpoint. |

**Training and evaluation**

| Command | What it does |
| --- | --- |
| `train [args]` | `train_blue_placement.py`; add `--dry-run` for the toy environment. |
| `train-adaptive`, `train-robust`, `train-balanced`, `train-temporal` | The four archived training arms. |
| `evaluate [args]`, `evaluate-adaptive [args]` | Checkpoint evaluation against the baselines. |

**Replays**

| Command | What it does |
| --- | --- |
| `demo-adaptive`, `demo-balanced`, `demo-temporal` | Render a self-contained HTML replay into `runs/`; open it in any browser. |

**Project tools**

| Command | What it does |
| --- | --- |
| `geometry` | Regenerate the Istana and environment OBJ geometry (standard library only). |
| `benchmarks` | Summarise the swarm benchmark CSVs. |
| `redteam-client [args]` | Wire client for a red-team bridge running in Unreal on the host. |
| `verify [args]` | `Tools/verify_release.py`; needs the repository bind-mounted. |
| `audit` | `Tools/audit_submission.py`; needs the repository bind-mounted. |

**Escape hatches**

| Command | What it does |
| --- | --- |
| `python [args]` | Python inside `RL/BlueTeam/Python`. |
| `bash` / `shell` | Interactive shell. |
| `exec <cmd> [args]` | Any command inside the container. |

Compose exposes the common ones as services: `test`, `smoke`, `plan`, `demo`,
`rl` (interactive shell) and `redteam-client`.

Flags pass through, so a longer training run is just:

```bash
docker run --rm -v "$PWD/runs:/workspace/runs" istana-rl \
  train-adaptive --episodes 200 --batch-size 8 --seed 42 --output /workspace/runs/adaptive
```

### Output, and file ownership on Linux

Everything a command writes goes to `/workspace/runs`, bind-mounted to `./runs`
on the host. The container runs unprivileged as uid 1000.

On macOS and Windows, Docker Desktop maps ownership for you. On Linux, if your
own uid is not 1000, tell Compose who you are before running anything that
writes:

```bash
export ISTANA_UID=$(id -u) ISTANA_GID=$(id -g)
docker compose run --rm smoke
```

With plain `docker run`, the equivalent is `--user "$(id -u):$(id -g)"`.

### What is pinned, and why

- **Dependency versions.** `docker/constraints.txt` pins NumPy to `2.4.6`, the
  version the published experiments recorded. Build against current upstream
  instead with `--build-arg CONSTRAINTS=docker/constraints-current.txt`, and
  expect last-bit numeric drift from the archived artifacts.
- **Thread count.** `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.
  Not a performance setting: the recorded runs used single-threaded BLAS, and
  letting thread count follow the host's core count changes float reduction
  order.
- **Hash seed.** `PYTHONHASHSEED=0`.
- **Python minor version.** `python:3.11-slim-bookworm`. Override with
  `--build-arg PYTHON_VERSION=3.12`.

Containerising does not erase host differences entirely.
`RL/BlueTeam/BALANCED_RESULTS.md` records 172 last-bit differences
(max ~5.7e-14) between Windows and Linux, which changed hashes of unrounded JSON
without changing a single sensor choice or action count. Running Linux in a
container gives you the *Linux* numbers consistently, which is the point, but it
does not make Windows-generated and Linux-generated hashes identical.

### Talking to Unreal running on the host

The red-team bridge runs inside Unreal on the host and binds loopback. A
container cannot reach host loopback directly, so the client takes a `--host`:

```bash
docker compose run --rm redteam-client
# equivalently
docker run --rm --add-host host.docker.internal:host-gateway istana-rl \
  redteam-client --host host.docker.internal --port 8765 --steps 100
```

### Unreal in a container (optional, amd64, entitlement-gated)

```bash
git lfs install && git lfs pull                  # .uasset files must be real
docker login ghcr.io -u <github-user> -p <pat>   # Epic-linked GitHub account
docker compose --profile unreal run --rm unreal all
```

This is not part of the reproducible path, and three constraints are worth
stating plainly: the base image is pullable only by accounts linked to Epic and
admitted to the EpicGames organisation; it exists for amd64 only; and this
project has only ever been built and tested on Windows, so a Linux build is
plausible but unverified. The interactive packaged viewer stays a Windows
desktop application. For the live browser and 3D simulation, use the
[Windows setup guide above](#set-up-the-browser-interface-and-live-3d-simulation-windows).

### Troubleshooting

**`entrypoint.sh: no such file or directory`.** CRLF line endings on the shell
scripts. `.gitattributes` forces LF; re-clone, or run
`git add --renormalize . && git checkout -- docker/`.

**Permission denied writing output (Linux).** See the ownership note above.

**`exec format error`, or the wrong architecture after switching platforms.**
Compose reuses the `istana-rl:local` tag, so an image built for the other
architecture can linger. Force a rebuild with `docker compose build --no-cache`,
and confirm with `docker compose run --rm rl versions`, which prints the
architecture it is actually running on.

**A pin has no wheel for your platform.** pip names the package and stops rather
than attempting a source build in an image with no compiler. Build with
`--build-arg CONSTRAINTS=docker/constraints-current.txt`.

**Published-artifact tests fail after you edited `RL/BlueTeam/Results/`.**
Expected: those tests verify archived bytes. `git checkout -- RL/BlueTeam/Results`.

**The image is ~700 MB.** About 184 MB is the published evidence tree,
deliberately baked in so verification works with no network.

[docker/README.md](docker/README.md) has the longer version of all of this.

## Saved-layout switching demo (Windows)

The **original simulation console** now also offers saved native previews in its
Blue planner selector, beneath the unchanged live models. Start
`simulation_console.py --bridge-port 8766` against the capture-enabled Unreal
instance and open port 9048. In Live Unreal mode, connect, select an archived
output and click **Apply saved layout**. The actual native overhead image appears
in the existing viewer. The optional **Presentation view** expands the model
list and hides performance panels for recording. Saved previews do not run
inference or Red simulation; live model choices retain the normal planner flow.
Native preview audit records are written to `Saved/ConsoleModelDemo`.

For a presentation-only interface demo, `model_switch_demo.py` loads the archived
RL, greedy, and initial-policy layouts from
`RL/BlueTeam/Results/model-switch-demo` (small extracts of the original pilot,
included in Git). It does **not** run inference, deploy Red, advance the solver,
or compare performance. Each Apply resets the same seed and commits the exact
saved layout through native placement validation, then captures Unreal's fixed
overhead view and a sensor close-up. Sensor types/counts are preserved from each
saved output; all share the same catalogue and placement rules.

After building the Editor, start `Tools/start_blue_capture.ps1 -Port 8766`.
In a second PowerShell terminal:

```powershell
cd RL/BlueTeam/Python
../.venv/Scripts/python.exe model_switch_demo.py --output ../../../Saved/ModelSwitchDemo
```

Open `http://127.0.0.1:9051/`, select a saved model and click **Apply layout**.
**Record guided switching demo** uses those same Apply operations and records
the actual interface canvas, including native captures, into one H.264 MP4.
The output folder contains `model-switching.mp4`, the WebM source and a native
deployment log with source/frame hashes. Choose a fresh output folder to record
again; existing MP4s are not overwritten. No performance metrics are displayed.

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
