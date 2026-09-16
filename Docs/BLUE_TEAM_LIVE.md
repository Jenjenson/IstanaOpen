# Blue + Red simulation console

## Validation status

The local browser console works now with all 18 unchanged published temporal-v6
replays. These are synthetic, offline Blue evaluations, **not recordings of the
Istana Unreal scene**. The map is a schematic objective-relative view, not a
geographical map of the grounds.

The experimental native integration passed both Game and Editor builds on
2026-09-16 with UE 5.5.4, MSVC 14.38 and Windows SDK 10.0.22621.0. All 18 native
`Istana.Simulation` automation tests passed, including Blue deployment guards
and the existing external Red bridge test.

Four end-to-end headless runs on `/Game/Maps/Istana` completed: the public
control and each explicitly selected checkpoint 406/407/408. Each deployed
two sensors and advanced 416 fixed steps (20.8 seconds), with measured terminal
metrics. These are one-seed smoke checks, not comparative policy evaluations.
Native test evidence is under `Saved/Automation/blue-integration-20260916`;
wire audits and reports are under `Saved/BlueLive/*native-20260916-*`.

The old downloadable landscape-only release does not contain this integration;
use the compiled source project's live launcher below.

## Open the console

From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\setup_blue_python.ps1
powershell -ExecutionPolicy Bypass -File .\Tools\start_simulation_console.ps1
```

Open <http://127.0.0.1:9048/>. Keep the console server running; Ctrl+C stops it.
The first setup command is only needed when creating/updating the environment.
It uses UE 5.5's bundled Python unless `-PythonExe` is explicitly supplied.

Recorded mode supports scenario/checkpoint/case selection, play/pause, speed,
single-frame stepping, restart, timeline scrubbing, sensor coverage, paths,
approved sites, sensor costs, an event log and explicitly labelled final metrics.
All cases, including failures, remain selectable. No retraining or selection of
a purported best checkpoint occurs.

## Live mode

Use UE **5.5.4**, the installed version associated with this project. Install the
compatible Visual Studio 2022 C++ build tools (MSVC v143 14.38, Windows SDK
10.0.22621.0, .NET Framework 4.8 SDK and targeting pack), then:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\build.ps1 -Target Editor
powershell -ExecutionPolicy Bypass -File .\Tools\start_blue_live.ps1
```

The native `Istana.Simulation.BlueTeam.InitialLayoutContract` test passed.
The map must contain exactly one Red Team Manager with a valid ObjectiveTarget.
`-IstanaBlueLive` creates/wires the optional Blue coordinator and loopback bridge
without editing the saved map; normal viewer/Red behavior remains opt-in unchanged.

In the console choose **Live Unreal → Connect to Unreal → Plan new episode**,
then **Run episode** or **Step**. Select an experimental checkpoint (406/407/408)
or **Greedy · non-RL**. Changing the selector applies on **Plan new episode**;
the active planner is shown separately. Greedy uses `TemporalPublicGreedy`,
repeatedly choosing the legal option with the greatest predicted marginal
return and stopping when no positive gain remains. It respects budget, site
count, separation and the native unsupported-site mask. `control` remains a
backward-compatible API alias; the CLI also accepts `--greedy`.
The Red demo places scripted radial
swarm centres; it is not a learned Red policy or joint adversarial training.

The browser displays 2D telemetry; the separate Unreal window renders the 3D
Istana scene. Native sensor markers/coverage are presentation only. A connection
failure clears live state and never substitutes recorded frames.

For a logged headless driver (Unreal must still be running):

```powershell
.\RL\BlueTeam\.venv\Scripts\python.exe .\RL\BlueTeam\Python\run_istana_live.py --temporal-public-control --paced --output-dir .\Saved\BlueLive\new-run
```

The output directory must not already exist. Use `--checkpoint` instead of
`--temporal-public-control` to select a published checkpoint explicitly.

## Integration contract and boundaries

- One owner, one fixed-step Red clock. Blue commits one initial profile/site
  layout before advancing; there is no mid-flight sensor relocation.
- Blue planner receives only validated public priors and public reported tracks.
  Private Red position/seed data is excluded from the planning envelope. The
  observer screen may separately display Red truth for debugging.
- Blue XY coordinates are objective-relative metres, Z is up. Native Red states
  are absolute Unreal centimetres; conversion uses the advertised objective.
- Synthetic sensing uses range falloff, weather, repeated confirmation looks and
  a deadline before objective-zone entry. No physical sensor control, weapons,
  interception or terrain occlusion is implemented.
- Live sensor mounts now rest on traced static collision surfaces. A downward
  complex trace samples the centre and four rim points of each 80 cm mount.
  Missing collision, steep surfaces (normal Z below .985), edges and height
  variation above 10 cm are rejected. The base uses the lowest sampled height
  to avoid floating; minor embedding on uneven ground is allowed. The visible
  mast connects that base to the profile's sensing height (normally 4 m).
  Unsupported sites are masked before planning and rejected again at deploy.
  Changed/lost support blocks stepping until reset. There is no imaginary
  fallback ground plane. `siteSurfacesWorldCm` exposes accepted bases; `null`
  means unsupported. Sensing and run-report positions use surface + mast height.
- The frozen offline planner/checkpoints still use their original 2D site and
  nominal-height forecasting model; surface feasibility is masked, but the
  forecast is not a terrain-aware performance estimate. Offline replays and
  trained artifacts remain unchanged.
- Native measured metrics/rewards are separate from archived offline metrics;
  no real-world calibration or transfer-performance claim follows from a run.
- Both servers bind to loopback. HTTP mutations require a same-origin session
  token. Bridge actions carry run/revision/request/step guards; transport failures
  are not automatically retried. Closing the console client cancels/freezes the
  native episode; the bridge also has a five-minute idle timeout.

Python regression tests (run from `RL/BlueTeam/Python`):

```powershell
..\.venv\Scripts\python.exe -m pytest tests/test_istana_live.py tests/test_simulation_console.py tests/test_recommend_temporal.py tests/test_temporal_inputs.py -q
```

These 90 checks passed on 2026-09-16. The tests cover the archived dataset, local
HTTP safeguards, controller failure behavior, public-only planning, all three
published checkpoints, coordinate conversion and malformed bridge transport.
The Unreal JSON boundary restores declared numeric types (e.g. `20` to `20.0`
for a double) before strict checkpoint validation. Checkpoint files and the
frozen offline policy contract are not modified. Mock tests complement the
separate native end-to-end runs described above.
The broader Python suite was started but stopped before completion; it is not
reported as passing.

Surface-placement update: both Game and Editor builds and all 18 native tests
passed again. Native assertions cover empty worlds, roof/ground elevation,
removed support, narrow ledges, steep roofs, and atomic airborne rejection.
The Istana map exposed 26 supported sites out of 32. A full greedy smoke run
accepted two mounted sensors and completed 416 steps; evidence is under
`Saved/BlueLive/greedy-surface-20260916` and
`Saved/Automation/surface-placement-20260916`. This is a functional smoke test,
not evidence that greedy outperforms a trained policy.
