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

**Compare placements** adds paired offline native evaluations of each archived
RL layout. Select either the original matched three-sensor common-sense layout
or the measured five-directional-sensor workbench start. Each pair uses matched
Red paths and seeds; the interface explicitly notes that the archived RL layouts
have not yet been retrained for the five-sensor contract. Unreal is not required
for playback. See the [comparison guide](SENSOR_PLACEMENT_COMPARISON.md).

**Training** runs the current native directional warning-time learner through
the same bridge. It shows episode progress, warning history, actual detection
fraction, the latest placement and retained checkpoint/output location. It is
an interactive run console, not a replacement for the audited CLI protocol.
The user supplies a unique model name before starting. After each policy update,
the console scores that checkpoint on the same fixed held-out native episode;
on successful completion it publishes the best checkpoint and its matched
selected-count directional-sensor comparison. The Training tab accepts an exact
count of one to five sensors and 4–10,000 episodes; all placements retain the
native budget, site, separation, surface and FOV rules. The named model then appears immediately in
**Compare placements** and as a saved-layout choice in **Live Unreal**.
The algorithm selector offers REINFORCE, Masked PPO and Masked A2C. They share
the exact environment, reward, warm starts, action mask and deployment contract.
PPO uses clipped categorical actor updates with a placement-step critic; A2C
uses synchronous advantage actor-critic updates. DDPG is intentionally omitted
because this console selects discrete approved mount/type/orientation actions,
not unconstrained continuous positions.

## Live mode

Use UE **5.5.4**, the installed version associated with this project. Install the
compatible Visual Studio 2022 C++ build tools (MSVC v143 14.38, Windows SDK
10.0.22621.0, .NET Framework 4.8 SDK and targeting pack), then:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\build.ps1 -Target Editor
powershell -ExecutionPolicy Bypass -File .\Tools\start_blue_live.ps1 -DelayedDetectionDemo -TrainingWorkbench
```

The native `Istana.Simulation.BlueTeam.InitialLayoutContract` test passed.
The map must contain exactly one Red Team Manager with a valid ObjectiveTarget.
`-IstanaBlueLive` creates/wires the optional Blue coordinator and loopback bridge
without editing the saved map; normal viewer/Red behavior remains opt-in unchanged.
The optional delayed-detection demo changes only this live run's permitted Red
spawn annulus from the map's 30–100 m to 560–580 m. With the map's 10 m swarm
spread, 45 m outer sensor sites and the Boson profile's 500 m simulation
evaluation boundary, every initial drone has at least 5 m conservative clearance;
the scripted radial control starts at 570 m and becomes observable as it moves
inward. The 500 m boundary is a documented simulation assumption, not a rated
camera detection range. The frozen temporal public
prior and 406/407/408 checkpoints are intentionally unchanged. Omit the switch
for live training/evaluation against the original map scenario; the standalone
viewer and `-IstanaWarningApproachV2` benchmark are also unaffected.

`-TrainingWorkbench` changes the opt-in live run's Blue allowance from three to
five sites/cost units. It leaves the catalogue, approved sites, surface
validation, Red speed/motion model and sensing model unchanged. It uses one
drone in each of five seeded full-circle random-bearing groups at the same 570 m
spawn radius so native training episodes vary without large-swarm congestion.
The radial approaches use the existing synthetic 120 m benchmark altitude, keeping
the exercise above local obstacle-avoidance geometry rather than tailoring
routes to the visual backdrop.
It evaluates the documented 60 Hz camera model at a conservative 5 Hz and uses
a 100 s episode, within the native 512-look validation bound.
In the browser choose **Training**, then select one of:

- **Untrained random policy** — zero initial actor logits.
- **Five directional sensors · balanced approaches** — the fixed public-only
  common-sense rule spread across representative full-circle coverage bearings.
- **Five directional sensors · public-prior weighted** — the same fixed rule
  using the scene's published approach weights.

Both common-sense choices use only the limited-FOV directional profile and
initialize the trainable policy logits. They do not lock the five placements.
Five static 24° fields of view cannot cover every direction continuously, so
the layout is described as sensible sector coverage—not a universal detection
guarantee—and the actual detected fraction is recorded per native episode.
Only one owner may use a bridge: while Training is active, Live Unreal controls
are disabled. Safe stop takes effect after the current native action. Artifacts
are written to `Saved/WarningTraining/console-*`; completed named models are
published atomically to `Saved/WarningTraining/models/trained-*`. Stopped or
failed runs remain in their run directory but are not added to the model lists.

In the console choose **Live Unreal → Connect to Unreal → Plan new episode**,
then **Run episode** or **Step**. Select an experimental checkpoint (406/407/408)
or **Greedy · non-RL**, or **Common sense · non-RL**. Changing the selector applies on **Plan new episode**;
the active planner is shown separately. Greedy uses `TemporalPublicGreedy`,
repeatedly choosing the legal option with the greatest predicted marginal
return and stopping when no positive gain remains. It respects budget, site
count, separation and the native unsupported-site mask. `control` remains a
backward-compatible API alias; the CLI also accepts `--greedy`.
Common sense uses a simpler public ingress-coverage rule with weather, cost and
overlap adjustments; it is an explainable heuristic, not a measured human
participant or a trained policy. The CLI accepts `--common-sense` as an alternative
to `--checkpoint` or `--greedy`.
Red placement defaults to the scripted radial control, while the CLI may
explicitly select a learned, random wedge or dispersed demo policy. See
[Red initial-placement RL](RED_TEAM_RL.md). This remains separate Red training
against frozen Blue, not joint adversarial training.

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
- Directional thermal sensing uses a 3D yaw/pitch frustum, dynamic
  pixels-on-target, smooth distance/weather/angle degradation and Unreal
  world-static LOS traces. Its 500 m boundary and probability parameters are
  simulator assumptions. Legacy RF/radar/EO/fused profiles remain radial.
  Repeated confirmation and the objective-zone deadline are unchanged. No
  physical sensor control, weapons or interception is implemented.
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
  forecast is not a terrain-aware performance estimate. When used with the new
  catalogue, a labelled public-prior adapter supplies orientation because those
  checkpoints never learned yaw/pitch. New native warning policies use the
  versioned joint profile/site/yaw/pitch action space. Offline replays and
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
