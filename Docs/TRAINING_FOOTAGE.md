# Training footage and evidence

New Training-tab runs retain the evidence needed to record the contractor
layout, individual sampled episodes, milestone policies and the selected best
policy. REINFORCE, Masked PPO and Masked A2C use the same recording format.
Recording does not require rerunning training or resampling a checkpoint.

## What can be filmed

| Shot | Support |
| --- | --- |
| Training timelapse with episode logs | Training-tab history and a top-down MP4 renderer; every completed episode has its actual layout, warning metrics, Red centers, event evidence and sparse native trajectory replay. |
| Contractor versus episodes 500, 1000, 1500 | Contractor and initial policy are evaluated before training; periodic checkpoint evaluations use the same fixed validation scenarios. Set the checkpoint interval to 500 for these milestones. Sampled episode recordings are available independently of checkpoint cadence. |
| Final best layout, with cinematic camera movement | Capture the retained best policy case in Unreal, then append a camera orbit around the frozen scene. Export with or without metric overlays. |
| Sensor and drone close-ups | Existing native capture views include sensor, drone, drone-model, overview and cosmetic showcase angles. New galleries show only the selected available limited-FOV sensor profiles and retain their catalogue specifications. |
| Sensor degradation in rain or reduced visibility | Native sensing already applies rain, visibility and humidity factors. These are simulation assumptions. A synchronized weather-preset demonstration with verified visible rain and paired clear/rain recordings is not supplied by the training recorder. |

An unchanged best layout is a valid recorded outcome. Do not label a lucky
training episode as general improvement: training scenarios differ. Use the
paired validation or separate final-test results for that claim. The best
observed sampled episode and best validated policy are different selections.

## Saved format

The console reports its run directory under `Saved/WarningTraining/console-*`.
It contains:

- `configuration.json`: selected sensors, initialization, optimizer and evaluation settings.
- `training.jsonl`: one flushed row per completed episode, including placements and warning metrics.
- `episodes/episode-000500.json`: the actual sampled episode 500, with the native context, Red context, placements, Red centers, warning evidence and recorded trajectories.
- `baseline-episode.json` and `initial-episode.json`: contractor and initial-policy cases before training.
- `policy-*.json`: initial and periodic optimizer checkpoints.
- `recording-manifest.json`: labels, paths, episode numbers, evaluation seed panels and aggregate metrics for the footage timeline.
- `best-episode.json` and, after publication, `best-test-episode.json`: selected best validation case and separate test case.

The log is useful while training is running. Completed episode files are
published atomically. Video generation defaults to the manifest's baseline,
initial policy, milestone evaluations and best results; it does not discard
unfavorable milestone results. `--episodes all` explicitly requests every
retained sampled episode. Large videos can take substantial time and disk.

Old CLI runs containing `summary.json`, `protocol` and `evaluation-*.json`
remain supported. Older console runs that did not save the recording manifest
and complete episodes cannot retroactively provide all missing layouts or
native evidence; start a new run for a complete film.

## Top-down timelapse with episode logs

From `RL/BlueTeam/Python`, using the project's Python environment:

```powershell
$run = '..\..\..\Saved\WarningTraining\console-YOUR-RUN'
..\.venv\Scripts\python.exe render_warning_timelapse.py $run --preview-only
..\.venv\Scripts\python.exe render_warning_timelapse.py $run
```

The default creates `warning-timelapse.mp4`, a poster, chapter metadata and an
HTML report in the run directory. It shows aggregate fixed-panel warning
results and a separate per-episode log chart. It is a **data replay**, not
Unreal camera footage. To make a rapid film of every sampled episode instead:

```powershell
..\.venv\Scripts\python.exe render_warning_timelapse.py $run --episodes all --seconds-per-stage 0.2
```

The movie output must not already exist; preserve or rename an earlier movie
before rendering another selection. A partial live run includes only completed
episodes, even if more episodes were requested.

## Native Unreal footage

Use a separate free bridge port, or stop the console's native session before
capture. Start a rendered scene with the **same** scenario flags, sensor
catalogue, weather and physical settings used for training. For a standard
delayed-detection workbench run, from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\start_blue_capture.ps1 -Port 8766 -DelayedDetectionDemo -TrainingWorkbench
```

Then from `RL/BlueTeam/Python`:

```powershell
$run = '..\..\..\Saved\WarningTraining\console-YOUR-RUN'
$capture = '..\..\..\Saved\WarningTraining\film-YOUR-RUN'
..\.venv\Scripts\python.exe capture_warning_3d.py $run --output $capture --port 8766 --detail-views
..\.venv\Scripts\python.exe render_warning_3d.py $capture
```

This defaults to the baseline, initial policy, checkpoint evaluations and best
results, with sensor gallery shots. Capture specific **sampled episodes** with
`--episodes 500,1000,1500`. These are distinct from milestone-policy evaluation
cases and retain their original training scenario. Missing episodes are an
error, not substituted with a nearby episode.

For a clean final-layout shot, use a new capture folder:

```powershell
$best = '..\..\..\Saved\WarningTraining\film-YOUR-RUN-best'
..\.venv\Scripts\python.exe capture_warning_3d.py $run --output $best --port 8766 --best-only --skip-gallery --cinematic-orbit
..\.venv\Scripts\python.exe render_warning_3d.py $best --clean
```

`--best-only` uses the separate test recording when present and otherwise the
retained best validation case. `--clean` produces `warning-3d-clean.mp4` without
burned-in captions so editors can add their own text. Keep `CREDITS.txt` with
the edited film. The camera orbit advances presentation frames only, with the
simulation frozen; it is not additional training or detection evidence.

Capture deploys saved actions directly and uses saved Red centers. It checks
the native physical/weather contract before replay and requires the complete
per-drone detection/arrival evidence to match afterward. A mismatch fails
capture rather than silently replacing the original metrics with a new score.
The manifest stores original metrics and replay measurements separately, plus
hashes of the native PNG frames.

## Equipment and weather footage

`capture_warning_3d.py --gallery-only` writes individual native sensor images
and `gallery.json`; it does not train or evaluate policies. When a console
manifest is present, the gallery uses exactly its selected limited-FOV sensor
IDs. Without a console manifest it defaults to the native scene's available
directional profiles. Omnidirectional profiles are excluded from new galleries.
For drone close-ups alongside the selected sensors, use `--detail-views` when
capturing the retained episode: the camera follows actual native drones before
returning to the sensor and overview views. Older cosmetic showcase scripts
use legacy props and are outside this selected-sensor recording workflow.

Current directional sensing computes weather attenuation in
`Source/IstanaOpen/Simulation/Sensing/DirectionalSensorModel.cpp`; the public
native snapshot records the weather values. Changing rendered rain alone does
not establish sensor degradation. A defensible weather demonstration needs
the same layout and Red scenario under saved clear/rain settings, matched
capture visuals, and the resulting probability/detection metrics.

Repository audit on 2026-09-24: the fetched `origin/weather-testing` points to
`c8d9d25`, an ancestor of the current application branch, with no unique
commits. This says nothing about a teammate's unpushed work or progress; no
teammate has been contacted by this change.
