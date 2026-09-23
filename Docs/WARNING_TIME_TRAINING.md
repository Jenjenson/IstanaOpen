# Native Blue warning-time training and timelapse

## Directional thermal training (current code)

The current live catalogue replaces the old 115 m omnidirectional thermal row
with the documented Boson+ directional profile. New
`istana.warning_directional_reinforce.v2` checkpoints learn profile, approved
site, yaw and pitch jointly; historical warning runs remain archival evidence
for their earlier radial contract and must not be relabelled as directional.
See [the sensor model](DIRECTIONAL_SENSORS.md) for the manufacturer/assumption
split and probability formula.

For a fresh long-approach directional run, start the scenario outside the 500 m
simulation boundary and train on its native terminal warning reward:

```powershell
# Repository root
powershell -ExecutionPolicy Bypass -File .\Tools\build.ps1 -Target Editor
powershell -ExecutionPolicy Bypass -File .\Tools\start_blue_live.ps1 -Port 8766 -DelayedDetectionDemo -TrainingWorkbench
cd RL\BlueTeam\Python
..\.venv\Scripts\python.exe train_warning_live.py --port 8766 --episodes 128 --batch-size 8 --eval-cases 8 --output ..\..\..\Saved\WarningTraining\my-directional-run
```

The browser console also has a **Training** tab for shorter interactive runs.
Start the scene on the console's configured bridge port with
`-TrainingWorkbench -DelayedDetectionDemo`, open `http://127.0.0.1:9048/`, and
select Training. Enter a unique model name, then choose an untrained start or a
five-directional-sensor public-only common-sense warm start. Warm starting
changes initial logits only; all selected and unselected actions remain
trainable. Select REINFORCE, Masked PPO or Masked A2C. Each optimizer consumes
the same complete native episode reward and legal categorical action records;
PPO and A2C add a placement-step value baseline without changing the simulator
or observations. DDPG is not exposed because the native contract is a discrete
choice among approved sensor/site/yaw/pitch options plus STOP. After each batch
update, the console evaluates that checkpoint on
the same fixed held-out native episode. A successful run saves the checkpoint
with the highest measured mean per-drone warning time under
`Saved/WarningTraining/models/trained-*` and immediately lists it in **Compare
placements** and the Live saved-layout selector. Its comparison replays that
best checkpoint and the fixed five-directional start with the same Red episode,
paths, speed, seed and sensing process. Browser run artifacts remain in
`Saved/WarningTraining/console-*`. Stopped and failed runs are not registered.
Use the CLI above for the full fixed multi-case evaluation protocol and audit
reports; the console's single held-out case is a selection aid, not a general
performance claim.

For native early-to-trained footage, close the training scene, start the capture
scene with the identical flag, and replay the fixed checkpoints:

```powershell
# Repository root
powershell -ExecutionPolicy Bypass -File .\Tools\start_blue_capture.ps1 -Port 8766 -DelayedDetectionDemo -TrainingWorkbench
cd RL\BlueTeam\Python
..\.venv\Scripts\python.exe capture_warning_3d.py ..\..\..\Saved\WarningTraining\my-directional-run --output ..\..\..\Saved\WarningTraining\my-directional-capture --port 8766
..\.venv\Scripts\python.exe render_warning_3d.py ..\..\..\Saved\WarningTraining\my-directional-capture
```

The native footage shows the physical sensor orientation and frustum. The film
records first detection, confirmation, target-zone arrival, warning time,
pixels-on-target and probability from terminal evidence. Checkpoint comparisons
reuse the same held-out Red seeds; measured regressions or ties are retained.
Do not claim improvement unless the generated paired summary actually shows it.

## Video provenance and result clarification

The presentation-only `native-20260918-quadcopter` video replays the **original
128-episode pilot**, not the later 256-episode experiment. Its single recorded
case has 12.3272 s team warning at every checkpoint (first detection 1 s,
first arrival 13.3272 s). This case statistic is not a held-out-set average.
The updated drone geometry is cosmetic only: no collision, navigation, sensing,
solver inputs or movement changes. Captures require exact original event evidence.

The separate `approach-v2-trained-20260918` result was 36.4155 to 41.1655 s
mean team warning across 16 paired held-out cases. The +4.75 s gain has a
95% bootstrap interval of [-0.5, 9.875] s, so it is not a confirmed positive
gain over the initial policy. Coverage fell from 100% to 52.9167%, and mean
per-drone warning fell from 33.4148 to 20.8164 s. Do not present this as an
overall successful improvement or put its means onto the old pilot's footage.

## Long-approach benchmark (v2)

The old September 17 pilot is retained, not overwritten. Its Red centers were
65 m from the objective, with 100–130 m synthetic sensors: all held-out cases
alerted on the first 1 s look. Its public forecast also described a different
320 m / 12 m/s scenario. Increasing the time limit alone cannot fix this.

`-IstanaWarningApproachV2` enables a **separate synthetic benchmark**:

- Red centers start 280 m away inside a 260–300 m allowed annulus. Five
  12-drone groups use a 0.4 rad fan around one of three synthetic sectors
  (0°, 90°, 180°; fixed probabilities 0.5, 0.3, 0.2), with seeded ±0.08 rad
  jitter. These are benchmark directions, not reconstructed real attack routes.
- Starts are 120 m above the objective. Initial low-altitude development
  probes encountered surrounding building collision hulls and unresolved
  navigation. Raising the start height gave complete approaches in the
  development route checks without disabling collision or changing terrain.
  The native collision-aware solver still controls motion toward the objective.
- Cruise speed, acceleration, drone count, all sensor capabilities/costs,
  budget, approved sites, surface validity, separation and the 20 m XY arrival
  boundary stay unchanged. The safety time limit is 180 s because the physical
  path is longer; the development arrivals were approximately 92 s.
- Public speed, altitude, spawn-radius and sector priors match the new scenario.
  No private per-episode Red coordinates enter the Blue actor.

The **v2 reward is team warning**, in seconds:
`max(0, first_20m_zone_entry - first_detection_of_any_Red)`. An entirely
undetected approach gets zero. Mean per-drone warning and detection fraction
are separate secondary metrics; they are not interchangeable with team warning.
The endpoint is still **arrival, not physical impact**, as requested. Unresolved
targets cause an error, never an optimistic reward or silently dropped case.

Every episode logs each drone's initial nearest deployed-sensor distance,
whether it starts in range, and a conservative clearance from **every**
supported site's available maximum sensor range (including combinations the
budget would disallow). Empty layouts log a null nearest distance, not zero.
The runner rejects starts with less than 50 m universal clearance and rejects
first-look saturation. First-detection times, first-look counts and fractions
are recorded as well.

All checkpoint comparisons use identical held-out Red/action seeds. They check
hashes of actual initial states, sampled positions **and velocities** every 5 s,
all arrival timestamps, and the frozen physical contract. Sensor randomness is
keyed by seed/site/drone/look, not by mutable RNG call order. Checkpoints are
0 (untrained), 1 (no update yet), and four fixed training milestones. No easiest
episode, best checkpoint, or favorable visual case is selected afterward.
Case 0 is preselected for the film; the report averages all held-out cases.

The fixed development comparison used two legal three-EO layouts with identical
cost 2.1. On one north-sector case, changing one site improved team warning by
9 s (24.07 → 33.07 s), with exactly matched sampled trajectories and arrivals.
The other three development cases tied. This proves placement sensitivity in
that test, **not a learned-policy gain**. Failed low-altitude development runs
remain in `Saved/WarningTraining/approach-v2-preflight-*`.

### Run the new experiment

```powershell
# Repository root; four independent local simulation clocks, no shared bridge.
powershell -ExecutionPolicy Bypass -File .\Tools\build.ps1 -Target Editor
8767..8770 | ForEach-Object { powershell -ExecutionPolicy Bypass -File .\Tools\start_warning_training.ps1 -Port $_ }
cd RL\BlueTeam\Python
..\.venv\Scripts\python.exe train_warning_approach.py --workers 4 --episodes 256 --batch-size 8 --eval-cases 16 --output ..\..\..\Saved\WarningTraining\my-approach-run
..\.venv\Scripts\python.exe audit_warning_run.py ..\..\..\Saved\WarningTraining\my-approach-run
```

Wait for each native port to be ready before launching Python. Workers only
accelerate independent episodes: actions are sampled serially before each
batch, results are ordered by seed, and the optimizer updates synchronously.
One worker on one port is supported. Every run gets a new output folder, its
source snapshot, native DLL hash and protocol written before evaluation.
Training seeds, held-out seeds and development seeds are disjoint. A failure
is retained with native evidence rather than automatically retried as a new
sample. This is a single-training-seed pilot, not proof of generalization.

### Clean 3D footage (no obstructing markers or burned-in statistics)

```powershell
# Repository root; use a free port distinct from active training workers.
powershell -ExecutionPolicy Bypass -File .\Tools\start_blue_capture.ps1 -Port 8766 -WarningApproachV2
cd RL\BlueTeam\Python
..\.venv\Scripts\python.exe capture_warning_3d.py ..\..\..\Saved\WarningTraining\my-approach-run --output ..\..\..\Saved\WarningTraining\my-raw-capture --port 8766 --step-batch 20
..\.venv\Scripts\python.exe render_warning_raw.py ..\..\..\Saved\WarningTraining\my-raw-capture
..\.venv\Scripts\python.exe serve_warning_report.py ..\..\..\Saved\WarningTraining\my-raw-capture --port 9050
```

The clean MP4 contains only native 1920×1080 frames: no boxes covering the
models, no label panels, no graph over the scene. Sensor close-ups and a drone
tracking camera make actual meshes visible; the overview shows placement.
Metrics, simulation clock, chapter controls and the results table sit **outside**
the video in the HTML report. “Hide statistics / enlarge footage” removes the
side panel. Timing metadata is a separate JSON sidecar, so a downloaded MP4
remains clean. The intended replay speed is 8×, identical for every checkpoint.
Native playback evidence must exactly match the original evaluation before a
capture is accepted. Earlier videos with markers must be recaptured; the
renderer refuses to pretend it can remove baked-in boxes.

For interactive replay, use `start_blue_live.ps1 -WarningApproachV2`, then
`replay_warning_live.py --checkpoint <run>/policy-0256.json`. It reads the
scenario and default held-out seeds from the run's protocol and rejects a
mismatched scene.

## Original short-approach pilot (v1)

This is a **separate from-scratch RL pilot**, not a continuation or replacement
of the published temporal checkpoints. It trains against the native Istana
simulator. Red uses seeded rotations of scripted radial swarms, **not a learned
Red policy or self-play**. Existing public priors and sensor capabilities stay
unchanged. Sensors must pass the native supported-surface placement checks.

## Metrics and learning objective

- **Team warning:** first arrival of any Red drone minus first detection of any
  Red drone, clamped to zero. This measures initial preparation time.
- **Per-drone warning:** that drone's zone-entry time minus its first detection,
  clamped to zero, averaged over **all** drones. Undetected arrivals contribute
  zero, not a missing value. **This is the training reward**, in seconds.
- Arrival is the interpolated first crossing of the **20 m objective zone** in
  XY. It is **not a simulated collision/crash into the building**. Detection
  is a first hit, not the later confirmation event.
- Unresolved episodes stop this training runner with an error rather than
  dropping difficult targets. Native observation aggregates are explicitly
  lower bounds when some arrivals are unresolved.
- Terminal `warningEvidenceForEvaluationOnly` contains per-drone event times.
  It never enters `publicSnapshot` or the actor's inputs. Evaluation recomputes
  both metrics and checks them against native aggregates.

The map-specific policy uses masked categorical type/site preferences with a
learnable STOP option. All initial logits are zero: it starts with uniform
probability over legal actions, not a hand-picked poor layout. REINFORCE uses
complete-episode warning rewards, a leave-one-out batch baseline, gradient
clipping and Adam. It receives only the validated public layout and legality
mask. Checkpoints include weights, optimizer state and random state.

## Reproduce a new run

Use the Windows setup in the main README and rebuild the editor after adding
the native warning instrumentation. Commands below start in the repository root.
The visible launcher works for training too; don't connect the browser console
to the same bridge while the trainer owns it.

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\build.ps1 -Target Editor
powershell -ExecutionPolicy Bypass -File .\Tools\start_blue_live.ps1 -Port 8766
cd RL\BlueTeam\Python
..\.venv\Scripts\python.exe train_warning_live.py --port 8766 --episodes 128 --batch-size 8 --eval-cases 8 --output ..\..\..\Saved\WarningTraining\my-new-run
..\.venv\Scripts\python.exe audit_warning_run.py ..\..\..\Saved\WarningTraining\my-new-run
..\.venv\Scripts\python.exe -m pip install pillow imageio-ffmpeg
..\.venv\Scripts\python.exe render_warning_timelapse.py ..\..\..\Saved\WarningTraining\my-new-run
..\.venv\Scripts\python.exe serve_warning_report.py ..\..\..\Saved\WarningTraining\my-new-run --port 9050
```

Open [the local report](http://127.0.0.1:9050/). The MP4 is a **top-down replay of
recorded native trajectories**, with actual accepted sensor positions and
event timestamps; it is not 3D camera footage. Its map shows held-out case 0,
chosen before any outcomes. Performance cards average all held-out cases.
Buttons jump to each fixed checkpoint. Gray trajectories/drones after zone
entry are visual context, not new sensing rewards.

The runner refuses an existing output folder. It writes its protocol before
training, retains all 128 episodes, saves checkpoints 0/32/64/96/128, and
evaluates the same eight Red/action-seed pairs at each checkpoint. Training,
evaluation and smoke-test seeds are disjoint. Evaluation does not update the
policy or consume its training random stream. No best-checkpoint selection,
automatic tuning, or resume is performed. Scripted approaches are rotated
within the scene's existing permitted spawn annulus/height.

Artifacts include `protocol.json`, `native_context.json`, `training.jsonl`,
`policy-*.json`, `evaluation-*.json`, `summary.json`, `warning-timelapse.mp4`,
`index.html`, `audit.json` and `artifact-sha256.json`. The audit verifies source
hashes, episode completeness, identical paired Red arrival times, recomputed
warning metrics, and reproduction of every checkpoint's evaluated layouts.
They stay under ignored `Saved/`;
copy them elsewhere if you want to archive/share the experiment. The source
hashes and base commit identify the training implementation; generated artifact
hashes detect subsequent changes. No artifacts are pushed automatically.

## Replay the learned placement in the actual 3D scene

Close the training scene or use a different free bridge port. Start the normal
visible launcher on 8765, with the browser console disconnected, then:

```powershell
# From RL\BlueTeam\Python; replace my-new-run with your output directory.
..\.venv\Scripts\python.exe replay_warning_live.py --checkpoint ..\..\..\Saved\WarningTraining\my-new-run\policy-0128.json
```

This reuses the displayed held-out Red/action seeds, deploys the saved policy's
layout, then advances native simulation in real time. The console's older
406/407/408 dropdown does not load this separate policy format.

## Interpreting improvement honestly

Compare the final fixed checkpoint against both the initial random legal
policy and the existing non-RL temporal-public greedy control. Report signed
changes and paired bootstrap intervals, including regressions. A single
training seed and eight scenarios are a pilot, not a robust generalization
study. The bootstrap describes scenario variation, not training-seed variation.

The current scene is easy to detect: a team-level hit can occur on the first
one-second sensing look even before learning. That puts team warning at its
first-look ceiling. Average individual warning may improve while the initial
team alert does not. Do not call those different outcomes the same gain.
The historical pilot described in this section used synthetic radial sensing.
Current directional thermal sensing adds Unreal world-static LOS, but its
probability curve remains a documented simulation assumption. Nothing here
establishes real-world warning time, interception success or crash behavior.

## Capture the actual 3D scene with detailed sensor models

The native sensor assemblies now use telescoping masts, bracing, anchored base
plates, equipment enclosures, fasteners and profile-specific EO/thermal/radar/RF
heads. They reuse authored PBR materials and engine meshes, so no marketplace
download is needed. They remain cosmetic: every component has collision and
navigation disabled. Approved surface, sensing height, costs and probabilities
are unchanged.

After training has finished, close its native scene or choose a different port.
From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\build.ps1 -Target Editor
powershell -ExecutionPolicy Bypass -File .\Tools\start_blue_capture.ps1 -Port 8766
cd RL\BlueTeam\Python
..\.venv\Scripts\python.exe -m pip install pillow imageio-ffmpeg
..\.venv\Scripts\python.exe capture_warning_3d.py ..\..\..\Saved\WarningTraining\my-new-run --output ..\..\..\Saved\WarningTraining\my-3d-capture --port 8766
..\.venv\Scripts\python.exe render_warning_3d.py ..\..\..\Saved\WarningTraining\my-3d-capture
..\.venv\Scripts\python.exe serve_warning_report.py ..\..\..\Saved\WarningTraining\my-3d-capture --port 9050
```

The capture scene renders offscreen with the GPU; **do not use `-nullrhi`**.
The `blue_capture` bridge operation is disabled unless `-IstanaAllowCapture` is
present. It validates the current episode/step, supports only an overview or a
deployed sensor close-up, and generates unique PNG paths under `Saved/BlueCapture`.
It does not accept arbitrary output paths or overwrite existing captures.

The recorder pauses the simulation clock while frames are written. It replays
all fixed checkpoints using the original preselected evaluation case and fails
if any per-drone detection/arrival evidence changes. Small colored markers are
observer overlays at native positions, not policy inputs. The film combines
actual scene frames with data labels; scene imagery is not AI-generated.
Sensor showcase resets are separate from the training/evaluation results.

Raw PNGs and `capture-manifest.json` preserve frame hashes. The 3D capture folder
contains the MP4, model preview, native replay records, report, video chapters,
and artifact hashes. The original training is replayed, not silently rerun or
selected for improvement. Do not interpret prettier models as improved RL.

The September 17 pilot remains unchanged. Before the visual update, its five
source-hashed files were preserved under its `training-source-snapshot` folder;
subsequent native presentation changes should be audited against that snapshot
and the exact replayed event evidence, not claimed to be the original binary.
