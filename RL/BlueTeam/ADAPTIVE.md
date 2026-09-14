# Adaptive Blue Team sensor placement

This is a new **synthetic sensing experiment**, separate from the historical
210-episode Unreal checkpoint. It trains one policy to choose both a sensor
profile and a legal deployment site, conditioned on a new scenario and the
layout it has already selected.

The objective is timely, confirmed detection before a threat reaches a
protected objective. There are no weapons, interception controls, or physical
deployment commands. Reported "defence" is an abstract sensing outcome.

The [measured results](ADAPTIVE_RESULTS.md) show 62.5% success on 200 unseen
randomized scenarios, versus 14.5% for the projected original policy and 67.5%
for greedy coverage. Severe stress performance is still poor; the broader
robustness objective remains open.

## Run the trained policy

Use the [Python setup](README.md#try-it-without-unreal), then run from the
repository root (PowerShell):

```powershell
.\RL\BlueTeam\.venv\Scripts\python.exe .\RL\BlueTeam\Python\demo_adaptive.py `
  --checkpoint .\RL\BlueTeam\Checkpoints\adaptive-v1 --episodes 6 `
  --output .\Saved\BlueRL\adaptive-demo.html
```

Open `Saved/BlueRL/adaptive-demo.html` in a browser. Choose a recorded scenario,
play/pause or scrub the approach, and inspect the selected sensor types,
coordinates, nominal ranges, detection links, threats, outcome and decomposed
reward. The page is self-contained and needs no server or internet connection.
It replays an actual frozen-policy run; it does not run inference in JavaScript.

To infer a complete layout from an external-format snapshot without Unreal:

```powershell
.\RL\BlueTeam\.venv\Scripts\python.exe .\RL\BlueTeam\Python\recommend_adaptive.py `
  --checkpoint .\RL\BlueTeam\Checkpoints\adaptive-v1 `
  --input .\RL\BlueTeam\Examples\public-snapshot.json --now 0 `
  --catalogue .\RL\BlueTeam\Examples\sensor-catalogue.json `
  --output .\Saved\BlueRL\recommendation.json
```

The example's clock is zero. For current observations, pass `--now` in the
provider's clock so old tracks are not treated as fresh. Supply your own legal
sites and calibrated capabilities through the [input contract](ADAPTIVE_INPUTS.md).

## Train and reproduce a comparison

```powershell
.\RL\BlueTeam\.venv\Scripts\python.exe -m triad_rl.train_adaptive `
  --episodes 4000 --batch-size 16 --seed 43 `
  --validation-every 800 --validation-episodes 100 `
  --output .\Saved\BlueRL\adaptive-run-43

.\RL\BlueTeam\.venv\Scripts\python.exe .\RL\BlueTeam\Python\evaluate_adaptive.py `
  --checkpoint .\Saved\BlueRL\adaptive-run-43\best `
  --episodes 200 --split heldout --seed 2000000000000000 `
  --output .\Saved\BlueRL\heldout.json
```

Run a shifted stress suite separately with `--split stress` and a new reserved
seed range. An evaluation against the bundled model should explicitly supply
`--initialized-checkpoint .\RL\BlueTeam\Checkpoints\adaptive-v1-initialized`.
Reports also include the preserved legacy checkpoint, fixed RF/radar, random
and greedy non-RL methods. Do not tune on final held-out results and then reuse
them as independent test evidence.

To reproduce the **published selected model** comparison without retraining:

```powershell
.\RL\BlueTeam\.venv\Scripts\python.exe .\RL\BlueTeam\Python\evaluate_adaptive.py `
  --checkpoint .\RL\BlueTeam\Checkpoints\adaptive-v1 `
  --initialized-checkpoint .\RL\BlueTeam\Checkpoints\adaptive-v1-initialized `
  --selection-report .\RL\BlueTeam\Results\adaptive-v1\common-validation-selection.json `
  --episodes 200 --split heldout --seed 2000000000000000 `
  --output .\Saved\BlueRL\published-model-comparison.json
```

Training saves `initialized/`, validation-selected `best/`, latest `last/`,
configuration, JSONL metrics and a summary. Resume with the same arguments and
`--resume ...\last`, increasing the total `--episodes`; exact resume requires
a full batch boundary and unchanged implementation/configuration fingerprints.
Use a fresh output directory for a new run.

For an optional local metrics dashboard, install `trackio` in a separate
environment and run `RL/BlueTeam/Python/track_adaptive.py --run-dir RUN_DIR --name RUN_NAME`.
Then run `trackio show --project blue-team-adaptive`. This imports the existing
JSONL record; Trackio is not a training dependency and no cloud sync is used.

## What changes from the original experiment

| Original experiment | Adaptive experiment |
| --- | --- |
| One northbound drone on a fixed path | Seeded directions, altitudes, speeds, path families and swarm sizes |
| Fixed sensing conditions | Changing emission behaviour, visibility and weather |
| A catalogue choice and continuous XY head | Joint sensor/site choices over a variable candidate set |
| Pooled placements and scenario bounds | Public threat estimates, weather, budget, placements and coverage features |
| Evaluation seeds repeat the same scenario | Separate training, validation, unseen evaluation and shifted stress scenarios |
| Live TRIAD bridge | Portable simulator plus a backend-neutral observation adapter |

The site set is not a learned fixed map. Training samples rotated sites in a
fictional local East/North arena; an input provider can supply its own legal
site coordinates. The policy scores sensor/site combinations with shared
weights. Adding or reordering profiles or sites does not resize the model,
provided the input feature meanings and supported modalities are unchanged.
This is candidate-site placement, not unrestricted continuous optimisation.

## Decision and sensing boundaries

1. An input provider supplies a versioned public snapshot: legal sites, sensor
   capabilities, existing placements, budget, threat estimates and environment.
2. The common observation builder estimates coverage from that snapshot and
   constructs masked sensor/site options, including a final stop option.
3. A small NumPy actor-critic chooses an option, updates the remaining layout
   state, and repeats until it stops or exhausts the legal budget/site choices.
4. The simulator runs the approach and measures detection, confirmation,
   coverage, resource costs and missed deadlines.

The placement policy does not receive exact future trajectories or the
simulator's random detection draws. Public threat estimates are prior inputs,
not magically generated from the sensors that have yet to be placed. A real
provider must supply those estimates or explicitly represent their absence.
Replay ground truth is for visualisation and scoring only.

The policy chooses an initial coordinated layout. It does **not** relocate
sensors during an approach. It can be called again with a new public snapshot,
including already deployed sensors, but moving hardware is outside this code.

## Learning and evaluation

Randomised training varies approach direction, multiple approach groups,
altitude, actual path speed, direct/curved/weaving trajectories, periodic
emission duty cycles, visibility, illumination, rain, humidity and RF noise.
It uses one to five threats, two deployment rings with a random rotation,
varying budgets from 2.0 to 4.0 units, and occasional unavailable sensor types.
Stress evaluation shifts to larger swarms, higher/faster targets, worse weather
and more divergent approaches. These shifts include physically difficult or
unobservable threats; the simulator does not guarantee every scenario is solvable.

The terminal reward is explicit and inspectable:

| Component | Contribution |
| --- | --- |
| Timely confirmed fraction / missed deadline fraction | +8 / −8 per fraction |
| Ever-detected fraction / confirmed fraction | +2 / +1 per fraction |
| Earlier confirmation | +3 × mean remaining approach-time fraction |
| Expected forecast coverage / its complement | +2 / −2.5 per fraction |
| Sensor cost | −0.55 × spent units |
| Near-zero marginal forecast contribution | Up to −0.3 per deployment |
| Invalid actions | −1 per attempt; repeated attempts terminate |

Confirmation requires two detections within three one-second ticks, at least
four seconds before reaching the objective for a successful defence.
Coverage is expected **per-look detection probability over forecast ingress
points**, not a measured fraction of terrain or a guaranteed detection rate.
Incremental coverage shaping is removed from the terminal payment, so the sum
of undiscounted step rewards equals the recorded reward-component sum.

The actor is a shared tanh network over each joint option. The critic pools
legal option representations. Training uses sampled episodes, discounted
reward-to-go, a learned value baseline, entropy regularisation, Adam updates,
and gradient clipping. The policy is not a renamed greedy coverage rule.

Validation selects a checkpoint without using final evaluation scenarios.
All evaluation methods run on paired scenario seeds with independent policy
randomness. Reports retain per-episode results and subgroup metrics; paired
bootstrap intervals quantify differences on the evaluated scenario sample.
These intervals do not establish reliability on a real deployment population.

Comparison methods include a fresh untrained policy, random legal placement,
a fixed RF/radar layout, a public-information greedy coverage heuristic, and
the preserved `Checkpoints/toy-210` policy. The old policy requires an explicit
lossy projection into its original observation and action contract. This is
a transfer comparison in the **new simulator**, not a rerun of the native
Unreal benchmark. Its earlier reported success rates remain in [RESULTS.md](RESULTS.md).

## Real-input boundary and limitations

Simulated inputs and external sensor snapshots use the same observation
builder and policy. External input is validated before inference. The adapter
does not connect to devices, authenticate a sensor feed, implement calibration,
or issue physical commands. A device-specific integration must populate the
documented contract and be separately tested for timestamp, coordinate-frame,
uncertainty, capability and identity correctness.

Sensor effects are analytical assumptions, not manufacturer specifications or
measured field performance. No terrain occlusion, realistic multipath,
hardware field of view, installation feasibility or communications latency is
validated. Real deployment needs those models and independent safety review.

The native Istana Open Red Team Manager is still a separate integration. See
[INTEGRATION.md](INTEGRATION.md); this work does not silently replace its
schema-1 interface or claim the new checkpoint works in the palace scene.
