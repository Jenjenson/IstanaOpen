# Compare reinforcement learning with sensible sensor placement

The simulation console can evaluate a trained RL layout against a simple,
explainable common-sense baseline or a layout that you place yourself. Open
**Compare placements** in the existing console. No Unreal process or training
run is needed for this comparison.

## Start the console

From the repository root, use the existing setup and launcher:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\setup_blue_python.ps1
powershell -ExecutionPolicy Bypass -File .\Tools\start_simulation_console.ps1
```

If Unreal is not installed, pass your Python 3.11+ executable to setup using
`-PythonExe 'C:\Path\To\python.exe'`. Open <http://127.0.0.1:9048/> and leave
the console server running. On other platforms, install and serve directly:

```sh
python -m pip install -e './RL/BlueTeam/Python[test,media]'
python RL/BlueTeam/Python/simulation_console.py
```

## Demonstrate a comparison

1. Choose **Compare placements**, then choose the swarm size, scenario conditions,
   RL checkpoint and case. **Large swarm · 60 drones** is the default demonstration;
   **Training range · 8 drones** uses the upper end of the available Temporal
   models' training population. **Original recorded swarm** retains all 18
   published cases at their original counts.
2. Keep **Common-sense layout** selected to use the suggested layout, or choose
   **My manual layout** to start with an empty map. Pick a sensor card, then click
   or tap the placement area on the map. The editor snaps to an approved location
   and shows the sensor's coverage before placement when using a mouse. No
   coordinate entry is needed. Use **Remove** to remove a sensor, or **Undo** to
   reverse an edit. **Use common-sense suggestion** provides a starting layout.
   The enlarged **Placement area** makes tapping easier. Amber arrows indicate
   distant public reports; **Full scene** shows their reported positions.
3. Evaluate the layouts. The results table compares detected adversary counts
   and rates, confirmation, timely confirmation and deployment cost.
   Positive RL-minus-baseline detection differences favor RL;
   negative differences favor the baseline. Ties are valid outcomes.
4. Use playback and the method selector to inspect both layouts at the same
   simulation time. Sensors and their coverage change, while adversary routes
   remain the same.
5. Choose **Edit my layout** or another case to try another comparison. Replay maps
   are read-only; editing returns to the placement preview and clears the previous
   result until you evaluate again. Budget is shown as a percentage, and the editor
   handles cost and spacing limits. The keyboard placement controls provide an
   alternative to clicking the map.

## Drone counts and training provenance

The three Temporal checkpoints in this console (406, 407 and 408) trained with
**1–8 drones**: normal cases contain 1–5, stress cases contain 3–8, and the
capability-varied cases use those same population families. Eight is within the
observed training population; it does not make every newly generated case a
reproduction of training. The generators are in `triad_rl/adaptive_env.py` and
`triad_rl/robust_scenarios.py`; training seed ranges are retained in the
`Results/temporal-v6-pilot` protocols.

The separate native warning-time training is documented with five groups of
twelve drones (60). Its map-specific policy, simulator and objective differ from
the Temporal experiment. The native trained weights and full run context are not
included in this repository's saved-layout preview fixtures.

The **60-drone mode is a new synthetic stress demonstration of the available
Temporal models**, outside their training population. It matches the documented
native drone count, not the native training experiment. It does not claim that
these Temporal models were trained on 60 drones. Reproducing the native experiment
requires its original checkpoint, run context and matching native simulator.

Both generated modes build a seeded new swarm and public reports, then run fresh
inference through the selected trained Temporal checkpoint. They do not reuse
the old small-case RL placements or simply draw extra copies of the same drone.
The common-sense and manual layouts are scored against the exact same generated
swarm as RL. Changing swarm size, case or checkpoint clears old results.

The editor receives public planning information and suggested placements before
evaluation. Future adversary trajectories and RL deployment are not included in
that editor response. After evaluation, the observer replay exposes the simulated
trajectories. A human can of course learn from previous attempts; repeated edits
on a revealed case are a demonstration, not a blind human-performance experiment.

## What common sense means here

This is a fixed rule that approximates sensible placement; it is not a claim
that a human participant made these choices. The planner:

- Covers likely approach directions using the public ingress priors.
- Uses the catalogue's declared range and strength, adjusted by public weather
  and emitter likelihood.
- Prefers additional coverage per unit cost and discounts overlap with sensors
  already placed. Equally scored options prefer greater separation, then stable
  catalogue/site order.
- Stops when no legal option provides useful additional predicted coverage.

Each suggestion includes its rationale. The numerical rule is
`marginal_coverage / cost * (1 - 0.5 * overlap / candidate_coverage)`. These are
static public coverage estimates, not measured future detection rates. This
baseline does not run RL, train on the selected case, inspect hidden adversaries,
or use the temporal greedy optimizer's predicted return. The existing **Greedy**
planner remains a separate method.

Common-sense and manual placements share the learner's placement validator:
sensor availability, approved and blocked sites, deployment radius, minimum
separation, sensor-count limit and budget. Manual input carries only sensor and
site identifiers; coordinates and costs are resolved by the server. An explicitly
empty manual layout is valid and produces zero detections.

## How the comparison stays fair

For original recorded swarms, the server copies the selected archived scenario
and independently rescores the RL's saved action sequence and the baseline
through the same existing sensing engine. Generated swarms use fresh RL inference
on the new public observations. For both methods, each comparison preserves
adversary identities, paths, emitter behavior, weather,
timing, sensor catalogue, budget and constraints. Sensing draws are indexed by
scenario seed, tick, target and modality, so different sensor counts do not shift
the random stream. No scenario is regenerated or selected based on who wins.

The response records the scenario/catalogue hash, shared randomness policy,
layout results and RL-minus-baseline differences. Original replay files and
checkpoint files are unchanged. The comparison is labelled as a new synthetic
evaluation; it is not appended to the published evaluation evidence. A single
case is illustrative, not evidence that either method is better in general.

The replay arena is schematic and uses the published offline analytical sensor
model, not the native Istana terrain. Sensor ranges are nominal visualization
aids; weather, range falloff and sensing probability determine scored detections.
Confirmation and timely confirmation are separate from simply detecting a target.
No interception is simulated.

## Use the baseline in Live Unreal

With the native project running, select **Live Unreal**, connect, choose
**Common sense · non-RL**, and plan a new episode. The same planner receives
only the validated native public snapshot and respects unsupported-site masks.
The existing Unreal bridge validates and deploys the resulting sensor/site IDs.
Live measured results remain separate from the offline comparison.

The existing logged driver also supports the baseline:

```powershell
.\RL\BlueTeam\.venv\Scripts\python.exe .\RL\BlueTeam\Python\run_istana_live.py --common-sense --paced --output-dir .\Saved\BlueLive\common-sense-run
```

Use a new output directory for each run. See [live setup](BLUE_TEAM_LIVE.md)
for native build and launch requirements.

## Verification

Run from `RL/BlueTeam/Python`:

```powershell
..\.venv\Scripts\python.exe -m pytest tests/test_common_sense.py tests/test_placement_comparison.py tests/test_swarm_comparison.py tests/test_console_comparison.py tests/test_simulation_console.py tests/test_istana_live.py -q
```

These tests exercise layout legality, public-only planning, paired sensing,
archived-result reproduction, empty layouts, HTTP validation and native bridge
planner integration. Browser checks cover scenario selection, editing, results,
method switching and playback. Mock native checks do not establish a new Unreal
build or native performance result.
