# Native Blue Team training workbench

Open **Training** in the simulation console to train sensor-placement policies
against the same Unreal Blue Team coordinator used by **Live Unreal**. There is
no browser-side training simulator. Native FOV, yaw/pitch, probability, world-static
occlusion, weather, repeated-hit confirmation and budget validation remain active.

## Start the application

After the source setup in the README, build the updated runtime:

```powershell
.\Tools\build.ps1 -Target Editor
# Headless, existing long-approach native benchmark on the Istana map:
.\Tools\start_warning_training.ps1 -Port 8766
.\Tools\start_simulation_console.ps1 -Port 9054 -BridgePort 8766
```

Open <http://127.0.0.1:9054/>. A visible native live launcher also works: point
the console at that process's bridge port. Training and Live share whichever
native scene you launched. The old packaged landscape viewer cannot train models.
The long-approach benchmark is an explicit synthetic threat scenario inside the
native Istana environment; it is not a claim of real-world detection performance.

1. Select **Training** and check that the native environment is ready.
2. Enter a run name, select REINFORCE or PPO, choose episodes, budget and sensor
   profiles. An optional seed is generated and saved when left blank.
3. Press **Start Training**. An initial three-scenario checkpoint evaluation runs
   before the first training episode. The status distinguishes training from
   checkpoint evaluation. All work runs in a background worker.
4. Watch raw and 20-episode-average graphs and recorded episode statistics.
5. Select **Stop Training** to cancel at the next native request boundary. Complete
   episodes are retained, and the last partial optimizer batch is learned/saved.
   An unfinished episode is discarded rather than recorded as a completed result.
6. Choose a saved checkpoint and **Load into Live Unreal**, then **Plan new episode**.
   Live applies its saved budget/types and performs deterministic model inference
   against the current native public state before deploying.

Live mutation controls are unavailable while training reserves the bridge. The
recorded replay/comparison views remain available. Run names receive `-2`, `-3`,
etc. when already used. History survives console restarts; an interrupted process
is labelled **interrupted**, with its completed logs/checkpoints retained.
An operating-system file lock prevents another console from starting a run in
the same directory or marking a still-running experiment as interrupted. The
lock is released automatically if its owning process exits.

Budget is passed into the native coordinator on every reset. It does not change
other scene constraints: the current scene permits at most three sensors, so a
budget of 10 with only a cost-1 thermal profile can still spend at most 3.
The actual threat count comes from the launched scene; the existing long-approach
scene uses five groups of twelve threats (60), and no display-only drone limit
is used for training.

## Algorithms and objectives

**REINFORCE** wraps the existing `WarningPolicy`: a masked categorical policy over
profile/site/yaw/pitch/STOP, batch return baseline, and Adam. It retains the
existing algorithm's update behavior. It has no critic, so value loss is unavailable.

**PPO** uses a shared tanh option encoder, masked categorical actor and pooled
value head, reusing the repository's NumPy network implementation. It implements
the [clipped PPO objective](https://arxiv.org/abs/1707.06347), terminal-layout
[GAE](https://arxiv.org/abs/1506.02438), shuffled minibatches, multiple epochs,
entropy regularization, gradient clipping and Adam. Defaults (saved with each
run) include gamma 1, lambda .95, clipping .2, four epochs and batch size 8
native episodes. Intermediate placement rewards are zero; the terminal native
warning objective supplies the return. No extra framework dependency is needed.

Historical temporal/adaptive checkpoints have incompatible observation/action
contracts and do not learn these directional angles; they are not offered as
native directional trainers. The algorithm registry/factory can accept future
implementations without putting optimizer logic in the UI.

The **training objective** preserves the existing warning trainer and simulator:

```
per-threat warning = max(0, zone_entry_time - first_detection_time)
training objective = mean per-threat warning (missed threats contribute zero)
```

An unresolved zone arrival is an error, matching `warning_metrics`; it is not
silently counted as successful warning. Confirmation time and a separate
confirmation-based mean warning are recorded without redefining the existing
warning metric. First detection/confirmation fields are the earliest measured
event in the episode; absence is null.

The primary **reward graph** shows the real native diagnostic episode reward:
`2*detected_fraction + 3*confirmed_fraction + 5*timely_fraction -
5*breached_fraction - cost/budget`. The optimizer targets warning, not this
diagnostic score. **Success rate** is native `timely_fraction`: confirmation with
at least the configured defence lead time remaining before zone entry.

Policy loss/value loss/entropy appear only on optimizer-update episodes and
describe that whole completed batch. They are not copied into other episodes.
Stopped partial-batch update metrics are saved separately. Thousands of graph
points are bounded to 1,000 displayed samples, retaining endpoints; moving averages
and summaries are computed from every episode before sampling. Full logs are saved.

## Checkpoints, history and reproducibility

At episode zero, at the chosen interval (default 10), and at completion, the
current model is evaluated deterministically on the same three held-out seeds.
**Best** maximizes the mean per-threat warning on this panel; ties retain the
earlier checkpoint. These seeds are disjoint from training seeds. This is a small
model-selection panel, not an unbiased final test or evidence of superiority.
An initial policy can legitimately remain best. Stopping before the first panel
finishes leaves no best checkpoint; latest/final remain available.

Each `training_runs/<unique_run_name>/` stores:

| File | Contents |
|---|---|
| `config.json` | Seed, episode/evaluation seed schedules, algorithm defaults, budget, enabled types, metric definitions |
| `algorithm_config.json` | Algorithm, architecture, observation/action spaces and compatibility contract |
| `native_context.json`, `red_context.json` | Native Blue and Red configuration snapshots |
| `sensor_config.json` | Full actual catalogue, temporal settings and enabled profiles |
| `reproducibility.json` | Git revision, relevant source hashes, Python version and available runtime metadata |
| `metrics.jsonl`, `metrics.csv` | Complete per-episode metrics, incrementally written |
| `evidence.jsonl` | Native terminal threat evidence, never exposed to the policy |
| `metrics.json`, `graph_data.json`, `summary.json` | Final/stop exports, graph data and summary |
| `evaluation-*.json` | Checkpoint selection episodes and native evidence |
| `checkpoints/` | Atomic latest, best, final and retained periodic models |
| `run.json` | Durable progress/status/checkpoint index |

Checkpoints store optimizer and random generator states in JSON, with hashes,
and retain enough metadata to reconstruct the policy. Loading rejects different
sites, placement constraints/budget, available profiles, catalogue, temporal
configuration, feature/action schema or sensor model. The console checks file
integrity before loading and rechecks compatibility before Live inference.
Keep the original source, scene assets and native build for repeatable experiments;
source hashes describe the console checkout and cannot certify an unrelated
externally launched Unreal binary. The capability check rejects older runtimes.
The UI starts new experiments; automatic interrupted-run resume is not provided.

Select prior runs to reopen graphs/statistics, or overlay up to three additional
runs. Compare like-for-like native scene configurations and adequate independent
seeds before drawing performance conclusions.

## Architecture and verification

`console/training.js` → HTTP console → `TrainingManager` → algorithm adapter →
`training_environment.run_episode` → existing `IstanaLiveClient`/Unreal coordinator.
The manager owns lifecycle/logging/checkpoint selection. Native `reset` accepts
validated atomic `blueConfiguration` budget/type overrides; failed resets restore
the previous settings. Neither UI nor Python replaces native sensing/physics.

Run Python tests from `RL/BlueTeam/Python`:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

Native Automation group `Istana.Simulation.BlueTeam` includes configuration,
directional sensing/LOS, layout validation and warning-time checks. Algorithm
tests check PPO gradients by finite differences, clipping/GAE, optimizer/RNG
round trips and compatibility rejection. Manager/controller tests cover native
configuration forwarding, cancellation, checkpoint/history/logging and HTTP guards.
