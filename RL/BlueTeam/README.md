# Blue Team RL: dynamic sensor placement

For the native limited-FOV sensor workbench, see
[Local refinement PPO](LOCAL_REFINEMENT.md). It learns legal adjustments to a
contractor layout using matched native scenario rewards. Historical trained
weights and their unchanged selected layouts are retained in the
[four-sensor model archive](Results/four-sensor-20260924/README.md).

Blue learns **which sensor profile to deploy and where to place it** around a
synthetic objective. It supports passive RF, search radar, electro-optical,
thermal, and combined radar/thermal profiles, including repeated deployments
of the same profile within the available budget.

**Latest experiment:** [temporal RL results](TEMPORAL_RESULTS.md) and the
[18-example offline replay](Results/temporal-v6-demo.html). Download the HTML
and open it in a browser to see every fixed endpoint choose sensor types/sites
and run through recorded threats. The new pilot improves over older baselines
but does not beat its own temporal control; it is not a promoted replacement.
For changing available sensors and supplying public snapshots, see the
[temporal input guide](TEMPORAL.md).

**Foundation: randomized adaptive training.** Start with the
[adaptive guide](ADAPTIVE.md) for scenario-conditioned sensor/site choices,
an offline replay demo, external-format sensor inputs, and held-out comparisons
against the original checkpoint and non-RL methods. The adaptive experiment
is separate from the historical live Unreal experiment documented below.

The follow-on [robust curriculum experiment](ROBUST.md) adds varying sensor
capabilities and severe cases. Its [v2 candidate](ROBUST_RESULTS.md) has modest
validation gains but an unresolved stopping/resource-use gap; it is not a
final-tested robustness release.

The additive [balanced deployment experiment](BALANCED.md) tests a learned
deploy-or-stop gate to address that gap, with an explicitly versioned actor,
matched initialized controls and unchanged simulation physics/rewards.
Its [v3 results and offline demo](BALANCED_RESULTS.md) document 24,000 further
training episodes and the measured cost/detection tradeoff; it remains an
experimental candidate, not a final-tested replacement.

The [credit-assignment pilot](CREDIT_PILOT.md) compares matched training runs
with and without critic gradients and a training-only paired STOP baseline.
Its [completed results](CREDIT_PILOT_RESULTS.md) show a small sensing gain but
higher cost and a failed predeclared scaling gate; it does not replace v3.

The follow-up [anchored later-placement pilot](ANCHORED_PILOT.md) preserves the
first decision's ordinary learning coefficient while changing later-placement
credit. Its [completed results](ANCHORED_PILOT_RESULTS.md) show better sensing
than the matched control but no demonstrated timely-sensing gain over v3; the fixed
scaling gate failed and no new policy was promoted.

The [rollout-guided ranking pilot](RANKING_PILOT.md) tests direct sensor/site
preferences with a learnable correction to public greedy, including STOP.
Its three-seed protocol keeps the original sensing physics and public inputs;
the [completed results and 18-example replay](RANKING_PILOT_RESULTS.md) show
higher detection but lower timely sensing and return than public greedy. The
fixed development gate failed; no endpoint was selected or promoted.

The [temporal-confirmation experiment](TEMPORAL.md) uses public track
motion, sensor height/strength and repeated-hit deadlines to inform placement.
It adds a bounded learned correction with an independent value baseline.
Its [fresh results](TEMPORAL_RESULTS.md) separate the benefit of forecasting
from learning; the full matched-control gate failed and no policy is promoted.

This package publishes the Python trainer, a trained checkpoint, recorded
results, and selected native source from the TRIAD experiment. You can run the
Python example immediately. Live Unreal training requires the original TRIAD
host and its dependencies; the package is not connected to Istana Open's Red
Team Manager or its schema-1 policy interface yet. See
[integration status](INTEGRATION.md).

## Try it without Unreal

Requires Python 3.11 or later. Run these PowerShell commands from the repository
root:

```powershell
py -3.11 -m venv .\RL\BlueTeam\.venv
.\RL\BlueTeam\.venv\Scripts\python.exe -m pip install -e ".\RL\BlueTeam\Python[test]"
.\RL\BlueTeam\.venv\Scripts\python.exe -m pytest -q .\RL\BlueTeam\Python\tests
.\RL\BlueTeam\.venv\Scripts\python.exe .\RL\BlueTeam\Python\train_blue_placement.py `
  --dry-run --episodes 40 --batch-size 10 --seed 1337 `
  --checkpoint-dir .\Saved\BlueRL\dry-run
```

The run prints episode metrics and saves `episode_000040_final/checkpoint.json`,
`arrays.npz`, and `training_metrics.jsonl` beneath `Saved/BlueRL/dry-run`.
Evaluate that newly trained example on a separate seed range:

```powershell
.\RL\BlueTeam\.venv\Scripts\python.exe .\RL\BlueTeam\Python\evaluate_checkpoint.py `
  --dry-run --checkpoint .\Saved\BlueRL\dry-run\episode_000040_final `
  --episodes 20 --seed 1500000000 --output .\Saved\BlueRL\dry-run-evaluation
```

Use a fresh output directory for each evaluation. On macOS/Linux, use
`python3 -m venv RL/BlueTeam/.venv` and `RL/BlueTeam/.venv/bin/python` with `/`
path separators.
The trainer runs on CPU with NumPy, Gymnasium, and PettingZoo; no GPU, PyTorch,
or Ray is required.

The dry-run environment is a small Python exercise for policy updates, action
masks, metrics, and checkpoint loading. Its observations and sensor behavior
differ from the Unreal experiment. **The bundled `Checkpoints/toy-210`
checkpoint cannot be used with `--dry-run`**: it has a different feature
contract and is intentionally rejected.

## How placement works

```mermaid
flowchart LR
    A[Sensor catalogue, budget and placed sites] --> B[Shared candidate scorer]
    B --> C[Choose profile and continuous position]
    C --> D[Validate placement and run scripted threat]
    D --> E[Detection, tracking and episode reward]
    E --> B
```

A Blue action contains a catalogue index and normalized East/North position.
The final catalogue index commits the layout. Dynamic placement is constrained
by objective standoff, terrain, site separation, site count, and cost; Unreal
makes the final decision about legality in live mode.

The policy scores each catalogue row with shared weights and pools existing
placements. Adding profiles does not require resizing the network, provided
the named observation features remain compatible. Live candidates are defined
in [DefaultTrainingConfig.json](DefaultTrainingConfig.json). Each has a stable
`SensorProfileId`, capability settings, cost, and `bAllowDynamicPosition`.
Changed configurations need their own training and evaluation; architectural
compatibility does not establish performance on unseen sensors.

Red follows a fixed or seeded script. Blue chooses the layout before Red
deployment; this version does not relocate sensors during an episode or train
a second Red policy. `train_self_play.py` is a compatibility entry point to the
same Blue trainer.

## Published experiment

The included checkpoint completed 210 live Unreal episodes on 10 September
2026. In 50 evaluation episodes, its sampled policy achieved **72% abstract
defence**, compared with **56%** for the initialized sampled baseline.
Deterministic deployment achieved 50/50 in the same fixed scenario.

These runs use one drone approaching from the north in a generic flat arena,
with analytical sensor models. They demonstrate learning in that experiment;
the seed changes do not make them 50 different threat scenarios. See
[results and limitations](RESULTS.md) for the evidence and interpretation.

| Included | Purpose |
| --- | --- |
| [Python/](Python/) | Policy, live bridge, training/evaluation commands, and tests |
| [ADAPTIVE.md](ADAPTIVE.md) | Randomized trainer, trained adaptive checkpoint, offline demo and benchmark commands |
| [ADAPTIVE_INPUTS.md](ADAPTIVE_INPUTS.md) | Shared simulated/external sensor input contract and recommendation API |
| [Examples/](Examples/) | Public observation and interchangeable sensor catalogue examples |
| [DefaultTrainingConfig.json](DefaultTrainingConfig.json) | Version-4 live scenario and five sensor profiles |
| [Checkpoints/toy-210/](Checkpoints/toy-210/) | Live-trained policy, optimizer and random-generator state |
| [Results/](Results/) | Training history, three evaluation reports, and native oracle |
| [NativeReference/](NativeReference/) | Selected TRIAD native RL sources for integration work |

For live host requirements, training commands, and the work needed to connect
this package to Istana Open, continue with [INTEGRATION.md](INTEGRATION.md).
