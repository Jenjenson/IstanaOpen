# Temporal confirmation: public motion, deadlines and sensor placement

This additive development work addresses a gap exposed by the
[ranking pilot](RANKING_PILOT_RESULTS.md): buying more detections did not produce
more **timely confirmations**, especially on fast/high-altitude stress cases.
It preserves the original simulator, reward, curriculum and published models.
There is not yet a promoted temporal-trained checkpoint.

The completed [three-seed results and offline replay](TEMPORAL_RESULTS.md)
improve over older baselines but show no meaningful learned timely-sensing
gain over the matched temporal control. The full gate failed; no promotion.

## What the new observation represents

The old features average one-look sensing coverage at four fixed radii. The new
`triad.temporal_placement_features.v2` keeps those columns and legal options
exactly, then adds public estimates of detection, rolling-window confirmation,
confirmation before the deadline, early confirmation and marginal return for
the existing layout and each possible extra sensor/site. STOP has zero marginal
gain. Actual catalogue strength and sensor height are explicit features too.

The same `TemporalObservationBuilder` accepts simulated or external
`triad.sensor_input.v1` snapshots. Public mission configuration supplies the
protected radius, look cadence, confirmation count/window and required lead
time. Old checkpoints reject this new feature schema. Changing mission rules
is not silently treated as the old policy's validated operating environment.

The probability calculation has two distinct levels:

1. Deterministic, uncertainty-weighted trajectory hypotheses use public bearing
   priors and fresh track position, velocity, altitude, confidence and emitter
   estimates. Closing tracks mix constant-velocity projection with an uncertain
   ingress corridor; nonclosing tracks retain their kinematics. These are
   assumptions, not knowledge of future curved/weaving paths.
2. Given each hypothetical per-tick hit sequence, a finite-state calculation
   exactly sums independent Bernoulli outcomes for the required number of hit
   ticks inside the rolling window. Multiple sensors/modalities on one tick
   still count as only one hit tick.

RF sensors share a single emission assumption before their probabilities are
combined. A declared mixture of independent-per-tick and pass-persistent
emission expresses uncertainty about temporal correlation. It does **not**
recover the simulator's hidden emitter period or phase.

Defaults use a 96-second horizon and at most 64 deterministic weighted trajectory
hypotheses. Horizon truncation, reduced quadrature and no-predicted-entry mass
are exposed. A no-entry hypothesis can have finite-horizon detection or
confirmation, but receives no invented timely/early confirmation utility.
`temporal_confirmation_without_timely` therefore includes both late confirmation
and confirmation without a predicted entry deadline. Already-confirmed public
tracks are treated as known at the current planning time, not assigned an
unknown historical confirmation timestamp.

These probabilities are exact only **conditional on the approximate forecast**.
They are not calibrated real-world probabilities or guarantees of defence.

## Public-input example

From `RL/BlueTeam/Python`, with the existing Python environment:

```python
import json
from pathlib import Path
from triad_rl.temporal_inputs import (
    TemporalConfig, TemporalObservationBuilder, TemporalPublicGreedy,
)

payload = json.loads(Path("../Examples/public-snapshot.json").read_text())
catalogue = json.loads(Path("../Examples/sensor-catalogue.json").read_text())
builder = TemporalObservationBuilder(TemporalConfig())
observation = builder.observe(payload, catalogue, now=0)
action = TemporalPublicGreedy().act(observation)
print(observation["options"][action])
```

This produces a local recommendation, not a device command. `TemporalPublicGreedy`
is explicitly a **non-RL control**, choosing positive expected marginal original
return and otherwise STOP. A learned policy must be compared with this
stronger control as well as the existing policies; a better forecast alone is
not proof that reinforcement learning improved deployment.

For an explicitly supplied temporal checkpoint, the standalone planner uses
the same builder and applies each selected placement to a local copy of the
snapshot before the next decision. From `RL/BlueTeam/Python`:

```powershell
python recommend_temporal.py --checkpoint ../Results/temporal-v6-pilot/training/seed-406/last `
  --input ../Examples/public-snapshot.json `
  --catalogue ../Examples/sensor-catalogue.json `
  --config ../Examples/temporal-config.json --now 0 --output ../runs/temporal-plan.json
```

Seed 406 is used only to make this command directly runnable, not because it
was selected as a winner; seeds 407 and 408 are retained alongside it.
The configuration must match the checkpoint exactly. Supply your provider's
clock through `--now` for fresh external data; zero is only this offline
example's clock. The output names sensor IDs, positions, costs and STOP, plus
clearly labelled forecast estimates. Existing output files are never replaced.
Input/catalogue objects and the policy RNG are unchanged. New catalogue
capabilities remain experimental until separately evaluated.

The three fixed trained endpoints were also checked on this same public example
with only `available_sensor_ids` changed (inference only, no outcome scoring):

| Available profiles | Recommended layout | Catalogue cost |
|---|---|---:|
| All five | Fused at site 10 | 2.0 |
| All except fused | Radar at site 10, RF at site 27 (seed 406) or 26 (407/408) | 2.0 |
| Radar only | Radar at site 10, then STOP | 1.2 |

Site indices are zero-based and refer to the supplied snapshot's positions.
These are all three endpoints, not a selected winner. This demonstrates
availability-conditioned planning, **not successful defence, calibration or
unseen-sensor generalization**. No device commands were sent.

## First development probe

Before sampling any new evaluation cases, `probe_temporal.py` scores the public
temporal control on exactly the first 20 **already-consumed** ranking-v5 cases
per profile. Existing greedy/v3/ranker outcomes remain the paired references.
All three profiles and every selected case are retained. This is a fixed,
60-case engineering diagnostic, **not unseen generalization or a promotion
screen**. Source and input hashes are saved with its results.

```powershell
python probe_temporal.py --output ../runs/temporal-v6-reused-probe.json
```

The completed [raw probe](Results/temporal-v6-development/reused-probe.json)
retains all 60 cases, input/source hashes, paired reference outcomes and means.
With profiles weighted equally:

| Method | Timely confirmation | Detection | Mean cost | Original return |
|---|---:|---:|---:|---:|
| Temporal public control (not RL) | 48.004% | 67.661% | 2.006 | 0.040 |
| Existing public greedy | 47.615% | 64.927% | 2.121 | -0.081 |
| Balanced v3 | 46.990% | 64.857% | 2.123 | -0.210 |

Temporal-control timely confirmation changed by **-2.583 percentage points**
on normal cases, **+0.417 pp** on stress and **+3.333 pp** on capability cases
against existing greedy. The +0.389 pp aggregate difference is small and mixed;
it does not establish generalization, statistical superiority or an RL gain.
All three prior ranking endpoints remain in the raw report, not just the best.

## Bounded learning pilot

The [protocol](Results/temporal-v6-pilot/protocol.json) declares three independent
512-episode training runs and a subsequent 600-case, three-profile development
evaluation. The policy starts with exactly the temporal control's deterministic
scores. A bounded shared neural correction can change both sensor/site choices
and STOP; a separate linear value baseline learns original reward-to-go without
changing the action network through critic gradients.

Every fixed endpoint must be compared with **its own non-RL temporal control**,
existing public greedy and v3, alongside the other published reference models.
The gate requires a timely-sensing improvement without lower aggregate
detection, all-threat success or return, and includes per-profile/seed guards.
There is no best-seed selection, early stopping or automatic promotion.

To reproduce the fixed pilot from `RL/BlueTeam/Python` (use new output
directories, and expect substantially more work than the unit tests):

```powershell
python train_temporal.py --protocol ../Results/temporal-v6-pilot/protocol.json --seed 406 --output ../runs/temporal-v6/seed-406
python train_temporal.py --protocol ../Results/temporal-v6-pilot/protocol.json --seed 407 --output ../runs/temporal-v6/seed-407
python train_temporal.py --protocol ../Results/temporal-v6-pilot/protocol.json --seed 408 --output ../runs/temporal-v6/seed-408
python evaluate_temporal.py --protocol ../Results/temporal-v6-pilot/protocol.json `
  --run 406=../runs/temporal-v6/seed-406 --run 407=../runs/temporal-v6/seed-407 `
  --run 408=../runs/temporal-v6/seed-408 --output ../runs/temporal-v6/evaluation
```

This driver intentionally enforces the published experiment, not arbitrary
hyperparameters. Each run records all 512 episodes and 32 batch updates, saves
`initialized/` and `last/`, and binds the raw records to the final checkpoint.
The evaluator requires all three complete runs. A replay of these published
seed slots is reproduction, not additional unseen evidence.

Portable JSONL metrics remain the training evidence. The optional existing
`track_adaptive.py` importer can send a completed run's events to a **local-only**
Trackio dashboard from a separate environment; it does not sync to a cloud
Space or affect the training process, optimizer or RNG.

The three reserved final tests remain unopened. Native integration, physical
deployment, real sensor calibration and mid-flight relocation are not implied
by this synthetic initial-layout planner.
