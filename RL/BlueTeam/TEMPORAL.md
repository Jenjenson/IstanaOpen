# Temporal confirmation: public motion, deadlines and sensor placement

This additive development work addresses a gap exposed by the
[ranking pilot](RANKING_PILOT_RESULTS.md): buying more detections did not produce
more **timely confirmations**, especially on fast/high-altitude stress cases.
It preserves the original simulator, reward, curriculum and published models.
There is not yet a promoted temporal-trained checkpoint.

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
return and otherwise STOP. A future learned policy must be compared with this
stronger control as well as the existing policies; a better forecast alone is
not proof that reinforcement learning improved deployment.

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

The three reserved final tests remain unopened. Native integration, physical
deployment, real sensor calibration and mid-flight relocation are not implied
by this synthetic initial-layout planner.
