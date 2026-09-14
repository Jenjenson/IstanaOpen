# Temporal RL pilot: better forecasting, no demonstrated learned gain

The three-seed temporal policy improved timely confirmation over v3 and the
old public greedy baseline on fresh development scenarios. It did **not**
improve meaningfully over its own non-RL temporal control: timely confirmation
was essentially unchanged, cost increased and original return declined.
The full predeclared gate therefore **failed**. No checkpoint was selected or
promoted, and the reserved final tests remain unopened.

[Offline replay: all 18 fixed examples](Results/temporal-v6-demo.html) ·
[Setup, public inputs and sensor-availability example](TEMPORAL.md) ·
[Protocol](Results/temporal-v6-pilot/protocol.json) ·
[Full aggregate](Results/temporal-v6-pilot/evaluation/aggregate.json) ·
[Artifact manifest](Results/temporal-v6-pilot/artifact-manifest.json)

Download the self-contained replay HTML and open it in a browser; no server,
Python installation, live connection or browser inference is required. The
selector includes every fixed endpoint and the first two scored cases from
each profile, including failures. The map shows sensor types/sites, sensing
ranges, threat movement and detection/confirmation; the panels show placement
decisions, budget, reward and the final outcome. This is recorded synthetic
sensing, not interception or physical deployment.

## What changed

The additive public feature schema models track motion, slant range, shared RF
emission and repeated-hit confirmation before a deadline. A bounded neural
correction learns sensor/site and STOP scores; its value baseline has separate
parameters, clipping and optimizer state. Initial deterministic sensor/site/STOP
decisions exactly match the temporal non-RL control. Physics, original reward, legality and curriculum are
unchanged. The same input builder accepts simulated and external-format public
snapshots, with explicit mission configuration and a variable sensor catalogue.

The [source and protocol were published](https://github.com/Jenjenson/IstanaOpen/commit/e2d4478c9eb5352640b317c4507a85667bce53c0)
before generating this pilot's training or validation scenarios. Three fixed
seeds each ran 512 episodes and 32 updates, followed by the same 200 cases in
each of normal, stress and capability-varied validation: **600 cases and 5,400
method-episodes**. There was no best-seed selection, early stopping, case
filtering, retraining or ensemble action selection.

## Fresh development comparison

Each scenario contributes its threat fractions equally; profiles are then
weighted equally. The primary statistic averages the three learned endpoints
**within each case**, not as 1,800 independent cases. All-threat success means
every threat in that episode achieved the required confirmation no later than
four seconds before zone entry. Cost uses synthetic
catalogue units, not real prices.

| Method | Timely fraction | Detection | All-threat success | Mean cost | Original return |
|---|---:|---:|---:|---:|---:|
| Temporal RL, three-seed mean | 53.303% | 70.760% | 38.333% | 2.154 | 1.035 |
| Temporal public control, not RL | 53.299% | 69.573% | 38.500% | 1.997 | 1.122 |
| Existing public greedy | 51.947% | 68.686% | 36.000% | 2.074 | 0.910 |
| Balanced v3 | 51.072% | 67.411% | 34.333% | 2.080 | 0.686 |
| Robust v2 | 51.533% | 68.506% | 34.333% | 2.477 | 0.439 |
| Adaptive v1 | 49.853% | 66.167% | 33.167% | 2.433 | 0.037 |
| Anchored later-placement control | 50.643% | 65.376% | 35.000% | 1.953 | 0.651 |
| Temporal seed 406 | 53.544% | 71.754% | 38.500% | 2.309 | 0.973 |
| Temporal seed 407 | 53.055% | 69.992% | 38.167% | 2.072 | 1.032 |
| Temporal seed 408 | 53.312% | 70.535% | 38.333% | 2.081 | 1.101 |

Primary timely-confirmation differences, in **percentage points**:

| Three-seed mean minus comparator | Difference | Paired 95% interval | Comparator gate |
|---|---:|---:|---|
| Temporal public control | +0.005 | [-0.247, +0.231] | Fail |
| Existing public greedy | +1.356 | [+0.489, +2.291] | Pass |
| Balanced v3 | +2.232 | [+1.130, +3.420] | Pass |

Intervals use 2,000 paired, profile-stratified bootstrap resamples with the
declared seed. They describe scenario uncertainty conditional on these three
endpoints, not uncertainty across the population of possible training seeds.
Passing the two older-comparator gates does not override the failed matched
control gate or constitute final-test evidence.

Against its temporal control, learning increased detection by **1.187 pp**
but timely confirmation by only **0.005 pp**. Mean cost rose **0.158** and
return fell **0.087** (paired 95% interval [-0.134, -0.044]); all-threat success
also missed its no-decline point guard. This suggests the new forecast prior,
not a demonstrated learned improvement over that prior, accounts for most of
the timely-sensing gain over the older baselines.

## Remaining weakness

| Profile, learned three-seed mean | Timely fraction | Detection | All-threat success | Mean cost | Return |
|---|---:|---:|---:|---:|---:|
| Normal | 86.583% | 96.333% | 69.333% | 2.731 | 7.951 |
| Stress | 14.099% | 36.238% | 1.500% | 1.461 | -7.478 |
| Capability-varied | 59.227% | 79.709% | 44.167% | 2.270 | 2.633 |

On stress cases, the learned mean spent **0.338** more than the temporal
control for just **+0.199 pp** timely confirmation. Detecting more threats
still often fails to deliver the required repeated hits before the deadline.
Seed 406 also spent more than the other endpoints. These are reasons to retain
the control and investigate resource-use credit, not select a favourable seed
or claim robustness from the aggregate alone.

## Training and verification

All 1,536 episodes were retained: **2,860 sampled decisions, 96 actor updates
and 96 independent critic updates**, with zero invalid placements. The runs
took approximately 341–357 seconds each while executing concurrently on the
local Windows CPU (Python 3.11.9, NumPy 2.4.6; OpenBLAS/OMP threads set to one).
These are not compute-matched comparisons with the older experiments.

An independent read-only reconstruction reproduced every sampled action,
before-update value, reward-to-go, batch update, optimizer state and final RNG
exactly from the saved public feature records. All five parameter groups
changed in each run. The audit generated no scenarios and changed no files.
An independent evaluation audit also reproduced target-derived metrics, paired
summaries, 528 confidence intervals and all gate checks across the 5,400 saved
episodes. It checked recorded accounting without resimulating sensing physics.
Portable logs contain 32 batch records per run; the optional local-only Trackio
import preserves all 96 without changing training or syncing a cloud Space.

The publication retains all three initialized/final checkpoints, compressed
episode records, logs, source-bound configuration and full paired reports.
The manifest binds 36 input artifacts byte-for-byte. Replay generation checks
the fixed scored scenarios, actions, layouts, metrics and outcomes before
exporting its separately derived HTML.

Browser verification checked all 18 scenario choices, play/pause, restart and
timeline scrubbing. Desktop and 390-pixel-wide layouts rendered without page
errors or horizontal overflow. This is a self-contained recorded demonstration,
not a browser-side policy or a live sensor connection.

This remains an initial-layout planner for synthetic scenarios. It does not
validate live hardware, native C2/Unreal integration, calibrated real-world
probabilities, mid-flight relocation or optimal placement. Architectural
support for changed catalogue capabilities is not proof of unseen-sensor
performance. The reserved final tests are still unopened.
