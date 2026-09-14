# Robust curriculum experiment (v2, experimental candidate)

The [adaptive-v1 result](ADAPTIVE_RESULTS.md) exposed two remaining problems:
poor severe-stress detection and excessive spending when sensing opportunities
are weak. Its structurally extensible catalogue had also not been trained or
evaluated with varying capabilities. This next experiment addresses those
gaps without changing the published v1 simulator, features, policy code,
trainer or artifacts.

The [predeclared protocol](Results/robust-v2/protocol.json) records transfer
initialization, candidate runs, validation-based selection and new reserved
final-test ranges. Three runs have completed 24,000 additional episodes; see
[the validation results and remaining gap](ROBUST_RESULTS.md). Independent
final-test suites remain unopened. The selected candidate is not promoted as
a solution to robust deployment.

## Run it

After the [Python setup](README.md#try-it-without-unreal), run from the repository
root in PowerShell:

```powershell
.\RL\BlueTeam\.venv\Scripts\python.exe .\RL\BlueTeam\Python\demo_robust.py `
  --checkpoint .\RL\BlueTeam\Checkpoints\robust-v2-candidate `
  --profile capability --episodes 6 --output .\Saved\BlueRL\robust-demo.html

.\RL\BlueTeam\.venv\Scripts\python.exe .\RL\BlueTeam\Python\recommend_adaptive.py `
  --checkpoint .\RL\BlueTeam\Checkpoints\robust-v2-candidate `
  --input .\RL\BlueTeam\Examples\public-snapshot.json `
  --catalogue .\RL\BlueTeam\Examples\sensor-catalogue.json --now 0 `
  --output .\Saved\BlueRL\robust-recommendation.json
```

The [bundled replay](Results/robust-v2/demo.html) is an offline HTML file with
six consecutive capability-varied validation cases, not selected successes.
Its play, scenario selector and keyboard timeline were checked in desktop
and mobile layouts. It shows recorded simulator outcomes, not live inference.

Reproduce one transfer run (choose a fresh output directory):

```powershell
.\RL\BlueTeam\.venv\Scripts\python.exe -m triad_rl.train_robust `
  --output .\Saved\BlueRL\robust-103 --seed 103 --episodes 8000 --batch-size 16 `
  --learning-rate 0.002 --entropy-coef 0.015 --gamma 1.0 `
  --validation-every 1000 --validation-episodes 60 --validation-run-seed 990100 `
  --initial-checkpoint .\RL\BlueTeam\Checkpoints\adaptive-v1 `
  --initial-selection-report .\RL\BlueTeam\Results\adaptive-v1\common-validation-selection.json
```

Seeds 101 and 102 are the other declared runs. Validation occurs at batch
boundaries crossing each cadence threshold (1008, 2000, 3008, and so on).
The complete run directories preserve `initialized`, `best`, `last`, logs,
configuration and summary. Exact resume requires the matching latest full-batch
checkpoint, unchanged source/configuration and intact initialization/logs.
Copy a complete published run to a new working directory before extending it;
do not alter the frozen published evidence in place.

`select_robust.py --candidate SEED=RUN/best` (one argument per run) reproduces
common-validation selection from the protocol. `evaluate_robust.py` provides
seven-method paired comparisons with explicit `--stage validation` or `test`;
use `--selection-report .../selection.json` to reserve all selection exposures.
Do not open the reserved final suites while further development uses validation.

Optional local tracking uses `track_adaptive.py` from an environment with
Trackio installed. Nested profile metrics are retained, for example
`validation/profiles/stress/mean_cost`. No cloud synchronization is configured.

## What changes

Training samples a fixed mixture: 40% ordinary randomized scenarios, 30%
severe stress, and 30% capability-varied scenarios. The last group independently
changes synthetic sensor ranges, strengths, costs, mounting heights,
availability, legal site geometry, budget and site-count constraints; half its
cases use ordinary threats and half use severe threats. The five canonical
RF/radar/EO/thermal/fused profiles retain their modality meanings. Cases are
not removed when no useful or affordable sensor exists.

The observation and reward contract stays identical to v1. The simulator and
external-format input adapter consume the same public catalogue and snapshot.
Private target paths, emission phases and detection draws remain unavailable
to the actor. The varied catalogue is used in scoring as well as observation
construction; it is not a cosmetic sensor-name change.

The learner starts from the published v1 weights, with a new optimizer and
independent policy random seed. This is explicitly **transfer training**, not
an independent from-scratch success claim. Complete sampled episodes still
drive actual on-policy actor-critic updates; no greedy labels or hidden-truth
actions are supplied. An undiscounted return (`gamma=1`) aligns optimization
with the recorded sum of reward components. Validation selects on mean reward
across all three profiles instead of ordinary-scenario success alone.

## Evidence boundaries

The v1 benchmark remains immutable and reproducible. New comparisons must
include the frozen v1 policy and non-RL baselines on paired cases. Catalogue
parameters and scenario truth both enter the evaluation fingerprint, while
neither evaluation metadata nor future truth enters the actor's observation.
Validation results are selection evidence, not independent final-test results.

Better resource control alone does not establish robust defence: reports must
retain success, timely confirmation, detection, cost and invalid-action rates
by profile. No amount of training guarantees sensing targets outside the
model's physical support. Real hardware calibration and native Istana scene
integration remain separate, unverified work.
