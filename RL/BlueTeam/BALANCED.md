# Balanced deployment: learn whether another sensor is worth its cost

This additive experiment keeps the v2 mixed curriculum, sensor physics,
public observation features and reward unchanged. It addresses a measured
exploration problem: one STOP action competed with hundreds of deployment
options. See the [predeclared experiment protocol](Results/balanced-v3/protocol.json).
Measured results, tradeoffs and execution history are recorded in the
[balanced-v3 results report](BALANCED_RESULTS.md).

The actor still chooses **both sensor type and offered site**. RF, radar, EO,
thermal and fused profiles can have different availability, cost, height,
modality strength and range in each scenario. Existing placements, remaining
budget, forecast coverage, public tracks and weather inform the next choice.

## Try the published candidate

After cloning this branch, open
[the recorded replay](Results/balanced-v3/demo.html) locally in a browser.
GitHub's file viewer shows the HTML source, so download the file or open the
cloned copy to use its scenario selector, play/restart buttons and timeline.
It is self-contained and works offline.

To request a new layout from the included example snapshot, install the
package using the [main guide](README.md), then run from `RL/BlueTeam/Python`:

```powershell
python recommend_balanced.py --checkpoint ../Checkpoints/balanced-v3-candidate `
  --input ../Examples/public-snapshot.json `
  --catalogue ../Examples/sensor-catalogue.json --now 0 `
  --output ../runs/balanced-plan.json
```

The [results](BALANCED_RESULTS.md) explain the measured tradeoff: lower stress
spending, but not uniformly stronger detection. This experimental checkpoint
does not replace the existing default policy.

## What changes

For every legal deployment, the policy subtracts `log(N)` from its learned
logit, where `N` is the number of legal sensor/site deployment options. STOP's
logit is unchanged. This balances the total deploy category against STOP;
duplicating every deployment option leaves the STOP probability unchanged.
It does not force equal probabilities after learning, and adding genuinely
different capabilities can still change the decision.

During training the actor samples the resulting joint distribution. At
demonstration/inference time it first chooses deploy versus STOP: STOP when
its probability is at least 0.5, otherwise the highest-scoring legal
deployment. This is intentionally **not** joint argmax: a single STOP row
can outweigh each deployment row while deploying remains more likely overall.
Exact deployment-score ties retain the first offered row.

The entropy bonus is `H(joint) - P(deploy) * log(N)`, or gate entropy minus
deployment probability times the conditional divergence from uniform. It
does not reward a large catalogue merely for containing more rows. The
pooled critic is otherwise unchanged. The implementation uses genuine
on-policy policy-gradient updates with a value baseline, Adam and gradient
clipping; no coverage threshold tells it to stop. Zero public forecast
coverage is not proof that actual sensing is impossible.

The new checkpoint schema prevents old loaders from silently applying the
wrong inference rule. V1/v2 checkpoints, source files and results remain
unchanged. No existing default policy is replaced.

## Train or resume

Install the package as described in the [main guide](README.md). From
`RL/BlueTeam/Python`, using the environment's Python:

```powershell
python -m triad_rl.train_balanced `
  --initial-checkpoint ../Checkpoints/robust-v2-candidate `
  --initial-selection-report ../Results/robust-v2/selection.json `
  --seed 201 --episodes 8000 --batch-size 16 `
  --validation-run-seed 991300 --output ../runs/balanced-201
```

The published protocol specifies seeds 201, 202 and 203, each with its own
empty output directory. All three transfer the same v2 parameter values,
reset optimizer state and use independent sampling RNGs. Their saved
`initialized` controls already have the new balanced inference rule but
have had **no additional weight training**. This distinguishes architecture
effects from learning; these are not independent from-scratch initializations.

Each run saves `config.json`, `training.jsonl`, `summary.json` and
`initialized`, `best`, `last` JSON/NPZ checkpoints. Best is selected by
equally weighted normal/stress/capability validation return, with earlier
checkpoints winning ties. Full inherited training and model-selection seed
exposure is preserved. Never present this selection validation as final tests.

An interrupted run can resume at its most recently saved full batch:

```powershell
python -m triad_rl.train_balanced `
  --resume ../runs/balanced-201/last --output ../runs/balanced-201 `
  --seed 201 --episodes 8000 --batch-size 16 --validation-run-seed 991300
```

The requested total must exceed completed episodes and be batch-aligned.
Keep source and hyperparameters unchanged. Resume restores exact weights,
Adam, sampling RNG and validation cadence; it rejects changed logs or
checkpoints. A transfer is a different operation and cannot be combined with
resume. The unopened final suites remain reserved in the protocol.

After all three protocol runs finish, compare their selected checkpoints on
the common validation cases:

```powershell
python evaluate_balanced.py `
  --candidate 201=../runs/balanced-201/best `
  --candidate 202=../runs/balanced-202/best `
  --candidate 203=../runs/balanced-203/best `
  --protocol ../Results/balanced-v3/protocol.json `
  --workers 3 `
  --output ../runs/balanced-common-selection
```

This includes all three matched initialized controls, frozen v1 and v2,
greedy public coverage, random legal placement, fixed RF/radar layouts and
the original toy checkpoint through its explicit transfer adapter. Every
method receives paired scenarios and sensor catalogues. The output includes
complete compressed episode records, source/checkpoint hashes, candidate
ranking and paired differences. Intervals on these model-selection cases
are descriptive, not selection-adjusted independent-test evidence.

`--workers 3` evaluates normal, stress and capability profiles in independent
processes. Worker count is only an execution setting: it does not change the
cases, selection criterion, policy decisions or statistical design. The default
is one worker; serial and parallel-resumed outputs are tested byte-for-byte.

After each profile finishes and passes the full source, checkpoint, run and
pairing checks, the evaluator atomically saves
`completed-validation-{profile}.json.gz` and updates `progress.json`. These
are operational recovery files. Final `common-validation-{profile}.json.gz`
reports and `selection.json` are written only after all three profiles finish;
the selection file is written last. A printed summary alone is not a saved
evaluation result.

If interrupted after at least one profile was saved, repeat the same command
with the same output directory and add **`--resume`**. For example:

```powershell
python evaluate_balanced.py `
  --candidate 201=../runs/balanced-201/best `
  --candidate 202=../runs/balanced-202/best `
  --candidate 203=../runs/balanced-203/best `
  --protocol ../Results/balanced-v3/protocol.json `
  --workers 3 --resume `
  --output ../runs/balanced-common-selection
```

Resume verifies the exact protocol, source, all complete training runs,
reference checkpoints, seed ranges and bootstrap settings. It checks saved
episode pairing, hashes, weights, summary metrics and comparison intervals
before reusing a completed profile; verified profiles are not sampled again.
An atomically completed profile whose progress update was interrupted can
also be recovered. Incomplete temporary files are left untouched, never used
as evidence. Changed or conflicting artifacts fail closed instead of being
overwritten. Worker count may change on resume. If no profile or progress
file was saved, use a fresh empty output directory without `--resume`.

For local metric visualization, the optional existing `track_adaptive.py`
imports a completed balanced run's numeric JSONL events into Trackio. This
does not change the training environment or RNG and does not upload to a
Hugging Face Space:

```powershell
python track_adaptive.py --run-dir ../runs/balanced-201 `
  --project blue-team-adaptive --name balanced-v3-transfer-201
```

## External inputs and presentation

Use the same [public snapshot and sensor catalogue contract](ADAPTIVE_INPUTS.md)
for simulated or external observations. This emits a coordinated plan only:

```powershell
python recommend_balanced.py --checkpoint ../runs/balanced-201/best `
  --input ../Examples/public-snapshot.json `
  --catalogue ../Examples/sensor-catalogue.json --now 0 `
  --output ../runs/balanced-plan.json

python demo_balanced.py --checkpoint ../runs/balanced-201/best `
  --profile capability --episodes 6 --output ../runs/balanced-demo.html
```

Open the generated HTML locally. Its controls show the actual recorded
sensor/site decisions, varied sensor ranges, threats, detections, outcome
and reward components. It is an offline simulator replay, not browser
inference or a live device-control interface. Replays are consecutive,
explicitly identified validation scenarios, not cherry-picked successes.

The validated bundle's selected checkpoint is at
`Checkpoints/balanced-v3-candidate`, its matched initialized control at
`Checkpoints/balanced-v3-candidate-initialized`, and its offline replay at
`Results/balanced-v3/demo.html`, all relative to `RL/BlueTeam`. Full training,
comparison and interpretation details are in [the results](BALANCED_RESULTS.md).

“Dynamic” means scenario-conditioned, sequential initial placement from the
currently offered catalogue and sites. This version does **not** relocate
sensors mid-flight, calibrate real devices, connect to live Istana/C2, or
perform physical interception. Synthetic sensing success is not field
reliability or proof of an optimal layout.
