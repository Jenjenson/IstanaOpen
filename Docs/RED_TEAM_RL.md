# Red initial-placement reinforcement learning

## Status and scope

This is a bounded hackathon implementation of Red reinforcement learning over
the live Unreal Red/Blue interface. Red makes one decision at the start of each
episode: where to place the swarm-group centres. The existing Unreal boid
controller owns all movement after placement, and Unreal remains authoritative
for placement validation, fixed-step transitions, sensing, termination and
native rewards.

Training is deliberately **not simultaneous self-play**. Only Red is updated;
Blue is frozen as either the temporal public greedy control or an explicitly
selected Blue checkpoint. Keeping the opponent fixed makes the short training
run reproducible and the comparison understandable.

| Component | Current implementation |
| --- | --- |
| Red observation | Advertised placement context plus public Blue context |
| Learned action | One of 16 legal wedge directions at the middle allowed radius |
| Red algorithm | Masked tabular softmax, one-step REINFORCE and Adam |
| Blue opponent | Frozen temporal public greedy by default |
| Movement | Existing Unreal boid controller; not learned |
| Training unit | One complete live Unreal episode per policy update |
| Evaluation | Scripted radial, random wedge and deterministic learned Red on identical seeds |

The implementation entry points are:

- `RL/BlueTeam/Python/triad_rl/red_policy.py`: policies, catalogue, optimizer,
  checkpoint format and public exposure calculation.
- `RL/BlueTeam/Python/triad_rl/istana_live.py`: live episode orchestration and
  Red-policy injection.
- `RL/BlueTeam/Python/train_red_placement.py`: training CLI.
- `RL/BlueTeam/Python/evaluate_red_placement.py`: frozen-seed evaluation CLI.
- `RL/BlueTeam/Python/run_istana_live.py`: single learned, random or dispersed
  live run.

## Policies

All policies implement the same `select(context, ...)` interface.

| Policy/CLI | Purpose |
| --- | --- |
| `ScriptedRadialRedPolicy` | Deterministic ring baseline with groups distributed around the objective |
| `RandomLegalRedPolicy` / `--red-random` | Randomly selects one legal wedge template; used by the formal evaluation |
| `LearnedRedPlacementPolicy` / `--red-checkpoint` | Loads and deterministically runs a trained 16-action policy |
| `DispersedRandomRedPolicy` / `--red-dispersed` | Independently randomizes legal group angles/radii for multi-direction footage; not an RL result |

Each learned template supplies exactly one centre per advertised Red group at
the native height. A conservative mask removes templates whose swarm regions
cannot meet the advertised spread and spacing. Unreal performs final terrain,
collision and member-placement validation; Python never clips or silently
replaces a rejected action.

## Learning objective

The model is a masked tabular softmax trained with one-step REINFORCE, a running
reward baseline, entropy regularization, gradient clipping and Adam. The first
return initializes the baseline without updating an action, preventing a large
negative-return bias at startup.

The training-only return is:

```text
training reward = native Red reward - exposure weight * public path exposure
```

The default exposure weight is `20`. Public path exposure is computed only from
the advertised Blue sensor layout, public capabilities and weather. It does not
inspect hidden Red/Blue truth or future observations. Native reward, exposure
and the combined training reward are logged separately.

This shaping is necessary in the current wedge scenario because Blue detects
all tested wedge placements and the native Red reward saturates at `-9`.
Therefore, the verified result below demonstrates improvement in the documented
public exposure signal, not a native-reward win or real-world sensor avoidance.

## Prerequisites

Run commands from the repository root in PowerShell. The verified environment
uses Unreal Engine 5.5.4 and the repository's isolated Python environment.

One-time setup:

```powershell
cd C:\path\to\IstanaOpen
powershell -ExecutionPolicy Bypass -File .\Tools\setup_blue_python.ps1
powershell -ExecutionPolicy Bypass -File .\Tools\build.ps1 -Target Editor
```

Start the live Unreal scene before every training, evaluation or CLI demo
session:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\start_blue_live.ps1
```

Wait for the map to load, then run Python commands from a second PowerShell
window. The bridge accepts one live client at a time. Disconnect the browser's
Live Unreal session before using a CLI trainer/evaluator.

## Reproduce the verified training run

The reference run used 120 episodes, Unreal seeds `91000..91119`, policy seed
`7301`, frozen temporal public greedy Blue and the checked-in default optimizer
settings. Do not add `--paced` when reproducing metrics; pacing is only for
recording and makes the run slower.

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$trainDir = ".\Saved\RedRL\reproduction-$stamp"

.\RL\BlueTeam\.venv\Scripts\python.exe `
  .\RL\BlueTeam\Python\train_red_placement.py `
  --temporal-public-control `
  --episodes 120 `
  --seed 91000 `
  --policy-seed 7301 `
  --learning-rate 0.03 `
  --baseline-rate 0.1 `
  --entropy-coefficient 0.01 `
  --exposure-weight 20 `
  --checkpoint-every 10 `
  --output-dir $trainDir
```

On the development machine this took roughly 10–15 minutes; runtime depends on
hardware and Unreal frame rate. The output directory must not already exist.
The command prints episode, reward, baseline, selected action and cumulative
invalid-placement count while it runs.

### Training output

```text
<trainDir>/
  initialized/             policy before the first update
  checkpoint-000010/       periodic recovery checkpoint
  ...
  checkpoint-000120/
  final/                   final inference/optimizer checkpoint
    checkpoint.json
    arrays.npz
  training.jsonl           one detailed record per completed episode
  summary.json              aggregate configuration and metrics
```

Checkpoints contain the action-catalogue contract, logits, Adam state, running
baseline, episode/update counts, RNG state and an array integrity hash.

`Saved/` is intentionally ignored by Git. A normal commit does not include the
locally trained checkpoint or evaluation JSON; reproduce them with these
commands or promote a deliberately selected checkpoint to a tracked artifact
location.

## Resume an interrupted run

The trainer writes `interrupted-NNNNNN/` if Unreal closes, the connection drops
or the process is interrupted after a completed update. Restart Unreal and
resume into a **new** output directory. `--episodes` is the original total
target, not the number of additional episodes.

Example for a run interrupted after episode 40:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"

.\RL\BlueTeam\.venv\Scripts\python.exe `
  .\RL\BlueTeam\Python\train_red_placement.py `
  --temporal-public-control `
  --resume ".\Saved\RedRL\reproduction-old\interrupted-000040" `
  --episodes 120 `
  --seed 91000 `
  --exposure-weight 20 `
  --output-dir ".\Saved\RedRL\reproduction-resumed-$stamp"
```

Keep the original episode seed and training hyperparameters when resuming.
The checkpoint restores policy/optimizer/RNG state, and training continues with
episode seed `91000 + completed episodes`.

## Reproduce the held-out evaluation

Keep Unreal running after training. The evaluator freezes all three policies
and gives each policy the same 20 Unreal seeds, `1500000000..1500000019`.
The random wedge baseline uses policy seed `8128`.

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$evalDir = ".\Saved\RedRL\evaluation-$stamp"

.\RL\BlueTeam\.venv\Scripts\python.exe `
  .\RL\BlueTeam\Python\evaluate_red_placement.py `
  --red-checkpoint "$trainDir\final" `
  --episodes 20 `
  --seed 1500000000 `
  --random-policy-seed 8128 `
  --output-dir $evalDir
```

The evaluator creates `evaluation.json` containing per-episode rows and policy
aggregates. A valid comparison requires all policies to use the same seed range,
the same map/build, the same frozen Blue opponent and no checkpoint selection
based on these evaluation results.

## Verified reference result

The corrected local run `Saved/RedRL/run-04` selected deterministic approach
sector 13. The following results came from 20 held-out seeds per policy:

| Policy | Valid episodes | Native Red reward | Public approach exposure |
| --- | ---: | ---: | ---: |
| Learned template softmax | 20/20 | -9.0 | **0.95931** |
| Random legal wedge | 20/20 | -9.0 | 0.96528 |
| Scripted radial | 20/20 | -9.0 | 0.96659 |

The learned policy reduced mean public exposure by approximately `0.62%`
relative to random wedge and `0.75%` relative to scripted radial. All 120
training placements and all 60 evaluation placements were valid. Native reward
remained `-9` for every evaluated policy, so do not describe this as a learned
breach or native-reward improvement.

Exact equality depends on the same source revision, Unreal build, map, public
sensor configuration and deterministic runtime behavior. Treat materially
different output as a result to investigate rather than silently replacing the
reference numbers.

## Run the learned checkpoint

After reproducing training, run one deterministic learned-policy episode:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"

.\RL\BlueTeam\.venv\Scripts\python.exe `
  .\RL\BlueTeam\Python\run_istana_live.py `
  --temporal-public-control `
  --red-checkpoint "$trainDir\final" `
  --seed 12345 `
  --paced `
  --output-dir ".\Saved\BlueLive\learned-red-$stamp"
```

Locally, the previously trained checkpoint is
`Saved/RedRL/run-04/final`; it is not version-controlled.

## Record training footage

`--paced` advances Unreal at visible simulation speed. A short five-episode
run is normally easier to record than the full 120-episode reproduction:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"

.\RL\BlueTeam\.venv\Scripts\python.exe `
  .\RL\BlueTeam\Python\train_red_placement.py `
  --temporal-public-control `
  --episodes 5 `
  --seed 92000 `
  --paced `
  --checkpoint-every 1 `
  --output-dir ".\Saved\RedRL\training-recording-$stamp"
```

This is real training from a fresh policy, but five episodes are not enough to
claim convergence. Use the 120-episode run and held-out evaluation for results.

## Record multi-direction swarm footage

The learned action catalogue intentionally concentrates groups into a wedge.
For footage showing swarms arriving from independent directions, use the
non-learning dispersed policy:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"

.\RL\BlueTeam\.venv\Scripts\python.exe `
  .\RL\BlueTeam\Python\run_istana_live.py `
  --temporal-public-control `
  --red-dispersed `
  --seed 24681 `
  --paced `
  --output-dir ".\Saved\BlueLive\dispersed-red-$stamp"
```

Changing the seed changes the legal randomized angles and radii. This policy is
a visual/demo baseline and must not be presented as the learned `run-04` model.

## Validation

From `RL/BlueTeam/Python`:

```powershell
..\.venv\Scripts\python.exe -m pytest -q `
  .\tests\test_red_policy.py `
  .\tests\test_istana_live.py
```

The current focused result is `42 passed`. The Unreal Editor Development build
also completed successfully:

```powershell
powershell -ExecutionPolicy Bypass -File .\Tools\build.ps1 -Target Editor
```

The broader Python suite was started but stopped before completion because of
long-running unrelated integration tests; it is not reported as passing.

## Troubleshooting

- **Connection refused:** start `Tools/start_blue_live.ps1` and wait for the map
  to finish loading before starting Python.
- **Port 8765 already in use:** reuse the existing live scene or close it before
  launching another. Only one bridge client should be active.
- **Output directory already exists:** choose a new timestamp. Output creation
  is intentionally exclusive to prevent overwriting evidence.
- **Unreal closed during training:** restart it and resume from the newest
  `interrupted-*` or periodic `checkpoint-*` directory.
- **Rejected placement:** retain the logged row and error. Unreal is
  authoritative; do not retry the mutation or substitute a random placement.
- **Different metrics:** confirm source revision, map, Blue opponent, training
  seeds, policy seed, hyperparameters and evaluation seeds before comparing.

## Known limitations and next step

- Red learns initial placement only, not flight control.
- The learned catalogue contains wedge directions only.
- Public exposure is a synthetic shaping proxy, not calibrated sensing risk.
- The evaluation does not establish real-world performance.
- Training is separate against frozen Blue, not joint Red/Blue self-play.

The highest-value next step is to introduce a controlled scenario where native
Red outcomes vary, then expand the learned action catalogue to include legal
multi-direction layouts. The dispersed demo produced varied approach behavior,
but it remains a random baseline until added to the learned action space and
evaluated on frozen seeds.
