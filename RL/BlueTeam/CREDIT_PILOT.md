# Credit-assignment pilot

This is a controlled training experiment, not a replacement for the published
[v3 candidate](BALANCED.md). It asks whether the shared critic or its training
baseline is limiting useful sensor-placement learning. The public inputs, sensor
physics, reward, actor architecture, action masks, and inference rule stay fixed.

## What changes

All arms copy the same selected v3 weights, reset Adam, reset the sampling RNG,
and train on the same 1,024 mixed-profile scenario seeds in batches of 16.

| Arm | Recorded training baseline | Critic gradient coefficient |
| --- | --- | ---: |
| `shared_critic` | Existing learned value prediction | 0.5 |
| `no_critic_gradient` | Existing learned value prediction | 0 |
| `paired_stop` | Simulated return from stopping at the current committed layout | 0 |

The paired STOP baseline is computed in an independent, training-only simulator
branch before the next sampled decision. It does not consume the live simulation
or actor RNG, expose private truth to actor observations, or label STOP as optimal.
Each branch costs additional simulation work; matched live episode counts are not
matched compute budgets.

The original optimizer and batch advantage normalization are unchanged. A sampled
STOP has zero *raw* advantage under its paired baseline but may have nonzero
normalized advantage after batch centering. The initial STOP return is always
`-10.5`; the state-dependent credit-assignment benefit can only start after a
placement. With no critic gradients, the critic head stays fixed but predictions
can still change because the actor and critic share an encoder.

## Predeclared decision rule

The [protocol](Results/credit-v4-pilot/protocol.json) fixes every endpoint and the
evaluation scenarios before training. There is no internal validation, best
checkpoint selection, or early stopping. The three endpoints are compared with
frozen v3, v2, v1, and a public-input greedy baseline on 200 new development cases
per profile: normal, stress, and changed sensor capabilities.

Only `paired_stop` is the primary pilot arm. To justify multi-seed replication,
it must meet **all** of these conditions against **both** the trained
`shared_critic` control and frozen v3:

- Improve the equal-profile mean timely-confirmed fraction by at least one
  percentage point, with a paired stratified 95% bootstrap interval above zero.
- Lose no more than one percentage point in any individual profile's point
  estimate of timely-confirmed fraction.
- Have no lower equal-profile all-threat success point estimate.

Timely-confirmed fraction means `1 - breached_fraction`. Success means timely
confirmed sensing of every simulated threat, **not interception**. Lower spending
or higher reward alone cannot pass this gate. The point-estimate safeguards are
not statistical noninferiority claims. The `paired_stop` versus
`no_critic_gradient` contrast is needed before attributing any improvement to the
baseline rather than removal of critic gradients.

This is a one-seed screening experiment. Scenario bootstrap intervals do not
measure training-seed uncertainty; passing does not promote the model or prove
robustness. Reserved final tests remain unopened.

## Reproduce

Use the Python environment from [BALANCED.md](BALANCED.md). From `RL/BlueTeam/Python`,
run each arm into a separate **new** directory:

```powershell
python -m triad_rl.train_credit_pilot --protocol ../Results/credit-v4-pilot/protocol.json --arm shared_critic --output ../runs/credit-pilot/shared_critic
python -m triad_rl.train_credit_pilot --protocol ../Results/credit-v4-pilot/protocol.json --arm no_critic_gradient --output ../runs/credit-pilot/no_critic_gradient
python -m triad_rl.train_credit_pilot --protocol ../Results/credit-v4-pilot/protocol.json --arm paired_stop --output ../runs/credit-pilot/paired_stop
```

The trainer checks the frozen source and publication hashes, never overwrites a
run, and saves `initialized`, `last`, the exact protocol, configuration, JSONL
metrics, and a summary. It does not support resume for this bounded pilot.
First-batch trajectory digests must match across all three arms; later policies
can choose different trajectory lengths. Logs separate first/later and
STOP/deployment credit statistics, critic/actor gradients, clipping, sensing,
cost, and extra branch work.

Evaluate only the fixed endpoints:

```powershell
python -m triad_rl.evaluate_credit_pilot --protocol ../Results/credit-v4-pilot/protocol.json --run shared_critic=../runs/credit-pilot/shared_critic --run no_critic_gradient=../runs/credit-pilot/no_critic_gradient --run paired_stop=../runs/credit-pilot/paired_stop --output ../runs/credit-pilot/evaluation
```

The evaluator writes one compressed report and receipt per profile, then
`aggregate.json`. Explicit `--resume` verifies completed profiles before running
missing ones. A partial file/receipt pair is rejected rather than overwritten.
`aggregate.json` contains the fixed-primary decision, paired confidence
intervals, and exploratory layout-change groups; those post-training groups
are descriptive, not causal estimates. The logged `critic_loss` remains the
learned head's prediction error even in the paired STOP arm; use the separate
credit statistics to inspect the substituted baseline.

Tracking is optional and local only. It is imported after training so dashboard
dependencies cannot affect optimization:

```powershell
python track_adaptive.py --run-dir ../runs/credit-pilot/paired_stop --project blue-team-adaptive --name credit-v4-pilot-paired-stop
```

The experiment does not add a live C2 connection, move sensors during a flight,
or send device commands. Existing modular public-input recommendation APIs and
the v3 replay demo remain available without these training-only branches.
