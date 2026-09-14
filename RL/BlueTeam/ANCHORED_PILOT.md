# Anchored later-placement pilot

This follow-up tests whether better credit for **later sensor placements** can
improve timely sensing without the inefficient extra deployment observed in the
[previous pilot](CREDIT_PILOT_RESULTS.md). It is a predeclared development
experiment, not a replacement policy or a final-tested robustness claim.

The [completed results](ANCHORED_PILOT_RESULTS.md) failed the fixed scaling gate:
the matched-control improvement did not translate into a clear gain over v3.

## The controlled change

Each arm starts from the same frozen v3 parameters, with fresh Adam state and
the same sampling RNG. All three train for 4,096 mixed-profile episodes, in 256
batches of 16. The endpoint is fixed: no internal validation, early stopping,
checkpoint selection, or training resume.

| Arm | First-decision credit | Later-decision credit | Critic gradient |
| --- | --- | --- | ---: |
| `shared_critic` | Ordinary learned baseline | Ordinary learned baseline | 0.5 |
| `no_critic_gradient` | Ordinary learned baseline | Ordinary learned baseline | 0 |
| `anchored_later_stop` | Ordinary learned baseline | Independent paired STOP baseline | 0 |

The new arm calculates the **ordinary reference batch mean and standard
deviation before substituting later baselines**. It applies that same reference
normalizer to the substituted advantages, without centering them again. This
keeps every first-decision learning coefficient byte-identical to its ordinary
reference on that same batch, including an initial STOP. The two controls keep
the original updater unchanged.

Paired STOP branches run only after an existing placement. They use a separate
simulator and do not consume the actor or live simulation RNG. Their reward is
training-only data: private simulator truth is never an actor feature, an
external inference input, or an optimal-STOP label. Equal live episode counts
are not equal compute budgets.

The first batch must have identical sampled trajectories and reference
coefficients across arms. Later trajectories can diverge. Shared parameters
mean that preserving first-decision coefficients does **not** guarantee equal
future stopping behavior. This remains a transition-averaged, batch-normalized
optimizer; it is not claimed to be an exactly unbiased policy-gradient estimate.

## Fixed evaluation and decision

The [machine-readable protocol](Results/anchored-v4-pilot/protocol.json) fixes
new training and validation seed ranges, implementation hashes and the rule
before any experiment scenarios are sampled. All published predecessor
exposure is checked; the three reserved final-test ranges remain unopened.

Eight methods are compared on the same 200 cases in each profile: normal,
stress, and varied sensor capabilities. They are the three trained arms,
frozen v3/v2/v1, the previous all-step STOP pilot, and public-input greedy
coverage. The previous pilot trained for 1,024 episodes on a different seed;
it is a diagnostic reference, **not** a matched training control.

The primary `anchored_later_stop` arm must satisfy every condition against
**both** its matched `no_critic_gradient` control and frozen v3:

- At least one percentage point more timely-confirmed threats, averaged equally
  across the three profiles, with a paired stratified 95% interval above zero.
- No profile's point estimate declines by more than one percentage point.
- Neither equal-profile all-threat success nor mean episode return is lower.

The return guard checks against trading a small sensing gain for inefficient
extra deployment. Higher return or lower spending alone cannot pass. Intervals
use 2,000 paired within-profile bootstrap resamples; the point guards are not
statistical noninferiority tests. A pass authorizes multi-seed replication only,
not promotion. One training seed cannot establish training-seed robustness.

## Run the experiment

Use the Python setup in [BALANCED.md](BALANCED.md). From `RL/BlueTeam/Python`,
run each arm into a separate new directory:

```powershell
python -m triad_rl.train_anchored_pilot --protocol ../Results/anchored-v4-pilot/protocol.json --arm shared_critic --output ../runs/anchored-pilot/shared_critic
python -m triad_rl.train_anchored_pilot --protocol ../Results/anchored-v4-pilot/protocol.json --arm no_critic_gradient --output ../runs/anchored-pilot/no_critic_gradient
python -m triad_rl.train_anchored_pilot --protocol ../Results/anchored-v4-pilot/protocol.json --arm anchored_later_stop --output ../runs/anchored-pilot/anchored_later_stop
python -m triad_rl.evaluate_anchored_pilot --protocol ../Results/anchored-v4-pilot/protocol.json --run shared_critic=../runs/anchored-pilot/shared_critic --run no_critic_gradient=../runs/anchored-pilot/no_critic_gradient --run anchored_later_stop=../runs/anchored-pilot/anchored_later_stop --output ../runs/anchored-pilot/evaluation
```

Training saves the initialized and fixed-endpoint checkpoints, exact protocol,
configuration, JSONL batch diagnostics, and summary. The evaluator verifies all
eight artifacts for every arm. It writes a compressed report and byte receipt
per profile, then the aggregate and fixed gate decision. Explicit evaluation
`--resume` validates saved inputs and all cached profile reports before sampling
missing profiles. Incomplete report/receipt pairs are rejected and never overwritten.

`Tools/archive_anchored_pilot.py` can verify and copy a complete terminal run
bundle into `Results/anchored-v4-pilot`. It recomputes the paired statistics and
gate from recorded reports, never calls training or inference, and copies exact
bytes only after input, privacy and destination checks. Existing identical
artifacts are accepted; differing artifacts are never replaced. Archival does
not select or promote a policy.

Optional local Trackio import runs after training, without affecting learning:

```powershell
python track_adaptive.py --run-dir ../runs/anchored-pilot/anchored_later_stop --project blue-team-adaptive --name anchored-v4-pilot-anchored-later-stop
```

## Scope

RF, radar, EO, thermal and fused sensor options, changing catalogues, public
observations, reward, physics and the inference interface are unchanged. The
existing [offline demo](Results/balanced-v3/demo.html) and external public-input
planner remain usable. This experiment concerns initial sequential sensor
type/site planning, not in-flight relocation. Timely confirmed sensing is not
interception. No live C2 connection, native Unreal validation, or physical
device commands are added or claimed.
