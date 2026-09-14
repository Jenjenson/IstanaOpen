# Balanced v3: deployment efficiency experiment

This experiment changes the learned deploy-versus-stop decision, not the
simulation physics, public observation contract, reward or mixed curriculum.
It is an experimental candidate, not a demonstrated optimal layout planner
or a field-validated robustness release. The broader training goal remains
open; the reserved final-test suites have not been opened.

## Published candidate and common validation

The predeclared comparison selected **run 201, episode 5008**, with an equally
weighted mean return of **0.898869**. Run 202 scored 0.858240 and run 203
scored 0.760190. Selection used 300 paired cases in each of the normal,
stress and capability-varied profiles: **900 scenarios, 14 methods and
12,600 scored rollouts**. These cases were unseen by the weight updates,
but were used to select this candidate; they are not final-test evidence.

The [candidate checkpoint](Checkpoints/balanced-v3-candidate/checkpoint.json),
[matched initialized control](Checkpoints/balanced-v3-candidate-initialized/checkpoint.json),
[selection report](Results/balanced-v3/selection.json) and
[41-file publication bundle](Results/balanced-v3/artifact-manifest.json)
preserve exact weights, complete runs and compressed paired results.
The selected weights fingerprint is
`a48d3c3a5b43e99dd4545cfb7321eb64077dceb09432a60658c2649dd19f21c0`.
No existing default checkpoint or v1/v2 evidence was replaced.

| Selected candidate | Mean return | All-threat timely sensing | Mean detected fraction | Mean sensor cost |
| --- | ---: | ---: | ---: | ---: |
| Normal | 7.885 | 195/300 (65.00%) | 92.44% | 2.547 |
| Stress | -7.742 | 4/300 (1.33%) | 32.81% | 1.517 |
| Capability-varied | 2.553 | 119/300 (39.67%) | 78.59% | 2.288 |

Detected fraction is averaged per scenario, not pooled over all threats.
Costs are synthetic budget units, not currency. Every evaluated method
reported zero invalid actions.

### Comparison with existing policies and non-RL methods

| Method | Normal return | Stress return | Capability return | Equal-profile mean |
| --- | ---: | ---: | ---: | ---: |
| Selected balanced v3 | 7.885 | -7.742 | 2.553 | 0.899 |
| Matched initialized balanced actor | 7.812 | -8.230 | 2.310 | 0.630 |
| Frozen robust v2 | 7.809 | -8.370 | 2.290 | 0.576 |
| Frozen adaptive v1 | 7.643 | -8.522 | 1.829 | 0.317 |
| Greedy public coverage | 8.169 | -7.646 | 2.822 | 1.115 |
| Random legal | -4.312 | -11.683 | -6.268 | -7.421 |
| Uniform fixed RF/radar | -5.441 | -10.997 | -6.291 | -7.577 |
| Fixed RF | -5.225 | -10.160 | -6.127 | -7.171 |
| Fixed radar | -5.888 | -11.630 | -7.493 | -8.337 |
| Original toy-210 projected adapter | -6.022 | -11.612 | -5.879 | -7.837 |

The complete reports also retain both other trained candidates and all
three initialized controls. The initialized controls had identical
deterministic actions on these cases, as expected from their equal weights.

The principal result is **an efficiency/sensing tradeoff**, not uniform
robustness improvement. On stress cases, cost fell 30.8% versus the matched
initialized actor (2.193 to 1.517) and 36.4% versus v2 (2.384 to 1.517).
Detection nevertheless fell from 35.86% / 35.99%, respectively, to 32.81%;
all three retained only 4/300 all-threat successes.

The paired 95% bootstrap interval for stress cost difference versus v2 is
[-0.991, -0.745], and for detected-fraction difference it is
[-5.01, -1.39] percentage points. The return difference is +0.628
[+0.457, +0.801]. These are **descriptive post-selection intervals**, not
selection-adjusted or independent-test confidence statements.

Greedy remains a meaningful unsolved comparison. It achieves 68.0% normal
and 42.0% capability all-threat success, versus 65.0% and 39.67% for v3.
V3's return differences versus greedy are -0.284 [-0.528, -0.050] on normal
and -0.268 [-0.485, -0.060] on capability cases. Higher return than v2 is
therefore not a reason to claim this is the best available planner.

### Sensor changes and the STOP diagnostic

The selected actor used all five types on capability-varied cases:
**272 RF, 172 thermal, 127 fused, 50 radar and 9 EO deployments**. These are
actual choices under changing capabilities, availability, sites and budgets,
not a requirement to select every sensor type equally. Stress cases used
RF, thermal and fused only.

The three archived public-input probes repeat v2's 60 diagnostic stress
snapshots. Among the same 19 cases with low public forecast coverage:

| Actual actor | Initial deterministic STOP | Mean STOP probability |
| --- | ---: | ---: |
| Frozen v2 | 0/19 | 0.35% |
| Matched initialized balanced actor | 0/19 | 28.26% |
| Selected balanced actor | 11/19 | 46.67% |

The count-balanced policy makes stopping a learned, reachable option. This
does not establish that any particular STOP decision is optimal. Architecture
and inference changes already affect the initialized control; subsequent
weight learning must be assessed against that control, not v2 alone.

### Demonstration

Download or locally open [demo.html](Results/balanced-v3/demo.html) in a
browser; GitHub itself shows HTML source. It contains six consecutive,
identified capability validation cases, not selected successes. The replay
shows chosen sensor types/sites, nominal ranges, threat motion, detections,
timely sensing outcomes and reward components. Desktop/mobile layout,
scenario switching, playback, restart and the timeline were browser-checked.
The [guide](BALANCED.md) also includes a command to plan from a new public
snapshot using a different sensor catalogue.

## Training

Three independent sampling seeds transferred the same frozen v2 parameter
values into the count-balanced actor and reset Adam. Each completed 8,000
episodes and 500 updates: **24,000 additional training episodes**. These are
transfer trajectories, not independent from-scratch initializations.

The matched initialized actors use the new inference rule immediately, but
have received no additional weight updates. All three therefore make the
same deterministic decisions before training. The full run directories
preserve initialized/best/last checkpoints, optimizer/RNG state, logs,
configuration, source hashes and inherited seed exposure.

| Run seed | Completed episodes | Best checkpoint episode | Internal validation mean return |
| --- | ---: | ---: | ---: |
| 201 | 8000 | 5008 | 0.987041 |
| 202 | 8000 | 8000 | 0.827233 |
| 203 | 8000 | 1008 | 0.824142 |
| Matched initialized control | 0 additional | 0 | 0.681372 |

These scores average 60 fixed cases per profile (normal, stress and
capability-varied). They selected each run's checkpoint and are **not**
independent test evidence. Every training and internal-validation batch
reported zero invalid actions.

The internal validation shows a real efficiency/sensing tradeoff. For run
201's best checkpoint, stress spending fell from 2.080 to 1.383 relative to
its matched initialized control, while threat detection fell from 36.90%
to 35.24%. Stress all-threat success remained 0/60, capability success
remained 23/60, and normal success remained 41/60. This is not evidence of
uniformly stronger sensing just because mean return increased.

## Execution and evidence scope

The [protocol](Results/balanced-v3/protocol.json) was recorded before these
training runs and the common validation. It keeps all generated cases,
including unaffordable and physically poor sensing opportunities. Success
means timely confirmed sensing of every simulated threat, not interception.

An initial common-validation process was interrupted by an app-session
change. It had emitted normal and stress summaries but had not saved final
reports or selected a candidate. Training weights and the statistical
protocol were not changed in response. The validation runner was then
updated to persist each completed profile, verify explicit resume against
the full experiment binding, and support parallel profile workers. The
interrupted validation was rerun on the same declared development seeds;
it is not presented as a second independent evaluation.

The first GitHub CI run passed all 568 tests and smoke checks on Windows,
but Linux passed 567 tests and failed the exact public-input hash check in
the remaining probe test. Local Linux reproduction isolated 172 last-bit
floating differences across 47 of the 60 snapshots (maximum absolute
difference about 5.7e-14). Hashes of unrounded JSON therefore differed;
there was no corresponding sensor-choice or action-count change.

An additive [public-input corpus](Results/balanced-v3/portable-public-inputs.json.gz)
and its [separate manifest](Results/balanced-v3/portable-public-inputs.manifest.json)
preserve the producer's exact public observations. The portable verifier
checks those input hashes against all three original probes, verifies
runtime-generated observations with a tight floating tolerance, and repeats
actor inference on the archived inputs. Original file hashes, masks,
sensor/site choices, counts and source bindings remain exact. The original
41-file bundle, weights, probabilities and evaluation results are unchanged;
this supplement performs no sensing rollout or further training.

Local Trackio imports initially failed because inherited RNG provenance
contains integers larger than its JSON encoder supports. The dashboard-only
adapter now preserves those integers' exact digits as strings. Original
logs/checkpoints and their numerical semantics are untouched; all three
509-event training logs were subsequently imported locally and a metric
query verified persistence. No Hugging Face Space was created.

The initial public-only STOP diagnostic intentionally reuses the 60 stress
validation inputs examined in v2. Low forecast coverage is a public input,
not proof that actual sensing is impossible or that stopping is optimal.
The probe invokes each actor's actual deterministic rule; joint argmax is
reported separately because it is not the balanced actor's gate-first rule.

All replays are recorded simulator output. Dynamic placement means choosing
types and offered sites sequentially before threat motion; there is no
mid-flight relocation or physical device command. The shared public input
adapter accepts changed catalogues, but real sensor calibration and live
integration require separate validation.

## Remaining learning work

The repository also contains an unused, training-only
`triad_rl/counterfactual_rollout.py` research helper. It recreates a training
case and the already chosen placements, then measures the return from stopping
there. That action-independent baseline may help learn the value of an
additional sensor. It is attached only to a detached training record; hidden
outcomes never enter the actor's public observation.

This helper did **not** produce any v3 checkpoint or reported result. It has
not been trained or shown to improve sensing. A future experiment needs a
separate predeclared, matched comparison, explicit optimizer and advantage
normalization settings, and accounting for the additional simulator work.
Initial STOP has the same return in every case, so this baseline cannot by
itself fix the ranking of the first sensor/site choice.
