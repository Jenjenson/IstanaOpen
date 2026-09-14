# Robust v2: modest validation gains, stopping gap unresolved

**Status: experimental candidate, not a final-tested robustness improvement.**
The training distribution now includes severe conditions and changing sensor
capabilities, but this iteration does not yet solve efficient deployment under
poor sensing opportunities. The independent final suites declared in the
[protocol](Results/robust-v2/protocol.json) have not been opened.

## Training and common validation

Three runs transferred the same published v1 weights, reset optimizer state,
and used independent policy random seeds 101/102/103. Each completed 8,000
episodes and 500 on-policy updates: **24,000 additional episodes**, all with
zero invalid actions. These are independent transfer trajectories, not three
independent from-scratch initializations. Reward, sensing physics and the
public feature contract remain identical to v1.

Each run selected its best checkpoint using 60 fixed validation cases per
profile. The frozen candidates then faced another common 300 cases per profile,
900 distinct validation cases in total. The equally weighted mean-return
criterion selected run **103, episode 8000**, with weights SHA-256:

`5c982c86a5da2e5b4960ba437021f751a36dc3af7187515903cf1b1ee2ab27ce`

The three candidates' common-validation balanced returns were −0.174, −0.292
and **−0.131**, respectively. No independent final-test evidence is implied by
selecting and reporting the best candidate on this validation set.

| Profile | Method | All-threat success | Threats ever detected | Mean return | Mean cost |
| --- | --- | ---: | ---: | ---: | ---: |
| Normal | Selected v2 candidate | 61.67% | 91.95% | 7.529 | 2.580 |
| Normal | Frozen adaptive v1 | 60.00% | 90.72% | 7.232 | 2.537 |
| Normal | Greedy public coverage | 66.67% | 93.97% | 7.891 | 2.696 |
| Severe stress | Selected v2 candidate | 1.67% | 32.24% | −9.031 | 2.359 |
| Severe stress | Frozen adaptive v1 | 1.33% | 30.23% | −9.315 | 2.321 |
| Severe stress | Greedy public coverage | 1.33% | 27.83% | −8.129 | 1.202 |
| Capability-varied | Selected v2 candidate | 33.33% | 76.77% | 1.110 | 2.474 |
| Capability-varied | Frozen adaptive v1 | 31.00% | 73.85% | 0.527 | 2.437 |
| Capability-varied | Greedy public coverage | 36.67% | 77.44% | 1.782 | 2.228 |

All methods faced paired scenario, catalogue and detection-draw seeds. Cases
were retained even if no affordable or useful sensor was available. Capability
variation includes both ordinary and severe threat families, so its percentage
cannot be directly compared with the ordinary profile as if difficulty were
the same. Success means timely confirmed sensing of every threat, not physical
interception or measured field reliability.

On the capability-varied validation set, the selected candidate used all five
profiles: 334 RF, 78 radar, 30 EO, 140 thermal and 108 fused deployments. This
demonstrates decisions conditioned on varied synthetic capabilities; it does
not establish that an arbitrary real device is correctly calibrated.

## Why more mixed training was not enough

The selected candidate's stress detection is slightly better, but its spending
is still almost twice greedy's and slightly higher than frozen v1's. Broader
training alone has not taught sensible abstention.

A public-input diagnostic of the transferred v1 policy found that 19 of the
60 stress-training-validation cases had maximum legal forecast marginal
coverage at most 0.01. Their initial mean STOP probability was only about
**0.103%**. Hundreds of sensor/site options collectively compete with one STOP
option; high action entropy can therefore reflect exploration among deployment
sites while almost never exploring stopping. This is an observed learning
bottleneck, not proof that STOP is mathematically unreachable or optimal.

On the same 19 public snapshots, the selected v2 candidate increased mean
initial STOP probability to **0.346%**, but still never chose STOP
deterministically. Neither policy chose initial STOP in any of the 60 cases;
all had legal deployment options, so this is not a forced-action effect.
The paired probe records and hashes are preserved alongside the results.

Forecast coverage is imperfect: zero forecast coverage does not prove that
true sensing is impossible. Consequently this evidence does **not** justify
hard-coding a zero-coverage stop rule. The next experiment should improve the
learned deploy-versus-stop exploration and then compare detection, cost and
success on validation before spending the reserved final-test suites.

## Preserved evidence

[Results/robust-v2](Results/robust-v2/) contains the selection report, complete
compressed paired validation records, full training directories and a
capability-varied offline replay. The candidate and its matched pretraining
weights are separately versioned; the original adaptive-v1 artifacts are
untouched. The broader robustness goal remains open.
