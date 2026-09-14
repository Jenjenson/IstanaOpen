# Ranking pilot: higher detection, but no timely-sensing improvement

The three-seed rollout-guided ranking experiment **failed its predeclared
development gate** against both public greedy and frozen v3. It spent more
resources and improved aggregate detection, but timely confirmation and original
return were lower. No endpoint is selected, scaled, or promoted as a replacement.

All three fixed endpoints completed 1,024 training episodes, followed by eight
methods on the same 600 new development cases: **3,072 training episodes and
4,800 evaluation episodes**. The three reserved final tests remain unopened.
See the [design guide](RANKING_PILOT.md),
[frozen protocol](Results/ranking-v5-pilot/protocol.json),
[complete aggregate](Results/ranking-v5-pilot/evaluation/aggregate.json), and
[18-example offline replay](Results/ranking-v5-pilot/demo.html).

## Held-out development results

Means weight normal, stress and changed-capability profiles equally, with 200
paired cases per profile. Timely sensing is `1 - breached_fraction`; all-threat
success means timely confirmation of every simulated threat, not interception.
Detection alone does not establish timely confirmation. Cost is in simulator
budget units, not currency. Every method had zero invalid actions.

| Method | Timely sensing | Detection | All-threat success | Cost | Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ranker 403 | 51.487% | 70.243% | 35.500% | 2.452 | 0.430 |
| Ranker 404 | 49.091% | 67.471% | 35.833% | 2.491 | -0.234 |
| Ranker 405 | 50.832% | 67.278% | 35.500% | 2.584 | 0.059 |
| Three-seed mean | 50.470% | 68.330% | 35.611% | 2.509 | 0.085 |
| Public-input greedy | 51.569% | 67.338% | 35.333% | 2.026 | 0.829 |
| Frozen balanced v3 | 50.789% | 66.319% | 31.833% | 2.052 | 0.585 |
| Frozen robust v2 | 50.142% | 67.149% | 31.667% | 2.406 | 0.166 |
| Frozen adaptive v1 | 48.715% | 65.832% | 30.333% | 2.376 | -0.174 |
| Anchored later-STOP pilot | 50.575% | 64.874% | 32.833% | 1.912 | 0.615 |

The three-seed mean averages outcomes within each shared case, **not actions**;
it is not an inference ensemble. Greedy has the highest mean timely sensing and
return in this comparison. No seed is selected after seeing these results.

Differences below are **three-seed mean minus reference**. Percentage-based
metrics and intervals use percentage points; cost and return retain their units.
Intervals use the declared 2,000 within-profile paired bootstrap resamples, seed
73043. They condition on these three trained endpoints, not the population of
training seeds, and are not multiplicity-adjusted superiority claims.

| Metric | Versus greedy [95% interval] | Versus v3 [95% interval] |
| --- | ---: | ---: |
| Timely sensing | -1.099 [-2.153, -0.088] | -0.319 [-1.440, +0.789] |
| Detection | +0.993 [+0.032, +1.980] | +2.011 [+0.979, +3.020] |
| All-threat success | +0.278 [-1.167, +1.778] | +3.778 [+1.889, +5.611] |
| Cost | +0.484 [+0.413, +0.553] | +0.457 [+0.393, +0.526] |
| Return | -0.744 [-0.944, -0.547] | -0.500 [-0.721, -0.290] |

## Why the fixed gate failed

Neither comparator meets the required one-percentage-point timely gain or
positive lower confidence bound. Mean detection and all-threat-success point
guards pass both comparisons; the mean-return guard fails both.

| Profile | Mean timely sensing | Difference vs greedy | Difference vs v3 | Mean cost |
| --- | ---: | ---: | ---: | ---: |
| Normal | 86.181% | -0.928 pp | +0.956 pp | 2.584 |
| Stress | 10.997% | -1.315 pp | -1.987 pp | 2.592 |
| Changed capabilities | 54.233% | -1.055 pp | +0.074 pp | 2.352 |

The one-percentage-point profile timely-decline guard fails on stress and
capability versus greedy, and stress versus v3. Capability detection versus
greedy declines **1.004365 percentage points**, narrowly beyond the exact limit;
rounding does not turn this into a pass. Seed 404 also fails the individual
timely safeguard (-2.478 pp versus greedy, -1.697 pp versus v3). Seeds 403 and
405 pass that safeguard, which does not establish superiority. These point
guards are not statistical noninferiority tests.

Cost rises 23.87% versus greedy and 22.28% versus v3. On stress cases, mean cost
is 2.592 versus greedy's 1.298 and v3's 1.505, despite lower timely sensing than
both. More detections or a small all-threat-success point gain therefore do not
resolve the sensing/resource-use tradeoff.

## What training changed

The policy began with exactly greedy's public utility and a zero neural
correction. Every endpoint completed 256 optimization updates. Rollout
preferences used private simulator outcomes only as training labels; deployed
actor features and the external-input planner remain public-only.

| Training diagnostic | Seed 403 | Seed 404 | Seed 405 |
| --- | ---: | ---: | ---: |
| Visited / informative states | 2,481 | 2,473 | 2,450 |
| Additional branch rollouts | 14,886 | 14,838 | 14,700 |
| Pairwise labels | 34,713 | 34,527 | 34,260 |
| Live choices differing from greedy on the same state | 1,472 | 1,385 | 1,302 |
| Deployments when greedy would STOP | 558 | 530 | 513 |
| Gradient-clipped updates / 256 | 20 | 12 | 5 |
| Collection seconds | 849.16 | 837.26 | 832.26 |
| Optimization seconds | 3.51 | 3.44 | 3.33 |
| Total trainer seconds | 1,025.66 | 1,015.23 | 1,003.61 |

The actual budget includes **44,424 additional branch rollouts and 103,500
pairwise labels**, beyond 3,072 live training episodes. All 7,404 visited states
were informative; all slates contained six alternatives. This is a changed
learning objective with extra simulator work, not a compute-matched v3 ablation.
Times were measured locally during concurrent CPU runs and are not portable
performance benchmarks.

Across the three runs, deployment differences from greedy included 1,279
site-only changes, 592 sensor-only changes and 687 changes to both. The actor
did learn different choices, but that does not imply those choices improved
unseen performance. Only seven actual STOP actions occurred, all early in
training and agreeing with greedy; the later tendency was to keep deploying.
There were zero invalid training actions. Falling training loss is not evidence
of improved held-out sensing or convergence.

## Evidence and practical use

The source and protocol were published in commit
[`bff3a77`](https://github.com/Jenjenson/IstanaOpen/commit/bff3a778655f2c35f6fb4c0c8666bca0e7c487ea)
before the new training or validation cases were sampled. The
[artifact manifest](Results/ranking-v5-pilot/artifact-manifest.json) binds 36
byte-exact inputs: protocol, three initialized/end checkpoints and training
records, evaluation input lock, raw reports, receipts and aggregate. The manifest
is a separate 37th file; the HTML demo is a separate derived artifact.

The frozen evaluator recomputes primary vectors, paired intervals and the gate,
and replays optimization from the archived comparisons without sampling new
scenarios. A separate read-only audit checked all 4,800 records for all 17
metrics, scenario/group/sequence hashes, nine subgroup categories, and inherited
paired intervals. Those secondary checks are not all performed by the frozen
evaluator alone. Independent NumPy recomputation also matched the ten primary
comparator-by-metric mean differences and intervals. No checkpoint was changed
or selected during these checks.

The complete Windows-generated experiment also passed the frozen verifier on
Linux (Python 3.12.3, NumPy 2.4.6), with all 36 input hashes unchanged and guards
rejecting new scenarios, deployment inference and filesystem writes. Derived
numeric replay uses the declared `1e-12` tolerance; artifact identities remain
exact. The training/publisher source commit passed
[Windows and Linux CI](https://github.com/Jenjenson/IstanaOpen/actions/runs/34842509393).
Five additional static publication tests pass on both platforms, pinning the
manifest and demo bytes and matching all 18 replays to their scored records.

The [offline HTML demo](Results/ranking-v5-pilot/demo.html) contains exactly the
first two already-scored cases per profile for **every** fixed endpoint: 18
examples, not selected successes. Recreated scenario hashes, actions, layouts,
metrics and outcomes must match the scored reports. Download the HTML and open
it locally; GitHub's file view displays source rather than running the viewer.
It shows sensor types, placements, nominal coverage, threats, detections,
episode outcomes and rewards. Threat trails and truth markers are evaluation
visuals, not policy inputs. A replay is not live control or independent final
testing.

Browser verification of the actual 18-case HTML passed at desktop and mobile
sizes: playback, restart, scenario selection and timeline scrubbing worked, with
no console errors, blank page, error overlay or horizontal mobile overflow.

The fixed seed-403 endpoint also passed an external-format example-snapshot
smoke test with the interchangeable catalogue: thermal sensors at sites 26 and
27, then STOP, with `physical_commands_sent: false`. This establishes interface
compatibility only, not calibration or real-world effectiveness. All 64 batch
events per endpoint were imported into local-only Trackio; the original JSONL
records remain authoritative.

Further development needs better timely sensing and resource use, particularly
on stress cases. This failed experiment does not justify simply increasing its
training budget, replacing the existing policy, or opening reserved final tests.
