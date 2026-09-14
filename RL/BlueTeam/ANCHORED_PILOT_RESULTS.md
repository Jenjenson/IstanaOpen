# Anchored pilot: no demonstrated timely-sensing gain over v3

The completed experiment **did not pass its predeclared scaling gate**. The
anchored arm improved timely sensing over its matched no-critic-gradient control, but
was essentially unchanged versus the frozen v3 policy. It is archived for
reproducibility, not promoted as a replacement.

Three fixed endpoints trained for 4,096 episodes each: **12,288 live training
episodes**, followed by **4,800 evaluation rollouts** covering eight methods
on the same 600 new development cases. All three reserved final tests remain
unopened. See the [design and reproduction guide](ANCHORED_PILOT.md),
[frozen protocol](Results/anchored-v4-pilot/protocol.json), and
[complete aggregate](Results/anchored-v4-pilot/evaluation/aggregate.json).

## Held-out development results

Means weight normal, stress and changed-capability profiles equally, with 200
paired cases per profile. Timely sensing is `1 - breached_fraction`; all-threat
success means timely confirmed sensing of every simulated threat, not interception.
Cost is in simulator budget units, not currency. Every method had zero invalid
actions.

| Method | Timely sensing | All-threat success | Detection | Cost | Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared critic control | 52.036% | 35.333% | 66.234% | 2.029 | 0.895 |
| No-critic-gradient control | 51.510% | 35.667% | 64.311% | 1.860 | 0.837 |
| **Anchored later STOP** | **52.452%** | **36.833%** | **65.944%** | **2.035** | **0.968** |
| Frozen balanced v3 | 52.428% | 35.833% | 67.469% | 2.149 | 0.921 |
| Frozen robust v2 | 52.667% | 35.667% | 67.644% | 2.499 | 0.627 |
| Frozen adaptive v1 | 50.919% | 34.000% | 66.036% | 2.457 | 0.229 |
| Prior all-step STOP pilot | 52.690% | 35.833% | 68.349% | 2.398 | 0.765 |
| Public-input greedy | 52.995% | 36.667% | 69.271% | 2.126 | 1.116 |

Paired differences below are **anchored minus reference**. Timely differences
and their 95% intervals are in percentage points; cost and return retain their
original units. Intervals use the fixed 2,000 within-profile paired bootstrap
resamples, seed 73042.

| Reference | Timely difference [95% interval] | Cost difference | Return difference |
| --- | ---: | ---: | ---: |
| Matched no-critic-gradient control | +0.942 [+0.249, +1.684] | +0.175 | +0.131 |
| Frozen v3 | +0.024 [-0.478, +0.541] | -0.114 | +0.048 |
| Shared critic control | +0.416 [-0.221, +1.073] | +0.005 | +0.073 |
| Prior all-step STOP pilot | -0.238 [-0.858, +0.419] | -0.363 | +0.203 |
| Public-input greedy | -0.543 [-1.306, +0.296] | -0.092 | -0.148 |

The matched-control timely gain falls short of the fixed **one-percentage-point**
threshold. Against v3, the gain is near zero and its interval includes a loss.
Both comparisons satisfy the profile, all-threat-success and mean-return point
guards; those are not noninferiority tests. The failed conditions remain failed:
the threshold is not rounded down or changed after seeing the data.

Greedy has the highest mean timely sensing and return among the evaluated
methods. The anchored arm's slightly higher all-threat success point estimate
does not establish superiority; its paired difference versus greedy is +0.167
percentage points, with a 95% interval of [-1.333, +1.667]. Scenario intervals
condition on these endpoints and do not measure training-seed uncertainty.

## Where the tradeoff remains

| Profile | Anchored timely sensing | Difference versus v3 | Cost difference versus v3 |
| --- | ---: | ---: | ---: |
| Normal | 84.192% | +0.200 pp | +0.031 |
| Stress | 14.632% | -0.773 pp | -0.269 |
| Changed capabilities | 58.532% | +0.645 pp | -0.104 |

The small normal/capability gains are offset by lower stress sensing. Overall
detection also falls by 1.525 percentage points versus v3. Lower spending and
a slightly higher mean return are therefore not evidence of stronger sensing.

The prior all-step STOP pilot is evaluated on these same cases, but trained for
only 1,024 episodes with a different seed. Its comparison cannot isolate the
baseline change. Post-treatment layout-change groups are included in the
aggregate for diagnosis, not as causal effects or extra pass criteria.

## What the training change actually did

All arms copied the same frozen v3 weights, reset Adam and the sampling RNG,
and completed 256 updates. The first batch contains 16 episodes / 38 decisions:
sampled trajectories and reference coefficient hashes match exactly across
all arms. Every batch preserved first-decision applied coefficients exactly
relative to its own ordinary reference.

| Training diagnostic | Shared critic | No critic gradient | Anchored |
| --- | ---: | ---: | ---: |
| Deployments | 7,924 | 7,793 | 7,742 |
| First / later STOP | 446 / 532 | 557 / 449 | 520 / 426 |
| Live decisions | 8,902 | 8,799 | 8,688 |
| Extra terminal branches | 0 | 0 | 4,592 |
| Clipped updates | 253 / 256 | 1 / 256 | 3 / 256 |
| Mean training return | 1.212 | 1.350 | 1.435 |
| Measured rollout seconds | 135.18 | 132.70 | 333.30 |
| Total trainer seconds | 235.23 | 230.47 | 424.00 |

Actual training used Python 3.11.9 and NumPy 2.4.6 on CPU. These elapsed times
were measured locally with concurrent runs; they are not
portable performance benchmarks. Total trainer time includes its preflight and
final verification, but not interpreter startup or held-out evaluation.

The anchored count is exact: **8,688 decisions - 4,096 first decisions = 4,592
later-only branches**. There are no initial STOP branches. All arms saw the same
profile counts: 1,660 normal, 1,153 stress and 1,283 capability cases. Anchored
deployments used RF/radar/EO/thermal/fused **4,173 / 398 / 157 / 1,377 / 1,637**
times. Zero invalid actions occurred in training.

In the first batch, later raw-advantage variance changes from 23.128 to 7.497;
first-decision coefficients remain identical. The substituted vector is not
recentered: all 22 later normalized coefficients in that first batch are
positive, including three raw-zero STOP coefficients. Over the full anchored
run, all 426 later STOP raw advantages are zero, while their normalized signs
split 231 negative / 195 positive. These scalar moments are not estimates of
policy-gradient variance or proof of improved learning.

Both zero-critic-gradient arms preserve the value head and its Adam moments
byte-exactly, but their shared encoders change. Weighted critic-gradient norms
exceed actor-plus-entropy gradient norms in all 256 shared-critic batches,
without consistent directional opposition. Removing
that dominance did not by itself produce stronger held-out sensing than v3.

## Evidence and practical use

The source and protocol were published in commit
[`2abd0cb`](https://github.com/Jenjenson/IstanaOpen/commit/2abd0cb9ea2cd01ac51e8669af7663d0ce0c9e1e)
before the new training scenarios were sampled. The
[artifact manifest](Results/anchored-v4-pilot/artifact-manifest.json) binds 33
exact artifacts: protocol, all three initialized/end checkpoints and logs,
input lock, raw paired profile reports, receipts and aggregate. It also binds
the evaluation and archive implementations. No endpoint was selected as "best".

An independent read-only audit matched all 4,800 records, 408 per-report scalar
summary means, all 85 paired confidence intervals, layout groups and the fixed
gate calculation. It did not invoke inference or sample additional scenarios.
Exact archive regeneration and aggregate recomputation also pass on Windows
and Linux with guards rejecting writes, inference and scenario execution. The
predeclared source commit passed the full GitHub Windows/Linux workflow.

The [existing v3 demo](Results/balanced-v3/demo.html) remains available; it is
not a replay of this new checkpoint. The anchored endpoint separately passed
an external-format example-snapshot smoke test: fused sensor at site 27, then
STOP, with `physical_commands_sent: false`. This confirms interface compatibility
only, not live-sensor calibration or real-world effectiveness. All 256 batch
events per arm were imported into local-only Trackio; original JSONL logs remain
authoritative.

The checkpoint remains experimental. Further work must improve sensing over
v3 and non-RL references, particularly on hard cases, rather than merely moving
along the cost/detection tradeoff. This pilot does not authorize scaling or
opening the reserved final tests.
