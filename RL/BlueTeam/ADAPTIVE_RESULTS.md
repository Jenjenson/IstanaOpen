# Adaptive v1: measured improvement, remaining robustness gap

The adaptive policy generalizes substantially better than the original
fixed-scenario policy **inside the new synthetic simulator**. It does not
outperform the greedy coverage heuristic, and severe shifted conditions remain
unsolved. This is a verified first adaptive model, not a claim that the broader
robust-deployment objective is complete.

## Training and selection

- Three independent initializations, seeds 42/43/44, trained for 4,000 episodes
  each: **12,000 randomized episodes**, all with zero invalid actions.
- Each run selected a checkpoint on its own 100 validation scenarios. The
  three frozen candidates then faced the same new 300-scenario validation set.
- Common validation success: 63.67%, **65.33%**, and 62.67%. Seed 43 was selected
  before opening either final test suite. Its selected checkpoint is episode
  4,000, with weights SHA-256
  `2062905477954b810516fa14e2009d6d25afb391950354780c6161644eebae36`.
- Training, candidate validation, common selection and final testing have
  separate recorded seed ranges. Source, weights and scenario hashes are
  included in the reports. Evaluation leaves weights unchanged.

## Final unseen scenarios

All six methods faced the same 200 unseen randomized scenarios, starting at
seed `2000000000000000`. Success requires timely confirmed detection of **all**
threats. Detection below is mean per-episode fraction of threats ever detected.

| Method | Episode success | Threat detection | Mean reward | Mean cost |
| --- | ---: | ---: | ---: | ---: |
| Adaptive RL | **62.5%** | 92.68% | 7.541 | 2.530 |
| Same policy before training | 17.0% | 55.72% | −3.747 | 2.842 |
| Random legal deployment | 16.0% | 66.42% | −5.158 | 2.526 |
| Greedy public-coverage heuristic | **67.5%** | 93.60% | 7.927 | 2.699 |
| Fixed RF/radar layout | 11.0% | 44.14% | −5.996 | 1.794 |
| Original toy-210, projected | 14.5% | 40.31% | −5.864 | 2.680 |

The paired success-rate difference versus the original projected policy is
**+48 percentage points**, with a 95% paired bootstrap interval of **+41 to
+55.5 points**. Against the matched untrained policy it is +45.5 points
(+38 to +53.5). Against greedy it is **−5 points** (−9.5 to −0.5). These intervals
use 2,000 episode resamples; they describe this scenario sample, not sensor
model fidelity or training-run uncertainty.

The three independently trained candidates also faced this same final set
after selection was frozen: success was 62.0%, 62.5% and 61.0% for seeds
42/43/44. The mean is 61.83%, with a descriptive sample standard deviation of
0.76 percentage points across only three runs; this is not a population
confidence interval. No model was reselected using these test results.

The original policy is the actual frozen toy-210 checkpoint, not a substitute
fixed rule. Its old observation contract cannot represent the new weather and
track information; its continuous placements are projected to nearest legal
sites of the same sensor type. The report records those displacements. This
comparison is not an architecture-only ablation or native Unreal rerun.

## Shifted stress scenarios

Another 200 scenarios start at seed `2000000000010000`. They include larger,
higher, faster and more divergent swarms under worse sensing conditions.

| Method | Episode success | Threat detection | Mean reward | Mean cost |
| --- | ---: | ---: | ---: | ---: |
| Adaptive RL | 1.0% | 31.42% | −9.163 | 2.333 |
| Greedy public coverage | 1.0% | 29.77% | −8.027 | 1.212 |
| Original toy-210, projected | 0.0% | 12.86% | −11.586 | 2.645 |
| Same policy before training | 0.0% | 14.03% | −11.388 | 2.744 |
| Random legal deployment | 0.0% | 14.32% | −11.878 | 2.523 |
| Fixed RF/radar layout | 0.0% | 11.83% | −11.062 | 1.746 |

The RL improves detection but still spends resources on layouts with little
chance of timely confirmation. Greedy spends substantially less for the same
episode success rate. Some stress threats exceed the standard sensors'
physical reach in this model; success cannot be assumed achievable just by
training longer. A feasibility audit separates optimistic sensing limits from
policy shortcomings. No failing episodes are removed from these tables.

The audit finds that **123/200 stress scenarios (61.5%) are provably unable to
confirm every threat in time**, even after ignoring joint budget, maximum
site count and sensor-separation constraints. In total, 479/1,099 threats have
no sufficient sensing support before their deadlines. The resulting 38.5%
full-defence support ceiling is optimistic, not an achievable success rate.
In the remaining 77 scenarios, RL and greedy each succeeded twice; that gap
still mixes budget, geometry, stochastic sensing and policy choices. The audit
uses terminal ground truth only and is not supplied to the policy.

## Does the policy actually react to scenario information?

Forty separate validation snapshots were used for controlled **public-input**
changes, with frozen weights:

- Reversing the approach changed the first selected position in 100% of cases.
- Changing estimated emitter activity from silent to active switched the
  selected sensor type in 87.5% of cases.
- Changing clear/day conditions to dark/rain switched sensor type in 50%.

These are recommendation-sensitivity checks, not intervention defence trials.
All five sensor options remain supported; the learned model is not forced to
use every option. On the 200 normal held-out episodes, it chose RF, thermal,
fused and one EO deployment, but no standalone radar. Catalogue extensibility
is tested structurally; performance on newly calibrated devices is unproven.

## Evidence and next stage

[Results/adaptive-v1](Results/adaptive-v1/) contains readable summaries,
compressed complete reports, validation-selection evidence and all three
training logs. The complete `.json.gz` files retain individual scenarios,
per-method actions, outcomes and selected replays; compression does not remove
episodes. [The offline replay](Results/adaptive-v1/demo.html) can be downloaded
and opened locally. See [ADAPTIVE.md](ADAPTIVE.md) for reproduction commands.

The robustness objective remains open. The next stage should learn sensible
abstention and cost control on hard/unobservable scenarios, broaden training
conditions and capability variations, and validate on **new** reserved test
sets. These published tests are now consumed evidence, not independent test
data for a model tuned in response to them. Real sensor calibration and native
Istana integration are not verified by these simulation results.
