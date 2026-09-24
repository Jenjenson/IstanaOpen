# Four-sensor learning diagnostics

Read completed saved files and reconstruct deterministic policy choices locally; no training or native simulator calls.

| Trial | Seed | Unique sampled layouts | Changed episodes | Batch / actor steps | Logit change L2 | Final greedy sensors changed | Selected episode | Test warning gain | Detection gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 917 | 272 | 292 / 500 | 25 / 100 | 26.112203 | 0 | 0 | +0.000000 s | +0.000 pp |
| B | 918 | 302 | 316 / 500 | 25 / 100 | 24.117253 | 0 | 0 | +0.000000 s | +0.000 pp |
| C | 919 | 260 | 281 / 500 | 25 / 100 | 35.364988 | 0 | 0 | +0.000000 s | +0.000 pp |

## Trial A

Initial legal placement actions: 624; actor parameters: 897.
First-step probability mass on initial layout actions: 0.800000 initially, 0.892312 finally.
Validation warning: initial 21.192357 s; final 21.192357 s; selected 21.192357 s.
Test cases improved/tied/worse: 0/10/0.

## Trial B

Initial legal placement actions: 624; actor parameters: 897.
First-step probability mass on initial layout actions: 0.800000 initially, 0.888187 finally.
Validation warning: initial 21.192357 s; final 21.192357 s; selected 21.192357 s.
Test cases improved/tied/worse: 0/10/0.

## Trial C

Initial legal placement actions: 624; actor parameters: 897.
First-step probability mass on initial layout actions: 0.800000 initially, 0.875406 finally.
Validation warning: initial 21.192357 s; final 21.192357 s; selected 21.192357 s.
Test cases improved/tied/worse: 0/10/0.

## Interpretation limits

- Every predeclared training seed is retained; no winning seed is selected.
- The ten scenarios repeat across trials and are not thirty independent scenarios.
- Held-out tests are excluded from selection but are a reused fixed benchmark, not a newly blind panel.
- Parameter movement and sampled layout diversity do not establish improved deterministic deployment performance.
- A selected episode of zero means no evaluated checkpoint exceeded the initial validation mean; it does not mean all gradients were zero.
- This study compares new PPO with the contractor; it does not compare new PPO with the previous algorithm.
