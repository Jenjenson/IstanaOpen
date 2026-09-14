# Credit-assignment pilot: measured results

**Decision: do not scale or promote this endpoint.** Paired STOP produced a small
timely-sensing gain over frozen v3, but did not clear the predeclared one percentage
point improvement against either primary comparator. Its difference from the
matched shared-critic control was uncertain, and it spent more than both.

This is completed development evidence, not a final robustness test. The
[protocol and methods](CREDIT_PILOT.md) were frozen and pushed in commit
`fe6120a4a4fb5b01f11e05f0c89cc925cf60ac99` before the actual runs. All three arms
started from identical selected v3 parameters with fresh Adam and identical
sampling RNG, then completed 1,024 episodes and 64 updates each. Only those fixed
endpoints were evaluated; no best checkpoint or winning arm was selected.

## Sensing and resource use

Each method received the same 200 new normal, 200 stress, and 200 changed sensor
capability cases: **4,200 scored rollouts** across seven methods. Means below give
each profile equal weight. Timely sensing is `1 - breached_fraction`; all-threat
success means timely confirmed sensing of every simulated threat, not interception.
Cost is the simulator's cost unit, not a currency amount.

| Method | Timely sensing | All-threat success | Detection | Mean cost | Mean return |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared critic, continued training (A) | 49.530% | 32.667% | 67.093% | 2.123 | 0.364 |
| No critic gradient (B) | 49.073% | 32.500% | 65.044% | 1.947 | 0.359 |
| Paired STOP (C, primary) | 49.691% | 33.000% | 68.564% | 2.388 | 0.212 |
| Frozen balanced v3 | 49.282% | 32.833% | 67.276% | 2.133 | 0.321 |
| Frozen robust v2 | 49.538% | 32.833% | 68.163% | 2.486 | 0.049 |
| Frozen adaptive v1 | 48.346% | 32.167% | 67.004% | 2.456 | -0.237 |
| Public-input greedy baseline | 50.182% | 33.500% | 68.671% | 2.126 | 0.557 |

These numbers use new development cases and should not be compared by subtraction
with the earlier v3 selection tables, which used different cases. All 4,200
evaluation rollouts had zero invalid actions.

Paired STOP differences on these same cases, with paired stratified 95% bootstrap
intervals (2,000 draws, fixed seed 73041):

| C minus reference | Timely sensing difference, percentage points | Cost difference | Return difference |
| --- | ---: | ---: | ---: |
| Shared critic A | +0.161 [-0.216, +0.535] | +0.265 [+0.217, +0.312] | -0.152 [-0.230, -0.071] |
| No critic gradient B | +0.618 [+0.164, +1.099] | +0.440 [+0.379, +0.502] | -0.146 [-0.242, -0.049] |
| Frozen v3 | +0.409 [+0.065, +0.758] | +0.255 [+0.211, +0.300] | -0.109 [-0.182, -0.032] |
| Greedy | -0.491 [-1.342, +0.428] | +0.262 [+0.205, +0.316] | -0.345 [-0.508, -0.165] |

The primary C-versus-A and C-versus-v3 gains are both below 1 percentage point;
the C-versus-A interval also crosses zero. The profile and all-threat-success
point-estimate safeguards passed, but the conjunctive gate still **failed**.
The diagnostic improvement over B does not change the predeclared primary gate.
These intervals measure scenario uncertainty for this one training seed, not
training-seed uncertainty or statistical noninferiority.

| Timely sensing by profile | Shared critic A | Paired STOP C | Frozen v3 | Greedy |
| --- | ---: | ---: | ---: | ---: |
| Normal | 83.783% | 83.408% | 83.308% | 84.467% |
| Stress | 11.692% | 12.249% | 11.670% | 11.705% |
| Changed capabilities | 53.115% | 53.416% | 52.868% | 54.374% |

## What the training diagnostics showed

The first batch is exactly paired: 16 scenarios, 36 decisions, and identical
feature/mask/action/reward trajectory digest
`191eb1ded18e4bb7e37365c19adc2c94be082d9366be23cc96fb45677eb8a9f0`.
Before any update, A and B therefore have identical credit statistics.

| First-batch advantage variance | A / B | Paired STOP C |
| --- | ---: | ---: |
| First decision, raw | 18.145 | 87.387 |
| Later decisions, raw | 19.008 | 16.644 |
| First decision, normalized | 0.971 | 1.119 |
| Later decisions, normalized | 1.017 | 0.213 |

C reduced later raw variance by 12.4% in this matched batch but increased the
first-decision variance 4.82-fold. Its constant initial baseline `-10.5` cannot remove
cross-scenario difficulty. Batch centering made all three zero-raw-advantage STOP
samples negative. Across C's full run, all 183 sampled STOPs likewise had zero raw
advantage and negative normalized advantage. Later trajectories differ between
arms, so full-run distribution comparisons are descriptive rather than matched
causal comparisons.

A's weighted critic gradient norm exceeded its actor-plus-entropy norm in all
64 updates; all 64 were clipped. B clipped once and C never clipped. However,
A's mean actor/critic gradient cosine was -0.002, not evidence of consistent
directional opposition. Adam step size is not determined by clipping scale alone.
B and C preserved their critic-head parameters and Adam moments byte-for-byte,
while all three shared encoders changed.

Each arm encountered 430 normal, 301 stress, and 293 capability training cases,
with zero invalid actions. All five sensor types were used:

| Training arm | RF | Radar | EO | Thermal | Fused | Deployments | Sampled STOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 994 | 120 | 51 | 423 | 397 | 1,985 | 254 |
| B | 939 | 90 | 66 | 467 | 389 | 1,951 | 274 |
| C | 1,104 | 118 | 67 | 436 | 412 | 2,137 | 183 |

C required **2,320 additional terminal branch rollouts** beyond its 1,024 live
episodes. Observed rollout time was 131.28 seconds versus 35.52 for A and 35.11
for B; complete process time was 161.54 versus 69.78 and 68.90 seconds. These were
concurrent local runs, not a controlled hardware benchmark or matched-compute
comparison. Local Trackio imports retained all 64 training events per arm; no
Hugging Face Space or cloud dashboard was created.

## Interpretation and next question

The result does not establish a robust improvement over continued v3 training
or the greedy baseline. It suggests a narrower follow-up question: can later-
placement credit be improved while preserving a useful first-decision baseline
and controlling the effect of batch centering? This is a hypothesis for a new,
separately predeclared experiment, not justification for extending or promoting C.

Exploratory layout groups help localize the tradeoff. Against frozen v3, C differed
in the initial STOP/deploy choice in 52 stress cases: mean cost increased by 1.352
with no mean timely-sensing gain in that group. In 36 stress cases with the same
first placement but changed later layout, timely sensing increased by 2.464
percentage points and cost by 0.814. These groups depend on the trained decisions;
they are descriptive, not causal estimates or additional pass criteria.

Overall, C chose initial STOP in only 2 of 600 validation cases, compared with
71 for frozen v3. All 69 initial-decision flips were C deploying where v3 stopped;
they account for 62.7% of the added cost but just 5.1% of the net timely-sensing
gain. This is why preserving useful stopping behavior matters in a follow-up.

The trained C endpoint was also smoke-tested through the unchanged external-input
planner using `Examples/public-snapshot.json`: it recommended fused sensor/site
27, then STOP, without private simulation inputs or physical commands. This is
interface compatibility, not live sensor or Unreal validation. The existing
[v3 replay demo](BALANCED.md) and default model are unchanged.

## Evidence and verification

- [Exact artifact manifest](Results/credit-v4-pilot/artifact-manifest.json),
  [aggregate scores and gate](Results/credit-v4-pilot/evaluation/aggregate.json),
  and [complete training records](Results/credit-v4-pilot/training/).
- Raw paired reports: [normal](Results/credit-v4-pilot/evaluation/validation-normal.json.gz),
  [stress](Results/credit-v4-pilot/evaluation/validation-stress.json.gz), and
  [changed capabilities](Results/credit-v4-pilot/evaluation/validation-capability.json.gz).
- The archive copies all 33 input artifacts byte-for-byte, with no redaction,
  checkpoint repacking, or selected replacement alias. The archived protocol's
  SHA256 is `f4ca8d76d17b55274025dec6587e8ee8481932dd441c8474206ef96d2c9b5aeb`.
- Independent review recomputed all paired means, all 70 aggregate metric
  intervals, layout groups, and the gate from the raw reports.
- Full local Windows suite: **712 passed**, six existing PettingZoo warnings.
  Linux WSL focused pilot suites: **103 passed** (58 trainer/lineage, 45
  evaluator/archive). An additional archived-evidence regression passed on both
  Windows and Linux, with sampling, inference, and new writes explicitly disabled.
  The predeclared source commit also passed the full Windows/Linux GitHub workflow.
  These are engineering checks, not evidence of RL quality.

Training consumed `401000000..401001023`. Validation consumed
`1000995100000000..1000995100000199`, `1000995101000000..1000995101000199`, and
`1000995102000000..1000995102000199`; they are now development exposure for any
follow-up. The three reserved final suites at `2000000500000000`,
`2000000501000000`, and `2000000502000000` (200 each) remain **unopened**.
