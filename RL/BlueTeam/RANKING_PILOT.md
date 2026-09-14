# Rollout-guided sensor/site ranking

This additive development experiment learns a correction to public-input greedy
coverage. It chooses **sensor type, site, or STOP** using the same public inputs
as the existing policies. It is not a released replacement or a claim of robust
real-world deployment.

The [protocol](Results/ranking-v5-pilot/protocol.json) fixes the training budget,
new scenario ranges, comparisons and decision rule before experiment runs.
The [completed results](RANKING_PILOT_RESULTS.md) failed the fixed development
gate: higher aggregate detection did not translate into stronger timely sensing
or return. All three endpoints remain experimental.

## What changes

The previous [anchored pilot](ANCHORED_PILOT_RESULTS.md) did not demonstrate a
timely-sensing improvement over v3. Its recorded layouts also showed losses
against public greedy when policies chose the same sensor types but different
sites. That descriptive comparison motivates testing placement ranking; it does
not prove that changing a particular site would cause an improvement.

The new policy starts with exactly greedy's utility and a zero neural correction.
A shared network scores each legal sensor/site option, including a learnable
STOP correction. Catalogue IDs are not fixed output neurons: options carry
sensor capabilities, cost, placement context and coverage features. Masks retain
the existing budget, availability and site constraints.

At each training decision:

1. Commit the current policy's public-input action before generating labels.
2. Propose up to six legal alternatives using STOP, greedy, current policy,
   frozen v3, another site, and sensor-balanced exploration.
3. Recreate the same case and placement prefix independently for each alternative.
   Take that alternative and finish with frozen public greedy.
4. Train a weighted pairwise ranking loss on the resulting sensing preferences.

Timely confirmation takes priority, then detection, then the original episode
return when the earlier metrics tie exactly. The original return still includes
cost, blind-spot and unnecessary-deployment penalties. Simulation physics,
rewards, curriculum and public feature semantics are unchanged; the **learning
objective** changes. This is not a compute-matched v3 ablation.

Rollout-generated action preferences are related to classification-based policy
improvement described by [Lagoudakis and Parr (IJCAI 2003)](https://www.ijcai.org/Proceedings/03/Papers/228.pdf).
This implementation is a different, single-paired-rollout neural ranking variant,
not a reproduction of that algorithm. Its noisy labels and frozen greedy
continuation do not guarantee optimal actions, monotonic improvement or convergence.

## Fixed comparison

Three independent training seeds each receive 1,024 mixed-profile episodes,
with four optimization passes per batch of 16. Every visited state is retained;
exact all-tier ties contribute no pair. No best checkpoint, seed selection,
training resume, internal validation or early stopping is allowed.

All three fixed endpoints are compared with public greedy, frozen v3/v2/v1 and
the anchored pilot on the same 200 new cases per profile: normal, stress and
varied sensor capabilities. Each ranker is reported separately. The three-seed
mean averages outcomes, **not actions**; it is not an inference ensemble.

To pass the development screen against both greedy and v3, the three-seed mean
must gain at least one percentage point in timely confirmation with a paired
95% interval above zero, satisfy the fixed profile/seed decline guards, and not
reduce aggregate detection, all-threat success or original return. Cost savings
alone cannot pass. Scenario bootstrap intervals are conditional on these three
endpoints; they do not measure the full population of training seeds.

The reserved final tests remain unopened. A development-screen pass is not an
automatic promotion or evidence of native Unreal/live-device performance.

## Reproduce

Use the environment setup in [BALANCED.md](BALANCED.md). From `RL/BlueTeam/Python`:

```powershell
python -m triad_rl.train_ranking --protocol ../Results/ranking-v5-pilot/protocol.json --seed 403 --output ../runs/ranking-v5-pilot/seed-403
python -m triad_rl.train_ranking --protocol ../Results/ranking-v5-pilot/protocol.json --seed 404 --output ../runs/ranking-v5-pilot/seed-404
python -m triad_rl.train_ranking --protocol ../Results/ranking-v5-pilot/protocol.json --seed 405 --output ../runs/ranking-v5-pilot/seed-405
python evaluate_ranking.py --protocol ../Results/ranking-v5-pilot/protocol.json --run 403=../runs/ranking-v5-pilot/seed-403 --run 404=../runs/ranking-v5-pilot/seed-404 --run 405=../runs/ranking-v5-pilot/seed-405 --output ../runs/ranking-v5-pilot/evaluation
```

Use fresh output directories. Source hashes and complete inherited exposure are
checked before sampling. Each run saves initialization, fixed endpoint, raw live
episode metrics, all slate labels/features and optimizer diagnostics. The
read-only validator replays optimization from archived comparisons; it does not
regenerate scenarios. A public-observation hash and selected feature rows do not
independently prove the original full snapshot without that snapshot or a replay.

Optional local training curves use `track_adaptive.py` with Trackio, without a
cloud Space. JSONL and checkpoint artifacts remain authoritative.

## See the decisions

`demo_ranking.py` recreates exactly the first two already-scored cases per
profile for every fixed endpoint: 18 examples, without selecting winners.
Actions, layouts, scenario hashes and outcomes must match the archived reports.
The offline viewer shows sensor types, placements, coverage, threats, detections
and rewards. Its examples are labeled experimental; a replay is not evidence of
policy promotion or live-device integration.
The [published 18-example HTML](Results/ranking-v5-pilot/demo.html) can be
downloaded and opened locally; GitHub's file page shows its source.

```powershell
python demo_ranking.py --protocol ../Results/ranking-v5-pilot/protocol.json --run 403=../runs/ranking-v5-pilot/seed-403 --run 404=../runs/ranking-v5-pilot/seed-404 --run 405=../runs/ranking-v5-pilot/seed-405 --evaluation ../runs/ranking-v5-pilot/evaluation --output ../runs/ranking-v5-demo.html
```

## Simulated and external inputs

The ranker uses the existing `triad.sensor_input.v1` observation adapter and
versioned public features. `recommend_ranking.py` loads a ranker checkpoint and
emits an offline plan from a public snapshot and optional capability catalogue.
It does not call the simulator for labels, send device commands, or connect C2.

```powershell
python recommend_ranking.py --checkpoint ../runs/ranking-v5-pilot/seed-403/last --input ../Examples/public-snapshot.json --catalogue ../Examples/sensor-catalogue.json --now 0 --output ../runs/ranking-plan.json
```

This is sequential **initial layout planning**, not mid-flight relocation.
Synthetic timely sensing is not interception. Real calibration, missing or
stale feeds, hardware deployment and native integration require separate tests.
