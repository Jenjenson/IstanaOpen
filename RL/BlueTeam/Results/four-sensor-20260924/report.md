# Four sensors against five drones

Status: complete

Three predeclared PPO seeds; 500 episodes each. Every trial is retained. Warning gains are matched to the contractor on identical scenario panels.

| Trial | Seed | Status | Selected episode | Validation warning / delta (s) | Test warning / delta (s) | Test detected |
| --- | --- | --- | --- | --- | --- | --- |
| A | 917 | complete | 0 | 21.192 / +0.000 | 27.199 / +0.000 | 58.0% |
| B | 918 | complete | 0 | 21.192 / +0.000 | 27.199 / +0.000 | 58.0% |
| C | 919 | complete | 0 | 21.192 / +0.000 | 27.199 / +0.000 | 58.0% |

## Trial A

Output: Saved\WarningTraining\console-20260924-063309-225823
Model: trained-four-sensor-study-2026-09-24-trial-a-4ebcf74784

Validation: mean warning 21.192357 s; matched delta +0.000000 s; detected fraction 0.460000.

| Scenario seed | Matched warning delta (s) |
| --- | --- |
| 2700000 | +0.000000 |
| 2700001 | +0.000000 |
| 2700002 | +0.000000 |
| 2700003 | +0.000000 |
| 2700004 | +0.000000 |
| 2700005 | +0.000000 |
| 2700006 | +0.000000 |
| 2700007 | +0.000000 |
| 2700008 | +0.000000 |
| 2700009 | +0.000000 |

Test: mean warning 27.199091 s; matched delta +0.000000 s; detected fraction 0.580000.

| Scenario seed | Matched warning delta (s) |
| --- | --- |
| 3800000 | +0.000000 |
| 3800001 | +0.000000 |
| 3800002 | +0.000000 |
| 3800003 | +0.000000 |
| 3800004 | +0.000000 |
| 3800005 | +0.000000 |
| 3800006 | +0.000000 |
| 3800007 | +0.000000 |
| 3800008 | +0.000000 |
| 3800009 | +0.000000 |

## Trial B

Output: Saved\WarningTraining\console-20260924-063312-011590
Model: trained-four-sensor-study-2026-09-24-trial-b-26860f6235

Validation: mean warning 21.192357 s; matched delta +0.000000 s; detected fraction 0.460000.

| Scenario seed | Matched warning delta (s) |
| --- | --- |
| 2700000 | +0.000000 |
| 2700001 | +0.000000 |
| 2700002 | +0.000000 |
| 2700003 | +0.000000 |
| 2700004 | +0.000000 |
| 2700005 | +0.000000 |
| 2700006 | +0.000000 |
| 2700007 | +0.000000 |
| 2700008 | +0.000000 |
| 2700009 | +0.000000 |

Test: mean warning 27.199091 s; matched delta +0.000000 s; detected fraction 0.580000.

| Scenario seed | Matched warning delta (s) |
| --- | --- |
| 3800000 | +0.000000 |
| 3800001 | +0.000000 |
| 3800002 | +0.000000 |
| 3800003 | +0.000000 |
| 3800004 | +0.000000 |
| 3800005 | +0.000000 |
| 3800006 | +0.000000 |
| 3800007 | +0.000000 |
| 3800008 | +0.000000 |
| 3800009 | +0.000000 |

## Trial C

Output: Saved\WarningTraining\console-20260924-063314-533939
Model: trained-four-sensor-study-2026-09-24-trial-c-f6564a4a6c

Validation: mean warning 21.192357 s; matched delta +0.000000 s; detected fraction 0.460000.

| Scenario seed | Matched warning delta (s) |
| --- | --- |
| 2700000 | +0.000000 |
| 2700001 | +0.000000 |
| 2700002 | +0.000000 |
| 2700003 | +0.000000 |
| 2700004 | +0.000000 |
| 2700005 | +0.000000 |
| 2700006 | +0.000000 |
| 2700007 | +0.000000 |
| 2700008 | +0.000000 |
| 2700009 | +0.000000 |

Test: mean warning 27.199091 s; matched delta +0.000000 s; detected fraction 0.580000.

| Scenario seed | Matched warning delta (s) |
| --- | --- |
| 3800000 | +0.000000 |
| 3800001 | +0.000000 |
| 3800002 | +0.000000 |
| 3800003 | +0.000000 |
| 3800004 | +0.000000 |
| 3800005 | +0.000000 |
| 3800006 | +0.000000 |
| 3800007 | +0.000000 |
| 3800008 | +0.000000 |
| 3800009 | +0.000000 |

## Reporting recovery

All three native runs completed 500 episodes and published models. The experiment driver's report aggregation accessed Trial A's validation field before finalization, causing a reporting-only failure. This report was reconstructed offline from the unchanged native summaries and predeclared protocol.

Original failed reporting evidence: `reporting-failure-evidence`.
No training or native episodes were rerun; saved native outputs and published models were unchanged.
