# Local refinement PPO: controlled pilot and confirmation

All arms used the same four limited-FOV thermal sensors, five drones, contractor layout and physics. Each received 200 sampled training episodes and 25 updates. Local PPO also used matched contractor replays for training rewards. Every result is retained, including regressions.

| Arm | Selected episode | Validation delta (s) | Pilot 8-case test delta (s) | Confirmation 100-case delta (s) | Descriptive 95% interval (s) |
|---|---:|---:|---:|---:|---|
| local-917 | 0 | +0.000 | +0.000 | +0.000 | [+0.000, +0.000] |
| local-918 | 0 | +0.000 | +0.000 | +0.000 | [+0.000, +0.000] |
| legacy-917 | 0 | +0.000 | +0.000 | +0.000 | [+0.000, +0.000] |

Both local final greedy policies changed three of four sensor poses; the legacy final greedy layout retained the contractor poses. Each arm made 25 nonzero parameter updates. The local runs explored 158 and 155 unique sampled layouts, but none of their validation checkpoints beat the contractor. All three selected policies therefore remained episode 0. Policy movement and learning updates are distinct from demonstrated performance improvement.

The contractor averaged 23.5826295383 seconds of per-drone warning over the 100 confirmation scenarios; all three selected policies matched that absolute mean. Deltas compare mean per-drone warning against the original contractor; undetected drones contribute zero. Positive values mean earlier warning on average. The 100 confirmation scenarios are shared across arms. Each interval is mean ± 1.96 SE of the 100 paired scenario deltas for that frozen policy. Zero primary intervals arise from identical selected layouts, not certainty about the whole algorithm.

Scenario panels and the confirmation panel were declared before pilot outcomes. Validation selected the checkpoint; neither the built-in eight-case test nor the 100-case confirmation selected a policy or tuned parameters. Two local action seeds and one legacy control remain a pilot, not definitive algorithm-wide evidence.

Actual native cost: 1723 pilot episodes plus 600 confirmation episodes. Wall time was 2383.2s for the parallel pilot and 860.1s for parallel confirmation. Per-arm calls, steps and timing are retained.

A separate secondary extension was declared after pilot completion and before evaluating any final checkpoint on the confirmation panel. All three final checkpoints were frozen and evaluated on the same 100 scenarios against the retained exact contractor records. These outcomes did not select or promote a model and are not part of the original primary protocol.

| Secondary final checkpoint | Mean warning (s) | Warning delta (s) | Descriptive 95% interval (s) |
|---|---:|---:|---|
| local-917 final | 21.856 | -1.727 | [-2.615, -0.840] |
| local-918 final | 21.459 | -2.123 | [-3.650, -0.596] |
| legacy-917 final | 23.583 | +0.000 | [+0.000, +0.000] |

Secondary cost: 300 additional native episodes, 510.7s parallel wall time. The same 100 cases are shared across primary and secondary results; they are not independent replication.

Portable contents: selected, final and initial policy weights; baseline public context and contractor placements; full validation/test/confirmation arrays; compressed training, evaluation and native-call logs; drivers and frozen Python sources. Raw replay frames remain in the Saved experiment directories. Source paths in historical metadata identify the original run; use the relative weight filenames in results.json for this bundle.

Source provenance: post-launch changes added a legality guard for separation > 300 m and three previously omitted fields to paired constraint validation. Exact deltas are retained as a patch. This study uses 20 m separation. A retrospective raw-context audit verifies budget_remaining and both deployment radius bounds on all 400 paired training cases and 600 primary/secondary confirmation pairs. All matched exactly at 8, 30 m and 150 m. Confirmation ran the exact launched core source from the frozen snapshot; source-provenance.json records the remaining unchanged dependencies and all hashes.

Fairness checks: contractor and initial layouts/metrics matched across arms; every paired scenario passed Red trajectory and physical-constraint hash checks; confirmation contractor metrics matched across all three processes. The selected policy hashes match the frozen confirmation weights.

Offline verification: run `python archive.py` from this directory with NumPy installed. The verifier loads bundled frozen sources and public context, checks every archive hash, policy inventory and legal layout, training/update/evaluation rows, paired test/confirmation hashes and aggregates. It requires neither Saved files nor a native port. Bundle-local .gitattributes preserves exact bytes.

Driver copies preserve the executed procedures and original workspace paths. Those paths are historical provenance and are not required by the offline verifier.
