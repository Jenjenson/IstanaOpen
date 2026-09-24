# Four-sensor PPO study — 24 September 2026

This portable research archive preserves **all three completed 500-episode runs** and **both policy choices from each run**. Four selected, limited-FOV **thermal** sensors faced five drones. No trial improved the selected layout's held-out warning time over the contractor layout.

| Trial | Training seed | Unique sampled layouts | Selected episode | Validation warning | Held-out warning | Held-out gain | Detected fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 917 | 272 | 0 | 21.192357 s | 27.199091 s | 0.000000 s | 0.58 |
| B | 918 | 302 | 0 | 21.192357 s | 27.199091 s | 0.000000 s | 0.58 |
| C | 919 | 260 | 0 | 21.192357 s | 27.199091 s | 0.000000 s | 0.58 |

Warning is the mean per-drone `max(0, arrival at the 20 m zone − first detection)`, with undetected drones contributing zero. These are synthetic benchmark measurements, not field performance estimates.

## The two models in each trial

- `trials/A-917/final-policy.json` (and B-918/C-919): the actual PPO weights and optimizer/RNG state **after episode 500**. Each completed **25 fresh batch updates and 100 Adam steps**: batch size 20, four PPO optimization passes per batch. Final weights changed, but their deterministic deployment remained the initial layout.
- `trials/A-917/best-policy.json` (and B-918/C-919): the **validation-selected checkpoint**, copied byte-for-byte from the model registry. In all trials `bestEpisode` is **0**, so this is the initial warm-start policy. No evaluated checkpoint exceeded its validation score. This does not imply zero gradients or unchanged final weights.

Both use schema `istana.warning_directional_masked_ppo.v1`, selected sensor IDs `["thermal"]`, and a four-sensor placement limit. Every policy file retains its exact original bytes. Final policies also match the original episode-500 checkpoints; selected policies match the original episode-0 checkpoints. Hashes are in `manifest.json`.

`native-context.json` contains the public planning context needed for the repository policy loader. The context catalogue may describe additional native sensor profiles; the archived policies' allowed-sensor mask restricts deployment to thermal sensors. `configuration.json` retains the initialization, sensor selection, exploration, reward, and scenario settings. `model-metadata.json` is historical registry metadata, **not an automatic registry import**; references to original comparison/run artifacts are provenance only. This archive does not modify which models appear in the application.

## Evidence and interpretation

Each trial includes all **500 training JSONL rows**, the **25 fixed-panel validation checkpoints**, the full evaluation summary, separate held-out evaluation, native summary, configuration, and policy metadata. JSONL files use standard gzip compression; decompression produces the exact original bytes, including every sampled placement, episode/action seed, Red decision, and metric. Full simulation frames and videos are intentionally excluded.

Validation uses seeds 2700000–2700009; the held-out panel uses 3800000–3800009 and was excluded from model selection. The trials share the **same 10 test scenarios, not 30 independent scenarios**. This is a reused fixed benchmark, not a newly blind test set. Per-case trajectory and constraint hashes support matched comparison. Every selected-policy test delta is zero. Stochastic training rewards and layout diversity are not evidence of generalization gains.

The full portable `protocol.json`, `results.json`, `report.md`, and diagnostics retain every predeclared trial. Embedded protocol/source hashes in those historical records refer to the **original source bytes**; `manifest.json` separately records each portable file's hash and transformation. Source paths are repository-relative identifiers, not required runtime dependencies. No local console or bridge connection is needed to inspect these models.

## Technical history

The first launch stopped before episode 1 because simultaneous native jobs collided on a timestamp-based output directory. The driver safely stopped its other jobs. Startup allocation was then serialized; the scientific parameters and trial names stayed unchanged, and training ran in parallel after allocation. Compact failed-attempt evidence is retained under `history/attempt-01-startup-failure/`.

All three subsequent native runs completed 500 episodes and published their selected models. A driver reporting race then accessed Trial A's `validation` field before report finalization and incorrectly marked the study failed. The final report was recovered offline from the unchanged native summaries and frozen protocol; **no training or native episodes were rerun**. `reporting-recovery.json` records original hashes and unchanged B/C copied-summary hashes. Compact original failed reports remain under `history/reporting-failure-evidence/`; bulky polling logs remain in the original local study only.

The original runs did not record a source commit. The manifest's `archiveAssemblyCommit` identifies packaging time only; it must not be cited as the training code revision.

## Verify or load

From the repository root, with any Python 3.10+ interpreter:

```sh
python RL/BlueTeam/Results/four-sensor-20260924/archive.py verify
```

This checks file hashes, 1,500 complete training records, selected sensor/target counts, all checkpoint counters, matched scenario panels, zero per-case gains, model IDs, and reported aggregates using the standard library. No original `Saved/` directories are required. To additionally load all six models and check their deterministic placements with the repository's installed Python dependencies:

```sh
python RL/BlueTeam/Results/four-sensor-20260924/archive.py verify --load-policies
```

To compare against still-available original source files, add `--source-root .`. The script's `build` command documents the packaging procedure and refuses to overwrite an existing archive. Read compressed logs with `gzip.open(path, "rt", encoding="utf-8")` and parse each line as JSON. Checkpoints are map/action-contract specific; a different simulator contract requires a compatibility check before using them for a new run.
