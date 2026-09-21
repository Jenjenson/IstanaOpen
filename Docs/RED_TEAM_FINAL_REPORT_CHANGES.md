# Red Team Changes for the Final Report

The current final report predates the completed Red initial-placement RL work.
The following changes should be made before submission so that the report is
consistent with the implementation on the `red-team-v2` branch.

## 1. Update the cover-page code reference

The cover currently references:

```text
branch codex/istana-open · head 05c8687
```

After committing the Red RL work, replace it with:

```text
branch red-team-v2 · head <new commit hash>
```

Update Reference 12 and any other repository references to the same commit.

## 2. Correct the Red-team description in Section 3.2

The report currently says that Red remains a scripted adversary with only an
interface for a future learned policy. That is now outdated.

Replace the outdated statement with:

> After the Blue experiments were frozen, we implemented a bounded learned Red
> initial-placement agent. The agent selects one of 16 legally separated wedge
> approach sectors, while Unreal retains authority over placement validation,
> swarm movement, sensing, termination and native reward. It is trained
> separately against frozen temporal-public-control Blue using one-step
> REINFORCE with a running baseline, entropy regularisation, gradient clipping
> and Adam. This is not simultaneous self-play and does not learn flight control.

Also clarify the information boundary:

> Red training uses the advertised placement context and a public path-exposure
> shaping signal derived from the declared Blue sensor layout and capabilities.
> It does not inspect hidden sensing draws, future observations or private Blue
> state. The final learned policy is a fixed distribution over legal placement
> sectors.

## 3. Correct the modularity discussion in Section 3.6

Remove or revise these now-outdated claims:

- “Red can therefore become a learned adversary later.”
- “We did not co-train one this cycle.”

Suggested replacement:

> The Red seam is now exercised by scripted radial, random legal wedge,
> dispersed random and learned placement policies through the same placement
> contract. The learned Red policy was trained separately against frozen Blue,
> rather than through simultaneous self-play, so Blue remained stationary and
> the short-run Red result stayed interpretable.

## 4. Add the Red RL method to Section 3.5

Add a short Red subsection after the Blue learning description:

> **Red initial-placement learning.** Red uses a masked tabular softmax over 16
> legal wedge directions. One full Unreal episode produces one policy update.
> The policy is trained with one-step REINFORCE, a running reward baseline,
> entropy regularisation, gradient clipping and Adam. The first return
> initialises the baseline without updating an action, avoiding a large
> negative-return bias at startup. Checkpoints preserve the policy parameters,
> optimiser state, baseline, episode count, random-number-generator state and
> an integrity hash.

Add the Red training return:

```text
Red training reward = native Red reward - 20 × public path exposure
```

Then explain:

> Native reward and public exposure are logged separately. Exposure shaping is
> required because native Red reward saturates in the evaluated wedge scenario.
> Public exposure is a synthetic training proxy and is not a calibrated sensing
> probability.

## 5. Add Red RL results to Section 5

Keep the existing Red pathfinding-performance paragraph, but add a separate
subsection titled **Red initial-placement learning**.

Use this table:

> **Configuration note:** these are the historical `run-04` results from before
> the Mavic 3E movement-preset change. Preserve them only with that qualification.
> Do not label them as Mavic-preset results. A new Mavic-preset run needs a new
> held-out evaluation because movement changes trajectories, sensing and reward.

| Red placement policy | Valid episodes | Mean native Red reward | Mean public approach exposure |
| --- | ---: | ---: | ---: |
| Learned template softmax | 20/20 | -9.0 | **0.95931** |
| Random legal wedge | 20/20 | -9.0 | 0.96528 |
| Scripted radial | 20/20 | -9.0 | 0.96659 |

Suggested result paragraph:

> A 120-episode live Unreal training run completed with zero invalid
> placements. On 20 held-out seeds per policy, learned Red reduced mean public
> approach exposure by 0.62% relative to random wedge placement and 0.75%
> relative to scripted radial placement. All 60 held-out evaluation placements
> were valid. Native Red reward remained saturated at -9 for every evaluated
> policy. This result therefore demonstrates learning in the documented
> synthetic exposure-shaping signal, not improved breach rate, native reward or
> real-world sensor avoidance.

Do not combine these results with the Blue planner tables. They use a different
policy, action space, training budget and evaluation objective.

## 6. Add Red reproduction commands to Appendix A

Add the following commands after the existing Red bridge example.

```powershell
# One-time setup
powershell -ExecutionPolicy Bypass -File .\Tools\setup_blue_python.ps1
powershell -ExecutionPolicy Bypass -File .\Tools\build.ps1 -Target Editor

# Start Unreal and wait for the map to load
powershell -ExecutionPolicy Bypass -File .\Tools\start_blue_live.ps1

# Train Red against frozen temporal-public-control Blue using a new output directory
$trainDir = ".\Saved\RedRL\mavic-training-$(Get-Date -Format 'yyyyMMdd-HHmmss-fff')"
.\RL\BlueTeam\.venv\Scripts\python.exe `
  .\RL\BlueTeam\Python\train_red_placement.py `
  --temporal-public-control `
  --episodes 120 `
  --seed 91000 `
  --policy-seed 7301 `
  --learning-rate 0.03 `
  --baseline-rate 0.1 `
  --entropy-coefficient 0.01 `
  --exposure-weight 20 `
  --checkpoint-every 10 `
  --output-dir $trainDir

# Compare scripted, random-wedge and learned Red on identical held-out seeds
$evalDir = ".\Saved\RedRL\mavic-evaluation-$(Get-Date -Format 'yyyyMMdd-HHmmss-fff')"
.\RL\BlueTeam\.venv\Scripts\python.exe `
  .\RL\BlueTeam\Python\evaluate_red_placement.py `
  --red-checkpoint "$trainDir\final" `
  --episodes 20 `
  --seed 1500000000 `
  --random-policy-seed 8128 `
  --output-dir $evalDir
```

Note that every output directory must be new and only one live bridge client
can connect at a time.

## 7. Update Appendix D: Evidence Index

Add rows for:

| Artifact | Location |
| --- | --- |
| Red RL implementation and methodology | `Docs/RED_TEAM_RL.md` |
| Red training CLI | `RL/BlueTeam/Python/train_red_placement.py` |
| Red evaluation CLI | `RL/BlueTeam/Python/evaluate_red_placement.py` |
| Red policy implementation | `RL/BlueTeam/Python/triad_rl/red_policy.py` |
| Selected Red checkpoint | Tracked artifact path, if promoted from `Saved/RedRL/run-04/final` |
| Red held-out evaluation | Tracked artifact path, if promoted from `Saved/RedRL/evaluation-04/evaluation.json` |

`Saved/` is ignored by Git. Do not claim the checkpoint or evaluation JSON is a
repository-archived artifact unless the selected files are copied into a
tracked results directory before the final commit.

The selected checkpoint consists of:

```text
checkpoint.json
arrays.npz
```

The minimum supporting result files are:

```text
summary.json
evaluation.json
```

## 8. Correct Appendix F: Conditions Not Tested

The current statement “No comparative evaluation in the live loop” is now too
broad.

Replace it with:

> No paired live comparison of Blue planners has been completed. A live
> comparison of Red initial-placement policies has been completed over 20
> held-out seeds per policy, but native Red reward was saturated and the
> measured improvement was limited to the synthetic public-exposure shaping
> metric.

Retain the other listed limitations, including synthetic sensing, no hardware
calibration, no terrain occlusion in live sensing and no interception model.

## 9. Update Appendix G: System Architecture

Expand the Red-team box with:

```text
External Python placement policies
Scripted radial · random wedge · dispersed · learned
16-action masked-softmax Red policy
Separate training against frozen Blue
```

The diagram should continue to show that Unreal owns geometry, placement
legality and movement. Do not imply that Red learned drone-level flight control.

## 10. Clarify the dispersed policy

The dispersed policy places swarm groups at independently randomized legal
angles and radii and is useful for multi-direction demonstration footage. It is
not the learned model and was not used as the random comparator in the formal
Red evaluation.

If mentioned in the report, label it as:

> A non-learning visual/demo baseline used to demonstrate multi-direction swarm
> approaches through the same Red placement interface.

Do not use the single dispersed demonstration's breach fraction as a formal
performance claim because it was not a paired multi-seed evaluation.

## 11. Update validation evidence

Add the focused Red validation result:

> On 21 September 2026, the focused Red policy and live-integration suite passed
> 42 tests in 6.06 seconds. The Unreal
> Editor Development target also builds successfully. The broader Python suite
> was started but stopped before completion because of long-running unrelated
> integration tests and is therefore not reported as passing for this change.

Do not add the 42 focused tests to the previously reported 1,682-test total
unless the complete combined suite is actually rerun and passes on the final
commit.

## 12. Maintain an explicit limitations statement

Include the following limitations with the Red result:

- Red learns initial placement only, not flight control.
- The learned action catalogue contains wedge directions only.
- Red and Blue were trained separately; this is not simultaneous self-play.
- The Blue opponent was frozen during Red training.
- Native Red reward was saturated in the evaluated wedge scenario.
- Public exposure is a synthetic shaping proxy, not calibrated sensing risk.
- The evaluation does not establish real-world performance.
- The trained checkpoint remains local unless promoted out of ignored `Saved/`.

## 13. Add the Mavic-inspired movement update

Add a short implementation note, without calling it a learned flight controller:

> Red swarm movement now uses a deterministic Mavic 3 Enterprise-inspired
> kinematic flight envelope. Published limits constrain horizontal airspeed,
> ascent/descent speed, tilt and angular velocity. The controller also applies
> a derived tilt-based acceleration limit, an explicitly synthetic jerk limit,
> steady-wind compensation and acceleration-derived visual pitch/bank. The
> checked-in `DA_Mavic3E_NormalFlight` preset is assigned to the saved Istana
> live map. This remains a boid/pathfinding controller, not motor, propeller,
> battery, CFD or learned flight-control simulation.

Reference the official DJI specifications and `Docs/SWARM_SIMULATION.md`. State
that the 566 cm/s2 acceleration is derived from `g * tan(30 degrees)` and the
1200 cm/s3 jerk value is simulation tuning, not a DJI-published measurement.

## Minimum changes if page count is constrained

If there is insufficient space for every update, make these five changes at a
minimum:

1. Correct Sections 3.2 and 3.6 so they no longer say learned Red is future work.
2. Add the Red result table and limitations paragraph to Section 5.
3. Add the training and evaluation commands to Appendix A.
4. Correct the live-comparison statement in Appendix F.
5. Update the branch, commit hash and Red artifact entries in the evidence index.
