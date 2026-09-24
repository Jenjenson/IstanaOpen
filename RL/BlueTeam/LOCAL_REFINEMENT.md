# Local refinement of the contractor layout

The Training tab defaults to **Local refinement PPO**. Choose a common-sense
starting layout, the allowed limited-FOV sensor profiles and an exact sensor
count. The learned deployment retains that starting inventory. Omnidirectional
profiles are rejected by this policy.

The previous global-placement PPO remains available as **Masked PPO**. Its
weights were updating, but its calibrated starting prior could give each of
four contractor poses 620 times the probability of each alternative pose.
All four placements also received the same absolute warning score, even though
random approach sectors changed between episodes. Those conditions made
exploration noisy and the deterministic deployment difficult to change.

## What changed

Local refinement gives each sensor a separate categorical actor over KEEP,
adjacent yaw/pitch bins and at most two nearest legal alternative sites. All
choices start with equal logits; KEEP wins initial deterministic ties. Earlier
edits update the legality mask for later sensors. A complete layout must remain
legal under the native site's support, deployment radius, spacing, budget and
sensor constraints. Several sensors can change in one episode.

This is genuine clipped PPO with recorded sampling probabilities, entropy
regularization and a value baseline. It is a restricted local search: each
sensor can make one edit from its fixed starting pose. It does not explore
every site, compound a site move with a rotation, or change sensor types.
The legacy starting-exploration setting is ignored for this algorithm.

Each training layout is run in Unreal and compared with the original contractor
on the **same scenario seed**. PPO receives the difference in mean per-drone
warning time. Trajectory and non-layout constraint hashes, target count, cost
and sensor inventory must match before the delta is accepted. Native sensing
remains stochastic but reproducible; changing a site can also change its
site-keyed sensing random stream. The policy never receives private drone
truth, contractor outcomes or evaluation results as planning inputs.

## Reading the results

The warning chart retains absolute warning seconds, counting missed detections
as zero. The reward log labels the paired contractor-relative training reward
separately. A large score on one sampled scenario is not evidence of a better
general policy.

Validation chooses the checkpoint by mean warning on a fixed panel. Ties retain
the earlier checkpoint, including episode zero. A separate test panel runs
only after selection and cannot select another checkpoint. The current narrow,
roughly symmetric eight-sector benchmark already favors the contractor's
outward-facing cameras, so a genuine improvement is not guaranteed.

For local PPO, fresh scenario panels are frozen before simulation starts.
`configuration.json` records `scenarioSeed`: training uses that seed plus
episode minus one, validation uses seed + 10,000, and test uses seed + 20,000.
An API caller can supply the same `scenarioSeed` to compare algorithms on the
same panels. The separate `seed` controls policy sampling. Legacy runs without
`scenarioSeed` retain their historical panels and checkpoint formats.

## Artifacts and cost

Every completed paired training episode retains both its sampled replay under
`episodes/` and its contractor control under `contractor-episodes/`. Training
logs contain the absolute warnings, paired reward and audit hashes. This uses
two native simulations per training episode, plus validation and final tests.

The selected checkpoint and the final trained weights are different artifacts.
The selected model may correctly retain the contractor if validation finds no
improvement. Old global-placement checkpoints remain loadable; local PPO uses
the separate `istana.warning_directional_local_ppo.v1` schema, including its
starting layout, candidate catalogue, optimizer and RNG state.
