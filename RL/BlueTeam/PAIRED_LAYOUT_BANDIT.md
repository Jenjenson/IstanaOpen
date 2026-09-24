# Paired action-value RL

`paired_bandit` is a one-step reinforcement learner for the native sensor
placement workbench. A complete sensor layout is its action, and the reward is
the difference in mean per-drone warning time against the original contractor
on the same native scenario. This is a finite-action bandit, not PPO or a neural
network. Missed detections count as zero warning time.

The public site and sensor contract defines the candidates before training:
KEEP the contractor, or edit one sensor's orientation or nearby site. The other
sensors remain fixed. Every action preserves the selected limited-FOV inventory,
exact count, deployment radius, spacing and budget. The learner does not receive
private drone trajectories as planning inputs, combine estimated edit gains,
or replace the contractor baseline with a more favorable opponent.

Each non-KEEP action receives eight initially balanced samples. Further actions
maximize `mean_delta_seconds + 2 * sqrt(2 * log(total_allocations) / allocations)`.
This fixed UCB exploration bonus is a heuristic measured in seconds, not a
confidence interval. Pending allocations spread a batch across actions without
counting unobserved rewards as samples. Only completed paired native results
update the running means, sample counts and variances. No gradient or Adam step
is involved. KEEP has a known zero paired advantage.

Deterministic deployment chooses the highest positive observed mean, otherwise
KEEP. Validation independently selects checkpoints; an optimistic training
estimate alone cannot promote a model. The separate native test panel does not
select or tune a checkpoint. Negative results and unchanged layouts are valid
outcomes and must remain visible.

The Training tab labels this method **Paired action-value RL**. The old starting
exploration and entropy parameters do not control it. Logs retain the chosen
action, its estimate at sampling, selection rule, allocation ID and paired
reward. Versioned checkpoints preserve observations, pending allocations, RNG,
candidate layouts and their public contract. Loading recomputes and validates
the candidates.

For larger validation panels, the API accepts `validationCases` up to 200 and
an optional `validationInterval` in episodes. The default interval remains the
batch size for compatibility. Checkpoint milestones and the last episode also
trigger validation; updates continue at the independent `batchSize` interval.
The configuration records both intervals and all scenario seeds before native
outcomes. Larger panels add evaluation cost, so they need not run after every
small training batch.

This method addresses credit assignment and exploration efficiency; it cannot
guarantee improvement when the legal neighborhood has little or no headroom.
The earlier [local PPO pilot](Results/local-refinement-20260924/README.md) and
its regressions remain archived separately.
