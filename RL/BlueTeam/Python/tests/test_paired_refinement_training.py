"""Contractor controls reduce scenario noise without exposing evaluation truth to the actor."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_training_evidence import Policy, evidence
from test_training_workbench import native_context
from triad_rl.training_evidence import paired_training_reward
from triad_rl.training_workbench import TrainingManager, scenario_panels


def config(**changes):
    return {"name": "Paired local policy", "algorithm": "local_ppo", "episodes": 4,
            "batchSize": 2, "sensorCount": 1, "seed": 31,
            "initialization": "directional_balanced_8", "validationCases": 2,
            "headroomProbes": 0, "checkpointInterval": 2, "scenarioSeed": 51000000,
            **changes}


@pytest.mark.parametrize("offset", [0., 20., 1000.])
def test_paired_reward_cancels_scenario_only_offset(offset):
    result = paired_training_reward(evidence(31, offset + 3.), evidence(31, offset + 1.))
    assert result["rewardSeconds"] == 2.
    assert result["policyWarningSeconds"] == offset + 3.
    assert result["contractorWarningSeconds"] == offset + 1.


@pytest.mark.parametrize("mismatch", ["seed", "trajectory", "weather", "inventory", "targets", "cost",
                                     "deployment_min_radius", "deployment_max_radius", "budget_remaining"])
def test_paired_training_rejects_unmatched_evidence(mismatch):
    actual, baseline = evidence(31, 5.), evidence(31, 4.)
    if mismatch == "seed": baseline["seed"] = 32
    if mismatch == "trajectory": baseline["trajectory_sha256"] = "different"
    if mismatch == "weather": baseline["context"]["publicSnapshot"]["weather"] = {"rain": .5}
    if mismatch == "inventory": baseline["placements"][0]["profileId"] = "omni"
    if mismatch == "targets": baseline["metrics"]["targets"] = 6
    if mismatch == "cost": baseline["metrics"]["cost"] = .5
    if mismatch in ("deployment_min_radius", "deployment_max_radius", "budget_remaining"):
        baseline["context"]["publicSnapshot"][mismatch] = {
            "deployment_min_radius": 40., "deployment_max_radius": 90., "budget_remaining": 1.}[mismatch]
    with pytest.raises(ValueError):
        paired_training_reward(actual, baseline)


def test_scenario_panels_are_reproducible_disjoint_and_persisted():
    validated = TrainingManager.validate(config(episodes=10000, batchSize=20))
    start, validation, test = scenario_panels(validated)
    assert start == 51000000
    assert validation == [51010000, 51010001] and test == [51020000, 51020001]
    assert start + validated["episodes"] - 1 < min(validation) < min(test)
    assert scenario_panels(TrainingManager.validate(validated)) == (start, validation, test)
    unspecified = config()
    del unspecified["scenarioSeed"]
    fresh = TrainingManager.validate(unspecified)
    assert fresh["scenarioSeed"] >= 10000000
    assert scenario_panels(TrainingManager.validate(fresh)) == scenario_panels(fresh)


@pytest.mark.parametrize("seed", [-1, True, 1.5, 2**31 - 20000, None])
def test_scenario_seed_rejects_overlap_or_overflow(seed):
    with pytest.raises(ValueError, match="scenarioSeed"):
        TrainingManager.validate(config(scenarioSeed=seed))


def test_local_policy_requires_a_base_layout():
    with pytest.raises(ValueError, match="starting layout"):
        TrainingManager.validate(config(initialization="untrained"))


def test_manager_updates_on_paired_deltas_and_retains_both_native_records(tmp_path):
    ctx = native_context()
    batches, calls = [], []

    class Local(Policy):
        reward_mode = "paired_contractor_delta"

        def update(self, episodes):
            batches.append([reward for _, reward in episodes])
            return super().update(episodes)

    class Client:
        def __init__(self, **_): pass
        def reset(self, _): return {}
        def get_blue_context(self): return deepcopy(ctx)
        def close(self): pass

    def runner(_client, policy, seed, **kwargs):
        generation = policy.updates if isinstance(policy, Local) else None
        calls.append((seed, generation, kwargs.get("deterministic", False)))
        gain = 0 if generation is None else generation + (not kwargs.get("deterministic", False))
        run = evidence(seed, 10. + seed % 97 + gain, context=ctx, placements=policy.placements)
        run["action_seed"] = kwargs.get("action_seed")
        return run, []

    manager = TrainingManager(output_root=tmp_path, client_factory=Client,
                              policy_factories={"local_ppo": Local}, episode_runner=runner)
    manager.start(config())
    manager.thread.join(timeout=10)
    status = manager.status()
    assert status["phase"] == "complete", status
    assert batches == [[1., 1.], [2., 2.]]
    assert status["rewardMode"] == "paired_contractor_delta"
    assert status["bestEpisode"] == 4
    assert status["testEvaluation"]["deltaSeconds"] == 2.
    output = Path(status["outputDirectory"])
    saved = json.loads((output / "configuration.json").read_text())
    assert saved["scenarioSeed"] == 51000000
    assert saved["rewardMode"] == "paired_contractor_delta"
    assert len(list((output / "contractor-episodes").glob("*.json"))) == 4
    for row in status["history"]:
        assert row["trainingReward"] == row["pairedTraining"]["rewardSeconds"]
        assert row["meanWarningSeconds"] - row["pairedTraining"]["contractorWarningSeconds"] == row["trainingReward"]
    # Each sampled training scenario has an independent contractor control.
    for seed in range(51000000, 51000004):
        matching = [row for row in calls if row[0] == seed]
        assert any(generation is None and deterministic for _, generation, deterministic in matching)
        assert any(generation is not None and not deterministic for _, generation, deterministic in matching)
    first_test = next(i for i, row in enumerate(calls) if row[0] == 51020000)
    assert all(row[0] < 51020000 for row in calls[:first_test])
    assert all(generation in (None, 2) for seed, generation, _ in calls if seed >= 51020000)
