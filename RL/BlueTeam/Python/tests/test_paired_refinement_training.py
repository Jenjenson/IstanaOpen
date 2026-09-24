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


@pytest.mark.parametrize("algorithm", ["local_ppo", "paired_bandit"])
def test_scenario_panels_are_reproducible_disjoint_and_persisted(algorithm):
    validated = TrainingManager.validate(config(algorithm=algorithm, episodes=10000, batchSize=20))
    start, validation, test = scenario_panels(validated)
    assert start == 51000000
    assert validation == [51010000, 51010001] and test == [51020000, 51020001]
    assert start + validated["episodes"] - 1 < min(validation) < min(test)
    assert scenario_panels(TrainingManager.validate(validated)) == (start, validation, test)
    unspecified = config(algorithm=algorithm)
    del unspecified["scenarioSeed"]
    fresh = TrainingManager.validate(unspecified)
    assert fresh["scenarioSeed"] >= 10000000
    assert scenario_panels(TrainingManager.validate(fresh)) == scenario_panels(fresh)


@pytest.mark.parametrize("seed", [-1, True, 1.5, 2**31 - 20000, None])
def test_scenario_seed_rejects_overlap_or_overflow(seed):
    with pytest.raises(ValueError, match="scenarioSeed"):
        TrainingManager.validate(config(scenarioSeed=seed))


@pytest.mark.parametrize("algorithm", ["local_ppo", "paired_bandit"])
def test_local_policy_requires_a_base_layout(algorithm):
    with pytest.raises(ValueError, match="starting layout"):
        TrainingManager.validate(config(algorithm=algorithm, initialization="untrained"))


@pytest.mark.parametrize("algorithm", ["local_ppo", "paired_bandit"])
def test_manager_updates_on_paired_deltas_and_retains_both_native_records(tmp_path, algorithm):
    ctx = native_context()
    batches, calls = [], []

    class Local(Policy):
        reward_mode = "paired_contractor_delta"

        def update(self, episodes):
            batches.append([reward for _, reward in episodes])
            return super().update(episodes)

        def episode_diagnostics(self, records):
            return {"arm": 1, "estimatedGainSeconds": float(self.updates)}

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
                              policy_factories={algorithm: Local}, episode_runner=runner)
    manager.start(config(algorithm=algorithm))
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
        assert row["learningDecision"]["arm"] == 1
        assert row["learningDecision"]["estimatedGainSeconds"] == (row["episode"] - 1) // 2
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


@pytest.mark.parametrize("interval, selected_episode", [(4, 4), (8, 8)])
def test_real_bandit_is_trained_saved_and_selected_through_manager(tmp_path, interval, selected_episode):
    import numpy as np
    from triad_rl.paired_layout_bandit import PairedLayoutBanditPolicy
    from triad_rl.training_evidence import layout_key
    from triad_rl.training_workbench import common_sense_start

    ctx = native_context()
    contractor = common_sense_start(ctx, "directional_balanced_8", count=1,
                                   allowed_sensor_ids=["thermal"])["placements"]
    calls = []

    class Client:
        def __init__(self, **_): pass
        def reset(self, _): return {}
        def get_blue_context(self): return deepcopy(ctx)
        def close(self): pass

    def runner(_client, policy, seed, **kwargs):
        placements, records = policy.plan(deepcopy(ctx),
            deterministic=kwargs.get("deterministic", False),
            rng=np.random.default_rng(kwargs.get("action_seed")))
        gain = 2. if layout_key(placements) != layout_key(contractor) else 0.
        run = evidence(seed, 20. + seed % 7 + gain, context=ctx, placements=placements)
        run["action_seed"] = kwargs.get("action_seed")
        calls.append((seed, isinstance(policy, PairedLayoutBanditPolicy)))
        return run, records

    manager = TrainingManager(output_root=tmp_path, client_factory=Client,
                              episode_runner=runner)
    manager.start(config(algorithm="paired_bandit", episodes=8, batchSize=4,
                         checkpointInterval=8, validationInterval=interval, sensorIds=["thermal"]))
    manager.thread.join(timeout=30)
    status = manager.status()
    assert not manager.thread.is_alive()
    assert status["phase"] == "complete", status
    assert status["bestEpisode"] == selected_episode
    assert status["testEvaluation"]["deltaSeconds"] == 2.
    assert status["exploration"]["actionValueBandit"]
    assert not status["exploration"]["localEdits"]
    assert all(row["trainingReward"] == 2. for row in status["history"])
    assert all(row["learningDecision"] for row in status["history"])
    evaluated = [row["episode"] for row in status["history"] if "validationWarningSeconds" in row]
    assert evaluated == list(range(interval, 9, interval))
    output = Path(status["outputDirectory"])
    restored = PairedLayoutBanditPolicy.load(output / "final-policy.json", ctx)
    placements, _ = restored.plan(ctx, deterministic=True)
    assert layout_key(placements) != layout_key(contractor)
    assert len(placements) == 1 and placements[0]["profileId"] == "thermal"
    first_test = next(index for index, (seed, _) in enumerate(calls) if seed >= 51020000)
    assert all(seed < 51020000 for seed, _ in calls[:first_test])
    # Publishing can replay a fixed best-observed layout for illustration,
    # but the learned model is never evaluated on training/validation again.
    assert all(seed >= 51020000 or not learned for seed, learned in calls[first_test:])


@pytest.mark.parametrize("field, value", [("validationCases", 201), ("validationCases", True),
    ("validationInterval", 0), ("validationInterval", True), ("validationInterval", 10001)])
def test_validation_schedule_rejects_invalid_limits(field, value):
    with pytest.raises(ValueError, match=field):
        TrainingManager.validate(config(**{field: value}))


def test_validation_schedule_keeps_legacy_default_and_supports_larger_panels():
    configured = TrainingManager.validate(config(validationCases=64, validationInterval=80))
    assert configured["validationCases"] == 64 and configured["validationInterval"] == 80
    assert len(scenario_panels(configured)[1]) == 64
    assert TrainingManager.validate(config())["validationInterval"] == 2
