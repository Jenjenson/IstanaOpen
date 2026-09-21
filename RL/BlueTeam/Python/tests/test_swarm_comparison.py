"""Generated populations, fresh public-only inference and paired fairness."""
from copy import deepcopy
import hashlib
import json

import pytest

import placement_comparison as comparison
import swarm_comparison as swarm
from simulation_console import load_replays


@pytest.fixture(scope="module")
def replays():
    return load_replays()


@pytest.fixture(scope="module")
def generated(replays):
    return {(profile, count): swarm.make_swarm_replay(
                next(row for row in replays if row["profile"] == profile), count)
            for profile in ("normal", "stress", "capability") for count in (8, 60)}


@pytest.fixture(scope="module")
def paired(replays):
    return swarm.compare_swarm_placements(replays[0], 60)


@pytest.mark.parametrize("count", [None, True, False, 0, 1, 7, 9, 59, 61, 64, 65, 8., 60., "60", [], {}])
def test_only_explicit_integer_supported_populations_are_accepted(count):
    for function in (swarm.make_swarm_case, swarm.swarm_scenario, swarm.make_swarm_replay,
                     swarm.compare_swarm_placements):
        with pytest.raises(ValueError, match="8 or 60"):
            function({}, count)


def test_generated_populations_have_distinct_paths_and_honest_public_reports(replays, generated):
    for (profile, count), replay in generated.items():
        base = next(row for row in replays if row["profile"] == profile)
        scenario, public = replay["scenario"], replay["scenario"]["public"]
        targets = scenario["targets"]
        assert 0 <= scenario["seed"] <= (1 << 53) - 1
        assert len(targets) == len({row["id"] for row in targets}) == count
        assert public["forecast"]["swarm_size"] == count
        assert len({tuple(row["position"]) for row in replay["frames"][0]["threats"]}) == count
        assert 0 < len(public["tracks"]) <= count
        truth = {row["id"]: row["position"] for row in replay["frames"][0]["threats"]}
        for track in public["tracks"]:
            assert track["id"] in truth
            assert track["position"] != truth[track["id"]]
            assert .45 <= track["confidence"] <= .9
            assert not track["confirmed"]
            assert not {"path", "curvature", "emitter_phase", "frames"} & track.keys()
        for key in ("sites", "blocked_sites", "weather", "budget_total", "max_sites", "min_separation",
                    "available_sensor_ids", "deployment_min_radius", "deployment_max_radius"):
            assert public[key] == base["scenario"]["public"][key]
        assert replay["catalogue"] == base["catalogue"]
        assert replay["metrics"]["invalid_actions"] == 0
        assert replay["metrics"]["cost"] <= public["budget_total"] + 1e-9
        assert len(replay["placements"]) <= public["max_sites"]
        assert replay["generation"]["policy_state_before"] == replay["generation"]["policy_state_after"]
        assert replay["generation"]["actor_updates"] > 0
        assert replay["swarm"]["withinTrainingPopulationRange"] == (count == 8)
        assert replay["swarm"]["modelTrainingDroneRange"] == [1, 8]
        assert replay["swarm"]["nativeReplay"] is False
        json.dumps(replay, allow_nan=False)


def test_preview_has_no_future_truth_and_never_runs_rl_or_scoring(replays, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Planning preview must not infer or score")
    monkeypatch.setattr(swarm.TemporalPolicy, "load", forbidden)
    monkeypatch.setattr(comparison.adaptive_env.AdaptivePlacementEnv, "_simulate", forbidden)
    original = deepcopy(replays[0])
    changed = deepcopy(original)
    changed["scenario"]["targets"] = [{"private": "must not affect generation"}]
    changed.update(placements=[], decisions=[], frames=[], metrics={"return": -999})
    first = swarm.swarm_scenario(original, 60)
    assert swarm.swarm_scenario(changed, 60) == first
    assert not {"targets", "frames", "metrics", "decisions", "scenario", "generation"} & first.keys()
    assert first["forecast"]["swarm_size"] == 60
    assert first["audit"]["public_only"] is True
    assert first["audit"]["scoring_performed"] is False
    assert first["audit"]["fresh_rl_inference"] is False
    assert first["audit"]["scenario_generation_performed"] is True
    assert original == replays[0]


def test_population_generation_is_deterministic_and_independent_of_model_choice(replays):
    cases = [row for row in replays if row["profile"] == "normal" and row["case_index"] == 1]
    first = swarm.make_swarm_case(cases[0], 60)
    assert first == swarm.make_swarm_case(cases[0], 60)
    assert first["scenario"] != swarm.make_swarm_case(cases[0], 8)["scenario"]
    assert first["scenario"] != swarm.make_swarm_case(
        next(row for row in replays if row["profile"] == "normal" and row["case_index"] == 2), 60)["scenario"]
    assert {row["temporal_seed"] for row in cases} == {406, 407, 408}
    for base in cases[1:]:
        assert swarm.make_swarm_case(base, 60)["scenario"] == first["scenario"]


def test_fresh_inference_uses_public_only_and_does_not_reuse_archive_actions(replays, generated, monkeypatch):
    seen, infer = [], swarm.recommend_layout
    def inspect(policy, public, *, config, catalogue):
        seen.append(deepcopy(public))
        assert not {"targets", "frames", "metrics", "decisions", "generation"} & public.keys()
        return infer(policy, public, config=config, catalogue=catalogue)
    monkeypatch.setattr(swarm, "recommend_layout", inspect)
    changed = deepcopy(replays[0])
    changed.update(decisions=[{"invalid": True}], placements=[], frames=[], metrics={"return": -999})
    before = deepcopy(changed)
    checkpoint = swarm.CHECKPOINT_ROOT / "seed-406/last"
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoint.iterdir()}
    replay = swarm.make_swarm_replay(changed, 60)
    assert replay == generated["normal", 60]
    assert seen == [replay["scenario"]["public"]]
    assert changed == before
    assert hashes == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoint.iterdir()}


def test_paired_comparison_has_identical_paths_resources_and_honest_provenance(paired):
    rl, baseline = paired["rl"], paired["baseline"]
    for key in ("seed", "catalogue", "sites", "budget", "weather", "objectiveRadius"):
        assert rl[key] == baseline[key]
    assert len(rl["frames"]) == len(baseline["frames"])
    for left, right in zip(rl["frames"], baseline["frames"]):
        assert left["time"] == right["time"]
        assert [{key: row[key] for key in ("id", "position", "emitting", "active")}
                for row in left["threats"]] == [
            {key: row[key] for key in ("id", "position", "emitting", "active")}
                for row in right["threats"]]
    assert paired["audit"]["fresh_rl_inference"] is True
    assert paired["audit"]["scenario_generation_performed"] is True
    assert paired["audit"]["generated_rl_result_reproduced"] is True
    assert paired["audit"]["published_evidence_modified"] is False
    assert paired["audit"]["training_performed"] is False
    assert "archived_rl_result_reproduced" not in paired["audit"]
    assert "Archived" not in rl["policy"]
    assert paired["metrics"]["rl"]["target_count"] == paired["metrics"]["baseline"]["target_count"] == 60
    for key, delta in paired["deltas"].items():
        assert delta == pytest.approx(paired["metrics"]["rl"][key] - paired["metrics"]["baseline"][key])


def test_manual_identical_layout_gets_identical_hits_and_outcomes(replays, generated):
    replay = generated["normal", 60]
    manual = [{"sensor_id": row["action"]["sensor_id"], "site_index": row["action"]["site_index"]}
              for row in replay["decisions"] if not row["action"]["stop"]]
    result = swarm.compare_swarm_placements(replays[0], 60, baseline="manual", placements=manual)
    for key in ("placements", "frames", "metrics"):
        assert result["rl"][key] == result["baseline"][key]
    assert all(delta == 0 for delta in result["deltas"].values())


def test_empty_manual_layout_detects_none_of_the_sixty_drones(replays):
    result = swarm.compare_swarm_placements(replays[0], 60, baseline="manual", placements=[])
    summary = result["metrics"]["baseline"]
    assert summary["target_count"] == summary["breached_count"] == 60
    assert summary["cost"] == summary["detected_count"] == summary["sensors_placed"] == 0
    assert all(not frame["detections"] for frame in result["baseline"]["frames"])


def test_generated_manual_constraints_are_enforced(replays):
    with pytest.raises(ValueError):
        swarm.compare_swarm_placements(replays[0], 8, baseline="manual", placements=[
            {"sensor_id": "rf", "site_index": 999}])


def test_missing_checkpoint_is_not_replaced_with_archived_or_greedy_choices(replays, tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        swarm.make_swarm_replay(replays[0], 8, checkpoint_root=tmp_path)
