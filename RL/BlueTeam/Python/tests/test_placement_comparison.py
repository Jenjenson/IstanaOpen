"""Paired fairness, public-input boundaries and archived evidence preservation."""
from copy import deepcopy
import json
import math

import pytest

import placement_comparison as comparison
from simulation_console import load_replays


@pytest.fixture(scope="module")
def replays():
    return load_replays()


@pytest.fixture(scope="module")
def paired(replays):
    before = deepcopy(replays)
    results = [comparison.compare_placements(replay) for replay in replays]
    assert replays == before
    return results


def assert_archived_result_matches(actual, archived):
    """Only continuous arithmetic may round differently on ARM versus x86.

    Use the published-result audit's 1e-9 relative/absolute tolerance for
    coordinates, travel duration and continuous scores. Sample times, sensor
    attribution, detection/confirmation ticks, status flags and outcome fractions
    remain exact; the tolerance must not turn a changed outcome into a match.
    """
    assert len(actual["frames"]) == len(archived["frames"])
    for left, right in zip(actual["frames"], archived["frames"]):
        assert {key: value for key, value in left.items() if key != "threats"} == {
            key: value for key, value in right.items() if key != "threats"}
        assert len(left["threats"]) == len(right["threats"])
        for target, expected in zip(left["threats"], right["threats"]):
            assert {key: value for key, value in target.items() if key != "position"} == {
                key: value for key, value in expected.items() if key != "position"}
            assert target["position"] == pytest.approx(expected["position"], rel=1e-9, abs=1e-9)
    metrics, expected = actual["metrics"], archived["metrics"]
    assert metrics.keys() == expected.keys()
    continuous = {"coverage", "sector_coverage", "early_detection", "cost", "total_cost",
                  "reward", "return", "reward_components"}
    for key, value in metrics.items():
        if key in continuous:
            assert value == pytest.approx(expected[key], rel=1e-9, abs=1e-9)
        elif key == "target_results":
            assert len(value) == len(expected[key])
            for target, stored in zip(value, expected[key]):
                assert {k: v for k, v in target.items() if k != "time_to_zone"} == {
                    k: v for k, v in stored.items() if k != "time_to_zone"}
                assert target["time_to_zone"] == pytest.approx(stored["time_to_zone"], rel=1e-9, abs=1e-9)
        else:
            assert value == expected[key]


def test_every_published_case_is_rescored_without_altering_the_archive(replays, paired):
    assert len(paired) == len(replays) == 18
    for replay, result in zip(replays, paired):
        assert result["audit"]["archived_rl_result_reproduced"]
        assert result["audit"]["published_evidence_modified"] is False
        assert result["audit"]["scenario_sha256"] == comparison._hash(replay["scenario"])
        assert result["rl"]["placements"] == replay["placements"]
        assert_archived_result_matches(result["rl"], replay)
        assert result["rl"]["mode"] == result["baseline"]["mode"] == "comparison"
        assert result["rl"]["audit"]["recorded"] is False
        json.dumps(result, allow_nan=False)


def test_archived_comparison_allows_only_continuous_roundoff(replays):
    archived = replays[0]
    rounded = deepcopy(archived)
    target = rounded["frames"][0]["threats"][0]
    target["position"][0] = math.nextafter(target["position"][0], math.inf)
    metrics = rounded["metrics"]
    metrics["coverage"] = math.nextafter(metrics["coverage"], math.inf)
    metrics["target_results"][0]["time_to_zone"] = math.nextafter(
        metrics["target_results"][0]["time_to_zone"], math.inf)
    metrics["reward_components"]["coverage"] = math.nextafter(
        metrics["reward_components"]["coverage"], math.inf)
    assert rounded != archived
    assert_archived_result_matches(rounded, archived)


@pytest.mark.parametrize("change", ["frame_time", "detection", "status", "detection_tick",
    "confirmation_tick", "outcome", "detection_fraction", "position", "coverage"])
def test_archived_comparison_rejects_changed_outcomes_or_excessive_roundoff(replays, change):
    archived, changed = replays[0], deepcopy(replays[0])
    frame, metrics = changed["frames"][0], changed["metrics"]
    if change == "frame_time":
        frame["time"] += 1e-12  # Even sub-tolerance differences in sample times must fail.
    elif change == "detection":
        frame["detections"].append({"sensor_index": 0, "target_id": "drone-1", "modalities": ["rf"]})
    elif change == "status":
        frame["threats"][0]["detected"] = not frame["threats"][0]["detected"]
    elif change in ("detection_tick", "confirmation_tick"):
        key = "first_detection" if change == "detection_tick" else "first_confirmation"
        metrics["target_results"][0][key] += 1e-12
    elif change == "outcome":
        metrics["outcome"] = "breached"
    elif change == "detection_fraction":
        metrics["detected_fraction"] -= 1e-12
    elif change == "position":
        frame["threats"][0]["position"][0] += 1e-5
    elif change == "coverage":
        metrics["coverage"] += 1e-7
    with pytest.raises(AssertionError):
        assert_archived_result_matches(changed, archived)


def test_both_layouts_receive_identical_paths_catalogue_and_resources(paired):
    for result in paired:
        rl, baseline = result["rl"], result["baseline"]
        for key in ("seed", "catalogue", "sites", "blockedSites", "budget", "weather", "objectiveRadius"):
            assert rl[key] == baseline[key]
        assert len(rl["frames"]) == len(baseline["frames"])
        for left, right in zip(rl["frames"], baseline["frames"]):
            assert left["time"] == right["time"]
            assert [{key: row[key] for key in ("id", "position", "emitting", "active")}
                    for row in left["threats"]] == [
                {key: row[key] for key in ("id", "position", "emitting", "active")}
                for row in right["threats"]]
        for view in (rl, baseline):
            assert view["metrics"]["cost"] <= view["budget"] + 1e-9
            assert view["metrics"]["invalid_actions"] == 0


def test_summary_counts_and_signed_deltas_match_measured_results(paired):
    for result in paired:
        for name in ("rl", "baseline"):
            summary = result["metrics"][name]
            targets = result[name]["metrics"]["target_results"]
            assert summary["target_count"] == len(targets)
            assert summary["detected_count"] == sum(row["first_detection"] is not None for row in targets)
            assert summary["confirmed_count"] == sum(row["first_confirmation"] is not None for row in targets)
            assert summary["timely_confirmed_count"] + summary["breached_count"] == len(targets)
            assert summary["detection_rate"] == pytest.approx(summary["detected_count"] / len(targets))
        assert result["deltaDirection"] == "RL minus baseline"
        for key, delta in result["deltas"].items():
            assert delta == pytest.approx(result["metrics"]["rl"][key] - result["metrics"]["baseline"][key])


@pytest.mark.parametrize("profile", ["normal", "stress", "capability"])
def test_identical_manual_layout_produces_identical_outcomes_and_sensing_draws(replays, profile):
    replay = next(row for row in replays if row["profile"] == profile)
    layout = [{"sensor_id": row["action"]["sensor_id"], "site_index": row["action"]["site_index"]}
              for row in replay["decisions"] if not row["action"]["stop"]]
    result = comparison.compare_placements(replay, baseline="manual", placements=layout)
    assert result["baseline"]["placements"] == result["rl"]["placements"]
    assert result["baseline"]["frames"] == result["rl"]["frames"]
    assert result["baseline"]["metrics"] == result["rl"]["metrics"]
    assert all(delta == 0 for delta in result["deltas"].values())


def test_empty_manual_layout_is_valid_and_does_not_receive_free_detections(replays):
    result = comparison.compare_placements(replays[0], baseline="manual", placements=[])
    metrics = result["metrics"]["baseline"]
    assert metrics["cost"] == metrics["sensors_placed"] == metrics["detected_count"] == 0
    assert metrics["breached_count"] == metrics["target_count"]
    assert result["baseline"]["placements"] == []
    assert all(not frame["detections"] for frame in result["baseline"]["frames"])


def test_editor_context_is_public_only_and_does_not_run_sensing(replays, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Editor context must not run the scorer")
    monkeypatch.setattr(comparison.adaptive_env.AdaptivePlacementEnv, "_simulate", forbidden)
    original = deepcopy(replays[0])
    changed = deepcopy(original)
    changed["scenario"]["seed"] += 900
    for target in changed["scenario"]["targets"]:
        target.update(bearing=2.3, altitude=170., path="weaving", emitter_phase=.123, emitter_duty=0.)
    changed["placements"] = []
    changed["frames"] = []
    changed["metrics"] = {"outcome": "made-up"}
    context = comparison.comparison_scenario(original)
    assert comparison.comparison_scenario(changed) == context
    assert not {"targets", "frames", "metrics", "decisions", "scenario"} & context.keys()
    assert context["audit"]["public_only"] is True
    assert context["audit"]["scoring_performed"] is False
    assert all(row["site_index"] in context["eligibleSites"] for row in context["suggestedPlacements"])
    assert original == replays[0]


def test_automatic_planner_receives_only_public_snapshot_and_no_case_generation(replays, monkeypatch):
    seen = []
    original_planner = comparison.plan_common_sense
    def planner(state, catalogue):
        seen.append(deepcopy(state))
        return original_planner(state, catalogue)
    def forbidden(*args, **kwargs):
        raise AssertionError("Comparison must reuse the archived case")
    monkeypatch.setattr(comparison, "plan_common_sense", planner)
    monkeypatch.setattr(comparison.adaptive_env, "generate_scenario", forbidden)
    first = comparison.compare_placements(replays[0])
    second = comparison.compare_placements(replays[0])
    assert first == second
    assert seen == [replays[0]["scenario"]["public"]] * 2


def test_new_scores_do_not_relabel_or_overwrite_old_evidence(replays):
    replay = deepcopy(replays[0])
    replay["metrics"]["return"] = -999.
    before = deepcopy(replay)
    result = comparison.compare_placements(replay)
    assert replay == before
    assert result["rl"]["metrics"]["return"] != -999.
    assert result["audit"]["archived_rl_result_reproduced"] is False
    assert result["audit"]["new_synthetic_comparison"] is True
    assert result["audit"]["fresh_rl_inference"] is False


@pytest.mark.parametrize("method", [None, "random", "oracle", True, [], {}])
def test_invalid_method_is_rejected(replays, method):
    with pytest.raises(ValueError, match="common_sense or manual"):
        comparison.compare_placements(replays[0], baseline=method)


@pytest.mark.parametrize("layout", [None, {}, [{"sensor_id": "rf", "site_index": True}],
    [{"sensor_id": "rf", "site_index": -1}], [{"sensor_id": "unavailable", "site_index": 0}],
    [{"sensor_id": "rf", "site_index": 0, "cost": 0}]])
def test_manual_inputs_cannot_override_server_constraints(replays, layout):
    with pytest.raises(ValueError):
        comparison.compare_placements(replays[0], baseline="manual", placements=layout)


def test_mixed_automatic_and_manual_payload_is_rejected(replays):
    with pytest.raises(ValueError, match="manual comparison"):
        comparison.compare_placements(replays[0], placements=[])


def test_corrupt_archived_actions_fail_instead_of_being_silently_replaced(replays):
    replay = deepcopy(replays[0])
    replay["decisions"][0]["action_index"] = -1
    with pytest.raises(ValueError, match="illegal"):
        comparison.compare_placements(replay)
