from copy import deepcopy
import gzip
import json

import pytest

from native_comparison import NativeComparisons, ROOT, _validate_episode, paired_timing
from triad_rl.trained_models import COMPARISON_SCHEMA, TrainedModelRegistry


def view(rows):
    return {"metrics": {"target_results": [dict(id=str(i), first_detection=d, first_confirmation=c)
                                           for i, d, c in rows]}}


def test_timing_uses_identical_targets_instead_of_biased_detected_group_averages():
    left = view([(1, 2., 4.), (2, None, None), (3, .1, None), (4, 9., 10.)])
    right = view([(1, 5., 7.), (2, 1., 2.), (3, None, None), (4, 11., None)])
    assert paired_timing(left, right) == {
        "shared_detected_count": 2, "mean_rl_detection_s": 5.5,
        "mean_baseline_detection_s": 8., "mean_detection_delta_s": -2.5,
        "shared_confirmed_count": 1, "mean_rl_confirmation_s": 4.,
        "mean_baseline_confirmation_s": 7., "mean_confirmation_delta_s": -3.}


def test_no_shared_detections_produces_unavailable_timing_not_zero():
    result = paired_timing(view([(1, 2., None)]), view([(1, None, None)]))
    assert result["shared_detected_count"] == result["shared_confirmed_count"] == 0
    assert all(value is None for key, value in result.items() if key.startswith("mean_"))


def test_target_mismatch_and_duplicate_ids_fail_closed():
    with pytest.raises(ValueError, match="same target IDs"):
        paired_timing(view([(1, 1., 2.)]), view([(2, 1., 2.)]))
    with pytest.raises(ValueError, match="unique target"):
        paired_timing(view([(1, 1., 2.), (1, 2., 3.)]), view([(1, 1., 2.)]))


def test_all_current_native_results_have_consistent_counts_and_delta_direction():
    store = NativeComparisons()
    for choice in store.list():
        result = store.get(choice["id"], layout_id=choice.get("defaultLayout", "matched_common_sense"))
        for key, delta in result["deltas"].items():
            assert delta == pytest.approx(result["metrics"]["rl"][key] - result["metrics"]["baseline"][key])
        assert result["audit"]["native_unreal_capture"] and not result["audit"]["live_unreal"]
        assert result["metrics"]["rl"]["target_count"] == result["metrics"]["baseline"]["target_count"]
        assert result["metrics"]["rl"]["target_count"] > 0
        if not choice.get("trainedModel"):
            assert result["metrics"]["rl"]["target_count"] == 60
        json.dumps(result, allow_nan=False)


def test_eight_directional_workbench_start_has_matched_native_warning_evidence():
    store = NativeComparisons()
    assert [row["id"] for row in store.layouts()] == [
        "matched_common_sense", "directional_balanced_8"]
    result = store.get("native-406-1", layout_id="directional_balanced_8")
    assert not result["layoutOnly"] and result["metrics"]["rl"]["target_count"] == 5
    assert result["audit"]["native_unreal_capture"]
    assert result["audit"]["eight_sensor_rl_trained"] is False
    assert result["fairness"]["matched"]
    assert len(result["baseline"]["placements"]) == 8
    assert result["baseline"]["budget"] == result["baseline"]["maxSensors"] == 8
    assert {row["sensor_id"] for row in result["baseline"]["placements"]} == {"thermal"}
    assert [row["yaw_deg"] for row in result["baseline"]["placements"]] == [
        0., 180., 90., 270., 45., 225., 135., 315.]
    assert all(row["pitch_deg"] == 20. for row in result["baseline"]["placements"])
    assert result["metrics"]["baseline"]["mean_warning_s"] > 0
    assert all(len(frame["threats"]) == 5 for side in ("rl", "baseline")
               for frame in result[side]["frames"])
    json.dumps(result, allow_nan=False)


def test_unknown_comparison_layout_fails_closed():
    with pytest.raises(ValueError, match="available fixed"):
        NativeComparisons().get("native-406-1", layout_id="manual")


def test_changed_native_trajectories_are_rejected():
    bundle = json.loads(gzip.decompress((ROOT / "bundle.json.gz").read_bytes()))
    episode = deepcopy(bundle["episodes"][0])
    episode["baseline"]["frames"][1]["threats"][0]["position"][0] += .1
    with pytest.raises(ValueError, match="trajectories"):
        _validate_episode(episode)


def test_checksum_prevents_serving_changed_evidence(tmp_path):
    raw = (ROOT / "bundle.json.gz").read_bytes()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "bundle.json.gz").write_bytes(raw + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        NativeComparisons(tmp_path).list()


def test_missing_capture_is_explicit_and_does_not_substitute_old_sensors(tmp_path):
    store = NativeComparisons(tmp_path)
    assert store.list() == []
    with pytest.raises(ValueError, match="unavailable"):
        store.get("native-406-1")


def test_completed_named_model_is_discovered_and_served_in_comparison(tmp_path):
    registry = TrainedModelRegistry(tmp_path / "models")
    source = deepcopy(NativeComparisons()._workbench_bundle()["episodes"][0])
    identifier = registry.identifier_for("Night Watch")
    source.update(schema=COMPARISON_SCHEMA, id=identifier, policy=identifier,
                  policyLabel="Night Watch", case=1,
                  label="Night Watch · held-out native episode")
    policy = tmp_path / "best.json"
    policy.write_text("{}", encoding="utf-8")
    registry.register(name="Night Watch", policy_path=policy, comparison_episode=source,
        metadata={"bestEpisode": 12, "bestWarningSeconds": 31.5,
                  "evaluationSeed": 2700000, "deploymentPlacements": []})
    store = NativeComparisons(registry=registry)
    row = next(row for row in store.list() if row["id"] == identifier)
    assert row["policyLabel"] == "Night Watch · REINFORCE"
    assert row["defaultLayout"] == "directional_balanced_8" and row["trainedModel"]
    result = store.get(identifier, layout_id="directional_balanced_8")
    assert result["trainedModel"] and result["policyLabel"] == "Night Watch"
    assert result["metrics"]["rl"]["target_count"] == 5
    with pytest.raises(ValueError, match="matched selected-count"):
        store.get(identifier, layout_id="matched_common_sense")
