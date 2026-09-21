"""Check that the portable comparison preserves its native paired evidence."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from build_native_comparison import frame, target_results, trajectory_digest


RESULTS = Path(__file__).resolve().parents[2] / "Results/native-placement-comparison"


def test_native_frames_convert_world_centimeters_only_for_presentation():
    row = frame({"drones": [{"droneId": 7, "positionCm": {"x": 500, "y": -200, "z": 1300}}]},
                {"elapsedSeconds": .5, "completedSteps": 10, "publicSnapshot": {"tracks": []}},
                {"x": 100, "y": 200, "z": 1000})
    assert row["threats"][0]["position"] == [4, -4, 3]
    assert row["threats"][0]["id"] == "drone-7"
    assert row["time"] == .5


def test_native_target_metrics_preserve_misses_and_unresolved():
    def evidence(index, detection, confirmation, entry, warning):
        return {"droneId": index, "firstDetectionSeconds": detection,
                "firstConfirmationSeconds": confirmation, "zoneEntrySeconds": entry,
                "warningSeconds": warning}
    rows = target_results([evidence(0, 1, 2, 6, 5), evidence(1, None, None, 6, 0),
                           evidence(2, 1, 2, None, None)], 4)
    assert rows[0]["timely_confirmed"] is True
    assert rows[1]["first_detection"] is None
    assert rows[1]["warning_time"] == 0
    assert rows[2]["unresolved"] is True
    assert rows[2]["timely_confirmed"] is False
    assert rows[2]["warning_time"] is None


@pytest.fixture(scope="module")
def bundle():
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    blob = (RESULTS / manifest["bundle"]).read_bytes()
    assert len(blob) == manifest["bytes"]
    assert hashlib.sha256(blob).hexdigest() == manifest["sha256"]
    result = json.loads(gzip.decompress(blob))
    assert result["protocol"] == manifest["protocol"]
    return result


def original_frames(view):
    rows = deepcopy(view["frames"])
    for row in rows:
        for threat in row["threats"]:
            threat.pop("detected")
            threat.pop("confirmed")
    return rows


def test_all_predeclared_policies_and_cases_are_retained(bundle):
    assert len(bundle["episodes"]) == 9
    assert {(row["policy"], row["case"]) for row in bundle["episodes"]} == {
        (policy, case) for policy in ("406", "407", "408") for case in (1, 2, 3)}
    assert {row["seed"] for row in bundle["episodes"]} == {2800001, 2800002, 2800003}


def test_native_paired_trajectories_catalogues_and_constraints_are_identical(bundle):
    for row in bundle["episodes"]:
        rl, baseline = row["rl"], row["baseline"]
        assert trajectory_digest(original_frames(rl)) == row["audit"]["trajectorySha256"]
        assert trajectory_digest(original_frames(baseline)) == row["audit"]["trajectorySha256"]
        assert len(rl["frames"][0]["threats"]) == 60
        for key in ("catalogue", "sites", "blockedSites", "budget", "maxSensors", "weather"):
            assert rl[key] == baseline[key]
        assert any(sensor["directional"] for sensor in rl["catalogue"])


def test_baseline_is_fixed_across_policies_and_episodes(bundle):
    expected = bundle["episodes"][0]["baseline"]["placements"]
    assert expected
    assert all(row["baseline"]["placements"] == expected for row in bundle["episodes"])


def test_terminal_metrics_and_playback_flags_follow_native_evidence(bundle):
    for episode in bundle["episodes"]:
        for side in ("rl", "baseline"):
            view = episode[side]
            targets = view["metrics"]["target_results"]
            count = len(targets)
            assert len(view["warningEvidence"]) == count == 60
            assert sum(t["first_detection"] is not None for t in targets) / count == pytest.approx(
                view["metrics"]["detected_fraction"])
            assert sum(t["timely_confirmed"] for t in targets) / count == pytest.approx(
                view["metrics"]["timely_fraction"])
            assert sum(t["unresolved"] for t in targets) / count == pytest.approx(
                view["metrics"]["unresolved_fraction"])
            indexed = {target["id"]: target for target in targets}
            for row in view["frames"]:
                for threat in row["threats"]:
                    target = indexed[threat["id"]]
                    assert threat["detected"] == (target["first_detection"] is not None
                        and target["first_detection"] <= row["time"] + 1e-8)
                    assert threat["confirmed"] == (target["first_confirmation"] is not None
                        and target["first_confirmation"] <= row["time"] + 1e-8)
