"""Validate the portable eight-sensor workbench comparison evidence."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from build_native_comparison import trajectory_digest


RESULTS = Path(__file__).resolve().parents[2] / "Results/workbench-placement-comparison"


@pytest.fixture(scope="module")
def bundle():
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    blob = (RESULTS / manifest["bundle"]).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == manifest["sha256"]
    assert len(blob) == manifest["bytes"]
    result = json.loads(gzip.decompress(blob))
    assert result["schema"] == "istana.workbench_layout_comparison_capture.v1"
    assert result["protocol"] == manifest["protocol"]
    return result


def truth_frames(view):
    rows = deepcopy(view["frames"])
    for frame in rows:
        for threat in frame["threats"]:
            threat.pop("detected")
            threat.pop("confirmed")
    return rows


def test_all_archived_layouts_and_predeclared_cases_are_retained(bundle):
    assert len(bundle["episodes"]) == 9
    assert {(row["policy"], row["case"]) for row in bundle["episodes"]} == {
        (policy, case) for policy in ("406", "407", "408") for case in (1, 2, 3)}
    assert {row["seed"] for row in bundle["episodes"]} == {1700000, 1700001, 1700002}


def test_archive_sources_match_the_original_capture_not_current_training_code(bundle):
    # The capture is immutable historical evidence. Training code can evolve;
    # changing its live files must not relabel old outcomes as a new capture.
    snapshot = json.loads((RESULTS / "source-snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["schema"] == "istana.archived_capture_sources.v1"
    assert snapshot["restoredFromCommit"] == "2aaf7b698087646a084afc435879df97a80a6ea6"
    assert snapshot["publishedManifestSha256"] == hashlib.sha256(
        (RESULTS / "manifest.json").read_bytes()).hexdigest()
    assert snapshot["publishedBundleSha256"] == hashlib.sha256(
        (RESULTS / "bundle.json.gz").read_bytes()).hexdigest()
    inventory = {entry["originalPath"]: entry for entry in snapshot["sources"]}
    assert len(inventory) == len(snapshot["sources"])
    assert set(inventory) == set(bundle["protocol"]["sourceSha256"])
    for relative, expected in bundle["protocol"]["sourceSha256"].items():
        entry = inventory[relative]
        assert entry["snapshotPath"] == f"source-snapshot/{relative}"
        path = (RESULTS / entry["snapshotPath"]).resolve()
        assert path.is_relative_to((RESULTS / "source-snapshot").resolve())
        raw = path.read_bytes()
        assert len(raw) == entry["bytes"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"] == expected
        # Verify the recorded Git blob identity without requiring git history
        # in a release archive or a shallow CI checkout.
        assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == entry["gitBlob"]


def test_workbench_pairs_have_identical_truth_and_real_warning_evidence(bundle):
    for episode in bundle["episodes"]:
        rl, baseline = episode["rl"], episode["baseline"]
        expected = episode["audit"]["trajectorySha256"]
        assert trajectory_digest(truth_frames(rl)) == expected
        assert trajectory_digest(truth_frames(baseline)) == expected
        assert episode["audit"]["eightSensorRlTrained"] is False
        for view in (rl, baseline):
            assert view["budget"] == view["maxSensors"] == 8
            assert len(view["metrics"]["target_results"]) == 5
            assert len(view["warningEvidence"]) == 5
            assert len(view["frames"][0]["threats"]) == 5
        assert len(baseline["placements"]) == 8
        assert baseline["metrics"]["mean_drone_warning_seconds_lower_bound"] > 0


def test_eight_sensor_layout_is_fixed_and_directional(bundle):
    expected = bundle["episodes"][0]["baseline"]["placements"]
    assert all(row["baseline"]["placements"] == expected for row in bundle["episodes"])
    assert [row["yaw_deg"] for row in expected] == [0, 180, 90, 270, 45, 225, 135, 315]
    assert all(row["sensor_id"] == "thermal" and row["pitch_deg"] == 20 for row in expected)
