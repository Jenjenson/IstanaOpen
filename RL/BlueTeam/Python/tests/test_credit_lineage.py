"""Compact lineage keeps exact artifact anchors without sampling scenarios."""
import builtins
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from triad_rl import credit_lineage as lineage


def test_range_union_keeps_exact_half_open_exposure():
    rows = [{"start": 20, "count": 3}, {"start": 11, "count": 4},
            {"start": 10, "count": 2}, {"start": 15, "count": 2}, {"start": 99, "count": 0}]
    assert lineage.merge_ranges(rows) == [{"start": 10, "count": 7}, {"start": 20, "count": 3}]


@pytest.mark.parametrize("row", [{"start": True, "count": 1}, {"start": 0, "count": -1},
                                  {"start": 0.5, "count": 3}, {"count": 1}, {"start": 1}])
def test_invalid_seed_ranges_fail_closed(row):
    with pytest.raises(ValueError):
        lineage.merge_ranges([row])


def test_collector_separates_reservations_and_never_uses_policy_rng_seeds():
    consumed, reserved = [], []
    value = {"seed": 999, "policy_seed": 888, "rng_state": {"seed": 777},
             "training": {"start": 100, "count": 5},
             "records": [{"scenario_seed": 150}, {"seed": 160, "scenario": {"targets": []}}],
             "reserved_final_tests": [{"profile": "stress", "start": 900, "count": 10}]}
    lineage._collect(value, consumed, reserved)
    assert lineage.merge_ranges(consumed) == [{"start": 100, "count": 5}, {"start": 150, "count": 1}, {"start": 160, "count": 1}]
    assert reserved == [{"start": 900, "count": 10}]


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "repo" / "RL" / "BlueTeam"
    files = {
        "Checkpoints/balanced-v3-candidate/checkpoint.json": json.dumps({"weights_sha256": lineage.SOURCE_WEIGHTS}).encode(),
        "Checkpoints/balanced-v3-candidate/arrays.npz": b"hash-bound opaque fixture arrays",
        "Results/balanced-v3/selection.json": json.dumps({"seed_provenance": {"training": {"start": 88000000, "count": 4}}}).encode(),
        "Results/balanced-v3/protocol.json": json.dumps({"reserved_final_tests": [{"start": 2 * 10**15 + 700000, "count": 3}]}).encode(),
        "Results/balanced-v3/probe.json": json.dumps({"records": [{"scenario_seed": 10**15 + 88000000}, {"scenario_seed": 10**15 + 88000001}]}).encode(),
        "Results/balanced-v3/demo.html": b'<script id="replay-data" type="application/json">{"replays":[{"seed":1000000099000000,"scenario":{"targets":[]}}]}</script>',
        "Results/balanced-v3/training/seed-88/config.json": json.dumps({"seed": 88, "target_episodes": 8}).encode(),
    }
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    source = root.parent.parent / "Tools" / "fixture_source.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"# frozen fixture source\n")
    manifest = {"files": [{"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                          for name, data in files.items()],
                "source_files_sha256": {"Tools/fixture_source.py": hashlib.sha256(source.read_bytes()).hexdigest()}}
    path = root / "Results/balanced-v3/artifact-manifest.json"
    data = json.dumps(manifest).encode()
    path.write_bytes(data)
    monkeypatch.setattr(lineage, "MANIFESTS", {"balanced-v3": hashlib.sha256(data).hexdigest()})
    return root, source, path


def test_compact_fixture_keeps_complete_run_probe_demo_and_reserved_ranges(bundle, monkeypatch):
    root, _, _ = bundle
    original = builtins.__import__
    def no_simulation(name, *args, **kwargs):
        if name.startswith(("triad_rl.adaptive_env", "triad_rl.robust_scenarios")):
            pytest.fail("Lineage may not import a simulation environment")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_simulation)
    value = lineage.load_published_lineage(root)
    assert value["consumed_seed_ranges"] == [
        {"start": 88000000, "count": 8}, {"start": 10**15 + 88000000, "count": 2},
        {"start": 1000000099000000, "count": 1}]
    assert value["reserved_final_seed_ranges"] == [{"start": 2 * 10**15 + 700000, "count": 3}]
    assert value["source_checkpoint"]["weights_sha256"] == lineage.SOURCE_WEIGHTS
    assert value["evidence_artifact_count"] == 8
    assert str(root) not in json.dumps(value)


@pytest.mark.parametrize("target", ["manifest", "artifact", "source", "missing"])
def test_changed_or_missing_published_bytes_rejected(bundle, target):
    root, source, manifest = bundle
    if target == "manifest":
        manifest.write_bytes(manifest.read_bytes() + b" ")
    elif target == "artifact":
        (root / "Results/balanced-v3/probe.json").write_text("{}")
    elif target == "source":
        source.write_bytes(b"# changed\n")
    else:
        (root / "Results/balanced-v3/probe.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        lineage.load_published_lineage(root)


def test_source_drift_during_read_is_rejected(bundle, monkeypatch):
    root, source, _ = bundle
    original = lineage._decode
    def drift(path, data):
        values = original(path, data)
        if path.name == "config.json":
            source.write_bytes(b"# changed during extraction\n")
        return values
    monkeypatch.setattr(lineage, "_decode", drift)
    with pytest.raises(ValueError, match="implementation changed"):
        lineage.load_published_lineage(root)


def test_collision_checks_cover_entire_intervals_and_preserve_inputs(bundle):
    root, _, _ = bundle
    value = lineage.load_published_lineage(root)
    before = deepcopy(value)
    for start, count in ((87999999, 2), (88000007, 2), (10**15 + 88000000, 1), (2 * 10**15 + 700001, 1)):
        with pytest.raises(ValueError, match="overlap"):
            lineage.assert_disjoint(start, count, value)
    lineage.assert_disjoint(88000008, 5, value)
    assert value == before


@pytest.mark.parametrize("data", [b'{"seed":1,"seed":2}', b'{"x":NaN}', b'{"x":1e999}'])
def test_json_rejects_ambiguous_or_nonfinite_evidence(data):
    with pytest.raises(ValueError):
        lineage._strict(data)


@pytest.mark.parametrize("name", ["../outside", "C:/outside", "/outside", "folder\\outside"])
def test_artifact_paths_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError):
        lineage._safe_path(tmp_path.resolve(), name)


def test_actual_publication_compacts_all_known_exposure_without_sampling():
    value = lineage.load_published_lineage()
    assert value["evidence_artifact_count"] == 107
    assert len(value["publication_manifests"]) == 3
    assert len(value["consumed_seed_ranges"]) == 30
    assert len(json.dumps(value)) < 4000
    expected = [
        {"start": 42000000, "count": 4000}, {"start": 103000000, "count": 8000},
        {"start": 203000000, "count": 8000}, {"start": 1000000998000000, "count": 40},
        {"start": 1000990600000000, "count": 6}, {"start": 1000991900000000, "count": 6},
        {"start": 2000000000000000, "count": 200}, {"start": 2000000000010000, "count": 200}]
    assert all(row in value["consumed_seed_ranges"] for row in expected)
    assert value["reserved_final_seed_ranges"] == [
        {"start": 2000000500000000, "count": 200}, {"start": 2000000501000000, "count": 200},
        {"start": 2000000502000000, "count": 200}]
    # This unit test checks only abstract disjointness, never pilot scenario seeds.
    lineage.assert_disjoint(880000000, 20, value)
