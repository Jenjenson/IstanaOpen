"""Small byte-exact publication tests on hand-case evaluation fixtures only."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil

import pytest

from test_evaluate_temporal import bindings, forbidden, temporal_evaluation_fixture
from triad_rl.temporal_policy import TemporalPolicy

REPO = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("archive_temporal_pilot", REPO / "Tools/archive_temporal_pilot.py")
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def tree(path):
    return {item.relative_to(path).as_posix(): item.read_bytes() for item in path.rglob("*") if item.is_file()}


@pytest.fixture
def source(temporal_evaluation_fixture, monkeypatch, tmp_path):
    fixture = temporal_evaluation_fixture
    bindings(monkeypatch, fixture)
    for owner, names in ((archive.evaluator, ("run_evaluation", "evaluate_robust_methods", "make_case")),
                         (TemporalPolicy, ("act", "sample", "update"))):
        for name in names: monkeypatch.setattr(owner, name, forbidden)
    return {"runs": fixture["runs"], "evaluation": fixture["output"], "protocol": fixture["protocol_path"],
            "output": tmp_path / "blue"}


def test_exact_36_inputs_manifest_ranges_and_no_mutation(source):
    before = {"evaluation": tree(source["evaluation"]), "runs": {seed: tree(path) for seed, path in source["runs"].items()}}
    manifest = archive.archive(**source)
    assert manifest["schema"] == "triad.temporal_pilot_publication.v1"
    assert manifest["predeclared_source_commit"] == "e2d4478c9eb5352640b317c4507a85667bce53c0"
    assert "not a claim of compute-matched" in manifest["runtime_reporting"]
    assert manifest["policy_seeds"] == [497, 498, 499]
    assert manifest["default_policy_changed"] is False and manifest["final_test_accessed"] is False
    assert manifest["scale_gate"]["not_policy_promotion"] is True
    assert len(manifest["files"]) == 36 and len(tree(source["output"])) == 37
    protocol = archive.evaluator._strict_json(source["protocol"].read_bytes())
    assert manifest["seed_provenance"]["training"] == protocol["training"]["scenario_seed_ranges"]
    assert manifest["seed_provenance"]["validation"] == protocol["validation"]["scenario_seed_ranges"]
    assert manifest["archive_implementation_sha256"] == archive.source_provenance()
    assert manifest["training_implementation_sha256"] == protocol["training_implementation_sha256"]
    assert manifest["evaluation_implementation_sha256"] == protocol["evaluation_implementation_sha256"]
    for row in manifest["files"]:
        data = (source["output"] / row["path"]).read_bytes()
        assert archive.evaluator._sha(data) == row["sha256"] and len(data) == row["bytes"]
    for name in archive.evaluator.EVALUATION_FILES:
        assert (source["output"] / archive.RESULTS / "evaluation" / name).read_bytes() == before["evaluation"][name]
    for seed, rows in before["runs"].items():
        for name, data in rows.items():
            assert (source["output"] / archive.RESULTS / f"training/seed-{seed}" / name).read_bytes() == data
    assert before == {"evaluation": tree(source["evaluation"]), "runs": {seed: tree(path) for seed, path in source["runs"].items()}}


def test_identical_protocol_only_destination_is_preserved_and_complete_archive_is_idempotent(source, monkeypatch):
    destination = source["output"] / archive.RESULTS
    destination.mkdir(parents=True)
    path = destination / "protocol.json"
    path.write_bytes(source["protocol"].read_bytes())
    before = path.stat().st_mtime_ns
    manifest = archive.archive(**source)
    assert path.stat().st_mtime_ns == before
    contents = tree(destination)
    monkeypatch.setattr(archive, "_new", forbidden)
    assert archive.archive(**source) == manifest
    assert tree(destination) == contents


@pytest.mark.parametrize("kind", ["conflicting_protocol", "unrelated_file", "partial_training", "empty_unrelated_directory"])
def test_existing_conflicts_and_partial_archives_are_not_overwritten(source, kind):
    destination = source["output"] / archive.RESULTS
    destination.mkdir(parents=True)
    (destination / "protocol.json").write_bytes(source["protocol"].read_bytes())
    if kind == "conflicting_protocol": (destination / "protocol.json").write_bytes(b"user protocol")
    elif kind == "unrelated_file": (destination / "notes.txt").write_bytes(b"user notes")
    elif kind == "partial_training":
        path = destination / "training/seed-499/summary.json"
        path.parent.mkdir(parents=True); path.write_bytes(b"unfinished")
    else: (destination / "unrelated").mkdir()
    before = tree(destination)
    with pytest.raises(FileExistsError): archive.archive(**source)
    assert tree(destination) == before and not (destination / "artifact-manifest.json").exists()


def test_missing_completed_aggregate_fails_before_any_publication(source, tmp_path):
    copied = tmp_path / "incomplete-evaluation"
    shutil.copytree(source["evaluation"], copied)
    (copied / "aggregate.json").unlink()
    with pytest.raises(FileNotFoundError): archive.archive(**{**source, "evaluation": copied})
    assert not source["output"].exists()


def test_input_change_after_read_only_verification_fails_before_writes(source, tmp_path, monkeypatch):
    copied = tmp_path / "changed-evaluation"
    shutil.copytree(source["evaluation"], copied)
    verify = archive.evaluator.verify_completed_evaluation
    def change_after_verification(*args, **kwargs):
        result = verify(*args, **kwargs)
        path = copied / "aggregate.json"
        path.write_bytes(path.read_bytes() + b" ")
        return result
    monkeypatch.setattr(archive.evaluator, "verify_completed_evaluation", change_after_verification)
    with pytest.raises(ValueError, match="changed"):
        archive.archive(**{**source, "evaluation": copied})
    assert not source["output"].exists()


def test_final_test_or_promotion_scope_is_refused(source, monkeypatch):
    verify = archive.evaluator.verify_completed_evaluation
    def wrong_scope(*args, **kwargs):
        result = deepcopy(verify(*args, **kwargs)); result["final_test_accessed"] = True
        return result
    monkeypatch.setattr(archive.evaluator, "verify_completed_evaluation", wrong_scope)
    with pytest.raises(ValueError, match="scope"): archive.archive(**source)
    assert not source["output"].exists()


def test_interrupted_write_never_publishes_manifest_or_overwrites_existing_protocol(source, monkeypatch):
    destination = source["output"] / archive.RESULTS
    destination.mkdir(parents=True)
    protocol = source["protocol"].read_bytes()
    (destination / "protocol.json").write_bytes(protocol)
    original, calls = archive._new, []
    def interrupt(path, data):
        calls.append(path)
        if len(calls) == 3: raise OSError("simulated interrupted write")
        original(path, data)
    monkeypatch.setattr(archive, "_new", interrupt)
    with pytest.raises(OSError, match="interrupted"): archive.archive(**source)
    assert not (destination / "artifact-manifest.json").exists()
    assert (destination / "protocol.json").read_bytes() == protocol
    before = tree(destination)
    monkeypatch.setattr(archive, "_new", forbidden)
    with pytest.raises(FileExistsError): archive.archive(**source)
    assert tree(destination) == before  # Partial files stay visible; no deletion/recovery magic.


def test_destination_symlink_is_refused_before_writing_elsewhere(source, tmp_path):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    destination = source["output"] / archive.RESULTS
    destination.parent.mkdir(parents=True)
    try:
        destination.symlink_to(unrelated, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are unavailable on this host")
    with pytest.raises(ValueError, match="symlink"): archive.archive(**source)
    assert not list(unrelated.iterdir())
