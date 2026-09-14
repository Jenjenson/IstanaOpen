"""Exact byte archival and no-sampling terminal verification for the anchored pilot."""
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from test_evaluate_anchored_pilot import anchored_evaluation_fixture

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "Tools"))
spec = importlib.util.spec_from_file_location("archive_anchored_pilot", REPO_ROOT / "Tools/archive_anchored_pilot.py")
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def tree(path):
    return {file.relative_to(path).as_posix(): file.read_bytes() for file in path.rglob("*") if file.is_file()}


@pytest.fixture
def stub(tmp_path, monkeypatch):
    evaluation, protocol, output = tmp_path / "evaluation", tmp_path / "protocol.json", tmp_path / "archive"
    evaluation.mkdir()
    protocol.write_bytes(b'{"test_only":true}\n')
    for name in archive.evaluator.EVALUATION_FILES:
        (evaluation / name).write_bytes(gzip.compress(b"{}", mtime=0) if name.endswith(".gz") else b"{}\n")
    runs = {}
    for arm in archive.evaluator.ARMS:
        runs[arm] = tmp_path / arm
        for name in archive.evaluator.trainer.RUN_FILES:
            path = runs[arm] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"opaque fixture" if name.endswith(".npz") else b"{}\n")
    calls = []
    def verify(*args):
        calls.append(args)
        return {"final_test_accessed": False, "scale_gate": {"passed": False}}
    monkeypatch.setattr(archive, "_verify", verify)
    monkeypatch.setattr(archive.evaluator, "source_provenance", lambda: {"test_only.py": "a" * 64})
    return {"runs": runs, "evaluation": evaluation, "protocol": protocol, "output": output}, calls


def test_complete_copy_is_exact_idempotent_and_does_not_promote(stub):
    args, calls = stub
    manifest = archive.archive(**args)
    assert len(manifest["files"]) == 33 and len(tree(args["output"])) == 34
    assert manifest["schema"] == "triad.anchored_pilot_publication.v1"
    assert manifest["default_policy_changed"] is False and manifest["final_test_accessed"] is False
    for row in manifest["files"]:
        data = (args["output"] / row["path"]).read_bytes()
        assert archive.sha256(data) == row["sha256"] and len(data) == row["bytes"]
    original = tree(args["output"])
    assert archive.archive(**args) == manifest and tree(args["output"]) == original
    assert len(calls) == 2


@pytest.mark.parametrize("name", ["aggregate.json", "validation-normal.json.gz", "validation-stress.receipt.json"])
def test_missing_terminal_report_fails_before_any_validation_or_copy(stub, name):
    args, calls = stub
    (args["evaluation"] / name).unlink()
    with pytest.raises(ValueError, match="complete terminal"):
        archive.archive(**args)
    assert not calls and not args["output"].exists()


def test_missing_endpoint_fails_before_validation_or_copy(stub):
    args, calls = stub
    (args["runs"]["anchored_later_stop"] / "last/arrays.npz").unlink()
    with pytest.raises(FileNotFoundError):
        archive.archive(**args)
    assert not calls and not args["output"].exists()


@pytest.mark.parametrize("raw", [b'{"password":"private"}', b'{"location":"C:/Users/private/experiment"}',
                                  b'{"duplicate":1,"duplicate":2}', b'{"number":1e999}'])
def test_privacy_and_ambiguous_json_fail_before_validation_or_copy(stub, raw):
    args, calls = stub
    args["protocol"].write_bytes(raw)
    with pytest.raises(ValueError):
        archive.archive(**args)
    assert not calls and not args["output"].exists()


def test_conflicting_destination_preserves_all_user_owned_bytes(stub):
    args, _ = stub
    target = args["output"] / archive.RESULTS / "training/anchored_later_stop/last/arrays.npz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user-owned conflicting evidence")
    before = tree(args["output"])
    with pytest.raises(FileExistsError):
        archive.archive(**args)
    assert tree(args["output"]) == before


def test_parent_file_conflict_creates_no_partial_archive(stub):
    args, _ = stub
    args["output"].mkdir()
    (args["output"] / "Results").write_bytes(b"user-owned file")
    before = tree(args["output"])
    with pytest.raises(ValueError, match="parent"):
        archive.archive(**args)
    assert tree(args["output"]) == before


def test_input_drift_during_verification_is_rejected(stub, monkeypatch):
    args, _ = stub
    def drift(*unused):
        args["protocol"].write_bytes(b'{"changed":true}')
        return {"scale_gate": {"passed": False}}
    monkeypatch.setattr(archive, "_verify", drift)
    with pytest.raises(ValueError, match="changed"):
        archive.archive(**args)
    assert not args["output"].exists()


def no_sampling(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Archival must not invoke any runner, policy decision or simulator")
    monkeypatch.setattr(archive.evaluator, "run_evaluation", forbidden)
    monkeypatch.setattr(archive.evaluator, "evaluate_robust_methods", forbidden)
    monkeypatch.setattr(archive.evaluator.trainer, "run_training", forbidden)
    for kind in (archive.evaluator.BalancedPolicy, archive.evaluator.AdaptivePolicy):
        monkeypatch.setattr(kind, "act", forbidden)
        monkeypatch.setattr(kind, "sample", forbidden)
    monkeypatch.setattr(archive.evaluator.trainer.RobustPlacementEnv, "reset", forbidden)
    monkeypatch.setattr(archive.evaluator.trainer.RobustPlacementEnv, "step", forbidden)


def test_real_terminal_evidence_archives_with_no_sampling_and_no_input_mutation(anchored_evaluation_fixture, tmp_path, monkeypatch):
    fixture = anchored_evaluation_fixture
    no_sampling(monkeypatch)
    # The session fixture already validated the actual published lineage. Reuse
    # it for this archive arithmetic/copy test, not another expensive extraction.
    monkeypatch.setattr(archive.evaluator, "load_published_lineage", lambda: fixture["lineage"])
    before = tree(fixture["root"])
    args = {"runs": fixture["runs"], "evaluation": fixture["output"], "protocol": fixture["protocol_path"], "output": tmp_path / "published"}
    manifest = archive.archive(**args)
    assert manifest["scale_gate"] == fixture["aggregate"]["scale_gate"]
    assert len(manifest["files"]) == 33 and tree(fixture["root"]) == before
    for row in manifest["files"]:
        path = args["output"] / row["path"]
        assert archive.sha256(path.read_bytes()) == row["sha256"]
    assert archive.archive(**args) == manifest


def test_real_first_batch_coefficient_identity_must_match_all_arms(anchored_evaluation_fixture, tmp_path, monkeypatch):
    fixture = anchored_evaluation_fixture
    no_sampling(monkeypatch)
    monkeypatch.setattr(archive.evaluator, "load_published_lineage", lambda: fixture["lineage"])
    original = archive.evaluator.trainer.validate_run
    def differing_identity(path, protocol, arm, lineage):
        policy, evidence = original(path, protocol, arm, lineage)
        if arm == "anchored_later_stop":
            evidence["first_batch_coefficient_identity"] = {
                **evidence["first_batch_coefficient_identity"], "reference_coefficients_sha256": "0" * 64}
        return policy, evidence
    monkeypatch.setattr(archive.evaluator.trainer, "validate_run", differing_identity)
    output = tmp_path / "published"
    with pytest.raises(ValueError, match="coefficient evidence"):
        archive.archive(runs=fixture["runs"], evaluation=fixture["output"], protocol=fixture["protocol_path"], output=output)
    assert not output.exists()


@pytest.mark.parametrize("mutation", ["gate", "paired_interval", "declared_lineage", "summary"])
def test_real_invalid_gate_or_rehashed_report_is_not_archived(anchored_evaluation_fixture, tmp_path, monkeypatch, mutation):
    fixture = anchored_evaluation_fixture
    no_sampling(monkeypatch)
    monkeypatch.setattr(archive.evaluator, "load_published_lineage", lambda: fixture["lineage"])
    evaluation = tmp_path / "evaluation"
    shutil.copytree(fixture["output"], evaluation)
    if mutation in ("gate", "paired_interval"):
        path = evaluation / "aggregate.json"
        value = json.loads(path.read_bytes())
        if mutation == "gate":
            value["scale_gate"]["passed"] = not value["scale_gate"]["passed"]
        else:
            next(iter(value["paired_stratified_differences"].values()))["timely_fraction"]["lower95"] += .1
        path.write_bytes(archive.evaluator._bytes(value))
    else:
        path = evaluation / "validation-normal.json.gz"
        value = json.loads(gzip.decompress(path.read_bytes()))
        if mutation == "declared_lineage":
            value["seed_provenance"]["declared_lineage"] = {}
        else:
            next(iter(value["methods"].values()))["summary"]["mean_cost"] += .1
        data = gzip.compress(archive.evaluator._bytes(value), mtime=0)
        path.write_bytes(data)
        receipt_path = evaluation / "validation-normal.receipt.json"
        receipt = json.loads(receipt_path.read_bytes())
        receipt.update(sha256=archive.sha256(data), bytes=len(data))
        receipt_path.write_bytes(archive.evaluator._bytes(receipt))
    output = tmp_path / "published"
    with pytest.raises(ValueError):
        archive.archive(runs=fixture["runs"], evaluation=evaluation, protocol=fixture["protocol_path"], output=output)
    assert not output.exists()
