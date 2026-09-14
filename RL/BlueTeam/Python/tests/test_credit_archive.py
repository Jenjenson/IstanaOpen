"""Byte-copy/preflight guards; numerical evidence validation belongs to evaluator tests."""
import gzip
import importlib.util
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "Tools"))
spec = importlib.util.spec_from_file_location("archive_credit_pilot", REPO_ROOT / "Tools/archive_credit_pilot.py")
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    evaluation, protocol, output = tmp_path / "evaluation", tmp_path / "protocol.json", tmp_path / "published"
    evaluation.mkdir()
    protocol.write_bytes(b'{"fixture":true}\n')
    for name in archive.EVALUATION_FILES:
        (evaluation / name).write_bytes(gzip.compress(b'{}', mtime=0) if name.endswith(".gz") else b'{}\n')
    runs = {}
    for arm in archive.evaluator.ARMS:
        path = tmp_path / arm
        runs[arm] = path
        for name in archive.evaluator.RUN_FILES:
            target = path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'opaque-array-fixture' if name.endswith(".npz") else b'{}\n')
    calls = []
    def verify(paths, *, protocol_path, output, resume):
        calls.append((paths, protocol_path, output, resume))
        return {"final_test_accessed": False, "scale_gate": {"passed": False}}
    monkeypatch.setattr(archive.evaluator, "run_evaluation", verify)
    monkeypatch.setattr(archive.evaluator, "source_provenance", lambda: {"fixture.py": "0" * 64})
    return dict(runs=runs, evaluation=evaluation, protocol=protocol, output=output), calls


def tree(path):
    return {file.relative_to(path).as_posix(): file.read_bytes() for file in path.rglob("*") if file.is_file()}


def test_exact_complete_copy_is_idempotent_and_never_promotes(fixture):
    args, calls = fixture
    manifest = archive.archive(**args)
    assert len(manifest["files"]) == 33
    assert manifest["default_policy_changed"] is False and manifest["final_test_accessed"] is False
    for row in manifest["files"]:
        data = (args["output"] / row["path"]).read_bytes()
        assert archive.sha256(data) == row["sha256"] and len(data) == row["bytes"]
    before = tree(args["output"])
    assert archive.archive(**args) == manifest
    assert tree(args["output"]) == before
    assert len(calls) == 2 and all(call[3] is True for call in calls)


def test_missing_completed_profile_never_invokes_evaluation(fixture):
    args, calls = fixture
    (args["evaluation"] / "validation-stress.receipt.json").unlink()
    with pytest.raises(FileNotFoundError):
        archive.archive(**args)
    assert calls == [] and not args["output"].exists()


def test_secret_fails_before_verification_or_output(fixture):
    args, calls = fixture
    args["protocol"].write_bytes(b'{"password":"not-for-publication"}')
    with pytest.raises(ValueError, match="credential"):
        archive.archive(**args)
    assert calls == [] and not args["output"].exists()


def test_conflicting_destination_leaves_all_existing_bytes_intact(fixture):
    args, _ = fixture
    target = args["output"] / archive.RESULTS / "training/paired_stop/last/arrays.npz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b'user-owned conflicting content')
    before = tree(args["output"])
    with pytest.raises(FileExistsError):
        archive.archive(**args)
    assert tree(args["output"]) == before


def test_input_drift_during_verification_never_copies(fixture, monkeypatch):
    args, _ = fixture
    def drift(*unused, **kwargs):
        args["protocol"].write_bytes(b'{}')
        return {"final_test_accessed": False, "scale_gate": {"passed": False}}
    monkeypatch.setattr(archive.evaluator, "run_evaluation", drift)
    with pytest.raises(ValueError, match="changed"):
        archive.archive(**args)
    assert not args["output"].exists()
