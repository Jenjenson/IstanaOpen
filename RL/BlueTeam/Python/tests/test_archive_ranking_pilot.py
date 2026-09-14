"""Byte-exact, manifest-last ranking publication without new scenario sampling."""
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from test_evaluate_ranking import fixture_bindings, ranking_evaluation_fixture


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "Tools"))
spec = importlib.util.spec_from_file_location("archive_ranking_pilot", REPO_ROOT / "Tools/archive_ranking_pilot.py")
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)
RUN_FILES = ("config.json", "protocol.json", "training.jsonl", "comparisons.jsonl.gz", "summary.json",
             "initialized/checkpoint.json", "initialized/arrays.npz", "last/checkpoint.json", "last/arrays.npz")


def tree(path):
    return {file.relative_to(path).as_posix(): file.read_bytes() for file in path.rglob("*") if file.is_file()}


@pytest.fixture
def stub(tmp_path, monkeypatch):
    evaluation, protocol, output = tmp_path / "evaluation", tmp_path / "protocol.json", tmp_path / "archive"
    evaluation.mkdir()
    sources = {"test_only.py": "a" * 64}
    document = {"schema": "triad.ranking_pilot_protocol.v1", "training": {"policy_seeds": [497, 498, 499]},
                "evaluation_implementation_sha256": sources, "training_implementation_sha256": sources}
    protocol.write_bytes(archive._canonical_bytes(document))
    aggregate = {"schema": "triad.ranking_aggregate.v1", "final_test_accessed": False,
                 "scale_gate": {"passed": False}}
    for name in archive.EVALUATION_FILES:
        data = archive._canonical_bytes(aggregate if name == "aggregate.json" else {})
        (evaluation / name).write_bytes(gzip.compress(data, mtime=0) if name.endswith(".gz") else data)
    runs = {}
    for seed in (497, 498, 499):
        runs[seed] = tmp_path / f"seed-{seed}"
        for name in RUN_FILES:
            path = runs[seed] / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = b"opaque test NPZ" if name.endswith(".npz") else b"{}\n"
            path.write_bytes(gzip.compress(data, mtime=0) if name.endswith(".gz") else data)
    calls = []
    def verify(runs, *, protocol_path, output):
        calls.append((runs, protocol_path, output))
        return json.loads((output / "aggregate.json").read_bytes())
    evaluator = SimpleNamespace(EVALUATION_FILES=archive.EVALUATION_FILES,
                                trainer=SimpleNamespace(RUN_FILES=RUN_FILES),
                                source_provenance=lambda: dict(sources),
                                verify_completed_evaluation=verify)
    monkeypatch.setattr(archive, "_evaluator", lambda: evaluator)
    return {"runs": runs, "evaluation": evaluation, "protocol": protocol, "output": output}, calls, evaluator


def test_exact_complete_archive_three_seed_inventory_and_idempotence(stub):
    args, calls, _ = stub
    source_before = {"evaluation": tree(args["evaluation"]), "runs": {seed: tree(path) for seed, path in args["runs"].items()}}
    manifest = archive.archive(**args)
    assert len(manifest["files"]) == 36
    assert len(tree(args["output"])) == 37
    assert manifest["schema"] == "triad.ranking_pilot_publication.v1"
    assert manifest["default_policy_changed"] is False and manifest["final_test_accessed"] is False
    assert manifest["policy_seeds"] == [497, 498, 499]
    for row in manifest["files"]:
        data = (args["output"] / row["path"]).read_bytes()
        assert archive.sha256(data) == row["sha256"] and len(data) == row["bytes"]
    for seed, path in args["runs"].items():
        for name in RUN_FILES:
            assert (args["output"] / archive.RESULTS / f"training/seed-{seed}" / name).read_bytes() == (path / name).read_bytes()
    for name in archive.EVALUATION_FILES:
        assert (args["output"] / archive.RESULTS / "evaluation" / name).read_bytes() == (args["evaluation"] / name).read_bytes()
    before = tree(args["output"])
    assert archive.archive(**args) == manifest
    assert tree(args["output"]) == before
    assert source_before == {"evaluation": tree(args["evaluation"]), "runs": {seed: tree(path) for seed, path in args["runs"].items()}}
    assert len(calls) == 2


def test_identical_protocol_only_destination_and_manifest_last(stub, monkeypatch):
    args, _, _ = stub
    target = args["output"] / archive.RESULTS
    target.mkdir(parents=True)
    protocol = args["protocol"].read_bytes()
    (target / "protocol.json").write_bytes(protocol)
    original, names = archive._rename_noreplace, []
    def traced(source, destination):
        names.append(destination.name)
        if destination.name == "artifact-manifest.json":
            assert (target / "training").is_dir() and (target / "evaluation").is_dir()
            assert not destination.exists()
        original(source, destination)
    monkeypatch.setattr(archive, "_rename_noreplace", traced)
    archive.archive(**args)
    assert names == ["training", "evaluation", "artifact-manifest.json"]
    assert (target / "protocol.json").read_bytes() == protocol


@pytest.mark.parametrize("extra", ["training/seed-499/config.json", "unrelated.txt", "empty_directory"])
def test_mixed_or_partial_destination_refused_unchanged(stub, extra):
    args, _, _ = stub
    target = args["output"] / archive.RESULTS
    target.mkdir(parents=True)
    (target / "protocol.json").write_bytes(args["protocol"].read_bytes())
    path = target / extra
    path.parent.mkdir(parents=True, exist_ok=True)
    if extra == "empty_directory":
        path.mkdir()
    else:
        path.write_bytes(b"user-owned")
    before = archive._tree(target)
    with pytest.raises(FileExistsError, match="protocol-only"):
        archive.archive(**args)
    assert archive._tree(target) == before


def test_different_protocol_or_complete_archive_not_replaced(stub):
    args, _, _ = stub
    archive.archive(**args)
    target = args["output"] / archive.RESULTS / "protocol.json"
    target.write_bytes(b'{"user":"modified"}\n')
    before = tree(args["output"])
    with pytest.raises(FileExistsError):
        archive.archive(**args)
    assert tree(args["output"]) == before


@pytest.mark.parametrize("failure", ["evaluation", "artifact-manifest.json"])
def test_failed_manifest_last_install_rolls_back_only_new_unchanged_artifacts(stub, monkeypatch, failure):
    args, _, _ = stub
    target = args["output"] / archive.RESULTS
    target.mkdir(parents=True)
    (target / "protocol.json").write_bytes(args["protocol"].read_bytes())
    before = tree(target)
    original = archive._rename_noreplace
    def fail(source, destination):
        if destination.parent == target and destination.name == failure:
            raise OSError("simulated interrupted publication")
        original(source, destination)
    monkeypatch.setattr(archive, "_rename_noreplace", fail)
    with pytest.raises(OSError, match="interrupted"):
        archive.archive(**args)
    assert tree(target) == before
    assert not list(target.parent.glob(".ranking-v5-pilot-staging-*"))


def test_rollback_preserves_concurrent_user_changes(stub, monkeypatch):
    args, _, _ = stub
    target = args["output"] / archive.RESULTS
    target.mkdir(parents=True)
    (target / "protocol.json").write_bytes(args["protocol"].read_bytes())
    original = archive._rename_noreplace
    def fail(source, destination):
        if destination.parent == target and destination.name == "evaluation":
            (target / "training/user-note.txt").write_bytes(b"do not remove me")
            raise OSError("interrupted")
        original(source, destination)
    monkeypatch.setattr(archive, "_rename_noreplace", fail)
    with pytest.raises(RuntimeError, match="preserved for manual recovery"):
        archive.archive(**args)
    assert (target / "training/user-note.txt").read_bytes() == b"do not remove me"
    assert not (target / "artifact-manifest.json").exists()


def test_atomic_new_archive_race_does_not_replace_created_destination(stub, monkeypatch):
    args, _, _ = stub
    original = archive._rename_noreplace
    def raced(source, destination):
        destination.mkdir()
        (destination / "user-file").write_bytes(b"race winner")
        original(source, destination)
    monkeypatch.setattr(archive, "_rename_noreplace", raced)
    with pytest.raises(FileExistsError):
        archive.archive(**args)
    assert tree(args["output"] / archive.RESULTS) == {"user-file": b"race winner"}


def test_low_level_no_replace_preserves_existing_empty_directory(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "evidence").write_bytes(b"new")
    with pytest.raises(FileExistsError):
        archive._rename_noreplace(source, target)
    assert target.is_dir() and not list(target.iterdir())
    assert (source / "evidence").read_bytes() == b"new"


@pytest.mark.parametrize("name", ["aggregate.json", "validation-normal.json.gz", "validation-stress.receipt.json"])
def test_missing_evaluation_refused_before_validation_or_output(stub, name):
    args, calls, _ = stub
    (args["evaluation"] / name).unlink()
    with pytest.raises(ValueError, match="complete terminal"):
        archive.archive(**args)
    assert not calls and not args["output"].exists()


def test_missing_training_dataset_refused_before_validation_or_output(stub):
    args, calls, _ = stub
    (args["runs"][499] / "comparisons.jsonl.gz").unlink()
    with pytest.raises(FileNotFoundError):
        archive.archive(**args)
    assert not calls and not args["output"].exists()


@pytest.mark.parametrize("name", ["extra.txt", "extra_directory"])
def test_unexpected_evaluation_entry_refused(stub, name):
    args, calls, _ = stub
    path = args["evaluation"] / name
    path.mkdir() if name == "extra_directory" else path.write_bytes(b"unrelated")
    with pytest.raises(ValueError):
        archive.archive(**args)
    assert not calls and not args["output"].exists()


@pytest.mark.parametrize("raw", [b'{"password":"secret"}', b'{"path":"C:/Users/private/experiment"}',
                                  b'{"duplicate":1,"duplicate":2}', b'{"number":1e999}', b'[]', b''])
def test_private_invalid_or_ambiguous_comparison_jsonl_refused(stub, raw):
    args, calls, _ = stub
    (args["runs"][499] / "comparisons.jsonl.gz").write_bytes(gzip.compress(raw, mtime=0))
    with pytest.raises(ValueError):
        archive.archive(**args)
    assert not calls and not args["output"].exists()


@pytest.mark.parametrize("mode", ["truncated", "line_bound", "expanded_bound", "file_bound"])
def test_bounded_compressed_input_validation(stub, monkeypatch, mode):
    args, calls, _ = stub
    path = args["runs"][499] / "comparisons.jsonl.gz"
    if mode == "truncated":
        path.write_bytes(path.read_bytes()[:-5])
    elif mode == "line_bound":
        monkeypatch.setattr(archive, "MAX_LINE_BYTES", 2)
    elif mode == "expanded_bound":
        monkeypatch.setattr(archive, "MAX_EXPANDED_BYTES", 2)
    else:
        monkeypatch.setattr(archive, "MAX_FILE_BYTES", 2)
    with pytest.raises(ValueError):
        archive.archive(**args)
    assert not calls and not args["output"].exists()


@pytest.mark.parametrize("runs", [{}, {True: "x"}, {"499": "x"}, {-1: "x"}, {i: "x" for i in range(4)}])
def test_invalid_run_keys(stub, runs):
    args, calls, _ = stub
    with pytest.raises(ValueError):
        archive.archive(**{**args, "runs": runs})
    assert not calls and not args["output"].exists()


def test_missing_declared_seed_and_bool_seed_declaration_rejected(stub):
    args, calls, _ = stub
    runs = {seed: path for seed, path in args["runs"].items() if seed != 499}
    with pytest.raises(ValueError, match="declared"):
        archive.archive(**{**args, "runs": runs})
    doc = json.loads(args["protocol"].read_bytes())
    doc["training"]["policy_seeds"] = [True, 498, 499]
    args["protocol"].write_bytes(archive._canonical_bytes(doc))
    with pytest.raises(ValueError, match="declared"):
        archive.archive(**args)
    assert not calls and not args["output"].exists()


def test_evaluator_rejection_leaves_no_output(stub):
    args, _, evaluator = stub
    def forbidden(*unused, **kwargs):
        raise ValueError("Recorded paired interval differs")
    evaluator.verify_completed_evaluation = forbidden
    with pytest.raises(ValueError, match="interval"):
        archive.archive(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("field,value", [("final_test_accessed", True), ("final_test_accessed", 0), ("passed", 0)])
def test_scope_flags_are_strict_booleans(stub, field, value):
    args, _, _ = stub
    path = args["evaluation"] / "aggregate.json"
    doc = json.loads(path.read_bytes())
    (doc["scale_gate"] if field == "passed" else doc)[field] = value
    path.write_bytes(archive._canonical_bytes(doc))
    with pytest.raises(ValueError, match="scope"):
        archive.archive(**args)
    assert not args["output"].exists()


def test_input_drift_during_verification_fails_before_output(stub):
    args, _, evaluator = stub
    original = evaluator.verify_completed_evaluation
    def drift(*items, **kwargs):
        result = original(*items, **kwargs)
        args["protocol"].write_bytes(b'{"changed":true}')
        return result
    evaluator.verify_completed_evaluation = drift
    with pytest.raises(ValueError, match="changed"):
        archive.archive(**args)
    assert not args["output"].exists()


def test_source_drift_during_validation_fails_before_output(stub):
    args, _, evaluator = stub
    original = evaluator.verify_completed_evaluation
    def drift(*items, **kwargs):
        result = original(*items, **kwargs)
        evaluator.source_provenance = lambda: {"different.py": "f" * 64}
        return result
    evaluator.verify_completed_evaluation = drift
    with pytest.raises(ValueError, match="changed"):
        archive.archive(**args)
    assert not args["output"].exists()


def test_parent_file_conflict_creates_no_partial_archive(stub):
    args, _, _ = stub
    args["output"].mkdir()
    (args["output"] / "Results").write_bytes(b"user owned")
    with pytest.raises(ValueError, match="parent"):
        archive.archive(**args)
    assert tree(args["output"]) == {"Results": b"user owned"}


def test_symlink_destination_refused_without_following_it(stub, tmp_path):
    args, _, _ = stub
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    try:
        args["output"].symlink_to(elsewhere, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Host does not permit symlink creation")
    with pytest.raises(ValueError, match="symlink"):
        archive.archive(**args)
    assert not list(elsewhere.iterdir())


def no_new_trajectory(monkeypatch):
    evaluator = archive._evaluator()
    def forbidden(*args, **kwargs):
        pytest.fail("Archive verification must not run a new training or behavior/simulator trajectory")
    monkeypatch.setattr(evaluator, "run_evaluation", forbidden)
    monkeypatch.setattr(evaluator, "evaluate_robust_methods", forbidden)
    monkeypatch.setattr(evaluator.trainer, "run_training", forbidden)
    monkeypatch.setattr(evaluator.trainer, "collect_episode", forbidden)
    for name in ("reset", "step"):
        monkeypatch.setattr(evaluator.trainer.RobustPlacementEnv, name, forbidden)
    for kind in (evaluator.RankPolicy, evaluator.BalancedPolicy, evaluator.AdaptivePolicy,
                 evaluator.GreedyPublicCoverage):
        monkeypatch.setattr(kind, "act", forbidden)
    for name in ("propose_slate", "evaluate_slate"):
        monkeypatch.setattr(evaluator.trainer.rollout, name, forbidden)
    # In-memory optimizer reconstruction is deliberately allowed: it checks
    # archived feature/label evidence without collecting or writing new data.


def test_real_completed_fixture_archives_exactly_without_new_trajectory(ranking_evaluation_fixture, tmp_path, monkeypatch):
    fixture = ranking_evaluation_fixture
    fixture_bindings(monkeypatch, fixture["lineage"])
    no_new_trajectory(monkeypatch)
    before = tree(fixture["root"])
    output = tmp_path / "published"
    target = output / archive.RESULTS
    target.mkdir(parents=True)
    (target / "protocol.json").write_bytes(fixture["protocol_path"].read_bytes())
    args = {"runs": fixture["runs"], "evaluation": fixture["output"],
            "protocol": fixture["protocol_path"], "output": output}
    manifest = archive.archive(**args)
    assert manifest["scale_gate"] == fixture["aggregate"]["scale_gate"]
    assert manifest["policy_seeds"] == [499] and len(manifest["files"]) == 18
    assert tree(fixture["root"]) == before
    for row in manifest["files"]:
        data = (output / row["path"]).read_bytes()
        assert archive.sha256(data) == row["sha256"] and len(data) == row["bytes"]
    assert archive.archive(**args) == manifest


@pytest.mark.parametrize("mutation", ["interval", "report_summary", "dataset"])
def test_real_tampered_evidence_refused_before_publication(ranking_evaluation_fixture, tmp_path, monkeypatch, mutation):
    fixture = ranking_evaluation_fixture
    fixture_bindings(monkeypatch, fixture["lineage"])
    no_new_trajectory(monkeypatch)
    evaluation, runs = tmp_path / "evaluation", dict(fixture["runs"])
    shutil.copytree(fixture["output"], evaluation)
    if mutation == "interval":
        path = evaluation / "aggregate.json"
        value = json.loads(path.read_bytes())
        next(iter(value["paired_stratified_differences"].values()))["timely_fraction"]["lower95"] += .02
        path.write_bytes(archive._canonical_bytes(value))
    elif mutation == "report_summary":
        path = evaluation / "validation-normal.json.gz"
        value = json.loads(gzip.decompress(path.read_bytes()))
        next(iter(value["methods"].values()))["summary"]["mean_cost"] += .1
        data = gzip.compress(archive._canonical_bytes(value), mtime=0)
        path.write_bytes(data)
        path = evaluation / "validation-normal.receipt.json"
        receipt = json.loads(path.read_bytes())
        receipt.update(sha256=archive.sha256(data), bytes=len(data))
        path.write_bytes(archive._canonical_bytes(receipt))
    else:
        runs[499] = tmp_path / "seed-499"
        shutil.copytree(fixture["runs"][499], runs[499])
        path = runs[499] / "comparisons.jsonl.gz"
        data = gzip.decompress(path.read_bytes()).splitlines()
        row = json.loads(data[0])
        row["outcomes"][0]["episode_return"] += 1.
        data[0] = archive._canonical_bytes(row).strip()
        path.write_bytes(gzip.compress(b"\n".join(data) + b"\n", mtime=0))
    output = tmp_path / "published"
    with pytest.raises(ValueError):
        archive.archive(runs=runs, evaluation=evaluation, protocol=fixture["protocol_path"], output=output)
    assert not output.exists()
