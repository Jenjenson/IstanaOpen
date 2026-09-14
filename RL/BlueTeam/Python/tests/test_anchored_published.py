"""Verify terminal published anchored evidence; never resample or rewrite it."""
import builtins
import importlib.util
import io
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "Tools"))
spec = importlib.util.spec_from_file_location("archive_anchored_pilot_published", REPO_ROOT / "Tools/archive_anchored_pilot.py")
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def test_published_anchored_bundle_recomputes_exactly_without_sampling_or_writes(monkeypatch):
    evaluator = archive.evaluator
    results = archive.BLUE_ROOT / archive.RESULTS
    manifest_path = results / "artifact-manifest.json"
    # No skip: once this publication test is enabled, missing evidence is a failure.
    assert manifest_path.is_file(), "The complete anchored publication must exist before running this test"
    manifest = archive._json_object(manifest_path.read_bytes())
    expected = {f"{archive.RESULTS}/protocol.json", *(
        f"{archive.RESULTS}/evaluation/{name}" for name in evaluator.EVALUATION_FILES), *(
        f"{archive.RESULTS}/training/{arm}/{name}" for arm in evaluator.ARMS for name in evaluator.trainer.RUN_FILES)}
    assert manifest["schema"] == "triad.anchored_pilot_publication.v1"
    assert manifest["experiment"] == "anchored-v4-pilot" and manifest["stage"] == "validation"
    assert manifest["default_policy_changed"] is False and manifest["final_test_accessed"] is False
    assert len(manifest["files"]) == len(expected) == 33
    assert {row["path"] for row in manifest["files"]} == expected
    assert manifest["evaluation_implementation_sha256"] == evaluator.source_provenance()
    assert set(manifest["archive_implementation_sha256"]) == set(archive.ARCHIVE_SOURCES)
    for name, digest in manifest["archive_implementation_sha256"].items():
        assert archive.sha256((REPO_ROOT / name).read_bytes()) == digest
    before = {manifest_path: archive.sha256(manifest_path.read_bytes())}
    for row in manifest["files"]:
        path = archive.BLUE_ROOT / row["path"]
        data = path.read_bytes()
        assert len(data) == row["bytes"] and archive.sha256(data) == row["sha256"]
        before[path] = row["sha256"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Published verification must not train, infer, sample, or write")

    # Forbid the runners themselves, not just a missing-profile fallback.
    for owner, names in ((evaluator, ("run_evaluation", "evaluate_robust_methods", "_new")),
                         (evaluator.trainer, ("run_training", "rollout")),
                         (evaluator.trainer.shared, ("run_training",)),
                         (evaluator.trainer.anchored_credit, ("rollout", "update"))):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    for kind in (evaluator.BalancedPolicy, evaluator.AdaptivePolicy):
        for name in ("act", "sample", "update"):
            monkeypatch.setattr(kind, name, forbidden)
    for name in ("__init__", "reset", "step"):
        monkeypatch.setattr(evaluator.trainer.RobustPlacementEnv, name, forbidden)

    def read_only(open_function):
        def guarded(file, mode="r", *args, **kwargs):
            if any(flag in mode for flag in "wax+"):
                forbidden()
            return open_function(file, mode, *args, **kwargs)
        return guarded
    monkeypatch.setattr(Path, "open", read_only(Path.open))
    monkeypatch.setattr(builtins, "open", read_only(builtins.open))
    monkeypatch.setattr(io, "open", read_only(io.open))
    for name in ("write_bytes", "write_text", "mkdir", "unlink", "rename", "replace"):
        monkeypatch.setattr(Path, name, forbidden)

    # Idempotent archive verification must find every existing byte unchanged;
    # any attempted creation, overwrite, normalization or promotion now fails.
    regenerated_manifest = archive.archive(
        runs={arm: results / "training" / arm for arm in evaluator.ARMS},
        evaluation=results / "evaluation", protocol=results / "protocol.json", output=archive.BLUE_ROOT)
    assert regenerated_manifest == manifest

    protocol = archive._json_object((results / "protocol.json").read_bytes())
    reports = {profile: archive._gzip_json((results / "evaluation" / f"validation-{profile}.json.gz").read_bytes())
               for profile in evaluator.PROFILES}
    raw_aggregate = evaluator.aggregate_reports(reports, protocol)
    stored = archive._json_object((results / "evaluation/aggregate.json").read_bytes())
    assert evaluator._same({key: stored[key] for key in raw_aggregate}, raw_aggregate)
    assert raw_aggregate["scale_gate"] == manifest["scale_gate"]
    assert raw_aggregate["final_test_accessed"] is False
    assert set(raw_aggregate["methods"]) == set(evaluator.METHODS) and len(evaluator.METHODS) == 8
    assert set(raw_aggregate["scale_gate"]["comparators"]) == {"no_critic_gradient", "balanced_v3"}
    assert raw_aggregate["scale_gate"]["not_policy_promotion"] is True
    assert all(archive.sha256(path.read_bytes()) == digest for path, digest in before.items())
