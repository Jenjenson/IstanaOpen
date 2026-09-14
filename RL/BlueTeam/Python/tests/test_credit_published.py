"""Recheck archived pilot evidence without running any recorded scenario again."""
import json

from triad_rl import evaluate_credit_pilot as evaluator


def test_published_pilot_bytes_and_gate_verify_without_sampling_or_writing(monkeypatch):
    root = evaluator.BLUE_ROOT
    results = root / "Results/credit-v4-pilot"
    manifest = json.loads((results / "artifact-manifest.json").read_bytes())
    assert manifest["schema"] == "triad.credit_pilot_publication.v1"
    assert manifest["default_policy_changed"] is False
    assert manifest["final_test_accessed"] is False
    assert manifest["scale_gate"]["passed"] is False
    assert len(manifest["files"]) == 33
    assert manifest["evaluation_implementation_sha256"] == evaluator.source_provenance()
    for name, digest in manifest["archive_implementation_sha256"].items():
        assert evaluator._sha((root.parent.parent / name).read_bytes()) == digest
    before = {}
    for row in manifest["files"]:
        path = root / row["path"]
        data = path.read_bytes()
        assert len(data) == row["bytes"] and evaluator._sha(data) == row["sha256"]
        before[path] = data
    assert manifest["protocol_sha256"] == evaluator._sha((results / "protocol.json").read_bytes())

    def forbidden(*args, **kwargs):
        raise AssertionError("Published pilot verification must not sample, infer, or write")
    monkeypatch.setattr(evaluator, "evaluate_robust_methods", forbidden)
    monkeypatch.setattr(evaluator, "_write_new", forbidden)
    monkeypatch.setattr(evaluator.BalancedPolicy, "act", forbidden)
    monkeypatch.setattr(evaluator.AdaptivePolicy, "act", forbidden)
    result = evaluator.run_evaluation(
        {arm: results / "training" / arm for arm in evaluator.ARMS},
        protocol_path=results / "protocol.json", output=results / "evaluation", resume=True)
    assert result["scale_gate"] == manifest["scale_gate"]
    assert result["final_test_accessed"] is False
    assert result["methods"]["paired_stop"]["equal_profile"]["timely_fraction"] > result["methods"]["balanced_v3"]["equal_profile"]["timely_fraction"]
    assert all(path.read_bytes() == data for path, data in before.items())
