"""Verify the actual archived candidate and evidence, not export fixtures only."""
import gzip
import hashlib
import json
from pathlib import Path

from select_robust import _run_evidence, select_from_reports
from triad_rl.adaptive_policy import AdaptivePolicy


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Results" / "robust-v2"


def test_actual_robust_publication_files_sources_and_candidate():
    manifest = json.loads((RESULTS / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "triad.robust_publication.v1"
    assert manifest["stage"] == "validation"
    assert manifest["independent_final_test_evidence"] is False
    assert manifest["default_policy_changed"] is False
    assert manifest["archived_run_seeds"] == [101, 102, 103]
    for artifact in manifest["files"]:
        path = (ROOT / artifact["path"]).resolve()
        assert path.is_relative_to(ROOT.resolve())
        data = path.read_bytes()
        assert len(data) == artifact["bytes"]
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
    for relative, fingerprint in manifest["source_files_sha256"].items():
        assert hashlib.sha256((ROOT.parents[1] / relative).read_bytes()).hexdigest() == fingerprint
    candidate = AdaptivePolicy.load(ROOT / "Checkpoints" / "robust-v2-candidate")
    assert candidate.weights_fingerprint() == manifest["selected_weights_sha256"]
    initialized = AdaptivePolicy.load(ROOT / "Checkpoints" / "robust-v2-candidate-initialized")
    assert initialized.weights_fingerprint() == manifest["initialized_weights_sha256"]
    assert initialized.update_count == 0
    assert candidate.training_state["config"]["seed"] == manifest["selected_seed"]


def test_actual_complete_runs_and_common_validation_selection_recompute():
    protocol = json.loads((RESULTS / "protocol.json").read_text(encoding="utf-8"))
    selection = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
    assert selection["final_test_accessed"] is False
    candidates = {}
    for seed in (101, 102, 103):
        path = RESULTS / "training" / f"seed-{seed}" / "best"
        policy = AdaptivePolicy.load(path)
        evidence = _run_evidence(path, policy, seed, protocol)
        assert evidence["run_summary"]["completed_episodes"] == 8000
        assert evidence == selection["candidate_metadata"][str(seed)]["complete_run"]
        candidates[seed] = {"weights_sha256": policy.weights_fingerprint()}
    reports = {profile: json.loads(gzip.decompress((RESULTS / info["path"]).read_bytes()))
               for profile, info in selection["reports"].items()}
    selected, scores = select_from_reports(reports, candidates)
    assert selected == selection["selected"]
    assert scores == selection["candidates"]
    assert selected["seed"] == 103
    assert all(report["seed_provenance"]["evaluation"]["count"] == 300 for report in reports.values())


def test_actual_stop_probes_are_paired_public_validation_only():
    before, after = [json.loads((RESULTS / f"stop-probe-{index}.json").read_text(encoding="utf-8"))
                     for index in (1, 2)]
    assert before["weights_sha256"] != after["weights_sha256"]
    assert [row["public_input_sha256"] for row in before["records"]] == [
        row["public_input_sha256"] for row in after["records"]]
    assert len(before["records"]) == 60
    assert before["groups"]["coverage_le_0.01"]["count"] == 19
    assert after["groups"]["coverage_le_0.01"]["deterministic_stop_count"] == 0
