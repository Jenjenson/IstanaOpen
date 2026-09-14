"""Verify actual published balanced-v3 evidence separately from export fixtures."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from evaluate_balanced import _run_evidence, select_from_reports
from triad_rl.balanced_policy import BalancedPolicy


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Results" / "balanced-v3"
MANIFEST = RESULTS / "artifact-manifest.json"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="balanced-v3 has not been published yet")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_actual_balanced_files_sources_and_versioned_aliases():
    manifest = read(MANIFEST)
    assert manifest["schema"] == "triad.balanced_publication.v1"
    assert manifest["stage"] == "validation"
    assert manifest["independent_final_test_evidence"] is False
    assert manifest["default_policy_changed"] is False
    assert manifest["archived_run_seeds"] == [201, 202, 203]
    for artifact in manifest["files"]:
        path = (ROOT / artifact["path"]).resolve()
        assert path.is_relative_to(ROOT.resolve())
        data = path.read_bytes()
        assert len(data) == artifact["bytes"]
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
    for relative, digest in manifest["source_files_sha256"].items():
        assert hashlib.sha256((ROOT.parents[1] / relative).read_bytes()).hexdigest() == digest
    for role, alias, key in (("best", "balanced-v3-candidate", "selected_weights_sha256"),
                            ("initialized", "balanced-v3-candidate-initialized", "initialized_weights_sha256")):
        path = ROOT / "Checkpoints" / alias
        actor = BalancedPolicy.load(path)
        assert actor.weights_fingerprint() == manifest[key]
        for name in ("checkpoint.json", "arrays.npz"):
            assert (path / name).read_bytes() == (
                RESULTS / "training" / f"seed-{manifest['selected_seed']}" / role / name).read_bytes()
        if role == "initialized":
            assert actor.update_count == 0


def test_actual_complete_runs_and_all_common_validation_methods_recompute():
    protocol, selection = read(RESULTS / "protocol.json"), read(RESULTS / "selection.json")
    assert selection["schema"] == "triad.balanced_model_selection.v1"
    assert selection["final_test_accessed"] is False
    candidates = {}
    for seed in (201, 202, 203):
        path = RESULTS / "training" / f"seed-{seed}" / "best"
        actor = BalancedPolicy.load(path)
        evidence = _run_evidence(path, actor, seed, protocol)
        assert evidence["run_summary"]["completed_episodes"] == 8000
        assert evidence == selection["candidate_metadata"][str(seed)]["complete_run"]
        candidates[seed] = {"weights_sha256": actor.weights_fingerprint(),
                            "initialized_weights_sha256": evidence["initialized_weights_sha256"]}
    reports = {profile: json.loads(gzip.decompress((RESULTS / entry["path"]).read_bytes()))
               for profile, entry in selection["reports"].items()}
    selected, scores = select_from_reports(reports, candidates)
    assert selected == selection["selected"] and scores == selection["candidates"]
    for report in reports.values():
        assert report["seed_provenance"]["evaluation"]["count"] == 300
        assert len(report["methods"]) == 14
        assert all(len(method["episodes"]) == 300 for method in report["methods"].values())


def test_actual_paired_stop_probes_recompute_actual_actor_actions():
    spec = importlib.util.spec_from_file_location("verify_balanced_publisher", ROOT.parents[1] / "Tools" / "publish_balanced_rl.py")
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    manifest = read(MANIFEST)
    paths = [ROOT / "Checkpoints" / "robust-v2-candidate",
             ROOT / "Checkpoints" / "balanced-v3-candidate-initialized",
             ROOT / "Checkpoints" / "balanced-v3-candidate"]
    probes = []
    for index, path in enumerate(paths):
        record = read(RESULTS / f"stop-probe-{index + 1}.json")
        probe = publisher._validate_stop_probe((RESULTS / f"stop-probe-{index + 1}.json").read_bytes(), {
            "policy_kind": "adaptive" if index == 0 else "balanced", "path": path,
            "weights_sha256": record["weights_sha256"], "files": publisher.checkpoint_files(path)})
        assert len(probe["records"]) == 60
        probes.append(probe)
    assert [entry["role"] for entry in manifest["stop_probes"]] == [
        "frozen_v2", "balanced_initialized", "balanced_selected"]
    hashes = [[row["public_input_sha256"] for row in probe["records"]] for probe in probes]
    assert hashes[0] == hashes[1] == hashes[2]
