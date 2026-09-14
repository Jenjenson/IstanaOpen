"""Validate the actual published adaptive evidence, not only export fixtures."""
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from triad_rl.adaptive_evaluation import summarize
from triad_rl.adaptive_policy import AdaptivePolicy


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Results" / "adaptive-v1"


def test_published_artifact_checksums_and_frozen_training_sources():
    manifest = json.loads((RESULTS / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "triad.adaptive_publication.v1"
    for artifact in manifest["files"]:
        path = (ROOT / artifact["path"]).resolve()
        assert path.is_relative_to(ROOT.resolve())
        data = path.read_bytes()
        assert len(data) == artifact["bytes"]
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
    for name, fingerprint in manifest["training_implementation_sha256"].items():
        assert hashlib.sha256((ROOT / "Python" / "triad_rl" / name).read_bytes()).hexdigest() == fingerprint
    policy = AdaptivePolicy.load(ROOT / "Checkpoints" / "adaptive-v1")
    assert policy.weights_fingerprint() == manifest["selected_weights_sha256"]
    assert policy.training_state["completed_episodes"] == 4000
    initialized = AdaptivePolicy.load(ROOT / "Checkpoints" / "adaptive-v1-initialized")
    assert initialized.update_count == 0
    assert initialized.feature_names == policy.feature_names
    assert initialized.weights_fingerprint() != policy.weights_fingerprint()


@pytest.mark.parametrize("split", ["heldout", "stress"])
def test_published_report_summaries_match_all_paired_episodes(split):
    report = json.loads(gzip.decompress((RESULTS / f"{split}.json.gz").read_bytes()))
    summary = json.loads((RESULTS / f"{split}-summary.json").read_text(encoding="utf-8"))
    assert report["split"] == split
    assert report["training_performed"] is False
    assert len(report["methods"]) == 6
    reference = report["methods"]["adaptive"]["episodes"]
    assert len(reference) == 200
    for name, method in report["methods"].items():
        assert [r["scenario_sha256"] for r in method["episodes"]] == [r["scenario_sha256"] for r in reference]
        recomputed = summarize(method["episodes"])
        assert recomputed == method["summary"] == summary["methods"][name]["summary"]
    assert summary["paired_differences"] == report["paired_differences"]
    assert report["methods"]["adaptive"]["summary"]["mean_invalid_actions"] == 0


def test_selection_and_three_training_runs_are_preserved():
    selection = json.loads((RESULTS / "common-validation-selection.json").read_text(encoding="utf-8"))
    assert selection["selection_split"] == "validation"
    assert selection["seed_provenance"]["validation"]["count"] == 300
    assert selection["selected_seed"] == 43
    assert {row["seed"] for row in selection["runs"]} == {42, 43, 44}
    for seed in (42, 43, 44):
        summary = json.loads((RESULTS / "training" / f"seed-{seed}" / "summary.json").read_text(encoding="utf-8"))
        assert summary["completed_episodes"] == 4000
