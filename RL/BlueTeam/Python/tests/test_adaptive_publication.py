"""Lossless, portable and provenance-checked adaptive artifact publication."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from triad_rl.adaptive_evaluation import evaluate_methods
from triad_rl.adaptive_policy import AdaptivePolicy
from triad_rl.train_adaptive import run_training


_SCRIPT = Path(__file__).resolve().parents[4] / "Tools" / "publish_adaptive_rl.py"
_SPEC = importlib.util.spec_from_file_location("adaptive_publisher", _SCRIPT)
publisher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(publisher)


@pytest.fixture
def evidence(tmp_path):
    run = tmp_path / "run"
    run_training(output=run, episodes=2, batch_size=2, seed=5, validation_every=2, validation_episodes=2)
    metadata = json.loads((run / "best/checkpoint.json").read_text(encoding="utf-8"))
    sources = tmp_path / "sources"
    sources.mkdir()
    package = _SCRIPT.parents[1] / "RL/BlueTeam/Python/triad_rl"
    for name in publisher.CORE_FILES:
        shutil.copyfile(package / name, sources / name)
    report = evaluate_methods({"adaptive": lambda seed: AdaptivePolicy.load(run / "best")},
                              episodes=2, seed=2_000_000_000_000_000, replay_count=0,
                              bootstrap_samples=10, seed_provenance=metadata["seed_provenance"])
    report["checkpoint"] = {"name": "best", "metadata": metadata}
    report["debug_source_path"] = str(run.resolve())
    report["unknown_path"] = "C:\\Users\\Example\\Private\\file.json"
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps(report), encoding="utf-8")
    stress_report = evaluate_methods({"adaptive": lambda seed: AdaptivePolicy.load(run / "best")},
                                    episodes=2, seed=2_000_000_000_010_000, split="stress", replay_count=0,
                                    bootstrap_samples=10, seed_provenance=metadata["seed_provenance"])
    stress_report["checkpoint"] = {"name": "best", "metadata": metadata}
    stress = tmp_path / "stress.json"
    stress.write_text(json.dumps(stress_report), encoding="utf-8")
    demo = tmp_path / "demo.html"
    demo_payload = json.dumps({"schema": "triad.adaptive_demo.v1", "weights_sha256": metadata["weights_sha256"], "replays": []})
    demo.write_text('<!doctype html><html><body><script id="replay-data" type="application/json">'
                    + demo_payload + '</script></body></html>', encoding="utf-8")
    return {"run_dir": run, "heldout": heldout, "stress": stress, "demo": demo,
            "output": tmp_path / "published", "source_root": sources}


def test_lossless_compressed_evidence_summary_and_checkpoints(evidence):
    result = publisher.publish(**evidence)
    root = evidence["output"]
    complete_bytes = gzip.decompress((root / "Results/adaptive-v1/heldout.json.gz").read_bytes())
    complete = json.loads(complete_bytes)
    summary = json.loads((root / "Results/adaptive-v1/heldout-summary.json").read_bytes())
    original = json.loads(evidence["heldout"].read_bytes())
    assert complete["methods"] == original["methods"]
    assert complete["paired_differences"] == original["paired_differences"]
    assert complete["seed_provenance"] == original["seed_provenance"]
    assert complete_bytes == publisher.canonical_json(complete)
    assert "episodes" not in summary["methods"]["adaptive"]
    assert summary["methods"]["adaptive"]["subgroups"] == complete["methods"]["adaptive"]["subgroups"]
    assert summary["detailed_evidence"]["canonical_json_sha256"] == hashlib.sha256(complete_bytes).hexdigest()
    assert b"C:\\" not in complete_bytes and b"Private" not in complete_bytes
    assert complete["debug_source_path"] == "RL/BlueTeam/Results/adaptive-v1/training/seed-5"
    assert complete["unknown_path"] == "<local-path-redacted>"
    assert (root / "Checkpoints/adaptive-v1/arrays.npz").read_bytes() == (evidence["run_dir"] / "best/arrays.npz").read_bytes()
    assert AdaptivePolicy.load(root / "Checkpoints/adaptive-v1").weights_fingerprint() == result["selected_weights_sha256"]
    for artifact in result["files"]:
        published = (root / artifact["path"]).read_bytes()
        assert hashlib.sha256(published).hexdigest() == artifact["sha256"]
        assert len(published) == artifact["bytes"]
    assert not (root / "source-manifest.json").exists()


def test_publication_idempotent_and_gzip_has_zero_mtime(evidence):
    first = publisher.publish(**evidence)
    before = {p.relative_to(evidence["output"]): p.read_bytes() for p in evidence["output"].rglob("*") if p.is_file()}
    assert publisher.publish(**evidence) == first
    after = {p.relative_to(evidence["output"]): p.read_bytes() for p in evidence["output"].rglob("*") if p.is_file()}
    assert before == after
    gz = (evidence["output"] / "Results/adaptive-v1/heldout.json.gz").read_bytes()
    assert gz[4:8] == b"\0\0\0\0" and gz[9] == 255


def test_conflicts_are_preflighted_before_any_write_and_replace_explicit(evidence):
    target = evidence["output"] / "Results/adaptive-v1/heldout-summary.json"
    target.parent.mkdir(parents=True)
    target.write_text("user-owned existing artifact", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--replace"):
        publisher.publish(**evidence)
    assert target.read_text() == "user-owned existing artifact"
    assert not (evidence["output"] / "Checkpoints").exists()
    publisher.publish(**evidence, replace=True)
    assert json.loads(target.read_bytes())["split"] == "heldout"


@pytest.mark.parametrize("problem", ["source", "arrays", "report_weights", "paired_scenario", "seed_overlap", "nonfinite"])
def test_invalid_provenance_rejected_before_writing(evidence, problem):
    if problem == "source":
        (evidence["source_root"] / "adaptive_env.py").write_text("modified physics", encoding="utf-8")
    elif problem == "arrays":
        (evidence["run_dir"] / "best/arrays.npz").write_bytes(b"corrupt")
    else:
        report = json.loads(evidence["heldout"].read_bytes())
        if problem == "report_weights":
            report["checkpoint"]["metadata"]["weights_sha256"] = "f" * 64
        elif problem == "paired_scenario":
            report["methods"]["adaptive"]["episodes"][0]["scenario"]["targets"][0]["speed"] += 1
        elif problem == "seed_overlap":
            report["seed_provenance"]["evaluation"] = {"start": 5_000_000, "count": 2}
        elif problem == "nonfinite":
            report["broken"] = float("nan")
        evidence["heldout"].write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        publisher.publish(**evidence)
    assert not evidence["output"].exists()


def test_common_validation_report_preserved_and_verified(evidence):
    run = evidence["run_dir"]
    metadata = json.loads((run / "best/checkpoint.json").read_bytes())
    report = {"schema": "triad.adaptive_model_selection.v1", "selection_split": "validation",
              "selected_seed": 5, "selection_rule": "success_rate then mean_return",
              "seed_provenance": {"validation": {"start": 1_000_000_999_000_000, "count": 300}},
              "runs": [{"seed": 5, "success_rate": .65, "mean_return": 7., "weights_sha256": metadata["weights_sha256"]}]}
    source = run.parent / "selection.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    publisher.publish(**evidence, selection_report=source)
    selection = json.loads((evidence["output"] / "Results/adaptive-v1/selection.json").read_bytes())
    assert selection["common_validation_evidence"] == report
    assert selection["heldout_used_for_selection"] is False
    raw = json.loads((evidence["output"] / "Results/adaptive-v1/common-validation-selection.json").read_bytes())
    assert raw == report and raw["schema"] == "triad.adaptive_model_selection.v1"
    report["selection_split"] = "heldout"
    source.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="validation split"):
        publisher.publish(**evidence, selection_report=source, replace=True)


def test_json_safety_rejects_duplicate_keys_and_preserves_nonpath_values():
    with pytest.raises(ValueError, match="Duplicate"):
        publisher.strict_json('{"x":1,"x":2}')
    value = {"score": 1.25, "hash": "a" * 64, "url": "https://example.com/result", "uri": "file:///C:/private"}
    safe = publisher.portable(value)
    assert safe["score"] == value["score"] and safe["hash"] == value["hash"] and safe["url"] == value["url"]
    assert safe["uri"] == "<local-path-redacted>"


def test_output_directory_collision_rejected_even_with_replace(evidence):
    collision = evidence["output"] / "Results/adaptive-v1/heldout-summary.json"
    collision.mkdir(parents=True)
    with pytest.raises(IsADirectoryError):
        publisher.publish(**evidence, replace=True)
    assert not (evidence["output"] / "Checkpoints").exists()


def test_optional_sensitivity_and_variability_evidence_are_preserved(evidence):
    metadata = json.loads((evidence["run_dir"] / "best/checkpoint.json").read_bytes())
    sensitivity = {"schema": "triad.adaptive_public_input_sensitivity.v1",
                   "weights_sha256": metadata["weights_sha256"],
                   "source_sha256": publisher.implementation_hashes(evidence["source_root"]),
                   "seed_provenance": {"validation": {"start": 1_000_000_998_000_000, "count": 40}},
                   "interpretation": "Recommendations only, not defence outcomes",
                   "training_performed": False, "final_test_accessed": False,
                   "variants": {"direction": {"position_change_rate": 1.}}}
    sensitivity_path = evidence["run_dir"].parent / "sensitivity.json"
    sensitivity_path.write_text(json.dumps(sensitivity), encoding="utf-8")
    variability = json.loads(evidence["heldout"].read_bytes())
    variability.update(selection_report={"selected": {"weights_sha256": metadata["weights_sha256"], "seed": 5}},
                       selected_training_seed_fixed_before_test=5, heldout_used_for_selection=False,
                       empirical_training_run_variability={"runs": 1, "success_rate_sample_std": None})
    variability_path = evidence["run_dir"].parent / "variability.json"
    variability_path.write_text(json.dumps(variability), encoding="utf-8")
    publisher.publish(**evidence, sensitivity=sensitivity_path, variability=variability_path)
    results = evidence["output"] / "Results/adaptive-v1"
    assert json.loads((results / "validation-sensitivity.json").read_bytes()) == sensitivity
    complete = json.loads(gzip.decompress((results / "training-run-variability.json.gz").read_bytes()))
    assert complete["methods"] == variability["methods"]
    summary = json.loads((results / "training-run-variability-summary.json").read_bytes())
    assert "episodes" not in summary["methods"]["adaptive"]
    assert summary["empirical_training_run_variability"] == variability["empirical_training_run_variability"]


def test_feasibility_audit_bound_to_exact_stress_report(evidence):
    stress = json.loads(evidence["stress"].read_bytes())
    audit = {"schema": "triad.adaptive_feasibility_audit.v1", "split": "stress",
             "training_performed": False, "new_independent_test": False, "private_truth_used": True,
             "source_report": {"sha256": hashlib.sha256(evidence["stress"].read_bytes()).hexdigest()},
             "implementation_sha256": stress["implementation_sha256"],
             "source_scenario_sequence_sha256": stress["scenario_sequence_sha256"],
             "interpretation": "Necessary-condition bound, not achievable defence",
             "scenario_reuse": "Post-hoc diagnostics, not a new independent test",
             "summary": {"episodes": 2},
             "episodes": [{"seed": e["seed"], "scenario_sha256": e["scenario_sha256"]}
                          for e in stress["methods"]["adaptive"]["episodes"]]}
    source = evidence["run_dir"].parent / "feasibility.json"
    source.write_text(json.dumps(audit), encoding="utf-8")
    publisher.publish(**evidence, feasibility=source)
    results = evidence["output"] / "Results/adaptive-v1"
    complete = json.loads(gzip.decompress((results / "stress-feasibility.json.gz").read_bytes()))
    summary = json.loads((results / "stress-feasibility-summary.json").read_bytes())
    assert complete["episodes"] == audit["episodes"]
    assert "episodes" not in summary and summary["summary"]["episodes"] == 2
    audit["source_report"]["sha256"] = "f" * 64
    source.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="source evaluation hash"):
        publisher.publish(**evidence, feasibility=source, replace=True)


def test_demo_cannot_silently_show_another_checkpoint(evidence):
    text = evidence["demo"].read_text(encoding="utf-8")
    metadata = json.loads((evidence["run_dir"] / "best/checkpoint.json").read_bytes())
    evidence["demo"].write_text(text.replace(metadata["weights_sha256"], "f" * 64), encoding="utf-8")
    with pytest.raises(ValueError, match="Demo replay checkpoint"):
        publisher.publish(**evidence)
    assert not evidence["output"].exists()
